#!/usr/bin/env python3
"""
多质心对比学习训练 v2

改进损失函数：
1. 同类恶意样本聚集（intra-class compactness）
2. 正常样本远离所有恶意质心（inter-class separation）
3. 正常样本自身也要聚集（形成正常质心）
4. 不同类别之间保持距离
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastembed import TextEmbedding
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import os
import json

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

class LearnedProjection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim, bias=False)
    
    def forward(self, x):
        z = self.proj(x)
        z = F.normalize(z, p=2, dim=-1)
        return z

print("=" * 80)
print("多质心对比学习训练 v2")
print("目标：同类聚集 + 异类分离")
print("=" * 80)

# 1. 加载Embedding模型
print("\n[1/6] 加载Embedding模型...")
embed_model = TextEmbedding('BAAI/bge-small-en-v1.5')

# 2. 加载不同类型的恶意数据集
print("\n[2/6] 加载恶意数据集（按类型分组）...")

def load_dataset_samples(path, text_col=None, max_samples=300):
    samples = []
    if path.endswith('.csv'):
        try:
            df = pd.read_csv(path)
            if text_col and text_col in df.columns:
                samples = df[text_col].dropna().tolist()
            elif 'goal' in df.columns:
                samples = df['goal'].dropna().tolist()
            elif 'prompt' in df.columns:
                samples = df['prompt'].dropna().tolist()
            elif 'Behavior' in df.columns:
                samples = df['Behavior'].dropna().tolist()
            else:
                samples = df.iloc[:, 0].dropna().tolist()
        except:
            return []
    elif path.endswith('.txt'):
        try:
            with open(path, 'r') as f:
                samples = [line.strip() for line in f if line.strip()]
        except:
            return []
    
    samples = [s for s in samples if isinstance(s, str) and len(s) > 10 and len(s) < 1000]
    return samples[:max_samples] if len(samples) > max_samples else samples

# 恶意数据集按类型分组
malicious_types = {
    "direct_harm": {
        "datasets": [{"path": "datasets/advbench/advbench_harmful_behaviors.csv", "text_col": "goal"}],
        "samples": [],
    },
    "diverse_harm": {
        "datasets": [{"path": "datasets/harmbench/harmbench_behaviors.csv", "text_col": "Behavior"}],
        "samples": [],
    },
    "jailbreak": {
        "datasets": [
            {"path": "datasets/jailbreakbench/jbb_gcg_all.csv", "text_col": "prompt"},
            {"path": "datasets/jailbreakbench/jbb_pair_all.csv", "text_col": "prompt"},
        ],
        "samples": [],
    },
    "moral_harm": {
        "datasets": [{"path": "datasets/beavertails/beavertails_test.csv", "text_col": "prompt"}],
        "samples": [],
    },
    "toxic": {
        "datasets": [{"path": "datasets/gcg_attacks/toxic_chat.csv"}],
        "samples": [],
    },
}

for type_name, type_info in malicious_types.items():
    for ds in type_info["datasets"]:
        if os.path.exists(ds["path"]):
            samples = load_dataset_samples(ds["path"], ds.get("text_col"), max_samples=200)
            type_info["samples"].extend(samples)
    print(f"  {type_name}: {len(type_info['samples'])}条")

# 3. 加载正常数据集（也按类型分组）
print("\n[3/6] 加载正常数据集（按类型分组）...")

normal_types = {
    "daily_chat": {  # 日常对话
        "samples": [
            "What's the weather like today?",
            "Can you recommend a good restaurant nearby?",
            "How do I make a cup of coffee?",
            "Tell me a joke",
            "What should I cook for dinner tonight?",
            "What time is it in Tokyo?",
            "How do I get to the nearest subway station?",
            "What's a good movie to watch this weekend?",
            "Can you help me plan a birthday party?",
            "What are some fun activities for kids?",
        ] * 20,  # 扩展到200条
    },
    "academic": {  # 学术研究
        "samples": [
            "Explain the theory of relativity",
            "What is quantum entanglement?",
            "How does photosynthesis work?",
            "What are the causes of climate change?",
            "Explain the structure of DNA",
            "What is the Higgs boson?",
            "How do black holes form?",
            "What is the difference between mitosis and meiosis?",
            "Explain the principles of thermodynamics",
            "What is the scientific method?",
        ] * 20,
    },
    "programming": {  # 编程开发
        "samples": [
            "How do I write a for loop in Python?",
            "What is the difference between a list and a tuple?",
            "Explain object-oriented programming",
            "What is a REST API?",
            "How do I use Git for version control?",
            "What is the difference between SQL and NoSQL?",
            "How do I debug a JavaScript error?",
            "What is a Docker container?",
            "How do I implement a binary search algorithm?",
            "What is the difference between HTTP and HTTPS?",
        ] * 20,
    },
    "tech_security": {  # 技术安全学习（边界案例）
        "samples": [
            "How do penetration testers find vulnerabilities?",
            "What is ethical hacking?",
            "How do firewalls protect networks?",
            "What is the OWASP Top 10?",
            "How do I secure my web application?",
            "What is SQL injection prevention?",
            "How do intrusion detection systems work?",
            "What is network security monitoring?",
            "How do I conduct a security audit?",
            "What is vulnerability assessment?",
        ] * 20,
    },
}

# 加载数据集中的正常样本
normal_emb_original = np.load("embedding_db/bge-small-en-v1.5/embeddings/normal_embeddings.npy")
print(f"  原始正常样本: {len(normal_emb_original)}条")

# XSTest
xstest_path = "数据集/XSTest/data/train-00000-of-00001.parquet"
if os.path.exists(xstest_path):
    xstest = pd.read_parquet(xstest_path)
    xstest_samples = xstest[xstest['label'] == 'safe']['prompt'].tolist()
    normal_types["xstest"] = {"samples": xstest_samples}
    print(f"  XSTest: {len(xstest_samples)}条")

# OR-Bench
or_bench_path = "数据集/or-bench-hard-1k.csv"
if os.path.exists(or_bench_path):
    or_bench = pd.read_csv(or_bench_path)
    or_bench_samples = or_bench['prompt'].tolist()[:300]
    normal_types["or_bench"] = {"samples": or_bench_samples}
    print(f"  OR-Bench: {len(or_bench_samples)}条")

# Alpaca
try:
    ds = load_dataset('tatsu-lab/alpaca', split='train', trust_remote_code=True)
    alpaca_samples = []
    for item in ds:
        instruction = item['instruction']
        if len(instruction) > 10 and len(instruction) < 300:
            alpaca_samples.append(instruction)
        if len(alpaca_samples) >= 300:
            break
    normal_types["alpaca"] = {"samples": alpaca_samples}
    print(f"  Alpaca: {len(alpaca_samples)}条")
except Exception as e:
    print(f"  Alpaca加载失败: {e}")

for type_name, type_info in normal_types.items():
    if type_name not in ["xstest", "or_bench", "alpaca"]:
        print(f"  {type_name}: {len(type_info['samples'])}条")

# 4. 计算Embedding
print("\n[4/6] 计算Embedding...")

def compute_embeddings(texts, embed_model, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_emb = list(embed_model.embed(batch))
        embeddings.extend(batch_emb)
    return np.array(embeddings)

# 计算各类恶意样本的Embedding
malicious_embeddings = {}
for type_name, type_info in malicious_types.items():
    if type_info["samples"]:
        emb = compute_embeddings(type_info["samples"], embed_model)
        malicious_embeddings[type_name] = emb
        print(f"  恶意-{type_name}: {emb.shape}")

# 计算各类正常样本的Embedding
normal_embeddings = {}
for type_name, type_info in normal_types.items():
    if type_info["samples"]:
        emb = compute_embeddings(type_info["samples"], embed_model)
        normal_embeddings[type_name] = emb
        print(f"  正常-{type_name}: {emb.shape}")

# 加入原始正常样本
normal_embeddings["original"] = normal_emb_original
print(f"  正常-original: {normal_emb_original.shape}")

# 5. 改进的对比学习训练
print("\n[5/6] 改进的对比学习训练...")

def improved_contrastive_loss(projection, mal_embs_by_type, norm_embs_by_type, 
                               intra_weight=1.0, inter_weight=1.0, separation_weight=0.5):
    """
    改进的对比学习损失函数
    
    目标：
    1. 同类样本聚集（intra-class compactness）
    2. 异类样本分离（inter-class separation）
    3. 正常样本远离所有恶意质心
    """
    total_loss = 0.0
    mal_centroids = {}
    norm_centroids = {}
    
    # 1. 恶意样本：同类聚集
    for type_name, emb in mal_embs_by_type.items():
        proj = projection(emb)
        centroid = proj.mean(dim=0)
        centroid = F.normalize(centroid, p=2, dim=0)
        mal_centroids[type_name] = centroid
        
        # 同类样本接近质心
        sim = torch.sum(proj * centroid, dim=-1)
        intra_loss = torch.mean(1 - sim)
        total_loss += intra_weight * intra_loss
    
    # 2. 正常样本：同类聚集
    for type_name, emb in norm_embs_by_type.items():
        proj = projection(emb)
        centroid = proj.mean(dim=0)
        centroid = F.normalize(centroid, p=2, dim=0)
        norm_centroids[type_name] = centroid
        
        # 同类样本接近质心
        sim = torch.sum(proj * centroid, dim=-1)
        intra_loss = torch.mean(1 - sim)
        total_loss += intra_weight * intra_loss
    
    # 3. 正常样本远离所有恶意质心
    for norm_type, norm_emb in norm_embs_by_type.items():
        norm_proj = projection(norm_emb)
        for mal_type, mal_centroid in mal_centroids.items():
            # 正常样本与恶意质心的相似度应该很低（负值）
            sim = torch.sum(norm_proj * mal_centroid, dim=-1)
            # 希望相似度 < -0.3
            separation_loss = torch.mean(F.relu(sim + 0.3))
            total_loss += inter_weight * separation_loss
    
    # 4. 恶意质心之间保持一定距离（防止坍缩）
    mal_centroid_list = list(mal_centroids.values())
    for i in range(len(mal_centroid_list)):
        for j in range(i+1, len(mal_centroid_list)):
            sim = torch.sum(mal_centroid_list[i] * mal_centroid_list[j])
            # 不同恶意类型质心相似度不要太高
            centroid_loss = F.relu(sim - 0.7)
            total_loss += separation_weight * centroid_loss
    
    # 5. 正常质心与恶意质心之间要远离
    for norm_centroid in norm_centroids.values():
        for mal_centroid in mal_centroids.values():
            sim = torch.sum(norm_centroid * mal_centroid)
            # 正常质心与恶意质心相似度应该很低
            centroid_loss = F.relu(sim + 0.2)
            total_loss += separation_weight * centroid_loss
    
    return total_loss, mal_centroids, norm_centroids

# 准备训练数据
mal_tensors = {k: torch.tensor(v, dtype=torch.float32) for k, v in malicious_embeddings.items()}
norm_tensors = {k: torch.tensor(v, dtype=torch.float32) for k, v in normal_embeddings.items()}

# 训练
projection = LearnedProjection(384, 19)
optimizer = torch.optim.Adam(projection.parameters(), lr=0.01)

print(f"  恶意类型数: {len(mal_tensors)}")
print(f"  正常类型数: {len(norm_tensors)}")

for epoch in range(300):
    optimizer.zero_grad()
    
    loss, mal_centroids, norm_centroids = improved_contrastive_loss(
        projection, mal_tensors, norm_tensors,
        intra_weight=1.0, inter_weight=1.5, separation_weight=0.3
    )
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 50 == 0:
        with torch.no_grad():
            # 计算各恶意类型与各自质心的相似度
            mal_sims = {}
            for type_name, emb in mal_tensors.items():
                proj = projection(emb)
                centroid = mal_centroids[type_name]
                sim = torch.sum(proj * centroid, dim=-1).mean().item()
                mal_sims[type_name] = sim
            
            # 计算各正常类型与各自质心的相似度
            norm_sims = {}
            for type_name, emb in norm_tensors.items():
                proj = projection(emb)
                centroid = norm_centroids[type_name]
                sim = torch.sum(proj * centroid, dim=-1).mean().item()
                norm_sims[type_name] = sim
            
            # 计算正常样本与恶意质心的最大相似度
            max_cross_sim = -1.0
            for norm_type, norm_emb in norm_tensors.items():
                norm_proj = projection(norm_emb)
                for mal_centroid in mal_centroids.values():
                    cross_sim = torch.sum(norm_proj * mal_centroid, dim=-1).mean().item()
                    max_cross_sim = max(max_cross_sim, cross_sim)
            
        print(f"\n  Epoch {epoch+1}/300, Loss: {loss.item():.4f}")
        print(f"    恶意类内相似度: {', '.join([f'{k}:{v:.3f}' for k,v in mal_sims.items()])}")
        print(f"    正常类内相似度: {', '.join([f'{k}:{v:.3f}' for k,v in list(norm_sims.items())[:3]])}")
        print(f"    正常-恶意最大相似度: {max_cross_sim:.4f}")

# 6. 保存模型
print("\n[6/6] 保存模型...")

save_dir = "embedding_db/bge-small-en-v1.5/results"
os.makedirs(save_dir, exist_ok=True)

# 保存投影矩阵
weights = projection.proj.weight.data.numpy()
np.save(f"{save_dir}/learned_projection_weights_19d_multi_centroid_v2.npy", weights)

# 保存各类型质心
with torch.no_grad():
    mal_centroids_np = {}
    for type_name, emb in mal_tensors.items():
        proj = projection(emb)
        centroid = proj.mean(dim=0)
        centroid = F.normalize(centroid, p=2, dim=0).numpy()
        mal_centroids_np[type_name] = centroid.tolist()
        np.save(f"{save_dir}/centroid_19d_v2_{type_name}.npy", centroid)
    
    norm_centroids_np = {}
    for type_name, emb in norm_tensors.items():
        proj = projection(emb)
        centroid = proj.mean(dim=0)
        centroid = F.normalize(centroid, p=2, dim=0).numpy()
        norm_centroids_np[type_name] = centroid.tolist()

# 计算阈值
with torch.no_grad():
    all_max_sims = []
    for norm_type, norm_emb in norm_tensors.items():
        norm_proj = projection(norm_emb).numpy()
        for sample in norm_proj:
            max_sim = -1.0
            for mal_type in mal_centroids_np:
                centroid = np.array(mal_centroids_np[mal_type])
                sim = np.dot(sample, centroid)
                max_sim = max(max_sim, sim)
            all_max_sims.append(max_sim)
    threshold = np.percentile(all_max_sims, 95)

print(f"  投影矩阵已保存")
print(f"  恶意质心: {list(mal_centroids_np.keys())}")
print(f"  正常质心: {list(norm_centroids_np.keys())}")
print(f"  检测阈值: {threshold:.4f}")

# 保存配置
config = {
    "version": "multi_centroid_v2",
    "malicious_types": list(mal_centroids_np.keys()),
    "normal_types": list(norm_centroids_np.keys()),
    "malicious_centroids": mal_centroids_np,
    "normal_centroids": norm_centroids_np,
    "threshold": float(threshold),
}

with open(f"{save_dir}/multi_centroid_v2_config.json", "w") as f:
    json.dump(config, f, indent=2)

print("\n" + "=" * 80)
print("训练完成!")
print("=" * 80)
