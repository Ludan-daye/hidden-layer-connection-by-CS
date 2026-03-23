#!/usr/bin/env python3
"""
多质心对比学习训练

改进：学习不同恶意类型的位置关系
- 每种恶意类型有自己的质心
- 投影空间中：
  - 同类恶意样本聚集在各自质心附近
  - 不同类恶意样本之间保持一定距离
  - 正常样本远离所有恶意质心
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
print("多质心对比学习训练")
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
    "direct_harm": {  # 直接恶意指令
        "datasets": [
            {"path": "datasets/advbench/advbench_harmful_behaviors.csv", "text_col": "goal"},
        ],
        "samples": [],
    },
    "diverse_harm": {  # 多样化恶意
        "datasets": [
            {"path": "datasets/harmbench/harmbench_behaviors.csv", "text_col": "Behavior"},
        ],
        "samples": [],
    },
    "jailbreak": {  # 越狱攻击
        "datasets": [
            {"path": "datasets/jailbreakbench/jbb_gcg_all.csv", "text_col": "prompt"},
            {"path": "datasets/jailbreakbench/jbb_pair_all.csv", "text_col": "prompt"},
        ],
        "samples": [],
    },
    "moral_harm": {  # 道德伤害
        "datasets": [
            {"path": "datasets/beavertails/beavertails_test.csv", "text_col": "prompt"},
        ],
        "samples": [],
    },
    "toxic": {  # 有毒对话
        "datasets": [
            {"path": "datasets/gcg_attacks/toxic_chat.csv"},
        ],
        "samples": [],
    },
}

for type_name, type_info in malicious_types.items():
    for ds in type_info["datasets"]:
        if os.path.exists(ds["path"]):
            samples = load_dataset_samples(ds["path"], ds.get("text_col"), max_samples=200)
            type_info["samples"].extend(samples)
    print(f"  {type_name}: {len(type_info['samples'])}条")

# 3. 加载正常数据集
print("\n[3/6] 加载正常数据集...")

# 原始正常样本
normal_emb = np.load("embedding_db/bge-small-en-v1.5/embeddings/normal_embeddings.npy")
print(f"  原始正常样本: {len(normal_emb)}条")

# XSTest
xstest_samples = []
xstest_path = "数据集/XSTest/data/train-00000-of-00001.parquet"
if os.path.exists(xstest_path):
    xstest = pd.read_parquet(xstest_path)
    xstest_samples = xstest[xstest['label'] == 'safe']['prompt'].tolist()
    print(f"  XSTest: {len(xstest_samples)}条")

# OR-Bench
or_bench_samples = []
or_bench_path = "数据集/or-bench-hard-1k.csv"
if os.path.exists(or_bench_path):
    or_bench = pd.read_csv(or_bench_path)
    or_bench_samples = or_bench['prompt'].tolist()[:500]
    print(f"  OR-Bench: {len(or_bench_samples)}条")

# Alpaca (HuggingFace)
alpaca_samples = []
try:
    ds = load_dataset('tatsu-lab/alpaca', split='train', trust_remote_code=True)
    for item in ds:
        instruction = item['instruction']
        if len(instruction) > 10 and len(instruction) < 300:
            alpaca_samples.append(instruction)
        if len(alpaca_samples) >= 500:
            break
    print(f"  Alpaca: {len(alpaca_samples)}条")
except Exception as e:
    print(f"  Alpaca加载失败: {e}")

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
        print(f"  {type_name}: {emb.shape}")

# 计算正常样本的Embedding
all_normal_samples = xstest_samples + or_bench_samples + alpaca_samples
if all_normal_samples:
    new_normal_emb = compute_embeddings(all_normal_samples, embed_model)
    all_normal_emb = np.vstack([normal_emb, new_normal_emb])
else:
    all_normal_emb = normal_emb
print(f"  正常样本总计: {all_normal_emb.shape}")

# 5. 多质心对比学习训练
print("\n[5/6] 多质心对比学习训练...")

def multi_centroid_contrastive_loss(projection, mal_embs_by_type, norm_emb, margin=0.3, inter_margin=0.5):
    """
    多质心对比学习损失函数
    
    Args:
        projection: 投影模型
        mal_embs_by_type: 按类型分组的恶意样本Embedding字典
        norm_emb: 正常样本Embedding
        margin: 正常样本与质心的最小距离
        inter_margin: 不同恶意类型质心之间的最小距离
    """
    total_loss = 0.0
    centroids = {}
    
    # 1. 计算每种恶意类型的质心
    for type_name, emb in mal_embs_by_type.items():
        proj = projection(emb)
        centroid = proj.mean(dim=0)
        centroid = F.normalize(centroid, p=2, dim=0)
        centroids[type_name] = centroid
        
        # 同类样本聚集损失：同类样本接近各自质心
        sim = torch.sum(proj * centroid, dim=-1)
        intra_loss = torch.mean(1 - sim)
        total_loss += intra_loss
    
    # 2. 正常样本远离所有恶意质心
    norm_proj = projection(norm_emb)
    for type_name, centroid in centroids.items():
        norm_sim = torch.sum(norm_proj * centroid, dim=-1)
        norm_loss = torch.mean(F.relu(norm_sim + margin))  # 希望相似度 < -margin
        total_loss += norm_loss
    
    # 3. 不同恶意类型质心之间保持距离（可选，防止坍缩）
    centroid_list = list(centroids.values())
    for i in range(len(centroid_list)):
        for j in range(i+1, len(centroid_list)):
            inter_sim = torch.sum(centroid_list[i] * centroid_list[j])
            # 不同类型质心相似度不要太高
            inter_loss = F.relu(inter_sim - inter_margin)
            total_loss += 0.1 * inter_loss
    
    return total_loss, centroids

# 准备训练数据
mal_tensors = {k: torch.tensor(v, dtype=torch.float32) for k, v in malicious_embeddings.items()}
norm_tensor = torch.tensor(all_normal_emb, dtype=torch.float32)

# 训练
projection = LearnedProjection(384, 19)
optimizer = torch.optim.Adam(projection.parameters(), lr=0.01)

print(f"  恶意类型数: {len(mal_tensors)}")
print(f"  正常样本数: {len(norm_tensor)}")

for epoch in range(200):
    optimizer.zero_grad()
    
    loss, centroids = multi_centroid_contrastive_loss(
        projection, mal_tensors, norm_tensor, 
        margin=0.3, inter_margin=0.5
    )
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 40 == 0:
        with torch.no_grad():
            # 计算各类型与各自质心的相似度
            type_sims = {}
            for type_name, emb in mal_tensors.items():
                proj = projection(emb)
                centroid = centroids[type_name]
                sim = torch.sum(proj * centroid, dim=-1).mean().item()
                type_sims[type_name] = sim
            
            # 计算正常样本与所有质心的最大相似度
            norm_proj = projection(norm_tensor)
            max_norm_sim = -1.0
            for centroid in centroids.values():
                norm_sim = torch.sum(norm_proj * centroid, dim=-1).mean().item()
                max_norm_sim = max(max_norm_sim, norm_sim)
            
        print(f"  Epoch {epoch+1}/200, Loss: {loss.item():.4f}")
        for type_name, sim in type_sims.items():
            print(f"    {type_name}: {sim:.4f}")
        print(f"    正常样本最大相似度: {max_norm_sim:.4f}")

# 6. 保存模型
print("\n[6/6] 保存模型...")

save_dir = "embedding_db/bge-small-en-v1.5/results"
os.makedirs(save_dir, exist_ok=True)

# 保存投影矩阵
weights = projection.proj.weight.data.numpy()
np.save(f"{save_dir}/learned_projection_weights_19d_multi_centroid.npy", weights)

# 保存各类型质心
with torch.no_grad():
    centroids_np = {}
    for type_name, emb in mal_tensors.items():
        proj = projection(emb)
        centroid = proj.mean(dim=0)
        centroid = F.normalize(centroid, p=2, dim=0).numpy()
        centroids_np[type_name] = centroid.tolist()
        np.save(f"{save_dir}/centroid_19d_{type_name}.npy", centroid)

# 计算阈值（使用正常样本的95%分位）
with torch.no_grad():
    norm_proj = projection(norm_tensor).numpy()
    max_sims = []
    for sample in norm_proj:
        max_sim = -1.0
        for type_name in centroids_np:
            centroid = np.array(centroids_np[type_name])
            sim = np.dot(sample, centroid)
            max_sim = max(max_sim, sim)
        max_sims.append(max_sim)
    threshold = np.percentile(max_sims, 95)

print(f"  投影矩阵已保存")
print(f"  质心已保存: {list(centroids_np.keys())}")
print(f"  检测阈值: {threshold:.4f}")

# 保存配置
config = {
    "version": "multi_centroid_v1",
    "malicious_types": list(centroids_np.keys()),
    "centroids": centroids_np,
    "threshold": float(threshold),
    "train_samples": {
        type_name: len(emb) for type_name, emb in malicious_embeddings.items()
    },
    "normal_samples": len(all_normal_emb),
}

with open(f"{save_dir}/multi_centroid_config.json", "w") as f:
    json.dump(config, f, indent=2)

print("\n" + "=" * 80)
print("训练完成!")
print("=" * 80)
