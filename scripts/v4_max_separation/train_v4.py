#!/usr/bin/env python3
"""
V4训练：最大化正常/恶意方向分离 + 类内聚集

目标：
1. 正常数据集和恶意数据集在投影空间中方向距离最远（余弦相似度→-1）
2. 正常样本聚在一起，恶意样本聚在一起
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastembed import TextEmbedding
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
print("V4训练：最大化正常/恶意方向分离 + 类内聚集")
print("=" * 80)

# 1. 加载Embedding模型
print("\n[1/6] 加载Embedding模型...")
embed_model = TextEmbedding('BAAI/bge-small-en-v1.5')

# 2. 加载恶意数据集
print("\n[2/6] 加载恶意数据集...")

def load_dataset_samples(path, text_col=None, max_samples=500):
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

malicious_datasets = [
    {"name": "AdvBench", "path": "datasets/advbench/advbench_harmful_behaviors.csv", "text_col": "goal"},
    {"name": "HarmBench", "path": "datasets/harmbench/harmbench_behaviors.csv", "text_col": "Behavior"},
    {"name": "BeaverTails", "path": "datasets/beavertails/beavertails_test.csv", "text_col": "prompt"},
    {"name": "JBB-GCG", "path": "datasets/jailbreakbench/jbb_gcg_all.csv", "text_col": "prompt"},
    {"name": "JBB-PAIR", "path": "datasets/jailbreakbench/jbb_pair_all.csv", "text_col": "prompt"},
    {"name": "ToxicChat", "path": "datasets/gcg_attacks/toxic_chat.csv"},
    {"name": "MaliciousInstruct", "path": "datasets/malicious_instruct/malicious_instruct.txt"},
]

all_malicious_samples = []
for ds in malicious_datasets:
    if os.path.exists(ds["path"]):
        samples = load_dataset_samples(ds["path"], ds.get("text_col"), max_samples=500)
        all_malicious_samples.extend(samples)
        print(f"  {ds['name']}: {len(samples)}条")

print(f"  恶意样本总计: {len(all_malicious_samples)}条")

# 3. 加载正常数据集
print("\n[3/6] 加载正常数据集...")

# 原始正常样本
normal_emb_original = np.load("embedding_db/bge-small-en-v1.5/embeddings/normal_embeddings.npy")
print(f"  原始正常样本: {len(normal_emb_original)}条")

all_normal_samples = []

# XSTest
xstest_path = "数据集/XSTest/data/train-00000-of-00001.parquet"
if os.path.exists(xstest_path):
    xstest = pd.read_parquet(xstest_path)
    xstest_samples = xstest[xstest['label'] == 'safe']['prompt'].tolist()
    all_normal_samples.extend(xstest_samples)
    print(f"  XSTest: {len(xstest_samples)}条")

# OR-Bench
or_bench_path = "数据集/or-bench-hard-1k.csv"
if os.path.exists(or_bench_path):
    or_bench = pd.read_csv(or_bench_path)
    or_bench_samples = or_bench['prompt'].tolist()
    all_normal_samples.extend(or_bench_samples)
    print(f"  OR-Bench: {len(or_bench_samples)}条")

# TruthfulQA
truthfulqa_path = "datasets/truthfulqa/truthfulqa.csv"
if os.path.exists(truthfulqa_path):
    truthfulqa = pd.read_csv(truthfulqa_path)
    if 'question' in truthfulqa.columns:
        truthfulqa_samples = truthfulqa['question'].dropna().tolist()[:500]
        all_normal_samples.extend(truthfulqa_samples)
        print(f"  TruthfulQA: {len(truthfulqa_samples)}条")

# DoNotAnswer
dna_path = "datasets/gcg_attacks/do_not_answer.csv"
if os.path.exists(dna_path):
    dna = pd.read_csv(dna_path)
    if 'prompt' in dna.columns:
        dna_samples = dna['prompt'].dropna().tolist()[:500]
    else:
        dna_samples = dna.iloc[:, 0].dropna().tolist()[:500]
    all_normal_samples.extend(dna_samples)
    print(f"  DoNotAnswer: {len(dna_samples)}条")

# Alpaca (HuggingFace)
try:
    ds = load_dataset('tatsu-lab/alpaca', split='train', trust_remote_code=True)
    alpaca_samples = []
    for item in ds:
        instruction = item['instruction']
        if len(instruction) > 10 and len(instruction) < 300:
            alpaca_samples.append(instruction)
        if len(alpaca_samples) >= 1000:
            break
    all_normal_samples.extend(alpaca_samples)
    print(f"  Alpaca: {len(alpaca_samples)}条")
except Exception as e:
    print(f"  Alpaca加载失败: {e}")

# 安全教育样本
SECURITY_EDUCATION_SAMPLES = [
    "What is the CIA triad in information security?",
    "Explain the difference between authentication and authorization",
    "How does public key cryptography work?",
    "What are the main types of cyber attacks?",
    "Explain how firewalls protect networks",
    "What is the purpose of an intrusion detection system?",
    "How do honeypots help in cybersecurity?",
    "What is defense in depth strategy?",
    "Explain the concept of least privilege principle",
    "What is zero trust architecture?",
    "How do penetration testers find vulnerabilities?",
    "What is ethical hacking?",
    "How do security researchers responsibly disclose vulnerabilities?",
    "What is the OWASP Top 10?",
    "How do bug bounty programs work?",
]
all_normal_samples.extend(SECURITY_EDUCATION_SAMPLES * 10)
print(f"  安全教育样本: {len(SECURITY_EDUCATION_SAMPLES) * 10}条")

print(f"  正常样本总计: {len(all_normal_samples)}条")

# 4. 计算Embedding
print("\n[4/6] 计算Embedding...")

def compute_embeddings(texts, embed_model, batch_size=32):
    # 过滤无效样本
    texts = [t for t in texts if isinstance(t, str) and len(t.strip()) > 0]
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = [str(t) for t in texts[i:i+batch_size] if t is not None]
        if batch:
            batch_emb = list(embed_model.embed(batch))
            embeddings.extend(batch_emb)
    return np.array(embeddings)

mal_emb = compute_embeddings(all_malicious_samples, embed_model)
print(f"  恶意样本Embedding: {mal_emb.shape}")

norm_emb_new = compute_embeddings(all_normal_samples, embed_model)
norm_emb = np.vstack([normal_emb_original, norm_emb_new])
print(f"  正常样本Embedding: {norm_emb.shape}")

# 5. V4训练
print("\n[5/6] V4训练：最大化分离 + 类内聚集...")

def v4_loss(projection, mal_emb, norm_emb, separation_weight=2.0, intra_weight=1.0):
    """
    V4损失函数
    
    目标一：最大化正常质心和恶意质心的分离（余弦相似度→-1）
    目标二：类内聚集（同类样本靠近各自质心）
    """
    mal_proj = projection(mal_emb)
    norm_proj = projection(norm_emb)
    
    # 计算质心
    mal_centroid = mal_proj.mean(dim=0)
    mal_centroid = F.normalize(mal_centroid, p=2, dim=0)
    
    norm_centroid = norm_proj.mean(dim=0)
    norm_centroid = F.normalize(norm_centroid, p=2, dim=0)
    
    # 目标一：最大化两个质心的分离（余弦相似度→-1）
    centroid_sim = torch.sum(mal_centroid * norm_centroid)
    separation_loss = (1 + centroid_sim)  # 目标是-1，此时loss=0
    
    # 目标二：类内聚集
    mal_sim = torch.sum(mal_proj * mal_centroid, dim=-1)
    mal_intra_loss = torch.mean(1 - mal_sim)
    
    norm_sim = torch.sum(norm_proj * norm_centroid, dim=-1)
    norm_intra_loss = torch.mean(1 - norm_sim)
    
    total_loss = separation_weight * separation_loss + intra_weight * (mal_intra_loss + norm_intra_loss)
    
    return total_loss, centroid_sim.item(), mal_sim.mean().item(), norm_sim.mean().item()

# 准备训练数据
mal_tensor = torch.tensor(mal_emb, dtype=torch.float32)
norm_tensor = torch.tensor(norm_emb, dtype=torch.float32)

# 训练
projection = LearnedProjection(384, 19)
optimizer = torch.optim.Adam(projection.parameters(), lr=0.01)

print(f"  恶意样本数: {len(mal_tensor)}")
print(f"  正常样本数: {len(norm_tensor)}")

for epoch in range(300):
    optimizer.zero_grad()
    
    loss, centroid_sim, mal_intra_sim, norm_intra_sim = v4_loss(
        projection, mal_tensor, norm_tensor,
        separation_weight=2.0, intra_weight=1.0
    )
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 50 == 0:
        print(f"\n  Epoch {epoch+1}/300, Loss: {loss.item():.4f}")
        print(f"    质心余弦相似度: {centroid_sim:.4f} (目标: -1)")
        print(f"    恶意类内相似度: {mal_intra_sim:.4f}")
        print(f"    正常类内相似度: {norm_intra_sim:.4f}")

# 6. 保存模型
print("\n[6/6] 保存模型...")

save_dir = "embedding_db/bge-small-en-v1.5/results/v4_max_separation"
os.makedirs(save_dir, exist_ok=True)

# 保存投影矩阵
weights = projection.proj.weight.data.numpy()
np.save(f"{save_dir}/learned_projection_weights_19d_v4.npy", weights)

# 计算并保存质心
with torch.no_grad():
    mal_proj = projection(mal_tensor)
    norm_proj = projection(norm_tensor)
    
    mal_centroid = mal_proj.mean(dim=0)
    mal_centroid = F.normalize(mal_centroid, p=2, dim=0).numpy()
    
    norm_centroid = norm_proj.mean(dim=0)
    norm_centroid = F.normalize(norm_centroid, p=2, dim=0).numpy()

np.save(f"{save_dir}/centroid_19d_malicious.npy", mal_centroid)
np.save(f"{save_dir}/centroid_19d_normal.npy", norm_centroid)

# 计算阈值（使用正常样本的95%分位）
with torch.no_grad():
    norm_proj_np = projection(norm_tensor).numpy()
    norm_sims = [np.dot(e, mal_centroid) for e in norm_proj_np]
    threshold = np.percentile(norm_sims, 95)

# 计算最终分离度
final_separation = np.dot(mal_centroid, norm_centroid)

print(f"  投影矩阵已保存")
print(f"  恶意质心已保存")
print(f"  正常质心已保存")
print(f"  检测阈值: {threshold:.4f}")
print(f"  质心分离度: {final_separation:.4f} (目标: -1)")

# 保存配置
config = {
    "version": "v4_max_separation",
    "malicious_samples": len(mal_tensor),
    "normal_samples": len(norm_tensor),
    "threshold": float(threshold),
    "centroid_separation": float(final_separation),
    "malicious_centroid": mal_centroid.tolist(),
    "normal_centroid": norm_centroid.tolist(),
}

with open(f"{save_dir}/v4_config.json", "w") as f:
    json.dump(config, f, indent=2)

print("\n" + "=" * 80)
print("V4训练完成!")
print(f"质心分离度: {final_separation:.4f}")
print("=" * 80)
