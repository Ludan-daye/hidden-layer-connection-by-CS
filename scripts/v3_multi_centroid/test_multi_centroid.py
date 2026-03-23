#!/usr/bin/env python3
"""
测试多质心模型的检测效果

对比：
- 单质心模型 (v3)
- 多质心模型 (multi_centroid)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastembed import TextEmbedding
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
print("多质心模型 vs 单质心模型 对比测试")
print("=" * 80)

# 1. 加载模型
print("\n[1] 加载模型...")
embed_model = TextEmbedding('BAAI/bge-small-en-v1.5')

# 单质心模型 (v3)
weights_v3 = np.load("embedding_db/bge-small-en-v1.5/results/learned_projection_weights_19d_contrastive_v3.npy")
projection_v3 = LearnedProjection(384, 19)
projection_v3.proj.weight.data = torch.tensor(weights_v3, dtype=torch.float32)
centroid_v3 = np.load("embedding_db/bge-small-en-v1.5/results/centroid_19d_contrastive_v3.npy")
threshold_v3 = -0.4118

# 多质心模型v2
weights_mc = np.load("embedding_db/bge-small-en-v1.5/results/learned_projection_weights_19d_multi_centroid_v2.npy")
projection_mc = LearnedProjection(384, 19)
projection_mc.proj.weight.data = torch.tensor(weights_mc, dtype=torch.float32)

with open("embedding_db/bge-small-en-v1.5/results/multi_centroid_v2_config.json", "r") as f:
    mc_config = json.load(f)

centroids_mc = {k: np.array(v) for k, v in mc_config["malicious_centroids"].items()}
threshold_mc = mc_config["threshold"]

print(f"  单质心模型: 阈值={threshold_v3:.4f}")
print(f"  多质心模型v2: 阈值={threshold_mc:.4f}, 恶意质心数={len(centroids_mc)}")

# 2. 数据集加载函数
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

def detect_v3(samples, embed_model, projection, centroid, threshold):
    """单质心检测"""
    if not samples:
        return None
    
    embeddings = np.array(list(embed_model.embed(samples)))
    
    with torch.no_grad():
        projected = projection(torch.tensor(embeddings, dtype=torch.float32)).numpy()
    
    similarities = [np.dot(e, centroid) for e in projected]
    predictions = [s > threshold for s in similarities]
    
    return {
        "total": len(samples),
        "detected": sum(predictions),
        "detection_rate": sum(predictions) / len(samples),
        "sim_mean": np.mean(similarities),
    }

def detect_multi_centroid(samples, embed_model, projection, centroids, threshold):
    """多质心检测：取与所有质心的最大相似度"""
    if not samples:
        return None
    
    embeddings = np.array(list(embed_model.embed(samples)))
    
    with torch.no_grad():
        projected = projection(torch.tensor(embeddings, dtype=torch.float32)).numpy()
    
    # 对每个样本，计算与所有质心的相似度，取最大值
    max_similarities = []
    best_types = []
    for emb in projected:
        max_sim = -1.0
        best_type = None
        for type_name, centroid in centroids.items():
            sim = np.dot(emb, centroid)
            if sim > max_sim:
                max_sim = sim
                best_type = type_name
        max_similarities.append(max_sim)
        best_types.append(best_type)
    
    predictions = [s > threshold for s in max_similarities]
    
    return {
        "total": len(samples),
        "detected": sum(predictions),
        "detection_rate": sum(predictions) / len(samples),
        "sim_mean": np.mean(max_similarities),
        "best_types": best_types,
    }

# 3. 测试恶意数据集
print("\n[2] 测试恶意数据集...")

malicious_datasets = [
    {"name": "AdvBench", "path": "datasets/advbench/advbench_harmful_behaviors.csv", "type": "直接恶意指令", "text_col": "goal"},
    {"name": "HarmBench", "path": "datasets/harmbench/harmbench_behaviors.csv", "type": "多样化恶意", "text_col": "Behavior"},
    {"name": "BeaverTails", "path": "datasets/beavertails/beavertails_test.csv", "type": "道德伤害", "text_col": "prompt"},
    {"name": "JBB-GCG", "path": "datasets/jailbreakbench/jbb_gcg_all.csv", "type": "GCG越狱", "text_col": "prompt"},
    {"name": "JBB-PAIR", "path": "datasets/jailbreakbench/jbb_pair_all.csv", "type": "PAIR越狱", "text_col": "prompt"},
    {"name": "ToxicChat", "path": "datasets/gcg_attacks/toxic_chat.csv", "type": "有毒对话"},
]

results_mal = []

for config in malicious_datasets:
    name = config["name"]
    path = config["path"]
    dtype = config["type"]
    text_col = config.get("text_col")
    
    if not os.path.exists(path):
        continue
    
    samples = load_dataset_samples(path, text_col, max_samples=300)
    if not samples:
        continue
    
    result_v3 = detect_v3(samples, embed_model, projection_v3, centroid_v3, threshold_v3)
    result_mc = detect_multi_centroid(samples, embed_model, projection_mc, centroids_mc, threshold_mc)
    
    results_mal.append({
        "name": name,
        "type": dtype,
        "total": result_v3["total"],
        "v3_rate": result_v3["detection_rate"],
        "mc_rate": result_mc["detection_rate"],
        "improvement": result_mc["detection_rate"] - result_v3["detection_rate"],
    })

# 4. 测试正常数据集
print("\n[3] 测试正常数据集...")

normal_datasets = [
    {"name": "TruthfulQA", "path": "datasets/truthfulqa/truthfulqa.csv", "text_col": "question"},
    {"name": "OR-Bench", "path": "数据集/or-bench-hard-1k.csv", "text_col": "prompt"},
]

results_norm = []

for config in normal_datasets:
    name = config["name"]
    path = config["path"]
    text_col = config.get("text_col")
    
    if not os.path.exists(path):
        continue
    
    samples = load_dataset_samples(path, text_col, max_samples=300)
    if not samples:
        continue
    
    result_v3 = detect_v3(samples, embed_model, projection_v3, centroid_v3, threshold_v3)
    result_mc = detect_multi_centroid(samples, embed_model, projection_mc, centroids_mc, threshold_mc)
    
    results_norm.append({
        "name": name,
        "total": result_v3["total"],
        "v3_fpr": result_v3["detection_rate"],
        "mc_fpr": result_mc["detection_rate"],
        "change": result_mc["detection_rate"] - result_v3["detection_rate"],
    })

# 5. 汇总结果
print("\n" + "=" * 80)
print("对比结果")
print("=" * 80)

print("\n恶意数据集检出率对比:")
print("-" * 90)
print(f"{'数据集':<15} {'类型':<12} {'样本数':>8} {'单质心(v3)':>12} {'多质心':>12} {'提升':>12}")
print("-" * 90)

total_v3 = 0
total_mc = 0
total_samples = 0

for r in sorted(results_mal, key=lambda x: -x['improvement']):
    print(f"{r['name']:<15} {r['type']:<12} {r['total']:>8} {r['v3_rate']*100:>11.1f}% {r['mc_rate']*100:>11.1f}% {r['improvement']*100:>+11.1f}%")
    total_v3 += r['v3_rate'] * r['total']
    total_mc += r['mc_rate'] * r['total']
    total_samples += r['total']

print("-" * 90)
print(f"{'总计':<15} {'':<12} {total_samples:>8} {total_v3/total_samples*100:>11.1f}% {total_mc/total_samples*100:>11.1f}% {(total_mc-total_v3)/total_samples*100:>+11.1f}%")

print("\n正常数据集误报率对比:")
print("-" * 70)
print(f"{'数据集':<15} {'样本数':>8} {'单质心(v3)':>12} {'多质心':>12} {'变化':>12}")
print("-" * 70)

for r in results_norm:
    print(f"{r['name']:<15} {r['total']:>8} {r['v3_fpr']*100:>11.1f}% {r['mc_fpr']*100:>11.1f}% {r['change']*100:>+11.1f}%")

# 6. 结论
print("\n" + "=" * 80)
print("结论")
print("=" * 80)
print(f"""
多质心模型通过学习不同恶意类型的位置关系：
- 直接恶意指令、有毒对话：检出率保持高水平
- 越狱攻击、道德伤害：检出率有所提升
- 正常样本误报率：需要关注变化

质心类型: {list(centroids_mc.keys())}
""")
