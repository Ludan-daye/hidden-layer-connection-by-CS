#!/usr/bin/env python3
"""
测试V4模型效果
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
print("V4模型测试")
print("=" * 80)

# 1. 加载模型
print("\n[1] 加载模型...")
embed_model = TextEmbedding('BAAI/bge-small-en-v1.5')

# V4模型
v4_dir = "embedding_db/bge-small-en-v1.5/results/v4_max_separation"
weights_v4 = np.load(f"{v4_dir}/learned_projection_weights_19d_v4.npy")
projection_v4 = LearnedProjection(384, 19)
projection_v4.proj.weight.data = torch.tensor(weights_v4, dtype=torch.float32)

with open(f"{v4_dir}/v4_config.json", "r") as f:
    v4_config = json.load(f)

mal_centroid = np.array(v4_config["malicious_centroid"])
norm_centroid = np.array(v4_config["normal_centroid"])
threshold = v4_config["threshold"]

print(f"  V4模型加载完成")
print(f"  质心分离度: {v4_config['centroid_separation']:.4f}")
print(f"  阈值: {threshold:.4f}")

# V3单质心模型（对比）
weights_v3 = np.load("embedding_db/bge-small-en-v1.5/results/learned_projection_weights_19d_contrastive_v3.npy")
projection_v3 = LearnedProjection(384, 19)
projection_v3.proj.weight.data = torch.tensor(weights_v3, dtype=torch.float32)
centroid_v3 = np.load("embedding_db/bge-small-en-v1.5/results/centroid_19d_contrastive_v3.npy")
threshold_v3 = -0.4118

# 2. 数据集加载
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

def detect_v4(samples, embed_model, projection, mal_centroid, threshold):
    if not samples:
        return None
    
    samples = [s for s in samples if isinstance(s, str) and len(s.strip()) > 0]
    embeddings = np.array(list(embed_model.embed(samples)))
    
    with torch.no_grad():
        projected = projection(torch.tensor(embeddings, dtype=torch.float32)).numpy()
    
    similarities = [np.dot(e, mal_centroid) for e in projected]
    predictions = [s > threshold for s in similarities]
    
    return {
        "total": len(samples),
        "detected": sum(predictions),
        "detection_rate": sum(predictions) / len(samples),
        "sim_mean": np.mean(similarities),
    }

def detect_v3(samples, embed_model, projection, centroid, threshold):
    if not samples:
        return None
    
    samples = [s for s in samples if isinstance(s, str) and len(s.strip()) > 0]
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

# 3. 测试恶意数据集
print("\n[2] 测试恶意数据集...")

malicious_datasets = [
    {"name": "AdvBench", "path": "datasets/advbench/advbench_harmful_behaviors.csv", "text_col": "goal"},
    {"name": "HarmBench", "path": "datasets/harmbench/harmbench_behaviors.csv", "text_col": "Behavior"},
    {"name": "BeaverTails", "path": "datasets/beavertails/beavertails_test.csv", "text_col": "prompt"},
    {"name": "JBB-GCG", "path": "datasets/jailbreakbench/jbb_gcg_all.csv", "text_col": "prompt"},
    {"name": "JBB-PAIR", "path": "datasets/jailbreakbench/jbb_pair_all.csv", "text_col": "prompt"},
    {"name": "ToxicChat", "path": "datasets/gcg_attacks/toxic_chat.csv"},
]

results_mal = []

for config in malicious_datasets:
    name = config["name"]
    path = config["path"]
    text_col = config.get("text_col")
    
    if not os.path.exists(path):
        continue
    
    samples = load_dataset_samples(path, text_col, max_samples=300)
    if not samples:
        continue
    
    result_v3 = detect_v3(samples, embed_model, projection_v3, centroid_v3, threshold_v3)
    result_v4 = detect_v4(samples, embed_model, projection_v4, mal_centroid, threshold)
    
    results_mal.append({
        "name": name,
        "total": result_v3["total"],
        "v3_rate": result_v3["detection_rate"],
        "v4_rate": result_v4["detection_rate"],
        "improvement": result_v4["detection_rate"] - result_v3["detection_rate"],
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
    result_v4 = detect_v4(samples, embed_model, projection_v4, mal_centroid, threshold)
    
    results_norm.append({
        "name": name,
        "total": result_v3["total"],
        "v3_fpr": result_v3["detection_rate"],
        "v4_fpr": result_v4["detection_rate"],
        "change": result_v4["detection_rate"] - result_v3["detection_rate"],
    })

# 5. 汇总结果
print("\n" + "=" * 80)
print("V4 vs V3 对比结果")
print("=" * 80)

print("\n恶意数据集检出率:")
print("-" * 70)
print(f"{'数据集':<15} {'样本数':>8} {'V3单质心':>12} {'V4':>12} {'变化':>12}")
print("-" * 70)

total_v3 = 0
total_v4 = 0
total_samples = 0

for r in sorted(results_mal, key=lambda x: -x['improvement']):
    print(f"{r['name']:<15} {r['total']:>8} {r['v3_rate']*100:>11.1f}% {r['v4_rate']*100:>11.1f}% {r['improvement']*100:>+11.1f}%")
    total_v3 += r['v3_rate'] * r['total']
    total_v4 += r['v4_rate'] * r['total']
    total_samples += r['total']

print("-" * 70)
print(f"{'总计':<15} {total_samples:>8} {total_v3/total_samples*100:>11.1f}% {total_v4/total_samples*100:>11.1f}% {(total_v4-total_v3)/total_samples*100:>+11.1f}%")

print("\n正常数据集误报率:")
print("-" * 70)
print(f"{'数据集':<15} {'样本数':>8} {'V3单质心':>12} {'V4':>12} {'变化':>12}")
print("-" * 70)

for r in results_norm:
    print(f"{r['name']:<15} {r['total']:>8} {r['v3_fpr']*100:>11.1f}% {r['v4_fpr']*100:>11.1f}% {r['change']*100:>+11.1f}%")

print("\n" + "=" * 80)
print(f"V4质心分离度: {v4_config['centroid_separation']:.4f} (目标: -1)")
print("=" * 80)
