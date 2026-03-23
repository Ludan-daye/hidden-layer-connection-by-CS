#!/usr/bin/env python3
"""
实验三：不同恶意类型检出率分析

分析模型对不同类型恶意输入的检出能力
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastembed import TextEmbedding
import os

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
print("实验三：不同恶意类型检出率分析")
print("=" * 80)

# 1. 加载模型
print("\n[1] 加载模型...")
embed_model = TextEmbedding('BAAI/bge-small-en-v1.5')

weights = np.load("embedding_db/bge-small-en-v1.5/results/learned_projection_weights_19d_contrastive_v3.npy")
projection = LearnedProjection(384, 19)
projection.proj.weight.data = torch.tensor(weights, dtype=torch.float32)

centroid_19d = np.load("embedding_db/bge-small-en-v1.5/results/centroid_19d_contrastive_v3.npy")
threshold_19d = -0.4118

print(f"  模型: v3对比学习")
print(f"  阈值: {threshold_19d:.4f}")

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

def detect_samples(samples, embed_model, projection, centroid, threshold):
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
        "sim_min": min(similarities),
        "sim_max": max(similarities),
    }

# 3. 测试恶意数据集
print("\n[2] 测试各类恶意数据集...")

malicious_datasets = [
    {"name": "AdvBench", "path": "datasets/advbench/advbench_harmful_behaviors.csv", "type": "直接恶意指令", "text_col": "goal"},
    {"name": "HarmBench", "path": "datasets/harmbench/harmbench_behaviors.csv", "type": "多样化恶意行为", "text_col": "Behavior"},
    {"name": "BeaverTails", "path": "datasets/beavertails/beavertails_test.csv", "type": "道德伤害", "text_col": "prompt"},
    {"name": "MaliciousInstruct", "path": "datasets/malicious_instruct/malicious_instruct.txt", "type": "恶意指令"},
    {"name": "JBB-Harmful", "path": "datasets/jailbreakbench/jbb_harmful.csv", "type": "越狱目标", "text_col": "Goal"},
    {"name": "JBB-GCG", "path": "datasets/jailbreakbench/jbb_gcg_all.csv", "type": "GCG越狱攻击", "text_col": "prompt"},
    {"name": "JBB-PAIR", "path": "datasets/jailbreakbench/jbb_pair_all.csv", "type": "PAIR越狱攻击", "text_col": "prompt"},
    {"name": "ToxicChat", "path": "datasets/gcg_attacks/toxic_chat.csv", "type": "有毒对话"},
    {"name": "PromptInjections", "path": "datasets/gcg_attacks/prompt_injections.csv", "type": "注入攻击", "text_col": "text"},
]

results = []

for config in malicious_datasets:
    name = config["name"]
    path = config["path"]
    dtype = config["type"]
    text_col = config.get("text_col")
    
    if not os.path.exists(path):
        continue
    
    samples = load_dataset_samples(path, text_col)
    if not samples:
        continue
    
    result = detect_samples(samples, embed_model, projection, centroid_19d, threshold_19d)
    if result:
        result["name"] = name
        result["type"] = dtype
        results.append(result)
        print(f"  {name:<20} 检出率: {result['detection_rate']*100:>6.1f}% ({result['detected']}/{result['total']})")

# 4. 汇总结果
print("\n" + "=" * 80)
print("恶意类型检出率分析结果")
print("=" * 80)

print(f"\n{'数据集':<20} {'类型':<15} {'样本数':>8} {'检出数':>8} {'检出率':>10} {'相似度均值':>12}")
print("-" * 85)

for r in sorted(results, key=lambda x: -x['detection_rate']):
    print(f"{r['name']:<20} {r['type']:<15} {r['total']:>8} {r['detected']:>8} {r['detection_rate']*100:>9.1f}% {r['sim_mean']:>12.4f}")

# 总计
total = sum(r['total'] for r in results)
detected = sum(r['detected'] for r in results)
print("-" * 85)
print(f"{'总计':<20} {'':<15} {total:>8} {detected:>8} {detected/total*100:>9.1f}%")

# 5. 按类型分组统计
print("\n按恶意类型分组:")
print("-" * 60)

type_groups = {}
for r in results:
    t = r['type']
    if t not in type_groups:
        type_groups[t] = {"total": 0, "detected": 0}
    type_groups[t]["total"] += r['total']
    type_groups[t]["detected"] += r['detected']

for t, g in sorted(type_groups.items(), key=lambda x: -x[1]['detected']/x[1]['total']):
    rate = g['detected'] / g['total'] * 100
    print(f"  {t:<20} {rate:>6.1f}% ({g['detected']}/{g['total']})")

print("\n" + "=" * 80)
print("实验三完成!")
