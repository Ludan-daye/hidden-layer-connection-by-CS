#!/usr/bin/env python3
"""
实验一：各数据集质心距离分析

分析不同类型数据集与恶意质心的余弦相似度分布
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
print("实验一：各数据集质心距离分析")
print("=" * 80)

# 1. 加载模型
print("\n[1] 加载模型...")
embed_model = TextEmbedding('BAAI/bge-small-en-v1.5')

# 加载v3投影矩阵
weights = np.load("embedding_db/bge-small-en-v1.5/results/learned_projection_weights_19d_contrastive_v3.npy")
projection = LearnedProjection(384, 19)
projection.proj.weight.data = torch.tensor(weights, dtype=torch.float32)

# 加载v3质心
centroid_19d = np.load("embedding_db/bge-small-en-v1.5/results/centroid_19d_contrastive_v3.npy")

print(f"  模型: v3对比学习 (384→19维)")

# 2. 定义数据集加载函数
def load_dataset_samples(path, text_col=None, max_samples=500):
    """加载数据集样本"""
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
            elif 'text' in df.columns:
                samples = df['text'].dropna().tolist()
            elif 'question' in df.columns:
                samples = df['question'].dropna().tolist()
            elif 'Behavior' in df.columns:
                samples = df['Behavior'].dropna().tolist()
            else:
                samples = df.iloc[:, 0].dropna().tolist()
        except Exception as e:
            print(f"    加载失败: {e}")
            return []
    elif path.endswith('.txt'):
        try:
            with open(path, 'r') as f:
                samples = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"    加载失败: {e}")
            return []
    elif path.endswith('.parquet'):
        try:
            df = pd.read_parquet(path)
            if text_col and text_col in df.columns:
                samples = df[text_col].dropna().tolist()
            elif 'prompt' in df.columns:
                samples = df['prompt'].dropna().tolist()
        except Exception as e:
            print(f"    加载失败: {e}")
            return []
    
    # 过滤有效样本
    samples = [s for s in samples if isinstance(s, str) and len(s) > 10 and len(s) < 1000]
    
    if len(samples) > max_samples:
        samples = samples[:max_samples]
    
    return samples

def compute_similarities(samples, embed_model, projection, centroid):
    """计算样本与质心的余弦相似度"""
    if not samples:
        return None
    
    # 计算Embedding
    embeddings = np.array(list(embed_model.embed(samples)))
    
    # 投影到19维
    with torch.no_grad():
        projected = projection(torch.tensor(embeddings, dtype=torch.float32)).numpy()
    
    # 计算余弦相似度
    similarities = [np.dot(e, centroid) for e in projected]
    
    return {
        "count": len(samples),
        "similarities": similarities,
        "min": min(similarities),
        "max": max(similarities),
        "mean": np.mean(similarities),
        "std": np.std(similarities),
        "median": np.median(similarities),
    }

# 3. 定义所有数据集
print("\n[2] 加载并分析所有数据集...")

datasets_config = [
    # 恶意数据集
    {"name": "AdvBench", "path": "datasets/advbench/advbench_harmful_behaviors.csv", "type": "恶意-直接指令", "text_col": "goal"},
    {"name": "HarmBench", "path": "datasets/harmbench/harmbench_behaviors.csv", "type": "恶意-多样化", "text_col": "Behavior"},
    {"name": "BeaverTails", "path": "datasets/beavertails/beavertails_test.csv", "type": "恶意-道德伤害", "text_col": "prompt"},
    {"name": "MaliciousInstruct", "path": "datasets/malicious_instruct/malicious_instruct.txt", "type": "恶意-指令"},
    {"name": "JBB-Harmful", "path": "datasets/jailbreakbench/jbb_harmful.csv", "type": "恶意-越狱目标", "text_col": "Goal"},
    {"name": "JBB-GCG", "path": "datasets/jailbreakbench/jbb_gcg_all.csv", "type": "恶意-GCG越狱", "text_col": "prompt"},
    {"name": "JBB-PAIR", "path": "datasets/jailbreakbench/jbb_pair_all.csv", "type": "恶意-PAIR越狱", "text_col": "prompt"},
    {"name": "ToxicChat", "path": "datasets/gcg_attacks/toxic_chat.csv", "type": "恶意-有毒对话"},
    {"name": "PromptInjections", "path": "datasets/gcg_attacks/prompt_injections.csv", "type": "恶意-注入攻击", "text_col": "text"},
    
    # 正常数据集
    {"name": "TruthfulQA", "path": "datasets/truthfulqa/truthfulqa.csv", "type": "正常-问答", "text_col": "question"},
    {"name": "XSTest", "path": "数据集/XSTest/data/train-00000-of-00001.parquet", "type": "正常-边界安全", "text_col": "prompt"},
    {"name": "OR-Bench", "path": "数据集/or-bench-hard-1k.csv", "type": "正常-过度拒绝", "text_col": "prompt"},
    {"name": "DoNotAnswer", "path": "datasets/gcg_attacks/do_not_answer.csv", "type": "正常-边界问题"},
]

results = []

for config in datasets_config:
    name = config["name"]
    path = config["path"]
    dtype = config["type"]
    text_col = config.get("text_col")
    
    print(f"\n  分析 {name} ({dtype})...")
    
    if not os.path.exists(path):
        print(f"    文件不存在: {path}")
        continue
    
    samples = load_dataset_samples(path, text_col)
    
    if not samples:
        print(f"    无有效样本")
        continue
    
    print(f"    样本数: {len(samples)}")
    
    result = compute_similarities(samples, embed_model, projection, centroid_19d)
    
    if result:
        result["name"] = name
        result["type"] = dtype
        results.append(result)
        
        print(f"    相似度: min={result['min']:.4f}, mean={result['mean']:.4f}, max={result['max']:.4f}, std={result['std']:.4f}")

# 4. 汇总结果
print("\n" + "=" * 80)
print("质心距离分析结果")
print("=" * 80)

# 按类型分组
malicious_results = [r for r in results if r["type"].startswith("恶意")]
normal_results = [r for r in results if r["type"].startswith("正常")]

print("\n恶意数据集与质心的余弦相似度:")
print("-" * 100)
print(f"{'数据集':<20} {'类型':<20} {'样本数':>8} {'最小值':>10} {'均值':>10} {'最大值':>10} {'标准差':>10}")
print("-" * 100)

for r in sorted(malicious_results, key=lambda x: -x['mean']):
    print(f"{r['name']:<20} {r['type']:<20} {r['count']:>8} {r['min']:>10.4f} {r['mean']:>10.4f} {r['max']:>10.4f} {r['std']:>10.4f}")

print("\n正常数据集与质心的余弦相似度:")
print("-" * 100)
print(f"{'数据集':<20} {'类型':<20} {'样本数':>8} {'最小值':>10} {'均值':>10} {'最大值':>10} {'标准差':>10}")
print("-" * 100)

for r in sorted(normal_results, key=lambda x: -x['mean']):
    print(f"{r['name']:<20} {r['type']:<20} {r['count']:>8} {r['min']:>10.4f} {r['mean']:>10.4f} {r['max']:>10.4f} {r['std']:>10.4f}")

# 5. 分析结论
print("\n" + "=" * 80)
print("分析结论")
print("=" * 80)

# 计算恶意和正常的整体统计
all_mal_sims = []
all_norm_sims = []

for r in malicious_results:
    all_mal_sims.extend(r['similarities'])
for r in normal_results:
    all_norm_sims.extend(r['similarities'])

print(f"\n恶意样本整体统计:")
print(f"  样本数: {len(all_mal_sims)}")
print(f"  相似度: min={min(all_mal_sims):.4f}, mean={np.mean(all_mal_sims):.4f}, max={max(all_mal_sims):.4f}")

print(f"\n正常样本整体统计:")
print(f"  样本数: {len(all_norm_sims)}")
print(f"  相似度: min={min(all_norm_sims):.4f}, mean={np.mean(all_norm_sims):.4f}, max={max(all_norm_sims):.4f}")

print(f"\n分离度: {np.mean(all_mal_sims) - np.mean(all_norm_sims):.4f}")

# 按相似度排序所有数据集
print("\n所有数据集按相似度均值排序:")
print("-" * 100)
for r in sorted(results, key=lambda x: -x['mean']):
    distance = "近" if r['mean'] > 0.5 else ("中" if r['mean'] > -0.3 else "远")
    print(f"  {r['name']:<20} 均值={r['mean']:>7.4f}  范围=[{r['min']:.2f}, {r['max']:.2f}]  与质心距离: {distance}")

print("\n" + "=" * 80)
print("实验一完成!")
