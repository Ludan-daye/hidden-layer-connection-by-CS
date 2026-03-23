#!/usr/bin/env python3
"""
使用datasets目录下所有数据集进行全面测试

测试数据集：
- AdvBench (恶意)
- HarmBench (恶意)
- BeaverTails (恶意)
- MaliciousInstruct (恶意)
- JailbreakBench GCG/PAIR/JBC (恶意)
- TruthfulQA (正常)
- Do Not Answer (混合)
- Toxic Chat (混合)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastembed import TextEmbedding
from sklearn.metrics import roc_auc_score
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
print("全面测试：datasets目录下所有数据集")
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
threshold_19d = -0.4118

print(f"  模型: v3对比学习")
print(f"  阈值: {threshold_19d:.4f}")

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
                # 尝试第一列
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
    
    # 过滤有效样本
    samples = [s for s in samples if isinstance(s, str) and len(s) > 10 and len(s) < 1000]
    
    if len(samples) > max_samples:
        samples = samples[:max_samples]
    
    return samples

def detect_samples(samples, embed_model, projection, centroid, threshold):
    """检测样本"""
    if not samples:
        return None
    
    # 计算Embedding
    embeddings = np.array(list(embed_model.embed(samples)))
    
    # 投影
    with torch.no_grad():
        projected = projection(torch.tensor(embeddings, dtype=torch.float32)).numpy()
    
    # 计算相似度
    similarities = [np.dot(e, centroid) for e in projected]
    
    # 判定
    predictions = [s > threshold for s in similarities]
    
    return {
        "similarities": similarities,
        "predictions": predictions,
        "detected_count": sum(predictions),
        "total": len(samples),
        "detection_rate": sum(predictions) / len(samples),
        "sim_min": min(similarities),
        "sim_max": max(similarities),
        "sim_mean": np.mean(similarities),
    }

# 3. 测试所有数据集
print("\n[2] 加载并测试所有数据集...")

datasets_config = [
    # 恶意数据集
    {"name": "AdvBench", "path": "datasets/advbench/advbench_harmful_behaviors.csv", "type": "malicious", "text_col": "goal"},
    {"name": "HarmBench", "path": "datasets/harmbench/harmbench_behaviors.csv", "type": "malicious", "text_col": "Behavior"},
    {"name": "BeaverTails", "path": "datasets/beavertails/beavertails_test.csv", "type": "malicious", "text_col": "prompt"},
    {"name": "MaliciousInstruct", "path": "datasets/malicious_instruct/malicious_instruct.txt", "type": "malicious"},
    {"name": "JBB-GCG", "path": "datasets/jailbreakbench/jbb_gcg_all.csv", "type": "malicious", "text_col": "prompt"},
    {"name": "JBB-PAIR", "path": "datasets/jailbreakbench/jbb_pair_all.csv", "type": "malicious", "text_col": "prompt"},
    {"name": "JBB-JBC", "path": "datasets/jailbreakbench/jbb_jbc_all.csv", "type": "malicious", "text_col": "prompt"},
    {"name": "JBB-Harmful", "path": "datasets/jailbreakbench/jbb_harmful.csv", "type": "malicious", "text_col": "Goal"},
    {"name": "ChatGPT-Jailbreak", "path": "datasets/gcg_attacks/chatgpt_jailbreak.csv", "type": "malicious", "text_col": "prompt"},
    {"name": "PromptInjections", "path": "datasets/gcg_attacks/prompt_injections.csv", "type": "malicious", "text_col": "text"},
    
    # 正常数据集
    {"name": "TruthfulQA", "path": "datasets/truthfulqa/truthfulqa.csv", "type": "normal", "text_col": "question"},
    {"name": "JBB-Benign", "path": "datasets/gcg_attacks/jbb_benign_behaviors.csv", "type": "normal", "text_col": "Goal"},
    
    # 混合数据集（需要特殊处理）
    {"name": "ToxicChat", "path": "datasets/gcg_attacks/toxic_chat.csv", "type": "mixed"},
    {"name": "DoNotAnswer", "path": "datasets/gcg_attacks/do_not_answer.csv", "type": "mixed"},
]

results = []

for config in datasets_config:
    name = config["name"]
    path = config["path"]
    dtype = config["type"]
    text_col = config.get("text_col")
    
    print(f"\n  测试 {name}...")
    
    if not os.path.exists(path):
        print(f"    文件不存在: {path}")
        continue
    
    samples = load_dataset_samples(path, text_col)
    
    if not samples:
        print(f"    无有效样本")
        continue
    
    print(f"    样本数: {len(samples)}")
    
    result = detect_samples(samples, embed_model, projection, centroid_19d, threshold_19d)
    
    if result:
        result["name"] = name
        result["type"] = dtype
        results.append(result)
        
        if dtype == "malicious":
            print(f"    检出率: {result['detection_rate']*100:.1f}% ({result['detected_count']}/{result['total']})")
        elif dtype == "normal":
            print(f"    误报率: {result['detection_rate']*100:.1f}% ({result['detected_count']}/{result['total']})")
        else:
            print(f"    检出率: {result['detection_rate']*100:.1f}% ({result['detected_count']}/{result['total']})")
        
        print(f"    相似度: min={result['sim_min']:.4f}, mean={result['sim_mean']:.4f}, max={result['sim_max']:.4f}")

# 4. 汇总结果
print("\n" + "=" * 80)
print("测试结果汇总")
print("=" * 80)

# 恶意数据集
print("\n恶意数据集（期望高检出率）:")
print("-" * 80)
print(f"{'数据集':<20} {'样本数':>8} {'检出数':>8} {'检出率':>10} {'相似度均值':>12} {'相似度范围':>20}")
print("-" * 80)

malicious_results = [r for r in results if r["type"] == "malicious"]
total_mal = 0
total_mal_detected = 0

for r in malicious_results:
    total_mal += r["total"]
    total_mal_detected += r["detected_count"]
    print(f"{r['name']:<20} {r['total']:>8} {r['detected_count']:>8} {r['detection_rate']*100:>9.1f}% {r['sim_mean']:>12.4f} [{r['sim_min']:.2f}, {r['sim_max']:.2f}]")

if malicious_results:
    print("-" * 80)
    print(f"{'总计':<20} {total_mal:>8} {total_mal_detected:>8} {total_mal_detected/total_mal*100:>9.1f}%")

# 正常数据集
print("\n正常数据集（期望低误报率）:")
print("-" * 80)
print(f"{'数据集':<20} {'样本数':>8} {'误报数':>8} {'误报率':>10} {'相似度均值':>12} {'相似度范围':>20}")
print("-" * 80)

normal_results = [r for r in results if r["type"] == "normal"]
total_norm = 0
total_norm_fp = 0

for r in normal_results:
    total_norm += r["total"]
    total_norm_fp += r["detected_count"]
    print(f"{r['name']:<20} {r['total']:>8} {r['detected_count']:>8} {r['detection_rate']*100:>9.1f}% {r['sim_mean']:>12.4f} [{r['sim_min']:.2f}, {r['sim_max']:.2f}]")

if normal_results:
    print("-" * 80)
    print(f"{'总计':<20} {total_norm:>8} {total_norm_fp:>8} {total_norm_fp/total_norm*100:>9.1f}%")

# 混合数据集
print("\n混合数据集:")
print("-" * 80)
print(f"{'数据集':<20} {'样本数':>8} {'检出数':>8} {'检出率':>10} {'相似度均值':>12} {'相似度范围':>20}")
print("-" * 80)

mixed_results = [r for r in results if r["type"] == "mixed"]
for r in mixed_results:
    print(f"{r['name']:<20} {r['total']:>8} {r['detected_count']:>8} {r['detection_rate']*100:>9.1f}% {r['sim_mean']:>12.4f} [{r['sim_min']:.2f}, {r['sim_max']:.2f}]")

# 总体统计
print("\n" + "=" * 80)
print("总体统计")
print("=" * 80)
if malicious_results and normal_results:
    print(f"\n恶意数据集总检出率: {total_mal_detected/total_mal*100:.1f}% ({total_mal_detected}/{total_mal})")
    print(f"正常数据集总误报率: {total_norm_fp/total_norm*100:.1f}% ({total_norm_fp}/{total_norm})")
    
    # 计算F1
    precision = total_mal_detected / (total_mal_detected + total_norm_fp) if (total_mal_detected + total_norm_fp) > 0 else 0
    recall = total_mal_detected / total_mal if total_mal > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n精确率 (Precision): {precision*100:.1f}%")
    print(f"召回率 (Recall): {recall*100:.1f}%")
    print(f"F1分数: {f1:.4f}")

print("\n" + "=" * 80)
print("测试完成!")
