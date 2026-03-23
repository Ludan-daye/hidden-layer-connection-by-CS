"""
全数据集测试实验
测试所有可用数据集的学习型投影检测效果
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastembed import TextEmbedding
import json
import os

# ============================================================
# 配置
# ============================================================

RESULTS_DIR = "embedding_db/bge-small-en-v1.5/results/all_datasets"
EMBEDDINGS_DIR = "embedding_db/bge-small-en-v1.5/embeddings"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# ============================================================
# 学习型投影模型
# ============================================================

class LearnedProjection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim, bias=False)
        nn.init.normal_(self.proj.weight, mean=0, std=1/np.sqrt(output_dim))
    
    def forward(self, x):
        z = self.proj(x)
        z = F.normalize(z, p=2, dim=-1)
        return z

def train_learned_projection(embeddings, target_dim, epochs=200, lr=0.01):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
    embeddings_norm = F.normalize(embeddings_tensor, p=2, dim=-1)
    
    model = LearnedProjection(embeddings.shape[1], target_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        original_sim = embeddings_norm @ embeddings_norm.T
        compressed = model(embeddings_tensor)
        compressed_sim = compressed @ compressed.T
        loss = F.mse_loss(compressed_sim, original_sim)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return model

def apply_projection(model, embeddings):
    device = next(model.parameters()).device
    with torch.no_grad():
        tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
        compressed = model(tensor).cpu().numpy()
    return compressed

def compute_detection_metrics(mal_emb, norm_emb):
    centroid_mal = mal_emb.mean(axis=0)
    centroid_mal_norm = centroid_mal / np.linalg.norm(centroid_mal)
    
    mal_sims = np.array([np.dot(e/np.linalg.norm(e), centroid_mal_norm) for e in mal_emb])
    norm_sims = np.array([np.dot(e/np.linalg.norm(e), centroid_mal_norm) for e in norm_emb])
    
    threshold = (mal_sims.mean() + norm_sims.mean()) / 2
    
    detection_rate = (mal_sims > threshold).sum() / len(mal_sims)
    false_positive_rate = (norm_sims > threshold).sum() / len(norm_sims)
    
    return {
        'detection_rate': float(detection_rate),
        'false_positive_rate': float(false_positive_rate),
    }

# ============================================================
# 加载数据集
# ============================================================

print("=" * 70)
print("全数据集测试实验")
print("=" * 70)

print("\n[1] 加载数据集...")

model_embed = TextEmbedding('BAAI/bge-small-en-v1.5')

datasets = {}

# 1. AdvBench (已有)
datasets['AdvBench'] = np.load("embedding_db/bge-small-en-v1.5/embeddings/advbench_embeddings.npy")
print(f"  AdvBench: {datasets['AdvBench'].shape[0]} 条")

# 2. HarmBench (已有)
datasets['HarmBench'] = np.load("embedding_db/bge-small-en-v1.5/embeddings/harmbench_embeddings.npy")
print(f"  HarmBench: {datasets['HarmBench'].shape[0]} 条")

# 3. AdvBench Strings (已有)
datasets['AdvBench_Strings'] = np.load("embedding_db/bge-small-en-v1.5/embeddings/advbench_strings_embeddings.npy")
print(f"  AdvBench_Strings: {datasets['AdvBench_Strings'].shape[0]} 条")

# 4. MaliciousInstruct (新)
if os.path.exists("datasets/malicious_instruct/malicious_instruct.txt"):
    with open("datasets/malicious_instruct/malicious_instruct.txt", 'r', encoding='utf-8') as f:
        mal_texts = [line.strip() for line in f if line.strip()]
    if len(mal_texts) > 0:
        mal_emb = np.array(list(model_embed.embed(mal_texts)))
        datasets['MaliciousInstruct'] = mal_emb
        np.save(f"{EMBEDDINGS_DIR}/malicious_instruct_embeddings.npy", mal_emb)
        print(f"  MaliciousInstruct: {len(mal_texts)} 条")

# 5. TruthfulQA (新)
if os.path.exists("datasets/truthfulqa/truthfulqa.csv"):
    try:
        tqa_df = pd.read_csv("datasets/truthfulqa/truthfulqa.csv")
        tqa_texts = tqa_df['Question'].dropna().tolist()[:500]  # 取前500条
        if len(tqa_texts) > 0:
            tqa_emb = np.array(list(model_embed.embed(tqa_texts)))
            datasets['TruthfulQA'] = tqa_emb
            np.save(f"{EMBEDDINGS_DIR}/truthfulqa_embeddings.npy", tqa_emb)
            print(f"  TruthfulQA: {len(tqa_texts)} 条")
    except Exception as e:
        print(f"  TruthfulQA: 加载失败 - {e}")

# 正常样本
normal_emb = np.load("embedding_db/bge-small-en-v1.5/embeddings/normal_embeddings.npy")
print(f"  正常样本: {normal_emb.shape[0]} 条")

# ============================================================
# 测试各数据集
# ============================================================

print("\n[2] 测试各数据集...")

results = {}

for dataset_name, mal_emb in datasets.items():
    print(f"\n--- {dataset_name} ({mal_emb.shape[0]} 条) ---")
    
    n_samples = min(len(mal_emb), len(normal_emb))
    norm_subset = normal_emb[:n_samples]
    mal_subset = mal_emb[:n_samples]
    all_emb = np.vstack([mal_subset, norm_subset])
    
    dataset_results = {
        'n_samples': int(mal_emb.shape[0]),
        'tests': {}
    }
    
    # 原始空间
    print("  原始空间 (384维)...", end=" ")
    orig_metrics = compute_detection_metrics(mal_subset, norm_subset)
    dataset_results['tests']['original_384d'] = orig_metrics
    print(f"检出率: {orig_metrics['detection_rate']*100:.1f}%, 误报率: {orig_metrics['false_positive_rate']*100:.1f}%")
    
    # 学习型投影 - 38维
    print("  学习型投影 (38维)...", end=" ")
    model_38 = train_learned_projection(all_emb, 38, epochs=200)
    mal_38 = apply_projection(model_38, mal_subset)
    norm_38 = apply_projection(model_38, norm_subset)
    metrics_38 = compute_detection_metrics(mal_38, norm_38)
    dataset_results['tests']['learned_38d'] = metrics_38
    print(f"检出率: {metrics_38['detection_rate']*100:.1f}%, 误报率: {metrics_38['false_positive_rate']*100:.1f}%")
    
    # 学习型投影 - 19维
    print("  学习型投影 (19维)...", end=" ")
    model_19 = train_learned_projection(all_emb, 19, epochs=200)
    mal_19 = apply_projection(model_19, mal_subset)
    norm_19 = apply_projection(model_19, norm_subset)
    metrics_19 = compute_detection_metrics(mal_19, norm_19)
    dataset_results['tests']['learned_19d'] = metrics_19
    print(f"检出率: {metrics_19['detection_rate']*100:.1f}%, 误报率: {metrics_19['false_positive_rate']*100:.1f}%")
    
    results[dataset_name] = dataset_results

# ============================================================
# 总结
# ============================================================

print("\n" + "=" * 70)
print("实验总结")
print("=" * 70)

print("\n检测效果对比 (检出率% / 误报率%):")
print("-" * 80)
print(f"{'数据集':<20} {'样本数':<10} {'原始(384维)':<18} {'学习型(38维)':<18} {'学习型(19维)':<18}")
print("-" * 80)

for name, res in results.items():
    n = res['n_samples']
    orig = res['tests']['original_384d']
    l38 = res['tests']['learned_38d']
    l19 = res['tests']['learned_19d']
    
    orig_str = f"{orig['detection_rate']*100:.1f} / {orig['false_positive_rate']*100:.1f}"
    l38_str = f"{l38['detection_rate']*100:.1f} / {l38['false_positive_rate']*100:.1f}"
    l19_str = f"{l19['detection_rate']*100:.1f} / {l19['false_positive_rate']*100:.1f}"
    
    print(f"{name:<20} {n:<10} {orig_str:<18} {l38_str:<18} {l19_str:<18}")

# 计算平均值
print("-" * 80)
avg_orig_det = np.mean([r['tests']['original_384d']['detection_rate'] for r in results.values()])
avg_orig_fpr = np.mean([r['tests']['original_384d']['false_positive_rate'] for r in results.values()])
avg_l38_det = np.mean([r['tests']['learned_38d']['detection_rate'] for r in results.values()])
avg_l38_fpr = np.mean([r['tests']['learned_38d']['false_positive_rate'] for r in results.values()])
avg_l19_det = np.mean([r['tests']['learned_19d']['detection_rate'] for r in results.values()])
avg_l19_fpr = np.mean([r['tests']['learned_19d']['false_positive_rate'] for r in results.values()])

print(f"{'平均':<20} {'':<10} {avg_orig_det*100:.1f} / {avg_orig_fpr*100:.1f}       {avg_l38_det*100:.1f} / {avg_l38_fpr*100:.1f}       {avg_l19_det*100:.1f} / {avg_l19_fpr*100:.1f}")

# 保存结果
with open(f"{RESULTS_DIR}/all_datasets_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n结果已保存到: {RESULTS_DIR}/all_datasets_results.json")
