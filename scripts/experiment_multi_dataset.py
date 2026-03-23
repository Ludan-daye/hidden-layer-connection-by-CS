"""
多数据集测试实验
对比不同恶意数据集在学习型投影下的检测效果
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

RESULTS_DIR = "embedding_db/bge-small-en-v1.5/results/multi_dataset"
EMBEDDINGS_DIR = "embedding_db/bge-small-en-v1.5/embeddings"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# ============================================================
# 加载数据集
# ============================================================

print("=" * 70)
print("多数据集测试实验")
print("=" * 70)

print("\n[1] 加载数据集...")

# 1. AdvBench (已有)
advbench_emb = np.load("embedding_db/bge-small-en-v1.5/embeddings/advbench_embeddings.npy")
print(f"  AdvBench: {advbench_emb.shape[0]} 条")

# 2. HarmBench
harmbench_df = pd.read_csv("datasets/harmbench/harmbench_behaviors.csv")
harmbench_texts = harmbench_df['Behavior'].dropna().tolist()
print(f"  HarmBench: {len(harmbench_texts)} 条")

# 3. AdvBench Strings
advbench_str_df = pd.read_csv("datasets/advbench_strings/advbench_strings.csv")
advbench_str_texts = advbench_str_df.iloc[:, 0].dropna().tolist()
print(f"  AdvBench Strings: {len(advbench_str_texts)} 条")

# 4. 正常样本
templates = [
    'Can you explain {}?', 'What is the meaning of {}?', 'How does {} work?',
    'Tell me about {}.', 'What are the benefits of {}?', 'Describe {} in simple terms.',
    'What is the history of {}?', 'How can I learn about {}?',
    'What are the main features of {}?', 'Explain the concept of {}.',
]
topics = [
    'mathematics', 'science', 'history', 'geography', 'literature',
    'music', 'art', 'cooking', 'gardening', 'photography',
    'programming', 'design', 'economics', 'psychology', 'philosophy',
    'biology', 'chemistry', 'physics', 'astronomy', 'medicine',
    'education', 'sports', 'travel', 'technology', 'engineering',
    'architecture', 'fashion', 'nutrition', 'fitness', 'meditation',
    'yoga', 'dance', 'painting', 'sculpture', 'poetry',
    'writing', 'reading', 'languages', 'culture', 'traditions',
    'festivals', 'holidays', 'nature', 'wildlife', 'environment',
    'sustainability', 'recycling', 'energy', 'transportation', 'communication',
]
normal_texts = []
for topic in topics:
    for template in templates:
        normal_texts.append(template.format(topic))
        if len(normal_texts) >= 1000:
            break
    if len(normal_texts) >= 1000:
        break
print(f"  正常样本: {len(normal_texts)} 条")

# ============================================================
# 计算Embeddings
# ============================================================

print("\n[2] 计算Embeddings...")

model_embed = TextEmbedding('BAAI/bge-small-en-v1.5')

# HarmBench
print("  计算HarmBench embeddings...")
harmbench_emb = np.array(list(model_embed.embed(harmbench_texts)))
np.save(f"{EMBEDDINGS_DIR}/harmbench_embeddings.npy", harmbench_emb)

# AdvBench Strings
print("  计算AdvBench Strings embeddings...")
advbench_str_emb = np.array(list(model_embed.embed(advbench_str_texts)))
np.save(f"{EMBEDDINGS_DIR}/advbench_strings_embeddings.npy", advbench_str_emb)

# 正常样本
print("  计算正常样本embeddings...")
normal_emb = np.array(list(model_embed.embed(normal_texts)))
np.save(f"{EMBEDDINGS_DIR}/normal_embeddings.npy", normal_emb)

print(f"\n  Embeddings已保存到 {EMBEDDINGS_DIR}/")

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

def compute_detection_metrics(mal_emb, norm_emb, method_name=""):
    """基于与恶意质心的距离进行检测"""
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
        'threshold': float(threshold),
    }

# ============================================================
# 测试各数据集
# ============================================================

print("\n[3] 测试各数据集...")

datasets = {
    'AdvBench': advbench_emb,
    'HarmBench': harmbench_emb,
    'AdvBench_Strings': advbench_str_emb,
}

results = {}

for dataset_name, mal_emb in datasets.items():
    print(f"\n--- {dataset_name} ({mal_emb.shape[0]} 条) ---")
    
    # 取与恶意样本数量匹配的正常样本
    n_samples = min(len(mal_emb), len(normal_emb))
    norm_subset = normal_emb[:n_samples]
    mal_subset = mal_emb[:n_samples]
    
    # 合并用于训练
    all_emb = np.vstack([mal_subset, norm_subset])
    
    dataset_results = {
        'n_malicious': int(mal_emb.shape[0]),
        'n_normal': int(n_samples),
        'tests': {}
    }
    
    # 原始空间
    print("  原始空间 (384维)...")
    orig_metrics = compute_detection_metrics(mal_subset, norm_subset)
    dataset_results['tests']['original_384d'] = orig_metrics
    print(f"    检出率: {orig_metrics['detection_rate']*100:.1f}%, 误报率: {orig_metrics['false_positive_rate']*100:.1f}%")
    
    # 学习型投影 - 38维 (10%)
    print("  学习型投影 (38维)...")
    model_38 = train_learned_projection(all_emb, 38, epochs=200)
    mal_38 = apply_projection(model_38, mal_subset)
    norm_38 = apply_projection(model_38, norm_subset)
    metrics_38 = compute_detection_metrics(mal_38, norm_38)
    dataset_results['tests']['learned_38d'] = metrics_38
    print(f"    检出率: {metrics_38['detection_rate']*100:.1f}%, 误报率: {metrics_38['false_positive_rate']*100:.1f}%")
    
    # 学习型投影 - 19维 (5%)
    print("  学习型投影 (19维)...")
    model_19 = train_learned_projection(all_emb, 19, epochs=200)
    mal_19 = apply_projection(model_19, mal_subset)
    norm_19 = apply_projection(model_19, norm_subset)
    metrics_19 = compute_detection_metrics(mal_19, norm_19)
    dataset_results['tests']['learned_19d'] = metrics_19
    print(f"    检出率: {metrics_19['detection_rate']*100:.1f}%, 误报率: {metrics_19['false_positive_rate']*100:.1f}%")
    
    results[dataset_name] = dataset_results

# ============================================================
# 总结
# ============================================================

print("\n" + "=" * 70)
print("实验总结")
print("=" * 70)

print("\n检出率对比:")
print("-" * 70)
print(f"{'数据集':<20} {'原始(384维)':<15} {'学习型(38维)':<15} {'学习型(19维)':<15}")
print("-" * 70)
for name, res in results.items():
    orig = res['tests']['original_384d']['detection_rate'] * 100
    l38 = res['tests']['learned_38d']['detection_rate'] * 100
    l19 = res['tests']['learned_19d']['detection_rate'] * 100
    print(f"{name:<20} {orig:<15.1f} {l38:<15.1f} {l19:<15.1f}")

print("\n误报率对比:")
print("-" * 70)
print(f"{'数据集':<20} {'原始(384维)':<15} {'学习型(38维)':<15} {'学习型(19维)':<15}")
print("-" * 70)
for name, res in results.items():
    orig = res['tests']['original_384d']['false_positive_rate'] * 100
    l38 = res['tests']['learned_38d']['false_positive_rate'] * 100
    l19 = res['tests']['learned_19d']['false_positive_rate'] * 100
    print(f"{name:<20} {orig:<15.1f} {l38:<15.1f} {l19:<15.1f}")

# 保存结果
with open(f"{RESULTS_DIR}/multi_dataset_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n结果已保存到: {RESULTS_DIR}/multi_dataset_results.json")
