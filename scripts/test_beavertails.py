"""
使用之前训练的投影矩阵测试BeaverTails数据集
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from fastembed import TextEmbedding
import json
import os

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

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

def train_learned_projection(train_embeddings, target_dim, epochs=300, lr=0.01):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embeddings_tensor = torch.tensor(train_embeddings, dtype=torch.float32).to(device)
    embeddings_norm = F.normalize(embeddings_tensor, p=2, dim=-1)
    
    model = LearnedProjection(train_embeddings.shape[1], target_dim).to(device)
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

def compute_detection_with_fixed_centroid(mal_emb, norm_emb, centroid_norm):
    mal_sims = np.array([np.dot(e/np.linalg.norm(e), centroid_norm) for e in mal_emb])
    norm_sims = np.array([np.dot(e/np.linalg.norm(e), centroid_norm) for e in norm_emb])
    
    threshold = (mal_sims.mean() + norm_sims.mean()) / 2
    detection_rate = (mal_sims > threshold).sum() / len(mal_sims)
    false_positive_rate = (norm_sims > threshold).sum() / len(norm_sims)
    
    y_true = np.concatenate([np.ones(len(mal_sims)), np.zeros(len(norm_sims))])
    y_scores = np.concatenate([mal_sims, norm_sims])
    auc = roc_auc_score(y_true, y_scores)
    
    return {
        'detection_rate': float(detection_rate),
        'false_positive_rate': float(false_positive_rate),
        'auc': float(auc),
    }

# ============================================================
# 加载数据
# ============================================================

print("=" * 70)
print("BeaverTails 数据集测试")
print("=" * 70)

print("\n[1] 加载数据...")

# 加载BeaverTails
beavertails_df = pd.read_csv("datasets/beavertails/beavertails_test.csv")
print(f"  BeaverTails原始数据: {len(beavertails_df)} 条")

# 筛选有害样本 (is_safe=False)
if 'is_safe' in beavertails_df.columns:
    harmful_df = beavertails_df[beavertails_df['is_safe'] == False]
    safe_df = beavertails_df[beavertails_df['is_safe'] == True]
    print(f"  有害样本: {len(harmful_df)} 条")
    print(f"  安全样本: {len(safe_df)} 条")
    
    # 提取文本
    if 'prompt' in harmful_df.columns:
        harmful_texts = harmful_df['prompt'].dropna().tolist()[:500]
    elif 'response' in harmful_df.columns:
        harmful_texts = harmful_df['response'].dropna().tolist()[:500]
    else:
        harmful_texts = harmful_df.iloc[:, 0].dropna().tolist()[:500]
else:
    # 如果没有is_safe列，取前500条作为测试
    print("  未找到is_safe列，使用全部数据")
    if 'prompt' in beavertails_df.columns:
        harmful_texts = beavertails_df['prompt'].dropna().tolist()[:500]
    elif 'response' in beavertails_df.columns:
        harmful_texts = beavertails_df['response'].dropna().tolist()[:500]
    else:
        harmful_texts = beavertails_df.iloc[:, 0].dropna().tolist()[:500]

print(f"  提取有害文本: {len(harmful_texts)} 条")

# 计算BeaverTails embedding
print("\n[2] 计算BeaverTails embedding...")
model_embed = TextEmbedding('BAAI/bge-small-en-v1.5')
beavertails_emb = np.array(list(model_embed.embed(harmful_texts)))
print(f"  BeaverTails embedding: {beavertails_emb.shape}")

# 保存embedding
np.save("embedding_db/bge-small-en-v1.5/embeddings/beavertails_embeddings.npy", beavertails_emb)

# 加载之前的训练数据
advbench_emb = np.load("embedding_db/bge-small-en-v1.5/embeddings/advbench_embeddings.npy")
normal_emb = np.load("embedding_db/bge-small-en-v1.5/embeddings/normal_embeddings.npy")

# 划分训练集（与之前一致）
advbench_train, advbench_test = train_test_split(advbench_emb, test_size=0.2, random_state=RANDOM_SEED)
normal_train, normal_test = train_test_split(normal_emb, test_size=0.2, random_state=RANDOM_SEED)

print(f"  AdvBench Train: {advbench_train.shape[0]} 条")
print(f"  正常样本 Test: {normal_test.shape[0]} 条")

# ============================================================
# 使用之前训练的投影矩阵
# ============================================================

print("\n[3] 训练投影矩阵（与之前相同配置）...")

train_emb = np.vstack([advbench_train, normal_train])

# 训练投影矩阵
model_19d = train_learned_projection(train_emb, 19, epochs=300)
print("  19维投影矩阵训练完成")

# 计算训练集质心
centroid_train = advbench_train.mean(axis=0)
centroid_train_norm = centroid_train / np.linalg.norm(centroid_train)

advbench_train_19d = apply_projection(model_19d, advbench_train)
centroid_train_19d = advbench_train_19d.mean(axis=0)
centroid_train_19d_norm = centroid_train_19d / np.linalg.norm(centroid_train_19d)

# ============================================================
# 测试BeaverTails
# ============================================================

print("\n[4] 测试BeaverTails (Zero-shot Transfer)...")

n_samples = min(len(beavertails_emb), len(normal_test))
beavertails_subset = beavertails_emb[:n_samples]
normal_subset = normal_test[:n_samples]

# 原始空间
metrics_orig = compute_detection_with_fixed_centroid(beavertails_subset, normal_subset, centroid_train_norm)
print(f"  原始(384维): 检出率={metrics_orig['detection_rate']*100:.1f}%, 误报率={metrics_orig['false_positive_rate']*100:.1f}%, AUC={metrics_orig['auc']:.4f}")

# 学习型投影 (19维)
beavertails_19d = apply_projection(model_19d, beavertails_subset)
normal_19d = apply_projection(model_19d, normal_subset)
metrics_19d = compute_detection_with_fixed_centroid(beavertails_19d, normal_19d, centroid_train_19d_norm)
print(f"  学习型(19维): 检出率={metrics_19d['detection_rate']*100:.1f}%, 误报率={metrics_19d['false_positive_rate']*100:.1f}%, AUC={metrics_19d['auc']:.4f}")

# ============================================================
# 与其他数据集对比
# ============================================================

print("\n" + "=" * 70)
print("与其他数据集对比 (Zero-shot Transfer)")
print("=" * 70)

# 加载其他数据集
other_datasets = {
    'AdvBench_Test': advbench_test,
    'HarmBench': np.load("embedding_db/bge-small-en-v1.5/embeddings/harmbench_embeddings.npy"),
    'AdvBench_Strings': np.load("embedding_db/bge-small-en-v1.5/embeddings/advbench_strings_embeddings.npy"),
    'BeaverTails': beavertails_emb,
}

print("\n检测效果对比 (检出率% / 误报率% / AUC):")
print("-" * 70)
print(f"{'数据集':<20} {'原始(384维)':<25} {'学习型(19维)':<25}")
print("-" * 70)

for dataset_name, mal_emb in other_datasets.items():
    n = min(len(mal_emb), len(normal_test))
    mal_subset = mal_emb[:n]
    norm_subset = normal_test[:n]
    
    # 原始
    orig = compute_detection_with_fixed_centroid(mal_subset, norm_subset, centroid_train_norm)
    
    # 学习型
    mal_19d = apply_projection(model_19d, mal_subset)
    norm_19d = apply_projection(model_19d, norm_subset)
    l19 = compute_detection_with_fixed_centroid(mal_19d, norm_19d, centroid_train_19d_norm)
    
    orig_str = f"{orig['detection_rate']*100:.1f}/{orig['false_positive_rate']*100:.1f}/{orig['auc']:.3f}"
    l19_str = f"{l19['detection_rate']*100:.1f}/{l19['false_positive_rate']*100:.1f}/{l19['auc']:.3f}"
    
    print(f"{dataset_name:<20} {orig_str:<25} {l19_str:<25}")

print("-" * 70)
print("完成!")
