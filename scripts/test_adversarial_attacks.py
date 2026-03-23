"""
测试对抗性攻击数据集
- GCG攻击样本 (与AdvBench相同，但用于对比)
- JailbreakBench (100条专门设计的越狱攻击)

目的：验证embedding检测方法对对抗性攻击的鲁棒性
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
        'mal_sim_mean': float(mal_sims.mean()),
        'norm_sim_mean': float(norm_sims.mean()),
    }

# ============================================================
# 主程序
# ============================================================

print("=" * 70)
print("对抗性攻击数据集测试")
print("=" * 70)

# 加载embedding模型
print("\n[1] 加载Embedding模型...")
embed_model = TextEmbedding('BAAI/bge-small-en-v1.5')

# 加载训练数据 (AdvBench)
print("\n[2] 加载训练数据...")
advbench_emb = np.load("embedding_db/bge-small-en-v1.5/embeddings/advbench_embeddings.npy")
normal_emb = np.load("embedding_db/bge-small-en-v1.5/embeddings/normal_embeddings.npy")

advbench_train, advbench_test = train_test_split(advbench_emb, test_size=0.2, random_state=RANDOM_SEED)
normal_train, normal_test = train_test_split(normal_emb, test_size=0.2, random_state=RANDOM_SEED)

print(f"  AdvBench Train: {advbench_train.shape[0]} 条")
print(f"  正常样本 Test: {normal_test.shape[0]} 条")

# 计算训练集质心
centroid_train = advbench_train.mean(axis=0)
centroid_train_norm = centroid_train / np.linalg.norm(centroid_train)

# 训练投影矩阵
print("\n[3] 训练投影矩阵...")
train_emb = np.vstack([advbench_train, normal_train])
model_19d = train_learned_projection(train_emb, 19, epochs=300)
print("  19维投影矩阵训练完成")

advbench_train_19d = apply_projection(model_19d, advbench_train)
centroid_train_19d = advbench_train_19d.mean(axis=0)
centroid_train_19d_norm = centroid_train_19d / np.linalg.norm(centroid_train_19d)

# 加载对抗性攻击数据集
print("\n[4] 加载对抗性攻击数据集...")

attack_datasets = {}

# JailbreakBench
jbb_df = pd.read_csv("datasets/jailbreakbench/jbb_harmful.csv")
jbb_texts = jbb_df['Goal'].dropna().tolist()
print(f"  JailbreakBench: {len(jbb_texts)} 条")
attack_datasets['JailbreakBench'] = jbb_texts

# 查看JailbreakBench的类别分布
print(f"  类别分布: {jbb_df['Category'].value_counts().to_dict()}")

# 计算对抗性攻击的embedding
print("\n[5] 计算对抗性攻击的Embedding...")

for name, texts in attack_datasets.items():
    emb = np.array(list(embed_model.embed(texts)))
    attack_datasets[name] = emb
    print(f"  {name}: {emb.shape}")

# 测试
print("\n" + "=" * 70)
print("对抗性攻击检测效果")
print("=" * 70)

results = {}

# 对比数据集
all_datasets = {
    'AdvBench_Test': advbench_test,
    'HarmBench': np.load("embedding_db/bge-small-en-v1.5/embeddings/harmbench_embeddings.npy"),
    'JailbreakBench': attack_datasets['JailbreakBench'],
}

print("\n检测效果对比:")
print("-" * 80)
print(f"{'数据集':<20} {'样本数':<8} {'原始(384维)':<25} {'学习型(19维)':<25}")
print("-" * 80)

for dataset_name, mal_emb in all_datasets.items():
    n = min(len(mal_emb), len(normal_test))
    mal_subset = mal_emb[:n]
    norm_subset = normal_test[:n]
    
    # 原始空间
    orig = compute_detection_with_fixed_centroid(mal_subset, norm_subset, centroid_train_norm)
    
    # 学习型投影 (19维)
    mal_19d = apply_projection(model_19d, mal_subset)
    norm_19d = apply_projection(model_19d, norm_subset)
    l19 = compute_detection_with_fixed_centroid(mal_19d, norm_19d, centroid_train_19d_norm)
    
    orig_str = f"{orig['detection_rate']*100:.1f}/{orig['false_positive_rate']*100:.1f}/{orig['auc']:.3f}"
    l19_str = f"{l19['detection_rate']*100:.1f}/{l19['false_positive_rate']*100:.1f}/{l19['auc']:.3f}"
    
    print(f"{dataset_name:<20} {len(mal_emb):<8} {orig_str:<25} {l19_str:<25}")
    
    results[dataset_name] = {
        'count': len(mal_emb),
        'original': orig,
        'learned_19d': l19,
    }

print("-" * 80)
print("格式: 检出率% / 误报率% / AUC")

# 分析JailbreakBench各类别
print("\n" + "=" * 70)
print("JailbreakBench 各类别检测效果")
print("=" * 70)

jbb_df = pd.read_csv("datasets/jailbreakbench/jbb_harmful.csv")
categories = jbb_df['Category'].unique()

print(f"\n{'类别':<30} {'样本数':<8} {'检出率':<10} {'与质心相似度':<15}")
print("-" * 70)

for cat in categories:
    cat_texts = jbb_df[jbb_df['Category'] == cat]['Goal'].tolist()
    cat_emb = np.array(list(embed_model.embed(cat_texts)))
    
    # 计算与恶意质心的相似度
    sims = np.array([np.dot(e/np.linalg.norm(e), centroid_train_norm) for e in cat_emb])
    
    # 使用之前的阈值
    threshold = (results['AdvBench_Test']['original']['mal_sim_mean'] + 
                 results['AdvBench_Test']['original']['norm_sim_mean']) / 2
    detection_rate = (sims > threshold).sum() / len(sims)
    
    print(f"{cat:<30} {len(cat_texts):<8} {detection_rate*100:.1f}%      {sims.mean():.4f}")

print("-" * 70)

# 保存结果
os.makedirs("embedding_db/bge-small-en-v1.5/results/adversarial", exist_ok=True)
with open("embedding_db/bge-small-en-v1.5/results/adversarial/adversarial_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n结果已保存到: embedding_db/bge-small-en-v1.5/results/adversarial/adversarial_results.json")
