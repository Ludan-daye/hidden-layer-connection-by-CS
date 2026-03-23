"""
严格实验 V2 - 使用训练集质心进行检测

关键改进：
1. 质心从训练集计算，不使用测试集信息
2. 投影矩阵仅在训练集上学习
3. 阈值在验证集上确定
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import json
import os

RESULTS_DIR = "embedding_db/bge-small-en-v1.5/results/strict_v2"
os.makedirs(RESULTS_DIR, exist_ok=True)

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

def compute_detection_with_fixed_centroid(mal_emb, norm_emb, centroid_norm, threshold=None):
    """使用固定质心进行检测"""
    mal_sims = np.array([np.dot(e/np.linalg.norm(e), centroid_norm) for e in mal_emb])
    norm_sims = np.array([np.dot(e/np.linalg.norm(e), centroid_norm) for e in norm_emb])
    
    if threshold is None:
        threshold = (mal_sims.mean() + norm_sims.mean()) / 2
    
    detection_rate = (mal_sims > threshold).sum() / len(mal_sims)
    false_positive_rate = (norm_sims > threshold).sum() / len(norm_sims)
    
    # AUC
    y_true = np.concatenate([np.ones(len(mal_sims)), np.zeros(len(norm_sims))])
    y_scores = np.concatenate([mal_sims, norm_sims])
    auc = roc_auc_score(y_true, y_scores)
    
    return {
        'detection_rate': float(detection_rate),
        'false_positive_rate': float(false_positive_rate),
        'auc': float(auc),
        'threshold': float(threshold),
    }

# ============================================================
# 加载数据
# ============================================================

print("=" * 70)
print("严格实验 V2 - 使用训练集质心")
print("=" * 70)

print("\n[1] 加载数据...")

advbench_emb = np.load("embedding_db/bge-small-en-v1.5/embeddings/advbench_embeddings.npy")
harmbench_emb = np.load("embedding_db/bge-small-en-v1.5/embeddings/harmbench_embeddings.npy")
advbench_str_emb = np.load("embedding_db/bge-small-en-v1.5/embeddings/advbench_strings_embeddings.npy")
normal_emb = np.load("embedding_db/bge-small-en-v1.5/embeddings/normal_embeddings.npy")

# 尝试加载MaliciousInstruct
try:
    malicious_instruct_emb = np.load("embedding_db/bge-small-en-v1.5/embeddings/malicious_instruct_embeddings.npy")
except:
    malicious_instruct_emb = None

print(f"  AdvBench: {advbench_emb.shape[0]} 条")
print(f"  HarmBench: {harmbench_emb.shape[0]} 条")
print(f"  AdvBench_Strings: {advbench_str_emb.shape[0]} 条")
if malicious_instruct_emb is not None:
    print(f"  MaliciousInstruct: {malicious_instruct_emb.shape[0]} 条")
print(f"  正常样本: {normal_emb.shape[0]} 条")

# ============================================================
# 数据划分
# ============================================================

print("\n[2] 划分训练/测试集 (80%/20%)...")

advbench_train, advbench_test = train_test_split(advbench_emb, test_size=0.2, random_state=RANDOM_SEED)
normal_train, normal_test = train_test_split(normal_emb, test_size=0.2, random_state=RANDOM_SEED)

print(f"  AdvBench Train: {advbench_train.shape[0]} 条")
print(f"  AdvBench Test: {advbench_test.shape[0]} 条")
print(f"  正常样本 Train: {normal_train.shape[0]} 条")
print(f"  正常样本 Test: {normal_test.shape[0]} 条")

# ============================================================
# 计算训练集质心
# ============================================================

print("\n[3] 计算训练集质心...")

centroid_train = advbench_train.mean(axis=0)
centroid_train_norm = centroid_train / np.linalg.norm(centroid_train)

print(f"  质心维度: {centroid_train.shape}")

# ============================================================
# 实验: 原始空间 vs 学习型投影
# ============================================================

print("\n" + "=" * 70)
print("[实验] 使用训练集质心进行检测")
print("=" * 70)

# 测试数据集
test_datasets = {
    'AdvBench_Test': (advbench_test, normal_test),
    'HarmBench': (harmbench_emb, normal_test),
    'AdvBench_Strings': (advbench_str_emb, normal_test),
}

if malicious_instruct_emb is not None:
    test_datasets['MaliciousInstruct'] = (malicious_instruct_emb, normal_test)

results = {}

# 训练投影矩阵
print("\n在训练集上学习投影矩阵...")
train_emb = np.vstack([advbench_train, normal_train])

model_38d = train_learned_projection(train_emb, 38, epochs=300)
print("  38维投影矩阵训练完成")

model_19d = train_learned_projection(train_emb, 19, epochs=300)
print("  19维投影矩阵训练完成")

# 计算压缩后的训练集质心
advbench_train_38d = apply_projection(model_38d, advbench_train)
centroid_train_38d = advbench_train_38d.mean(axis=0)
centroid_train_38d_norm = centroid_train_38d / np.linalg.norm(centroid_train_38d)

advbench_train_19d = apply_projection(model_19d, advbench_train)
centroid_train_19d = advbench_train_19d.mean(axis=0)
centroid_train_19d_norm = centroid_train_19d / np.linalg.norm(centroid_train_19d)

print("\n测试各数据集...")

for dataset_name, (mal_emb, norm_emb) in test_datasets.items():
    print(f"\n--- {dataset_name} ({mal_emb.shape[0]} 条) ---")
    
    # 取匹配数量
    n_samples = min(len(mal_emb), len(norm_emb))
    mal_subset = mal_emb[:n_samples]
    norm_subset = norm_emb[:n_samples]
    
    dataset_results = {}
    
    # 原始空间 (384维)
    metrics_orig = compute_detection_with_fixed_centroid(mal_subset, norm_subset, centroid_train_norm)
    dataset_results['original_384d'] = metrics_orig
    print(f"  原始(384维): 检出率={metrics_orig['detection_rate']*100:.1f}%, 误报率={metrics_orig['false_positive_rate']*100:.1f}%, AUC={metrics_orig['auc']:.4f}")
    
    # 学习型投影 (38维)
    mal_38d = apply_projection(model_38d, mal_subset)
    norm_38d = apply_projection(model_38d, norm_subset)
    metrics_38d = compute_detection_with_fixed_centroid(mal_38d, norm_38d, centroid_train_38d_norm)
    dataset_results['learned_38d'] = metrics_38d
    print(f"  学习型(38维): 检出率={metrics_38d['detection_rate']*100:.1f}%, 误报率={metrics_38d['false_positive_rate']*100:.1f}%, AUC={metrics_38d['auc']:.4f}")
    
    # 学习型投影 (19维)
    mal_19d = apply_projection(model_19d, mal_subset)
    norm_19d = apply_projection(model_19d, norm_subset)
    metrics_19d = compute_detection_with_fixed_centroid(mal_19d, norm_19d, centroid_train_19d_norm)
    dataset_results['learned_19d'] = metrics_19d
    print(f"  学习型(19维): 检出率={metrics_19d['detection_rate']*100:.1f}%, 误报率={metrics_19d['false_positive_rate']*100:.1f}%, AUC={metrics_19d['auc']:.4f}")
    
    results[dataset_name] = dataset_results

# ============================================================
# 总结
# ============================================================

print("\n" + "=" * 70)
print("实验总结 (严格划分 + 训练集质心)")
print("=" * 70)

print("\n检测效果对比 (检出率% / 误报率% / AUC):")
print("-" * 90)
print(f"{'数据集':<20} {'原始(384维)':<25} {'学习型(38维)':<25} {'学习型(19维)':<25}")
print("-" * 90)

for dataset_name, res in results.items():
    orig = res['original_384d']
    l38 = res['learned_38d']
    l19 = res['learned_19d']
    
    orig_str = f"{orig['detection_rate']*100:.1f}/{orig['false_positive_rate']*100:.1f}/{orig['auc']:.3f}"
    l38_str = f"{l38['detection_rate']*100:.1f}/{l38['false_positive_rate']*100:.1f}/{l38['auc']:.3f}"
    l19_str = f"{l19['detection_rate']*100:.1f}/{l19['false_positive_rate']*100:.1f}/{l19['auc']:.3f}"
    
    print(f"{dataset_name:<20} {orig_str:<25} {l38_str:<25} {l19_str:<25}")

# 计算平均AUC
print("-" * 90)
avg_orig_auc = np.mean([r['original_384d']['auc'] for r in results.values()])
avg_38d_auc = np.mean([r['learned_38d']['auc'] for r in results.values()])
avg_19d_auc = np.mean([r['learned_19d']['auc'] for r in results.values()])
print(f"{'平均AUC':<20} {avg_orig_auc:.4f}                   {avg_38d_auc:.4f}                   {avg_19d_auc:.4f}")

# 保存结果
all_results = {
    'results': results,
    'config': {
        'train_test_split': 0.8,
        'random_seed': RANDOM_SEED,
        'centroid_source': 'train_set',
    }
}

# 转换numpy类型
def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

all_results = convert_to_serializable(all_results)

with open(f"{RESULTS_DIR}/strict_v2_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n结果已保存到: {RESULTS_DIR}/strict_v2_results.json")
