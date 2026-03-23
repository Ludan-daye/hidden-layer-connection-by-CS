"""
严格划分训练/测试集的实验

实验设计：
1. 数据划分：
   - AdvBench: 80% Train / 20% Test
   - 正常样本: 80% Train / 20% Test
   
2. 训练阶段：
   - 仅在Train集上学习投影矩阵Φ
   
3. 测试阶段：
   - 在Test集上评估相似度保持
   - 在Test集上评估检测率/误报率
   
4. 跨数据集验证（Zero-shot Transfer）：
   - 用AdvBench训练的Φ，直接在HarmBench/MaliciousInstruct上测试
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import json
import os

# ============================================================
# 配置
# ============================================================

RESULTS_DIR = "results/strict_split"
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
    """仅在训练集上学习投影矩阵"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embeddings_tensor = torch.tensor(train_embeddings, dtype=torch.float32).to(device)
    embeddings_norm = F.normalize(embeddings_tensor, p=2, dim=-1)
    
    model = LearnedProjection(train_embeddings.shape[1], target_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_loss = float('inf')
    patience = 0
    
    for epoch in range(epochs):
        original_sim = embeddings_norm @ embeddings_norm.T
        compressed = model(embeddings_tensor)
        compressed_sim = compressed @ compressed.T
        loss = F.mse_loss(compressed_sim, original_sim)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Early stopping
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience = 0
        else:
            patience += 1
            if patience > 30:
                break
    
    return model

def apply_projection(model, embeddings):
    device = next(model.parameters()).device
    with torch.no_grad():
        tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
        compressed = model(tensor).cpu().numpy()
    return compressed

def compute_similarity_preservation(original, compressed):
    """计算相似度保持程度"""
    orig_norm = original / np.linalg.norm(original, axis=1, keepdims=True)
    orig_sim = orig_norm @ orig_norm.T
    comp_sim = compressed @ compressed.T
    
    # 只取上三角（排除对角线）
    mask = np.triu(np.ones_like(orig_sim, dtype=bool), k=1)
    orig_flat = orig_sim[mask]
    comp_flat = comp_sim[mask]
    
    corr = np.corrcoef(orig_flat, comp_flat)[0, 1]
    mse = np.mean((orig_flat - comp_flat) ** 2)
    
    return {'correlation': corr, 'mse': mse}

def compute_detection_metrics(mal_emb, norm_emb):
    """计算检测指标，包括AUC"""
    centroid_mal = mal_emb.mean(axis=0)
    centroid_mal_norm = centroid_mal / np.linalg.norm(centroid_mal)
    
    mal_sims = np.array([np.dot(e/np.linalg.norm(e), centroid_mal_norm) for e in mal_emb])
    norm_sims = np.array([np.dot(e/np.linalg.norm(e), centroid_mal_norm) for e in norm_emb])
    
    # 构建标签和分数
    y_true = np.concatenate([np.ones(len(mal_sims)), np.zeros(len(norm_sims))])
    y_scores = np.concatenate([mal_sims, norm_sims])
    
    # 计算AUC
    auc = roc_auc_score(y_true, y_scores)
    
    # 使用阈值计算检出率和误报率
    threshold = (mal_sims.mean() + norm_sims.mean()) / 2
    detection_rate = (mal_sims > threshold).sum() / len(mal_sims)
    false_positive_rate = (norm_sims > threshold).sum() / len(norm_sims)
    
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
print("严格划分训练/测试集实验")
print("=" * 70)

print("\n[1] 加载数据...")

# 恶意样本
advbench_emb = np.load("data/embeddings/advbench_embeddings.npy")
harmbench_emb = np.load("data/embeddings/harmbench_embeddings.npy")
advbench_str_emb = np.load("data/embeddings/advbench_strings_embeddings.npy")

# 正常样本
normal_emb = np.load("data/embeddings/normal_embeddings.npy")

print(f"  AdvBench: {advbench_emb.shape[0]} 条")
print(f"  HarmBench: {harmbench_emb.shape[0]} 条")
print(f"  AdvBench_Strings: {advbench_str_emb.shape[0]} 条")
print(f"  正常样本: {normal_emb.shape[0]} 条")

# ============================================================
# 数据划分
# ============================================================

print("\n[2] 划分训练/测试集 (80%/20%)...")

# AdvBench 划分
advbench_train, advbench_test = train_test_split(
    advbench_emb, test_size=0.2, random_state=RANDOM_SEED
)

# 正常样本划分
normal_train, normal_test = train_test_split(
    normal_emb, test_size=0.2, random_state=RANDOM_SEED
)

print(f"  AdvBench Train: {advbench_train.shape[0]} 条")
print(f"  AdvBench Test: {advbench_test.shape[0]} 条")
print(f"  正常样本 Train: {normal_train.shape[0]} 条")
print(f"  正常样本 Test: {normal_test.shape[0]} 条")

# 合并训练数据
train_emb = np.vstack([advbench_train, normal_train])
print(f"  训练集总计: {train_emb.shape[0]} 条")

# ============================================================
# 实验1: 在测试集上评估（严格划分）
# ============================================================

print("\n" + "=" * 70)
print("[实验1] 严格划分 - 在AdvBench测试集上评估")
print("=" * 70)

target_dims = [384, 38, 19]  # 原始, 10%, 5%
results_exp1 = {}

for target_dim in target_dims:
    dim_name = f"{target_dim}d" if target_dim < 384 else "original_384d"
    print(f"\n--- {dim_name} ---")
    
    if target_dim == 384:
        # 原始空间，不压缩
        mal_test = advbench_test
        norm_test = normal_test
        sim_pres = {'correlation': 1.0, 'mse': 0.0}
    else:
        # 仅在训练集上学习投影矩阵
        print(f"  在训练集上学习投影矩阵...", end=" ")
        model = train_learned_projection(train_emb, target_dim, epochs=300)
        print("完成")
        
        # 在测试集上应用投影
        mal_test = apply_projection(model, advbench_test)
        norm_test = apply_projection(model, normal_test)
        
        # 在测试集上评估相似度保持
        test_emb = np.vstack([advbench_test, normal_test])
        test_compressed = apply_projection(model, test_emb)
        sim_pres = compute_similarity_preservation(test_emb, test_compressed)
        print(f"  相似度保持 - Corr: {sim_pres['correlation']:.4f}, MSE: {sim_pres['mse']:.6f}")
    
    # 在测试集上评估检测效果
    metrics = compute_detection_metrics(mal_test, norm_test)
    print(f"  检测效果 - 检出率: {metrics['detection_rate']*100:.1f}%, 误报率: {metrics['false_positive_rate']*100:.1f}%, AUC: {metrics['auc']:.4f}")
    
    results_exp1[dim_name] = {
        'similarity_preservation': sim_pres,
        'detection': metrics,
    }

# ============================================================
# 实验2: 跨数据集验证 (Zero-shot Transfer)
# ============================================================

print("\n" + "=" * 70)
print("[实验2] 跨数据集验证 (Zero-shot Transfer)")
print("=" * 70)
print("使用AdvBench训练的投影矩阵，直接在其他数据集上测试")

# 在AdvBench训练集上训练投影矩阵
print("\n在AdvBench训练集上学习投影矩阵 (19维)...", end=" ")
model_19d = train_learned_projection(train_emb, 19, epochs=300)
print("完成")

results_exp2 = {}

# 测试数据集
test_datasets = {
    'AdvBench_Test': advbench_test,
    'HarmBench': harmbench_emb,
    'AdvBench_Strings': advbench_str_emb,
}

for dataset_name, mal_emb in test_datasets.items():
    print(f"\n--- {dataset_name} ({mal_emb.shape[0]} 条) ---")
    
    # 取匹配数量的正常样本
    n_samples = min(len(mal_emb), len(normal_test))
    norm_subset = normal_test[:n_samples]
    mal_subset = mal_emb[:n_samples]
    
    dataset_results = {}
    
    # 原始空间
    orig_metrics = compute_detection_metrics(mal_subset, norm_subset)
    dataset_results['original_384d'] = orig_metrics
    print(f"  原始(384维): 检出率={orig_metrics['detection_rate']*100:.1f}%, 误报率={orig_metrics['false_positive_rate']*100:.1f}%, AUC={orig_metrics['auc']:.4f}")
    
    # 应用投影（Zero-shot，不重新训练）
    mal_compressed = apply_projection(model_19d, mal_subset)
    norm_compressed = apply_projection(model_19d, norm_subset)
    
    compressed_metrics = compute_detection_metrics(mal_compressed, norm_compressed)
    dataset_results['learned_19d_zeroshot'] = compressed_metrics
    print(f"  学习型(19维,Zero-shot): 检出率={compressed_metrics['detection_rate']*100:.1f}%, 误报率={compressed_metrics['false_positive_rate']*100:.1f}%, AUC={compressed_metrics['auc']:.4f}")
    
    results_exp2[dataset_name] = dataset_results

# ============================================================
# 总结
# ============================================================

print("\n" + "=" * 70)
print("实验总结")
print("=" * 70)

print("\n[实验1] 严格划分 - AdvBench测试集结果:")
print("-" * 60)
print(f"{'维度':<15} {'检出率':<12} {'误报率':<12} {'AUC':<10} {'相似度Corr':<12}")
print("-" * 60)
for dim_name, res in results_exp1.items():
    det = res['detection']['detection_rate'] * 100
    fpr = res['detection']['false_positive_rate'] * 100
    auc = res['detection']['auc']
    corr = res['similarity_preservation']['correlation']
    print(f"{dim_name:<15} {det:<12.1f} {fpr:<12.1f} {auc:<10.4f} {corr:<12.4f}")

print("\n[实验2] 跨数据集验证 (Zero-shot Transfer):")
print("-" * 70)
print(f"{'数据集':<20} {'方法':<20} {'检出率':<12} {'误报率':<12} {'AUC':<10}")
print("-" * 70)
for dataset_name, res in results_exp2.items():
    for method, metrics in res.items():
        det = metrics['detection_rate'] * 100
        fpr = metrics['false_positive_rate'] * 100
        auc = metrics['auc']
        print(f"{dataset_name:<20} {method:<20} {det:<12.1f} {fpr:<12.1f} {auc:<10.4f}")

# 保存结果
all_results = {
    'experiment_1_strict_split': results_exp1,
    'experiment_2_zeroshot_transfer': results_exp2,
    'config': {
        'train_test_split': 0.8,
        'random_seed': RANDOM_SEED,
        'target_dims': target_dims,
    }
}

with open(f"{RESULTS_DIR}/strict_split_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n结果已保存到: {RESULTS_DIR}/strict_split_results.json")
