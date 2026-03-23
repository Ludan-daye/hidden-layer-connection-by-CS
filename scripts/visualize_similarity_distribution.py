"""
可视化正常样本和恶意样本与质心的余弦相似度分布

包括：
1. 原始384维空间的分布
2. 19维投影空间的分布
3. JailbreakBench攻击样本的分布
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from fastembed import TextEmbedding
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
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

def compute_similarities(embeddings, centroid):
    centroid_norm = centroid / np.linalg.norm(centroid)
    sims = []
    for e in embeddings:
        e_norm = e / np.linalg.norm(e)
        sims.append(np.dot(e_norm, centroid_norm))
    return np.array(sims)

# ============================================================
# 主程序
# ============================================================

print("=" * 70)
print("正常/恶意样本与质心的余弦相似度分布可视化")
print("=" * 70)

# 1. 加载数据
print("\n[1] 加载Embedding数据...")
advbench_emb = np.load("embedding_db/bge-small-en-v1.5/embeddings/advbench_embeddings.npy")
normal_emb = np.load("embedding_db/bge-small-en-v1.5/embeddings/normal_embeddings.npy")

advbench_train, advbench_test = train_test_split(advbench_emb, test_size=0.2, random_state=RANDOM_SEED)
normal_train, normal_test = train_test_split(normal_emb, test_size=0.2, random_state=RANDOM_SEED)

print(f"  恶意样本: 训练{advbench_train.shape[0]}条, 测试{advbench_test.shape[0]}条")
print(f"  正常样本: 训练{normal_train.shape[0]}条, 测试{normal_test.shape[0]}条")

# 2. 加载JailbreakBench攻击数据
print("\n[2] 加载JailbreakBench攻击数据...")
embed_model = TextEmbedding('BAAI/bge-small-en-v1.5')

jbb_data = {}
for method in ['gcg', 'pair', 'jbc']:
    csv_path = f"datasets/jailbreakbench/jbb_{method}_all.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        prompts = [p for p in df['prompt'].tolist() if p is not None and isinstance(p, str)]
        jbb_data[method.upper()] = np.array(list(embed_model.embed(prompts)))
        print(f"  {method.upper()}: {len(prompts)}条")

# 3. 训练投影矩阵
print("\n[3] 训练投影矩阵...")
train_emb = np.vstack([advbench_train, normal_train])
model_19d = train_learned_projection(train_emb, 19, epochs=300)
print("  19维投影矩阵训练完成")

# 4. 计算质心
centroid_orig = advbench_train.mean(axis=0)
advbench_train_19d = apply_projection(model_19d, advbench_train)
centroid_19d = advbench_train_19d.mean(axis=0)

# 5. 计算相似度分布
print("\n[4] 计算相似度分布...")

# 原始空间
mal_sims_orig = compute_similarities(advbench_emb, centroid_orig)
norm_sims_orig = compute_similarities(normal_emb, centroid_orig)

# 19维空间
advbench_19d = apply_projection(model_19d, advbench_emb)
normal_19d = apply_projection(model_19d, normal_emb)
mal_sims_19d = compute_similarities(advbench_19d, centroid_19d)
norm_sims_19d = compute_similarities(normal_19d, centroid_19d)

# JailbreakBench攻击
jbb_sims_orig = {}
jbb_sims_19d = {}
for method, embs in jbb_data.items():
    jbb_sims_orig[method] = compute_similarities(embs, centroid_orig)
    embs_19d = apply_projection(model_19d, embs)
    jbb_sims_19d[method] = compute_similarities(embs_19d, centroid_19d)

# 6. 计算阈值
threshold_orig = (mal_sims_orig.mean() + norm_sims_orig.mean()) / 2
threshold_19d = (mal_sims_19d.mean() + norm_sims_19d.mean()) / 2

# 7. 打印统计数据
print("\n" + "=" * 70)
print("统计数据")
print("=" * 70)

print(f"\n{'样本类型':<20} {'原始384维均值':<15} {'原始384维标准差':<15} {'19维均值':<15} {'19维标准差':<15}")
print("-" * 80)
print(f"{'恶意样本(AdvBench)':<20} {mal_sims_orig.mean():.4f}         {mal_sims_orig.std():.4f}           {mal_sims_19d.mean():.4f}         {mal_sims_19d.std():.4f}")
print(f"{'正常样本(Alpaca)':<20} {norm_sims_orig.mean():.4f}         {norm_sims_orig.std():.4f}           {norm_sims_19d.mean():.4f}         {norm_sims_19d.std():.4f}")

for method in jbb_sims_orig:
    print(f"{'JBB-'+method:<20} {jbb_sims_orig[method].mean():.4f}         {jbb_sims_orig[method].std():.4f}           {jbb_sims_19d[method].mean():.4f}         {jbb_sims_19d[method].std():.4f}")

print(f"\n阈值: 原始空间={threshold_orig:.4f}, 19维空间={threshold_19d:.4f}")

# 8. 创建可视化
print("\n[5] 创建可视化图...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 图1: 原始空间 - 正常 vs 恶意
ax1 = axes[0, 0]
ax1.hist(norm_sims_orig, bins=50, alpha=0.7, label=f'Normal (Alpaca) μ={norm_sims_orig.mean():.3f}', color='green')
ax1.hist(mal_sims_orig, bins=50, alpha=0.7, label=f'Malicious (AdvBench) μ={mal_sims_orig.mean():.3f}', color='red')
ax1.axvline(threshold_orig, color='black', linestyle='--', linewidth=2, label=f'Threshold={threshold_orig:.3f}')
ax1.set_xlabel('Cosine Similarity to Malicious Centroid', fontsize=12)
ax1.set_ylabel('Count', fontsize=12)
ax1.set_title('Original Space (384D): Normal vs Malicious', fontsize=14)
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# 图2: 19维空间 - 正常 vs 恶意
ax2 = axes[0, 1]
ax2.hist(norm_sims_19d, bins=50, alpha=0.7, label=f'Normal (Alpaca) μ={norm_sims_19d.mean():.3f}', color='green')
ax2.hist(mal_sims_19d, bins=50, alpha=0.7, label=f'Malicious (AdvBench) μ={mal_sims_19d.mean():.3f}', color='red')
ax2.axvline(threshold_19d, color='black', linestyle='--', linewidth=2, label=f'Threshold={threshold_19d:.3f}')
ax2.set_xlabel('Cosine Similarity to Malicious Centroid', fontsize=12)
ax2.set_ylabel('Count', fontsize=12)
ax2.set_title('Learned Projection (19D): Normal vs Malicious', fontsize=14)
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

# 图3: 原始空间 - JailbreakBench攻击
ax3 = axes[1, 0]
ax3.hist(norm_sims_orig, bins=50, alpha=0.5, label=f'Normal μ={norm_sims_orig.mean():.3f}', color='green')
colors = {'GCG': 'red', 'PAIR': 'orange', 'JBC': 'purple'}
for method, sims in jbb_sims_orig.items():
    ax3.hist(sims, bins=50, alpha=0.5, label=f'{method} μ={sims.mean():.3f}', color=colors.get(method, 'blue'))
ax3.axvline(threshold_orig, color='black', linestyle='--', linewidth=2, label=f'Threshold={threshold_orig:.3f}')
ax3.set_xlabel('Cosine Similarity to Malicious Centroid', fontsize=12)
ax3.set_ylabel('Count', fontsize=12)
ax3.set_title('Original Space (384D): JailbreakBench Attacks', fontsize=14)
ax3.legend(loc='upper left')
ax3.grid(True, alpha=0.3)

# 图4: 19维空间 - JailbreakBench攻击
ax4 = axes[1, 1]
ax4.hist(norm_sims_19d, bins=50, alpha=0.5, label=f'Normal μ={norm_sims_19d.mean():.3f}', color='green')
for method, sims in jbb_sims_19d.items():
    ax4.hist(sims, bins=50, alpha=0.5, label=f'{method} μ={sims.mean():.3f}', color=colors.get(method, 'blue'))
ax4.axvline(threshold_19d, color='black', linestyle='--', linewidth=2, label=f'Threshold={threshold_19d:.3f}')
ax4.set_xlabel('Cosine Similarity to Malicious Centroid', fontsize=12)
ax4.set_ylabel('Count', fontsize=12)
ax4.set_title('Learned Projection (19D): JailbreakBench Attacks', fontsize=14)
ax4.legend(loc='upper left')
ax4.grid(True, alpha=0.3)

plt.tight_layout()

# 保存图片
os.makedirs("embedding_db/bge-small-en-v1.5/results/figures", exist_ok=True)
fig_path = "embedding_db/bge-small-en-v1.5/results/figures/similarity_distribution.png"
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"\n图片已保存到: {fig_path}")

# 9. 创建箱线图对比
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

# 原始空间箱线图
ax_box1 = axes2[0]
data_orig = [norm_sims_orig, mal_sims_orig] + [jbb_sims_orig[m] for m in ['GCG', 'PAIR', 'JBC'] if m in jbb_sims_orig]
labels_orig = ['Normal\n(Alpaca)', 'Malicious\n(AdvBench)'] + [f'JBB\n{m}' for m in ['GCG', 'PAIR', 'JBC'] if m in jbb_sims_orig]
bp1 = ax_box1.boxplot(data_orig, labels=labels_orig, patch_artist=True)
colors_box = ['green', 'red', 'red', 'orange', 'purple']
for patch, color in zip(bp1['boxes'], colors_box[:len(data_orig)]):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax_box1.axhline(threshold_orig, color='black', linestyle='--', linewidth=2, label=f'Threshold={threshold_orig:.3f}')
ax_box1.set_ylabel('Cosine Similarity to Malicious Centroid', fontsize=12)
ax_box1.set_title('Original Space (384D)', fontsize=14)
ax_box1.legend()
ax_box1.grid(True, alpha=0.3)

# 19维空间箱线图
ax_box2 = axes2[1]
data_19d = [norm_sims_19d, mal_sims_19d] + [jbb_sims_19d[m] for m in ['GCG', 'PAIR', 'JBC'] if m in jbb_sims_19d]
labels_19d = ['Normal\n(Alpaca)', 'Malicious\n(AdvBench)'] + [f'JBB\n{m}' for m in ['GCG', 'PAIR', 'JBC'] if m in jbb_sims_19d]
bp2 = ax_box2.boxplot(data_19d, labels=labels_19d, patch_artist=True)
for patch, color in zip(bp2['boxes'], colors_box[:len(data_19d)]):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax_box2.axhline(threshold_19d, color='black', linestyle='--', linewidth=2, label=f'Threshold={threshold_19d:.3f}')
ax_box2.set_ylabel('Cosine Similarity to Malicious Centroid', fontsize=12)
ax_box2.set_title('Learned Projection (19D)', fontsize=14)
ax_box2.legend()
ax_box2.grid(True, alpha=0.3)

plt.tight_layout()

fig_path2 = "embedding_db/bge-small-en-v1.5/results/figures/similarity_boxplot.png"
plt.savefig(fig_path2, dpi=150, bbox_inches='tight')
print(f"箱线图已保存到: {fig_path2}")

print("\n完成!")
