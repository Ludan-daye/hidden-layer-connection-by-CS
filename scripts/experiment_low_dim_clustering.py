"""
恶意攻击Embedding低维聚集检测实验

实验目标：检测恶意攻击语句是否聚集在embedding空间的低维子空间中

实验设计：
1. PCA方差分布分析
2. 有效维度计算
3. 质心距离分布
4. CS压缩可分性测试
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from fastembed import TextEmbedding
import json
import os

# ============================================================
# 数据准备
# ============================================================

print("=" * 70)
print("实验：恶意攻击Embedding低维聚集检测")
print("=" * 70)

# 加载恶意样本 (AdvBench)
print("\n[1] 加载数据...")
malicious_emb = np.load("embedding_db/bge-small-en-v1.5/embeddings/advbench_embeddings.npy")
print(f"    恶意样本: {malicious_emb.shape}")

# 生成正常样本 - 使用随机正常问题
normal_texts = [
    "What is the capital of France?",
    "How do I cook pasta?",
    "Explain photosynthesis to me.",
    "What are the benefits of exercise?",
    "How does a computer work?",
    "What is the weather like today?",
    "Tell me about the solar system.",
    "How do I learn a new language?",
    "What is machine learning?",
    "Explain the water cycle.",
    "How do plants grow?",
    "What causes earthquakes?",
    "How do airplanes fly?",
    "What is the history of the internet?",
    "How do I improve my writing skills?",
    "What are the benefits of reading?",
    "Explain how vaccines work.",
    "What is climate change?",
    "How do I start a small business?",
    "What is artificial intelligence?",
]

# 扩展正常样本到520条
extended_normal = []
templates = [
    "Can you explain {}?",
    "What is the meaning of {}?",
    "How does {} work?",
    "Tell me about {}.",
    "What are the benefits of {}?",
    "Describe {} in simple terms.",
    "What is the history of {}?",
    "How can I learn about {}?",
    "What are the main features of {}?",
    "Explain the concept of {}.",
]

topics = [
    "mathematics", "science", "history", "geography", "literature",
    "music", "art", "cooking", "gardening", "photography",
    "programming", "design", "economics", "psychology", "philosophy",
    "biology", "chemistry", "physics", "astronomy", "medicine",
    "education", "sports", "travel", "technology", "engineering",
    "architecture", "fashion", "nutrition", "fitness", "meditation",
    "yoga", "dance", "painting", "sculpture", "poetry",
    "writing", "reading", "languages", "culture", "traditions",
    "festivals", "holidays", "nature", "wildlife", "environment",
    "sustainability", "recycling", "energy", "transportation", "communication",
]

for topic in topics:
    for template in templates:
        extended_normal.append(template.format(topic))
        if len(extended_normal) >= 520:
            break
    if len(extended_normal) >= 520:
        break

print(f"    生成正常样本: {len(extended_normal)} 条")

# 计算正常样本embedding
print("    计算正常样本embedding...")
model = TextEmbedding("BAAI/bge-small-en-v1.5")
normal_emb = np.array(list(model.embed(extended_normal)))
print(f"    正常样本embedding: {normal_emb.shape}")

# ============================================================
# 实验1: PCA方差分布分析
# ============================================================

print("\n" + "=" * 70)
print("[实验1] PCA方差分布分析")
print("=" * 70)

# 对恶意样本做PCA
pca_mal = PCA()
pca_mal.fit(malicious_emb)
var_ratio_mal = pca_mal.explained_variance_ratio_
cumsum_mal = np.cumsum(var_ratio_mal)

# 对正常样本做PCA
pca_norm = PCA()
pca_norm.fit(normal_emb)
var_ratio_norm = pca_norm.explained_variance_ratio_
cumsum_norm = np.cumsum(var_ratio_norm)

print("\n累积方差解释比例:")
print("-" * 50)
print(f"{'维度':<10} {'恶意样本':<15} {'正常样本':<15} {'差异':<10}")
print("-" * 50)

checkpoints = [10, 20, 50, 100, 200, 384]
for k in checkpoints:
    if k <= len(cumsum_mal):
        diff = cumsum_mal[k-1] - cumsum_norm[k-1]
        print(f"{k:<10} {cumsum_mal[k-1]:.4f}         {cumsum_norm[k-1]:.4f}         {diff:+.4f}")

# 找到解释90%方差需要的维度
dim_90_mal = np.argmax(cumsum_mal >= 0.90) + 1
dim_90_norm = np.argmax(cumsum_norm >= 0.90) + 1
dim_95_mal = np.argmax(cumsum_mal >= 0.95) + 1
dim_95_norm = np.argmax(cumsum_norm >= 0.95) + 1

print("\n关键发现:")
print(f"  解释90%方差所需维度: 恶意={dim_90_mal}, 正常={dim_90_norm}")
print(f"  解释95%方差所需维度: 恶意={dim_95_mal}, 正常={dim_95_norm}")

if dim_90_mal < dim_90_norm:
    print(f"  → 恶意样本更集中在低维空间 (少{dim_90_norm - dim_90_mal}维)")
else:
    print(f"  → 正常样本更集中在低维空间 (少{dim_90_mal - dim_90_norm}维)")

# ============================================================
# 实验2: 有效维度计算
# ============================================================

print("\n" + "=" * 70)
print("[实验2] 有效维度计算 (Participation Ratio)")
print("=" * 70)

def participation_ratio(eigenvalues):
    """计算参与比率 (有效维度)"""
    eigenvalues = eigenvalues / eigenvalues.sum()  # 归一化
    return 1.0 / np.sum(eigenvalues ** 2)

# 计算协方差矩阵的特征值
cov_mal = np.cov(malicious_emb.T)
cov_norm = np.cov(normal_emb.T)

eigenvalues_mal = np.linalg.eigvalsh(cov_mal)[::-1]
eigenvalues_norm = np.linalg.eigvalsh(cov_norm)[::-1]

pr_mal = participation_ratio(eigenvalues_mal[eigenvalues_mal > 0])
pr_norm = participation_ratio(eigenvalues_norm[eigenvalues_norm > 0])

print(f"\n有效维度 (Participation Ratio):")
print(f"  恶意样本: {pr_mal:.2f} / 384 ({pr_mal/384*100:.1f}%)")
print(f"  正常样本: {pr_norm:.2f} / 384 ({pr_norm/384*100:.1f}%)")
print(f"  差异: {pr_mal - pr_norm:+.2f}")

if pr_mal < pr_norm:
    print(f"  → 恶意样本有效维度更低，更集中在低维子空间")
else:
    print(f"  → 正常样本有效维度更低")

# ============================================================
# 实验3: 质心距离分布
# ============================================================

print("\n" + "=" * 70)
print("[实验3] 质心距离分布分析")
print("=" * 70)

# 计算质心
centroid_mal = malicious_emb.mean(axis=0)
centroid_norm = normal_emb.mean(axis=0)

# 归一化
centroid_mal_norm = centroid_mal / np.linalg.norm(centroid_mal)
centroid_norm_norm = centroid_norm / np.linalg.norm(centroid_norm)

# 计算每个样本与质心的余弦相似度
def cosine_to_centroid(embeddings, centroid):
    centroid_norm = centroid / np.linalg.norm(centroid)
    sims = []
    for emb in embeddings:
        emb_norm = emb / np.linalg.norm(emb)
        sims.append(np.dot(emb_norm, centroid_norm))
    return np.array(sims)

sims_mal = cosine_to_centroid(malicious_emb, centroid_mal)
sims_norm = cosine_to_centroid(normal_emb, centroid_norm)

print(f"\n与质心的余弦相似度统计:")
print("-" * 50)
print(f"{'指标':<15} {'恶意样本':<15} {'正常样本':<15}")
print("-" * 50)
print(f"{'均值':<15} {sims_mal.mean():.4f}         {sims_norm.mean():.4f}")
print(f"{'标准差':<15} {sims_mal.std():.4f}         {sims_norm.std():.4f}")
print(f"{'最小值':<15} {sims_mal.min():.4f}         {sims_norm.min():.4f}")
print(f"{'最大值':<15} {sims_mal.max():.4f}         {sims_norm.max():.4f}")

if sims_mal.mean() > sims_norm.mean():
    print(f"\n  → 恶意样本与质心更接近 (聚集性更强)")
else:
    print(f"\n  → 正常样本与质心更接近")

# 样本间平均相似度
print(f"\n样本间平均余弦相似度:")
sim_matrix_mal = cosine_similarity(malicious_emb)
sim_matrix_norm = cosine_similarity(normal_emb)

# 排除对角线
n_mal = sim_matrix_mal.shape[0]
n_norm = sim_matrix_norm.shape[0]
mask_mal = ~np.eye(n_mal, dtype=bool)
mask_norm = ~np.eye(n_norm, dtype=bool)
avg_sim_mal = sim_matrix_mal[mask_mal].mean()
avg_sim_norm = sim_matrix_norm[mask_norm].mean()

print(f"  恶意样本内部: {avg_sim_mal:.4f}")
print(f"  正常样本内部: {avg_sim_norm:.4f}")

if avg_sim_mal > avg_sim_norm:
    print(f"  → 恶意样本内部相似度更高 (更聚集)")
else:
    print(f"  → 正常样本内部相似度更高")

# ============================================================
# 实验4: CS压缩可分性测试
# ============================================================

print("\n" + "=" * 70)
print("[实验4] CS压缩可分性测试")
print("=" * 70)

def random_projection(data, target_dim, seed=42):
    """随机投影压缩"""
    np.random.seed(seed)
    original_dim = data.shape[1]
    # 高斯随机矩阵
    proj_matrix = np.random.randn(target_dim, original_dim) / np.sqrt(target_dim)
    compressed = data @ proj_matrix.T
    # L2归一化
    compressed = compressed / np.linalg.norm(compressed, axis=1, keepdims=True)
    return compressed

def compute_separability(emb_mal, emb_norm):
    """计算可分性指标"""
    # 质心
    c_mal = emb_mal.mean(axis=0)
    c_norm = emb_norm.mean(axis=0)
    
    # 质心间距离
    inter_dist = np.linalg.norm(c_mal - c_norm)
    
    # 类内平均距离
    intra_mal = np.mean([np.linalg.norm(e - c_mal) for e in emb_mal])
    intra_norm = np.mean([np.linalg.norm(e - c_norm) for e in emb_norm])
    intra_avg = (intra_mal + intra_norm) / 2
    
    # Fisher判别比
    fisher_ratio = inter_dist / (intra_avg + 1e-8)
    
    return {
        'inter_dist': inter_dist,
        'intra_mal': intra_mal,
        'intra_norm': intra_norm,
        'fisher_ratio': fisher_ratio
    }

compression_ratios = [1.0, 0.5, 0.25, 0.1, 0.05]  # 100%, 50%, 25%, 10%, 5%
target_dims = [int(384 * r) for r in compression_ratios]

print(f"\n不同压缩比下的可分性 (Fisher判别比):")
print("-" * 60)
print(f"{'压缩比':<10} {'维度':<10} {'Fisher比':<12} {'类间距离':<12} {'类内距离':<12}")
print("-" * 60)

results = []
for ratio, dim in zip(compression_ratios, target_dims):
    if dim < 1:
        dim = 1
    
    if ratio == 1.0:
        # 原始空间
        sep = compute_separability(malicious_emb, normal_emb)
    else:
        # 压缩空间
        mal_compressed = random_projection(malicious_emb, dim)
        norm_compressed = random_projection(normal_emb, dim)
        sep = compute_separability(mal_compressed, norm_compressed)
    
    results.append((ratio, dim, sep))
    print(f"{ratio*100:>6.0f}%    {dim:<10} {sep['fisher_ratio']:<12.4f} {sep['inter_dist']:<12.4f} {(sep['intra_mal']+sep['intra_norm'])/2:<12.4f}")

# 分析趋势
print("\n关键发现:")
fisher_original = results[0][2]['fisher_ratio']
fisher_10pct = results[3][2]['fisher_ratio']
fisher_5pct = results[4][2]['fisher_ratio']

if fisher_10pct > fisher_original * 0.8:
    print(f"  → 压缩到10%维度后，可分性保持良好 ({fisher_10pct/fisher_original*100:.1f}%)")
else:
    print(f"  → 压缩到10%维度后，可分性下降明显 ({fisher_10pct/fisher_original*100:.1f}%)")

# ============================================================
# 实验5: 低维投影检测
# ============================================================

print("\n" + "=" * 70)
print("[实验5] 基于PCA的低维检测")
print("=" * 70)

# 使用恶意样本的主成分方向
n_components = 20
pca_detector = PCA(n_components=n_components)
pca_detector.fit(malicious_emb)

# 投影
mal_proj = pca_detector.transform(malicious_emb)
norm_proj = pca_detector.transform(normal_emb)

# 计算投影后的L2范数（在恶意样本主成分空间的能量）
mal_energy = np.linalg.norm(mal_proj, axis=1)
norm_energy = np.linalg.norm(norm_proj, axis=1)

print(f"\n在恶意样本前{n_components}主成分空间的投影能量:")
print(f"  恶意样本: 均值={mal_energy.mean():.4f}, 标准差={mal_energy.std():.4f}")
print(f"  正常样本: 均值={norm_energy.mean():.4f}, 标准差={norm_energy.std():.4f}")

# 简单阈值检测
threshold = (mal_energy.mean() + norm_energy.mean()) / 2
mal_detected = (mal_energy > threshold).sum()
norm_detected = (norm_energy > threshold).sum()

print(f"\n使用阈值 {threshold:.4f} 进行检测:")
print(f"  恶意样本检出率: {mal_detected}/{len(mal_energy)} ({mal_detected/len(mal_energy)*100:.1f}%)")
print(f"  正常样本误报率: {norm_detected}/{len(norm_energy)} ({norm_detected/len(norm_energy)*100:.1f}%)")

# ============================================================
# 总结
# ============================================================

print("\n" + "=" * 70)
print("实验总结")
print("=" * 70)

print("""
1. PCA方差分布:
   - 恶意样本解释90%方差需要 {} 维
   - 正常样本解释90%方差需要 {} 维
   
2. 有效维度:
   - 恶意样本有效维度: {:.1f}
   - 正常样本有效维度: {:.1f}
   
3. 聚集性:
   - 恶意样本内部平均相似度: {:.4f}
   - 正常样本内部平均相似度: {:.4f}
   
4. CS压缩可分性:
   - 原始空间Fisher比: {:.4f}
   - 10%压缩后Fisher比: {:.4f}
   
5. 低维检测:
   - 恶意样本检出率: {:.1f}%
   - 正常样本误报率: {:.1f}%
""".format(
    dim_90_mal, dim_90_norm,
    pr_mal, pr_norm,
    avg_sim_mal, avg_sim_norm,
    fisher_original, fisher_10pct,
    mal_detected/len(mal_energy)*100,
    norm_detected/len(norm_energy)*100
))

# 保存结果
results_dict = {
    "pca_analysis": {
        "dim_90_malicious": int(dim_90_mal),
        "dim_90_normal": int(dim_90_norm),
        "dim_95_malicious": int(dim_95_mal),
        "dim_95_normal": int(dim_95_norm),
    },
    "effective_dimension": {
        "malicious": float(pr_mal),
        "normal": float(pr_norm),
    },
    "clustering": {
        "avg_sim_malicious": float(avg_sim_mal),
        "avg_sim_normal": float(avg_sim_norm),
        "centroid_sim_malicious_mean": float(sims_mal.mean()),
        "centroid_sim_normal_mean": float(sims_norm.mean()),
    },
    "compression_separability": [
        {"ratio": r, "dim": d, "fisher_ratio": s['fisher_ratio']}
        for r, d, s in results
    ],
    "detection": {
        "threshold": float(threshold),
        "malicious_detection_rate": float(mal_detected/len(mal_energy)),
        "normal_false_positive_rate": float(norm_detected/len(norm_energy)),
    }
}

os.makedirs("embedding_db/bge-small-en-v1.5/results", exist_ok=True)
with open("embedding_db/bge-small-en-v1.5/results/low_dim_clustering_experiment.json", "w") as f:
    json.dump(results_dict, f, indent=2)

print("结果已保存到: embedding_db/bge-small-en-v1.5/results/low_dim_clustering_experiment.json")
