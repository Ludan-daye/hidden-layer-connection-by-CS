"""
学习型投影矩阵实验

目标：学习一个投影矩阵Φ，使压缩后的余弦相似度与原始保持一致
对比随机投影 vs 学习型投影的效果
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import json
import os

# ============================================================
# 学习型投影模型
# ============================================================

class LearnedProjection(nn.Module):
    """学习型投影矩阵"""
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim, bias=False)
        # 高斯初始化
        nn.init.normal_(self.proj.weight, mean=0, std=1/np.sqrt(output_dim))
    
    def forward(self, x):
        # 投影
        z = self.proj(x)
        # L2归一化
        z = F.normalize(z, p=2, dim=-1)
        return z


def train_learned_projection(embeddings, target_dim, epochs=200, lr=0.01, batch_size=128):
    """
    训练学习型投影矩阵
    
    目标：保持压缩前后的余弦相似度一致
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 转换为tensor
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
    n_samples = embeddings_tensor.shape[0]
    input_dim = embeddings_tensor.shape[1]
    
    # 预计算原始相似度矩阵（归一化后）
    embeddings_norm = F.normalize(embeddings_tensor, p=2, dim=-1)
    
    # 模型
    model = LearnedProjection(input_dim, target_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    history = {'loss': [], 'sim_corr': []}
    
    for epoch in range(epochs):
        # 随机采样batch
        if n_samples > batch_size:
            indices = torch.randperm(n_samples)[:batch_size]
            batch = embeddings_tensor[indices]
            batch_norm = embeddings_norm[indices]
        else:
            batch = embeddings_tensor
            batch_norm = embeddings_norm
        
        # 原始相似度矩阵
        original_sim = batch_norm @ batch_norm.T
        
        # 压缩后相似度矩阵
        compressed = model(batch)
        compressed_sim = compressed @ compressed.T
        
        # 损失：MSE
        loss = F.mse_loss(compressed_sim, original_sim)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 记录
        with torch.no_grad():
            # 计算相关系数
            orig_flat = original_sim.flatten().cpu().numpy()
            comp_flat = compressed_sim.flatten().cpu().numpy()
            corr = np.corrcoef(orig_flat, comp_flat)[0, 1]
            
            history['loss'].append(loss.item())
            history['sim_corr'].append(corr)
        
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Loss: {loss.item():.6f} | SimCorr: {corr:.4f}")
    
    return model, history


# ============================================================
# 主实验
# ============================================================

print("=" * 70)
print("学习型投影矩阵实验")
print("=" * 70)

# 加载数据
print("\n[1] 加载数据...")
malicious_emb = np.load("embedding_db/bge-small-en-v1.5/embeddings/advbench_embeddings.npy")
print(f"    恶意样本: {malicious_emb.shape}")

# 生成正常样本
from fastembed import TextEmbedding

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
]
extended_normal = []
for topic in topics:
    for template in templates:
        extended_normal.append(template.format(topic))
        if len(extended_normal) >= 520:
            break
    if len(extended_normal) >= 520:
        break

model_embed = TextEmbedding('BAAI/bge-small-en-v1.5')
normal_emb = np.array(list(model_embed.embed(extended_normal[:len(malicious_emb)])))
print(f"    正常样本: {normal_emb.shape}")

# 合并数据用于训练
all_emb = np.vstack([malicious_emb, normal_emb])
print(f"    合并数据: {all_emb.shape}")

# ============================================================
# 对比实验
# ============================================================

def random_projection(data, target_dim, seed=42):
    """随机投影"""
    np.random.seed(seed)
    proj_matrix = np.random.randn(target_dim, data.shape[1]) / np.sqrt(target_dim)
    compressed = data @ proj_matrix.T
    compressed = compressed / np.linalg.norm(compressed, axis=1, keepdims=True)
    return compressed

def compute_similarity_preservation(original, compressed):
    """计算相似度保持程度"""
    # 归一化
    orig_norm = original / np.linalg.norm(original, axis=1, keepdims=True)
    
    # 原始相似度
    orig_sim = orig_norm @ orig_norm.T
    # 压缩后相似度
    comp_sim = compressed @ compressed.T
    
    # 相关系数
    corr = np.corrcoef(orig_sim.flatten(), comp_sim.flatten())[0, 1]
    # MSE
    mse = np.mean((orig_sim - comp_sim) ** 2)
    # 最大误差
    max_err = np.max(np.abs(orig_sim - comp_sim))
    
    return {'correlation': corr, 'mse': mse, 'max_error': max_err}

def compute_separability(emb_mal, emb_norm):
    """计算可分性"""
    c_mal = emb_mal.mean(axis=0)
    c_norm = emb_norm.mean(axis=0)
    inter_dist = np.linalg.norm(c_mal - c_norm)
    intra_mal = np.mean([np.linalg.norm(e - c_mal) for e in emb_mal])
    intra_norm = np.mean([np.linalg.norm(e - c_norm) for e in emb_norm])
    fisher_ratio = inter_dist / ((intra_mal + intra_norm) / 2 + 1e-8)
    return fisher_ratio

# 测试不同压缩维度
target_dims = [192, 96, 38, 19]  # 50%, 25%, 10%, 5%

print("\n" + "=" * 70)
print("[2] 训练学习型投影矩阵")
print("=" * 70)

results = []

for target_dim in target_dims:
    ratio = target_dim / 384
    print(f"\n--- 压缩到 {target_dim} 维 ({ratio*100:.0f}%) ---")
    
    # 随机投影
    print("  随机投影...")
    rand_compressed = random_projection(all_emb, target_dim)
    rand_mal = rand_compressed[:len(malicious_emb)]
    rand_norm = rand_compressed[len(malicious_emb):]
    
    rand_sim_pres = compute_similarity_preservation(all_emb, rand_compressed)
    rand_fisher = compute_separability(rand_mal, rand_norm)
    
    # 学习型投影
    print("  学习型投影训练...")
    learned_model, history = train_learned_projection(
        all_emb, target_dim, epochs=200, lr=0.01, batch_size=256
    )
    
    # 应用学习型投影
    with torch.no_grad():
        all_tensor = torch.tensor(all_emb, dtype=torch.float32)
        if torch.cuda.is_available():
            all_tensor = all_tensor.cuda()
            learned_model = learned_model.cuda()
        learned_compressed = learned_model(all_tensor).cpu().numpy()
    
    learned_mal = learned_compressed[:len(malicious_emb)]
    learned_norm = learned_compressed[len(malicious_emb):]
    
    learned_sim_pres = compute_similarity_preservation(all_emb, learned_compressed)
    learned_fisher = compute_separability(learned_mal, learned_norm)
    
    # 记录结果
    result = {
        'target_dim': target_dim,
        'ratio': ratio,
        'random': {
            'sim_correlation': rand_sim_pres['correlation'],
            'sim_mse': rand_sim_pres['mse'],
            'fisher_ratio': rand_fisher,
        },
        'learned': {
            'sim_correlation': learned_sim_pres['correlation'],
            'sim_mse': learned_sim_pres['mse'],
            'fisher_ratio': learned_fisher,
        }
    }
    results.append(result)
    
    print(f"\n  对比结果:")
    print(f"    相似度相关系数: 随机={rand_sim_pres['correlation']:.4f}, 学习={learned_sim_pres['correlation']:.4f}")
    print(f"    相似度MSE:      随机={rand_sim_pres['mse']:.6f}, 学习={learned_sim_pres['mse']:.6f}")
    print(f"    Fisher判别比:   随机={rand_fisher:.4f}, 学习={learned_fisher:.4f}")

# ============================================================
# 总结
# ============================================================

print("\n" + "=" * 70)
print("实验总结")
print("=" * 70)

print("\n相似度保持 (相关系数，越高越好):")
print("-" * 60)
print(f"{'维度':<10} {'随机投影':<15} {'学习型投影':<15} {'提升':<10}")
print("-" * 60)
for r in results:
    rand_corr = r['random']['sim_correlation']
    learn_corr = r['learned']['sim_correlation']
    improve = (learn_corr - rand_corr) / rand_corr * 100
    print(f"{r['target_dim']:<10} {rand_corr:<15.4f} {learn_corr:<15.4f} {improve:+.2f}%")

print("\n可分性 (Fisher判别比，越高越好):")
print("-" * 60)
print(f"{'维度':<10} {'随机投影':<15} {'学习型投影':<15} {'提升':<10}")
print("-" * 60)
for r in results:
    rand_fisher = r['random']['fisher_ratio']
    learn_fisher = r['learned']['fisher_ratio']
    improve = (learn_fisher - rand_fisher) / rand_fisher * 100
    print(f"{r['target_dim']:<10} {rand_fisher:<15.4f} {learn_fisher:<15.4f} {improve:+.2f}%")

# 保存结果
os.makedirs("embedding_db/bge-small-en-v1.5/results", exist_ok=True)
with open("embedding_db/bge-small-en-v1.5/results/learned_projection_experiment.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n结果已保存到: embedding_db/bge-small-en-v1.5/results/learned_projection_experiment.json")
