#!/usr/bin/env python3
"""
使用对比学习训练投影矩阵

目标：
- 拉近恶意样本与恶意质心的距离
- 推远正常样本与恶意质心的距离
- 让模型学会区分"学习安全"和"实施攻击"
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from fastembed import TextEmbedding
from sklearn.model_selection import train_test_split
import os
import json

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

def load_tech_normal_samples():
    """加载技术类正常样本"""
    samples = []
    
    # 1. XSTest
    xstest_path = "数据集/XSTest/data/train-00000-of-00001.parquet"
    if os.path.exists(xstest_path):
        xstest = pd.read_parquet(xstest_path)
        safe_samples = xstest[xstest['label'] == 'safe']['prompt'].tolist()
        samples.extend(safe_samples)
        print(f"  XSTest safe样本: {len(safe_samples)}条")
    
    # 2. OR-Bench Hard
    or_bench_path = "数据集/or-bench-hard-1k.csv"
    if os.path.exists(or_bench_path):
        or_bench = pd.read_csv(or_bench_path)
        or_samples = or_bench['prompt'].tolist()
        samples.extend(or_samples)
        print(f"  OR-Bench Hard样本: {len(or_samples)}条")
    
    return samples

def compute_embeddings(texts, embed_model, batch_size=32):
    """计算文本的Embedding"""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_emb = list(embed_model.embed(batch))
        embeddings.extend(batch_emb)
        if (i // batch_size) % 10 == 0:
            print(f"    处理进度: {min(i+batch_size, len(texts))}/{len(texts)}")
    return np.array(embeddings)

def contrastive_loss(mal_proj, norm_proj, centroid, margin=0.3):
    """
    对比损失函数
    - 恶意样本应该接近质心（相似度高）
    - 正常样本应该远离质心（相似度低）
    """
    # 计算与质心的相似度
    mal_sim = torch.sum(mal_proj * centroid, dim=-1)  # 恶意样本相似度
    norm_sim = torch.sum(norm_proj * centroid, dim=-1)  # 正常样本相似度
    
    # 恶意样本：希望相似度尽量高（接近1）
    mal_loss = torch.mean(1 - mal_sim)
    
    # 正常样本：希望相似度尽量低（低于margin）
    norm_loss = torch.mean(F.relu(norm_sim - margin))
    
    # 分离损失：恶意样本相似度应该比正常样本高
    # 使用hinge loss
    separation_loss = torch.mean(F.relu(margin - (mal_sim.mean() - norm_sim.mean())))
    
    return mal_loss + norm_loss + separation_loss

def train_projection_contrastive(mal_emb, norm_emb, input_dim=384, output_dim=19, epochs=200, lr=0.01, margin=0.3):
    """使用对比学习训练投影矩阵"""
    projection = LearnedProjection(input_dim, output_dim)
    optimizer = torch.optim.Adam(projection.parameters(), lr=lr)
    
    mal_tensor = torch.tensor(mal_emb, dtype=torch.float32)
    norm_tensor = torch.tensor(norm_emb, dtype=torch.float32)
    
    print(f"\n  开始对比学习训练 ({input_dim}→{output_dim})...")
    print(f"  恶意样本: {len(mal_emb)}, 正常样本: {len(norm_emb)}")
    print(f"  Margin: {margin}")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 投影
        mal_proj = projection(mal_tensor)
        norm_proj = projection(norm_tensor)
        
        # 计算质心（动态更新）
        centroid = mal_proj.mean(dim=0)
        centroid = F.normalize(centroid, p=2, dim=0)
        
        # 计算损失
        loss = contrastive_loss(mal_proj, norm_proj, centroid, margin)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 40 == 0:
            with torch.no_grad():
                mal_sim = torch.sum(mal_proj * centroid, dim=-1).mean().item()
                norm_sim = torch.sum(norm_proj * centroid, dim=-1).mean().item()
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, "
                  f"Mal_sim: {mal_sim:.4f}, Norm_sim: {norm_sim:.4f}, Gap: {mal_sim-norm_sim:.4f}")
    
    return projection

def evaluate(projection, mal_emb, norm_emb, threshold):
    """评估检测效果"""
    with torch.no_grad():
        mal_proj = projection(torch.tensor(mal_emb, dtype=torch.float32))
        norm_proj = projection(torch.tensor(norm_emb, dtype=torch.float32))
        
        # 计算质心
        centroid = mal_proj.mean(dim=0)
        centroid = F.normalize(centroid, p=2, dim=0)
        
        mal_sims = torch.sum(mal_proj * centroid, dim=-1).numpy()
        norm_sims = torch.sum(norm_proj * centroid, dim=-1).numpy()
    
    # 计算指标
    TP = sum(1 for s in mal_sims if s > threshold)
    FP = sum(1 for s in norm_sims if s > threshold)
    TN = sum(1 for s in norm_sims if s <= threshold)
    FN = sum(1 for s in mal_sims if s <= threshold)
    
    detection_rate = TP / (TP + FN) if (TP + FN) > 0 else 0
    false_positive_rate = FP / (FP + TN) if (FP + TN) > 0 else 0
    
    return {
        "detection_rate": detection_rate,
        "false_positive_rate": false_positive_rate,
        "mal_sim_mean": float(np.mean(mal_sims)),
        "mal_sim_min": float(np.min(mal_sims)),
        "norm_sim_mean": float(np.mean(norm_sims)),
        "norm_sim_max": float(np.max(norm_sims)),
        "threshold": threshold
    }

def main():
    print("=" * 70)
    print("使用对比学习训练投影矩阵")
    print("=" * 70)
    
    # 1. 加载Embedding模型
    print("\n[1/6] 加载Embedding模型...")
    embed_model = TextEmbedding('BAAI/bge-small-en-v1.5')
    
    # 2. 加载原始数据
    print("\n[2/6] 加载原始训练数据...")
    advbench_emb = np.load("embedding_db/bge-small-en-v1.5/embeddings/advbench_embeddings.npy")
    normal_emb = np.load("embedding_db/bge-small-en-v1.5/embeddings/normal_embeddings.npy")
    print(f"  AdvBench: {len(advbench_emb)}条")
    print(f"  原始正常样本: {len(normal_emb)}条")
    
    # 3. 加载技术类正常样本
    print("\n[3/6] 加载技术类正常样本...")
    tech_samples = load_tech_normal_samples()
    print(f"  技术类正常样本总计: {len(tech_samples)}条")
    
    # 计算技术类样本的Embedding
    print("\n  计算技术类样本Embedding...")
    tech_emb = compute_embeddings(tech_samples, embed_model)
    
    # 4. 合并正常样本
    print("\n[4/6] 合并正常样本...")
    all_normal_emb = np.vstack([normal_emb, tech_emb])
    print(f"  合并后正常样本: {len(all_normal_emb)}条")
    
    # 划分训练/测试集
    advbench_train, advbench_test = train_test_split(advbench_emb, test_size=0.2, random_state=RANDOM_SEED)
    normal_train, normal_test = train_test_split(all_normal_emb, test_size=0.2, random_state=RANDOM_SEED)
    
    print(f"  恶意训练集: {len(advbench_train)}条")
    print(f"  正常训练集: {len(normal_train)}条")
    
    # 5. 对比学习训练
    print("\n[5/6] 对比学习训练...")
    projection = train_projection_contrastive(
        advbench_train, normal_train, 
        input_dim=384, output_dim=19, 
        epochs=200, lr=0.01, margin=0.3
    )
    
    # 计算质心和阈值
    with torch.no_grad():
        advbench_proj = projection(torch.tensor(advbench_train, dtype=torch.float32))
        normal_proj = projection(torch.tensor(normal_train, dtype=torch.float32))
        
        centroid = advbench_proj.mean(dim=0)
        centroid_norm = F.normalize(centroid, p=2, dim=0).numpy()
        
        mal_sims = torch.sum(advbench_proj * torch.tensor(centroid_norm), dim=-1).numpy()
        norm_sims = torch.sum(normal_proj * torch.tensor(centroid_norm), dim=-1).numpy()
    
    # 使用正常样本95百分位作为阈值
    threshold = np.percentile(norm_sims, 95)
    
    print(f"\n  恶意样本相似度: min={np.min(mal_sims):.4f}, mean={np.mean(mal_sims):.4f}, max={np.max(mal_sims):.4f}")
    print(f"  正常样本相似度: min={np.min(norm_sims):.4f}, mean={np.mean(norm_sims):.4f}, max={np.max(norm_sims):.4f}")
    print(f"  相似度差距: {np.mean(mal_sims) - np.mean(norm_sims):.4f}")
    print(f"  检测阈值 (95%分位): {threshold:.4f}")
    
    # 6. 评估
    print("\n[6/6] 评估检测效果...")
    
    # 在测试集上评估
    test_results = evaluate(projection, advbench_test, normal_test, threshold)
    print(f"\n  测试集结果:")
    print(f"    检出率: {test_results['detection_rate']*100:.1f}%")
    print(f"    误报率: {test_results['false_positive_rate']*100:.1f}%")
    
    # 保存模型
    save_dir = "embedding_db/bge-small-en-v1.5/results"
    os.makedirs(save_dir, exist_ok=True)
    
    weights = projection.proj.weight.data.numpy()
    np.save(f"{save_dir}/learned_projection_weights_19d_contrastive.npy", weights)
    np.save(f"{save_dir}/centroid_19d_contrastive.npy", centroid_norm)
    
    results = {
        "version": "contrastive_with_tech_samples",
        "train_samples": {
            "advbench": len(advbench_train),
            "normal_original": len(normal_emb),
            "normal_tech": len(tech_emb),
            "normal_total": len(normal_train)
        },
        "threshold": float(threshold),
        "similarity_stats": {
            "mal_sim_mean": float(np.mean(mal_sims)),
            "norm_sim_mean": float(np.mean(norm_sims)),
            "gap": float(np.mean(mal_sims) - np.mean(norm_sims))
        },
        "test_results": {
            "detection_rate": test_results['detection_rate'],
            "false_positive_rate": test_results['false_positive_rate']
        }
    }
    
    with open(f"{save_dir}/training_contrastive_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  模型已保存到:")
    print(f"    - {save_dir}/learned_projection_weights_19d_contrastive.npy")
    print(f"    - {save_dir}/centroid_19d_contrastive.npy")
    
    print("\n" + "=" * 70)
    print("训练完成!")
    print("=" * 70)

if __name__ == "__main__":
    main()
