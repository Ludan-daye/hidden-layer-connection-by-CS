#!/usr/bin/env python3
"""
使用技术类正常样本重新训练投影矩阵

训练数据：
- 恶意样本：AdvBench (416条)
- 正常样本：原始Alpaca (160条) + XSTest (450条) + OR-Bench Hard (1319条)

目标：让模型学会区分"学习安全"和"实施攻击"
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

def train_projection(mal_emb, norm_emb, input_dim=384, output_dim=19, epochs=100, lr=0.01):
    """训练投影矩阵"""
    projection = LearnedProjection(input_dim, output_dim)
    optimizer = torch.optim.Adam(projection.parameters(), lr=lr)
    
    # 合并数据
    all_emb = np.vstack([mal_emb, norm_emb])
    all_emb_tensor = torch.tensor(all_emb, dtype=torch.float32)
    
    # 计算原始相似度矩阵
    all_emb_norm = F.normalize(all_emb_tensor, p=2, dim=-1)
    original_sim = all_emb_norm @ all_emb_norm.T
    
    dataset = TensorDataset(all_emb_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    print(f"\n  开始训练投影矩阵 ({input_dim}→{output_dim})...")
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x = batch[0]
            optimizer.zero_grad()
            
            # 投影
            z = projection(x)
            
            # 计算投影后的相似度
            proj_sim = z @ z.T
            
            # 损失：保持相似度结构
            loss = F.mse_loss(proj_sim, original_sim[:len(x), :len(x)])
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {total_loss:.6f}")
    
    return projection

def evaluate(projection, mal_emb, norm_emb, centroid_norm, threshold):
    """评估检测效果"""
    with torch.no_grad():
        mal_proj = projection(torch.tensor(mal_emb, dtype=torch.float32)).numpy()
        norm_proj = projection(torch.tensor(norm_emb, dtype=torch.float32)).numpy()
    
    mal_sims = [np.dot(e / np.linalg.norm(e), centroid_norm) for e in mal_proj]
    norm_sims = [np.dot(e / np.linalg.norm(e), centroid_norm) for e in norm_proj]
    
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
        "mal_sim_mean": np.mean(mal_sims),
        "norm_sim_mean": np.mean(norm_sims),
        "threshold": threshold
    }

def main():
    print("=" * 70)
    print("使用技术类正常样本重新训练投影矩阵")
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
    print(f"  技术类Embedding形状: {tech_emb.shape}")
    
    # 4. 合并正常样本
    print("\n[4/6] 合并正常样本...")
    all_normal_emb = np.vstack([normal_emb, tech_emb])
    print(f"  合并后正常样本: {len(all_normal_emb)}条")
    
    # 划分训练/测试集
    advbench_train, advbench_test = train_test_split(advbench_emb, test_size=0.2, random_state=RANDOM_SEED)
    normal_train, normal_test = train_test_split(all_normal_emb, test_size=0.2, random_state=RANDOM_SEED)
    
    print(f"  恶意训练集: {len(advbench_train)}条")
    print(f"  正常训练集: {len(normal_train)}条")
    print(f"  恶意测试集: {len(advbench_test)}条")
    print(f"  正常测试集: {len(normal_test)}条")
    
    # 5. 训练投影矩阵
    print("\n[5/6] 训练投影矩阵...")
    projection = train_projection(advbench_train, normal_train, input_dim=384, output_dim=19, epochs=100)
    
    # 计算质心和阈值
    with torch.no_grad():
        advbench_proj = projection(torch.tensor(advbench_train, dtype=torch.float32)).numpy()
        normal_proj = projection(torch.tensor(normal_train, dtype=torch.float32)).numpy()
    
    centroid = advbench_proj.mean(axis=0)
    centroid_norm = centroid / np.linalg.norm(centroid)
    
    mal_sims = [np.dot(e / np.linalg.norm(e), centroid_norm) for e in advbench_proj]
    norm_sims = [np.dot(e / np.linalg.norm(e), centroid_norm) for e in normal_proj]
    
    # 使用95百分位阈值
    threshold = np.percentile(norm_sims, 95)
    
    print(f"\n  恶意样本平均相似度: {np.mean(mal_sims):.4f}")
    print(f"  正常样本平均相似度: {np.mean(norm_sims):.4f}")
    print(f"  检测阈值 (95%分位): {threshold:.4f}")
    
    # 6. 评估
    print("\n[6/6] 评估检测效果...")
    
    # 在测试集上评估
    test_results = evaluate(projection, advbench_test, normal_test, centroid_norm, threshold)
    print(f"\n  测试集结果:")
    print(f"    检出率: {test_results['detection_rate']*100:.1f}%")
    print(f"    误报率: {test_results['false_positive_rate']*100:.1f}%")
    
    # 保存模型
    save_dir = "embedding_db/bge-small-en-v1.5/results"
    os.makedirs(save_dir, exist_ok=True)
    
    weights = projection.proj.weight.data.numpy()
    np.save(f"{save_dir}/learned_projection_weights_19d_v2.npy", weights)
    np.save(f"{save_dir}/centroid_19d_v2.npy", centroid_norm)
    
    results = {
        "version": "v2_with_tech_samples",
        "train_samples": {
            "advbench": len(advbench_train),
            "normal_original": len(normal_emb),
            "normal_tech": len(tech_emb),
            "normal_total": len(normal_train)
        },
        "threshold": float(threshold),
        "test_results": {
            "detection_rate": test_results['detection_rate'],
            "false_positive_rate": test_results['false_positive_rate']
        }
    }
    
    with open(f"{save_dir}/training_v2_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  模型已保存到:")
    print(f"    - {save_dir}/learned_projection_weights_19d_v2.npy")
    print(f"    - {save_dir}/centroid_19d_v2.npy")
    print(f"    - {save_dir}/training_v2_results.json")
    
    print("\n" + "=" * 70)
    print("训练完成!")
    print("=" * 70)

if __name__ == "__main__":
    main()
