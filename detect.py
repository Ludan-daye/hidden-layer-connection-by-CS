#!/usr/bin/env python3
"""
交互式恶意输入检测接口

使用方法:
    python detect.py

输入文本后，系统会显示:
- 原始Embedding维度和前几维数值
- 压缩后19维向量
- 与恶意质心的余弦相似度
- 检测阈值
- 最终判定结果（恶意/正常）
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastembed import TextEmbedding
# from sklearn.model_selection import train_test_split  # v3模型不需要
import os

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

def load_detector():
    """加载检测器所需的所有组件"""
    print("=" * 60)
    print("加载恶意输入检测器...")
    print("=" * 60)
    
    # 1. 加载Embedding模型
    print("\n[1/4] 加载Embedding模型 (bge-small-en-v1.5)...")
    embed_model = TextEmbedding('BAAI/bge-small-en-v1.5')
    print("      ✓ Embedding模型加载完成 (384维)")
    
    # 2. 加载投影矩阵 (v3对比学习模型)
    print("\n[2/4] 加载投影矩阵 (v3对比学习模型)...")
    weights_path = "embedding_db/bge-small-en-v1.5/results/learned_projection_weights_19d_contrastive_v3.npy"
    weights = np.load(weights_path)
    projection = LearnedProjection(384, 19)
    projection.proj.weight.data = torch.tensor(weights, dtype=torch.float32)
    print(f"      ✓ 投影矩阵加载完成 (384→19维)")
    
    # 3. 加载质心和阈值
    print("\n[3/4] 加载恶意质心和检测阈值...")
    centroid_norm = np.load("embedding_db/bge-small-en-v1.5/results/centroid_19d_contrastive_v3.npy")
    threshold = -0.4118  # v3模型的95%分位阈值
    
    print(f"      ✓ 恶意质心: 19维向量")
    print(f"      ✓ 检测阈值: {threshold:.4f}")
    
    print("\n[4/4] 检测器初始化完成!")
    
    return embed_model, projection, centroid_norm, threshold

def detect(text, embed_model, projection, centroid_norm, threshold):
    """检测单条文本"""
    
    print("\n" + "=" * 60)
    print("输入文本:")
    print(f"  \"{text}\"")
    print("=" * 60)
    
    # 1. 计算Embedding
    embedding = np.array(list(embed_model.embed([text])))[0]
    print(f"\n[Step 1] 原始Embedding")
    print(f"  维度: {embedding.shape[0]}")
    print(f"  前5维: [{', '.join([f'{x:.4f}' for x in embedding[:5]])}]")
    print(f"  L2范数: {np.linalg.norm(embedding):.4f}")
    
    # 2. 投影到19维
    with torch.no_grad():
        emb_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
        projected = projection(emb_tensor).numpy()[0]
    
    print(f"\n[Step 2] 压缩后Embedding")
    print(f"  维度: {projected.shape[0]}")
    print(f"  完整向量: [{', '.join([f'{x:.4f}' for x in projected])}]")
    print(f"  L2范数: {np.linalg.norm(projected):.4f} (已归一化)")
    
    # 3. 计算相似度
    similarity = np.dot(projected, centroid_norm)
    
    print(f"\n[Step 3] 与恶意质心的余弦相似度")
    print(f"  相似度: {similarity:.4f}")
    print(f"  阈值:   {threshold:.4f}")
    
    # 4. 判定
    is_malicious = similarity > threshold
    
    print(f"\n[Step 4] 判定结果")
    print("=" * 60)
    if is_malicious:
        print(f"  ⚠️  【恶意】 相似度 {similarity:.4f} > 阈值 {threshold:.4f}")
        print(f"  该输入被判定为恶意攻击!")
    else:
        print(f"  ✅  【正常】 相似度 {similarity:.4f} ≤ 阈值 {threshold:.4f}")
        print(f"  该输入被判定为正常请求。")
    print("=" * 60)
    
    return is_malicious, similarity

def main():
    # 加载检测器
    embed_model, projection, centroid_norm, threshold = load_detector()
    
    print("\n" + "=" * 60)
    print("交互式恶意输入检测器")
    print("=" * 60)
    print("输入文本进行检测，输入 'quit' 或 'exit' 退出")
    print("=" * 60)
    
    while True:
        try:
            print("\n")
            text = input("请输入要检测的文本: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("\n再见!")
                break
            
            if not text:
                print("输入为空，请重新输入。")
                continue
            
            detect(text, embed_model, projection, centroid_norm, threshold)
            
        except KeyboardInterrupt:
            print("\n\n检测器已退出。")
            break
        except Exception as e:
            print(f"\n错误: {e}")

if __name__ == "__main__":
    main()
