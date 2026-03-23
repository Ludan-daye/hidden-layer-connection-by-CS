#!/usr/bin/env python3
"""
测试V4模型在完整ToxicChat数据集上的表现
ToxicChat包含10万+条对话，分为有毒和无毒两类
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastembed import TextEmbedding
from datasets import load_dataset
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

print("=" * 80)
print("V4模型测试 - 完整ToxicChat数据集")
print("=" * 80)

# 1. 加载模型
print("\n[1/4] 加载V4模型...")
embed_model = TextEmbedding('BAAI/bge-small-en-v1.5')

v4_dir = "embedding_db/bge-small-en-v1.5/results/v4_max_separation"
weights_v4 = np.load(f"{v4_dir}/learned_projection_weights_19d_v4.npy")
projection_v4 = LearnedProjection(384, 19)
projection_v4.proj.weight.data = torch.tensor(weights_v4, dtype=torch.float32)

with open(f"{v4_dir}/v4_config.json", "r") as f:
    v4_config = json.load(f)

mal_centroid = np.array(v4_config["malicious_centroid"])
threshold = v4_config["threshold"]

print(f"  V4模型加载完成")
print(f"  阈值: {threshold:.4f}")

# 2. 加载ToxicChat完整数据集
print("\n[2/4] 加载ToxicChat完整数据集...")

try:
    # 从HuggingFace加载
    ds = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="train", trust_remote_code=True)
    print(f"  ToxicChat总样本数: {len(ds)}")
    
    # 分离有毒和无毒样本
    toxic_samples = []
    benign_samples = []
    
    for item in ds:
        text = item.get('user_input', item.get('human', ''))
        if not text or not isinstance(text, str) or len(text.strip()) < 10:
            continue
        
        # toxicity标签：1=有毒，0=无毒
        is_toxic = item.get('toxicity', 0) == 1 or item.get('human_annotation', False)
        
        if is_toxic:
            toxic_samples.append(text)
        else:
            benign_samples.append(text)
    
    print(f"  有毒样本: {len(toxic_samples)}条")
    print(f"  无毒样本: {len(benign_samples)}条")
    
except Exception as e:
    print(f"  HuggingFace加载失败: {e}")
    print("  使用本地数据...")
    
    # 使用本地数据
    df = pd.read_csv("datasets/gcg_attacks/toxic_chat.csv")
    
    toxic_samples = df[df['toxicity'] == 1]['user_input'].dropna().tolist()
    benign_samples = df[df['toxicity'] == 0]['user_input'].dropna().tolist()
    
    print(f"  有毒样本: {len(toxic_samples)}条")
    print(f"  无毒样本: {len(benign_samples)}条")

# 3. 计算Embedding并检测
print("\n[3/4] 计算Embedding并检测...")

def compute_embeddings_batch(texts, embed_model, batch_size=64):
    texts = [t for t in texts if isinstance(t, str) and len(t.strip()) > 0]
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_emb = list(embed_model.embed(batch))
        embeddings.extend(batch_emb)
        if (i + batch_size) % 1000 == 0:
            print(f"    已处理 {min(i+batch_size, len(texts))}/{len(texts)} 条")
    return np.array(embeddings)

def detect_batch(embeddings, projection, centroid, threshold):
    with torch.no_grad():
        projected = projection(torch.tensor(embeddings, dtype=torch.float32)).numpy()
    
    similarities = np.dot(projected, centroid)
    predictions = similarities > threshold
    
    return predictions, similarities

# 测试有毒样本（应该被检出）
print("\n  测试有毒样本...")
if len(toxic_samples) > 0:
    toxic_emb = compute_embeddings_batch(toxic_samples, embed_model)
    toxic_pred, toxic_sim = detect_batch(toxic_emb, projection_v4, mal_centroid, threshold)
    toxic_detection_rate = np.mean(toxic_pred)
    print(f"  有毒样本检出率: {toxic_detection_rate*100:.2f}% ({sum(toxic_pred)}/{len(toxic_pred)})")
    print(f"  有毒样本相似度: mean={np.mean(toxic_sim):.4f}, std={np.std(toxic_sim):.4f}")

# 测试无毒样本（应该不被检出，即误报）
print("\n  测试无毒样本...")
if len(benign_samples) > 0:
    # 限制测试数量避免太慢
    test_benign = benign_samples[:10000] if len(benign_samples) > 10000 else benign_samples
    benign_emb = compute_embeddings_batch(test_benign, embed_model)
    benign_pred, benign_sim = detect_batch(benign_emb, projection_v4, mal_centroid, threshold)
    benign_fpr = np.mean(benign_pred)
    print(f"  无毒样本误报率: {benign_fpr*100:.2f}% ({sum(benign_pred)}/{len(benign_pred)})")
    print(f"  无毒样本相似度: mean={np.mean(benign_sim):.4f}, std={np.std(benign_sim):.4f}")

# 4. 汇总结果
print("\n" + "=" * 80)
print("ToxicChat测试结果汇总")
print("=" * 80)

print(f"\n数据集统计:")
print(f"  有毒样本: {len(toxic_samples)}条")
print(f"  无毒样本: {len(benign_samples)}条")

print(f"\nV4模型性能:")
if len(toxic_samples) > 0:
    print(f"  有毒样本检出率 (TPR): {toxic_detection_rate*100:.2f}%")
if len(benign_samples) > 0:
    print(f"  无毒样本误报率 (FPR): {benign_fpr*100:.2f}%")

# 计算F1分数
if len(toxic_samples) > 0 and len(benign_samples) > 0:
    tp = sum(toxic_pred)
    fp = sum(benign_pred)
    fn = len(toxic_pred) - tp
    tn = len(benign_pred) - fp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    print(f"\n  Precision: {precision*100:.2f}%")
    print(f"  Recall: {recall*100:.2f}%")
    print(f"  F1 Score: {f1*100:.2f}%")
    print(f"  Accuracy: {accuracy*100:.2f}%")

print("\n" + "=" * 80)
