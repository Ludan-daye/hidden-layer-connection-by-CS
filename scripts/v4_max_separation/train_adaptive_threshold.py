#!/usr/bin/env python3
"""
自适应灰色地带阈值学习
- 三分类：恶意(>上界) / 灰色地带(下界~上界) / 正常(<下界)
- 通过学习自动调整上下界阈值，最大化有效样本的分类准确率
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastembed import TextEmbedding
from datasets import load_dataset
import json
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

class AdaptiveThreshold:
    """自适应阈值学习器"""
    def __init__(self, init_lower=-0.3, init_upper=0.7):
        self.lower = init_lower  # 下界：低于此值判为正常
        self.upper = init_upper  # 上界：高于此值判为恶意
    
    def classify(self, similarities):
        """三分类"""
        malicious = similarities > self.upper
        benign = similarities < self.lower
        gray = ~malicious & ~benign
        return malicious, gray, benign
    
    def compute_metrics(self, mal_sim, benign_sim):
        """计算指标"""
        # 恶意样本分类
        mal_pred_mal = np.sum(mal_sim > self.upper)
        mal_pred_gray = np.sum((mal_sim >= self.lower) & (mal_sim <= self.upper))
        mal_pred_benign = np.sum(mal_sim < self.lower)
        
        # 正常样本分类
        benign_pred_mal = np.sum(benign_sim > self.upper)
        benign_pred_gray = np.sum((benign_sim >= self.lower) & (benign_sim <= self.upper))
        benign_pred_benign = np.sum(benign_sim < self.lower)
        
        # 指标
        tpr = mal_pred_mal / len(mal_sim) if len(mal_sim) > 0 else 0
        tnr = benign_pred_benign / len(benign_sim) if len(benign_sim) > 0 else 0
        fpr = benign_pred_mal / len(benign_sim) if len(benign_sim) > 0 else 0
        fnr = mal_pred_benign / len(mal_sim) if len(mal_sim) > 0 else 0
        
        # 灰色地带占比
        total = len(mal_sim) + len(benign_sim)
        gray_ratio = (mal_pred_gray + benign_pred_gray) / total
        
        # 有效样本准确率（排除灰色地带）
        effective_correct = mal_pred_mal + benign_pred_benign
        effective_total = mal_pred_mal + mal_pred_benign + benign_pred_mal + benign_pred_benign
        effective_acc = effective_correct / effective_total if effective_total > 0 else 0
        
        return {
            'tpr': tpr, 'tnr': tnr, 'fpr': fpr, 'fnr': fnr,
            'gray_ratio': gray_ratio, 'effective_acc': effective_acc,
            'coverage': 1 - gray_ratio
        }
    
    def optimize(self, mal_sim, benign_sim, target_coverage=0.8, lr=0.01, epochs=100):
        """
        优化阈值
        目标：在保证覆盖率的前提下，最大化有效准确率
        """
        print(f"\n开始优化阈值...")
        print(f"目标覆盖率: {target_coverage*100:.0f}%")
        
        best_acc = 0
        best_lower, best_upper = self.lower, self.upper
        
        # 网格搜索 + 梯度优化
        for lower in np.arange(-0.8, 0.5, 0.05):
            for upper in np.arange(lower + 0.2, 1.0, 0.05):
                self.lower = lower
                self.upper = upper
                
                metrics = self.compute_metrics(mal_sim, benign_sim)
                
                # 约束：覆盖率 >= 目标
                if metrics['coverage'] >= target_coverage:
                    if metrics['effective_acc'] > best_acc:
                        best_acc = metrics['effective_acc']
                        best_lower = lower
                        best_upper = upper
        
        self.lower = best_lower
        self.upper = best_upper
        
        final_metrics = self.compute_metrics(mal_sim, benign_sim)
        print(f"\n优化完成:")
        print(f"  下界: {self.lower:.3f}")
        print(f"  上界: {self.upper:.3f}")
        print(f"  有效准确率: {final_metrics['effective_acc']*100:.1f}%")
        print(f"  覆盖率: {final_metrics['coverage']*100:.1f}%")
        print(f"  恶意检出率(TPR): {final_metrics['tpr']*100:.1f}%")
        print(f"  正常正确率(TNR): {final_metrics['tnr']*100:.1f}%")
        
        return final_metrics

print("=" * 80)
print("自适应灰色地带阈值学习")
print("=" * 80)

# 1. 加载模型
print("\n[1/4] 加载V4增强版模型...")
embed_model = TextEmbedding('BAAI/bge-small-en-v1.5')
v4_dir = "embedding_db/bge-small-en-v1.5/results/v4_max_separation"

weights = np.load(f"{v4_dir}/learned_projection_weights_19d_v4_enhanced.npy")
projection = LearnedProjection(384, 19)
projection.proj.weight.data = torch.tensor(weights, dtype=torch.float32)

with open(f"{v4_dir}/v4_enhanced_config.json", "r") as f:
    config = json.load(f)
mal_centroid = np.array(config["malicious_centroid"])

# 2. 加载ToxicChat数据
print("\n[2/4] 加载ToxicChat数据...")
ds = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="train", trust_remote_code=True)

toxic_samples, benign_samples = [], []
for item in ds:
    text = item.get('user_input', '')
    if not text or len(text.strip()) < 10:
        continue
    if item.get('toxicity', 0) == 1:
        toxic_samples.append(text)
    else:
        benign_samples.append(text)

print(f"  有毒样本: {len(toxic_samples)}条")
print(f"  无毒样本: {len(benign_samples)}条")

# 3. 计算相似度
print("\n[3/4] 计算相似度...")

def compute_emb_batch(texts, model, batch_size=64):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_emb = list(model.embed(batch))
        embeddings.extend(batch_emb)
        if (i + batch_size) % 500 == 0 or i + batch_size >= len(texts):
            print(f"    已处理 {min(i+batch_size, len(texts))}/{len(texts)} 条")
    return np.array(embeddings)

print("  计算有毒样本Embedding...")
toxic_emb = compute_emb_batch(toxic_samples, embed_model)
print("  计算无毒样本Embedding...")
benign_emb = compute_emb_batch(benign_samples, embed_model)

with torch.no_grad():
    toxic_proj = projection(torch.tensor(toxic_emb, dtype=torch.float32)).numpy()
    benign_proj = projection(torch.tensor(benign_emb, dtype=torch.float32)).numpy()

toxic_sim = np.dot(toxic_proj, mal_centroid)
benign_sim = np.dot(benign_proj, mal_centroid)

print(f"  有毒样本相似度: mean={np.mean(toxic_sim):.3f}, std={np.std(toxic_sim):.3f}")
print(f"  无毒样本相似度: mean={np.mean(benign_sim):.3f}, std={np.std(benign_sim):.3f}")

# 4. 优化阈值
print("\n[4/4] 优化灰色地带阈值...")

# 测试不同覆盖率目标
print("\n" + "=" * 80)
print("不同覆盖率目标下的最优阈值")
print("=" * 80)

results = []
for target_cov in [0.6, 0.7, 0.8, 0.9, 0.95]:
    threshold = AdaptiveThreshold()
    metrics = threshold.optimize(toxic_sim, benign_sim, target_coverage=target_cov)
    results.append({
        'target_coverage': target_cov,
        'lower': threshold.lower,
        'upper': threshold.upper,
        **metrics
    })

# 打印结果表格
print("\n" + "=" * 80)
print("结果汇总")
print("=" * 80)
print("{:>10} {:>8} {:>8} {:>10} {:>10} {:>10} {:>10}".format(
    '目标覆盖', '下界', '上界', '有效Acc', '实际覆盖', 'TPR', 'TNR'))
print("-" * 80)

for r in results:
    print("{:>9.0f}% {:>8.2f} {:>8.2f} {:>9.1f}% {:>9.1f}% {:>9.1f}% {:>9.1f}%".format(
        r['target_coverage']*100, r['lower'], r['upper'],
        r['effective_acc']*100, r['coverage']*100,
        r['tpr']*100, r['tnr']*100))

print("-" * 80)

# 保存最优配置（80%覆盖率）
best_result = [r for r in results if r['target_coverage'] == 0.8][0]
adaptive_config = {
    "version": "v4_enhanced_adaptive",
    "lower_threshold": best_result['lower'],
    "upper_threshold": best_result['upper'],
    "target_coverage": 0.8,
    "effective_accuracy": best_result['effective_acc'],
    "actual_coverage": best_result['coverage'],
    "tpr": best_result['tpr'],
    "tnr": best_result['tnr'],
    "malicious_centroid": config["malicious_centroid"],
}

with open(f"{v4_dir}/v4_adaptive_threshold_config.json", "w") as f:
    json.dump(adaptive_config, f, indent=2)

print(f"\n最优配置已保存到: {v4_dir}/v4_adaptive_threshold_config.json")
print(f"  下界: {best_result['lower']:.3f} (低于此值判为正常)")
print(f"  上界: {best_result['upper']:.3f} (高于此值判为恶意)")
print(f"  有效准确率: {best_result['effective_acc']*100:.1f}%")

print("\n" + "=" * 80)
