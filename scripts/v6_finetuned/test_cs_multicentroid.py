#!/usr/bin/env python3
"""
测试V6 + CS压缩 + 多质心检测的效果
对比不同配置和阈值
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from transformers import AutoTokenizer
from datasets import load_dataset

sys.path.append(str(Path(__file__).parent))
from model import V6HarmfulDetector


class LearnedCSProjection(nn.Module):
    """学习型CS压缩投影"""
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim, bias=False)
    
    def forward(self, x):
        z = self.proj(x)
        z = F.normalize(z, p=2, dim=-1)
        return z


def load_v6_model(model_path, device='cuda'):
    """加载V6模型"""
    config_path = os.path.join(model_path, 'config.json')
    weights_path = os.path.join(model_path, 'best_model.pt')
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model = V6HarmfulDetector(
        model_name=config['model_name'],
        projection_dim=config['projection_dim']
    )
    
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    
    return model, tokenizer


def extract_embeddings(model, tokenizer, texts, device='cuda', batch_size=32):
    """提取V6 fine-tuned embeddings"""
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="提取embeddings"):
        batch_texts = texts[i:i+batch_size]
        
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            embeddings, _, _ = model(inputs['input_ids'], inputs['attention_mask'])
        
        all_embeddings.append(embeddings.cpu().numpy())
    
    return np.vstack(all_embeddings)


def predict_multicentroid(embeddings, centroids, benign_centroid, threshold=0.0):
    """多质心预测"""
    # 与恶意质心的最大相似度
    harmful_sims = embeddings @ centroids.T
    max_harmful_sim = np.max(harmful_sims, axis=1)
    
    # 与正常质心的相似度
    benign_sim = embeddings @ benign_centroid
    
    # 差值判定
    diff = max_harmful_sim - benign_sim
    preds = (diff > threshold).astype(int)
    
    return preds, max_harmful_sim, benign_sim, diff


def evaluate(preds, labels, name):
    """评估"""
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    return {
        'name': name,
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'fpr': fpr,
        'fnr': fnr,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
    }


def find_optimal_threshold(diffs, labels, target_fpr=0.05):
    """寻找最优阈值"""
    thresholds = np.percentile(diffs, np.arange(0, 100, 1))
    
    best_f1 = 0
    best_threshold = 0
    best_result = None
    
    for threshold in thresholds:
        preds = (diffs > threshold).astype(int)
        result = evaluate(preds, labels, f"threshold={threshold:.4f}")
        
        if result['fpr'] <= target_fpr and result['f1'] > best_f1:
            best_f1 = result['f1']
            best_threshold = threshold
            best_result = result
    
    return best_threshold, best_result


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = '/home/vicuna/ludan/CSonEmbedding/models/v6_finetuned'
    cs_path = '/home/vicuna/ludan/CSonEmbedding/models/v6_cs_multicentroid'
    
    print("="*70)
    print("V6 + CS压缩 + 多质心检测 - 详细测试")
    print("="*70)
    
    # 加载V6模型
    print("\n加载V6模型...")
    v6_model, tokenizer = load_v6_model(model_path, device)
    
    # 测试配置: 32D + 3质心 (训练集上最佳)
    config = "32d_3c"
    print(f"\n使用配置: {config}")
    
    saved = np.load(os.path.join(cs_path, f'cs_multicentroid_{config}.npz'))
    
    cs_model = LearnedCSProjection(768, 32).to(device)
    cs_model.proj.weight.data = torch.tensor(saved['cs_weights'], dtype=torch.float32).to(device)
    
    centroids = saved['centroids']
    benign_centroid = saved['benign_centroid']
    
    results = {}
    
    # ============ 测试数据集 ============
    
    # 1. AdvBench (恶意)
    print("\n[1] AdvBench...")
    try:
        ds = load_dataset("walledai/AdvBench", split='train')
        texts = [x['prompt'] for x in ds if x.get('prompt')][:500]
        labels = [1] * len(texts)
        
        embeddings = extract_embeddings(v6_model, tokenizer, texts, device)
        with torch.no_grad():
            compressed = cs_model(torch.tensor(embeddings, dtype=torch.float32).to(device)).cpu().numpy()
        
        preds, harmful_sims, benign_sims, diffs = predict_multicentroid(compressed, centroids, benign_centroid, threshold=0.0)
        result = evaluate(preds, labels, "AdvBench")
        results['advbench'] = result
        print(f"  F1: {result['f1']:.4f}, Recall: {result['recall']:.4f}")
    except Exception as e:
        print(f"  失败: {e}")
    
    # 2. Alpaca (正常)
    print("\n[2] Alpaca...")
    try:
        ds = load_dataset("tatsu-lab/alpaca", split='train')
        texts = []
        for item in ds:
            if item.get('instruction'):
                text = item['instruction']
                if item.get('input'):
                    text += " " + item['input']
                texts.append(text)
                if len(texts) >= 500:
                    break
        labels = [0] * len(texts)
        
        embeddings = extract_embeddings(v6_model, tokenizer, texts, device)
        with torch.no_grad():
            compressed = cs_model(torch.tensor(embeddings, dtype=torch.float32).to(device)).cpu().numpy()
        
        preds, harmful_sims, benign_sims, diffs = predict_multicentroid(compressed, centroids, benign_centroid, threshold=0.0)
        result = evaluate(preds, labels, "Alpaca")
        results['alpaca'] = result
        print(f"  Accuracy: {result['accuracy']:.4f}, FPR: {result['fpr']:.4f}")
    except Exception as e:
        print(f"  失败: {e}")
    
    # 3. ToxicChat (混合)
    print("\n[3] ToxicChat...")
    try:
        ds = load_dataset("lmsys/toxic-chat", "toxicchat0124", split='test')
        texts = [x['user_input'] for x in ds if x.get('user_input')][:1000]
        labels = np.array([1 if x.get('toxicity', 0) == 1 else 0 for x in ds if x.get('user_input')][:1000])
        
        embeddings = extract_embeddings(v6_model, tokenizer, texts, device)
        with torch.no_grad():
            compressed = cs_model(torch.tensor(embeddings, dtype=torch.float32).to(device)).cpu().numpy()
        
        # 测试不同阈值
        print("\n  阈值调优...")
        preds, harmful_sims, benign_sims, diffs = predict_multicentroid(compressed, centroids, benign_centroid, threshold=0.0)
        
        for threshold in [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3]:
            preds = (diffs > threshold).astype(int)
            result = evaluate(preds, labels, f"ToxicChat (threshold={threshold})")
            print(f"    threshold={threshold:+.1f}: F1={result['f1']:.4f}, Prec={result['precision']:.4f}, Rec={result['recall']:.4f}, FPR={result['fpr']:.4f}")
        
        # 寻找最优阈值 (目标FPR<10%)
        best_threshold, best_result = find_optimal_threshold(diffs, labels, target_fpr=0.10)
        if best_result:
            print(f"\n  最优阈值 (FPR<10%): {best_threshold:.4f}")
            print(f"    F1={best_result['f1']:.4f}, Prec={best_result['precision']:.4f}, Rec={best_result['recall']:.4f}, FPR={best_result['fpr']:.4f}")
            results['toxicchat_optimized'] = best_result
        
        # 默认阈值结果
        preds = (diffs > 0.0).astype(int)
        result = evaluate(preds, labels, "ToxicChat")
        results['toxicchat'] = result
        
    except Exception as e:
        print(f"  失败: {e}")
    
    # 4. Gray样本
    print("\n[4] Gray样本...")
    data_path = '/home/vicuna/ludan/CSonEmbedding/datasets'
    
    # Gray-Benign
    gray_benign_path = os.path.join(data_path, 'v6_training', 'gray_benign.jsonl')
    if os.path.exists(gray_benign_path):
        texts = []
        with open(gray_benign_path, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if item.get('text'):
                        texts.append(item['text'])
                except:
                    pass
        labels = [0] * len(texts)
        
        embeddings = extract_embeddings(v6_model, tokenizer, texts, device)
        with torch.no_grad():
            compressed = cs_model(torch.tensor(embeddings, dtype=torch.float32).to(device)).cpu().numpy()
        
        preds, _, _, _ = predict_multicentroid(compressed, centroids, benign_centroid, threshold=0.0)
        result = evaluate(preds, labels, "Gray-Benign")
        results['gray_benign'] = result
        print(f"  Gray-Benign: Accuracy={result['accuracy']:.4f}, FPR={result['fpr']:.4f}")
    
    # Gray-Harmful
    gray_harmful_path = os.path.join(data_path, 'v6_training', 'gray_harmful.jsonl')
    if os.path.exists(gray_harmful_path):
        texts = []
        with open(gray_harmful_path, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if item.get('text'):
                        texts.append(item['text'])
                except:
                    pass
        labels = [1] * len(texts)
        
        embeddings = extract_embeddings(v6_model, tokenizer, texts, device)
        with torch.no_grad():
            compressed = cs_model(torch.tensor(embeddings, dtype=torch.float32).to(device)).cpu().numpy()
        
        preds, _, _, _ = predict_multicentroid(compressed, centroids, benign_centroid, threshold=0.0)
        result = evaluate(preds, labels, "Gray-Harmful")
        results['gray_harmful'] = result
        print(f"  Gray-Harmful: Recall={result['recall']:.4f}, F1={result['f1']:.4f}")
    
    # ============ 汇总 ============
    print("\n" + "="*70)
    print("结果汇总 (V6 + CS压缩32D + 3质心)")
    print("="*70)
    print(f"\n{'数据集':<20} {'Accuracy':<12} {'F1':<12} {'Precision':<12} {'Recall':<12} {'FPR':<10}")
    print("-"*80)
    
    for name, result in results.items():
        if isinstance(result, dict) and 'accuracy' in result:
            print(f"{result['name']:<20} {result['accuracy']:<12.4f} {result['f1']:<12.4f} {result['precision']:<12.4f} {result['recall']:<12.4f} {result['fpr']:<10.4f}")
    
    # 与V6分类器对比
    print("\n" + "="*70)
    print("与V6分类器对比")
    print("="*70)
    print("""
| 方法 | AdvBench Recall | Alpaca FPR | Gray-Benign FPR | Gray-Harmful Recall |
|------|-----------------|------------|-----------------|---------------------|
| V6分类器 | 99% | 0% | 0% | 100% |
| V6+CS+多质心 | 见上表 | 见上表 | 见上表 | 见上表 |
""")


if __name__ == '__main__':
    main()
