#!/usr/bin/env python3
"""
分析V7模型的概率分布
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from datasets import load_dataset

sys.path.append(str(Path(__file__).parent))
from model import V6HarmfulDetector


def load_v7_model(model_path, device='cuda'):
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


def predict_batch(model, tokenizer, texts, device='cuda', batch_size=32):
    all_probs = []
    
    for i in range(0, len(texts), batch_size):
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
            probs = model.predict(inputs['input_ids'], inputs['attention_mask'])
        
        all_probs.extend(probs.cpu().numpy().tolist())
    
    return np.array(all_probs)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = '/home/vicuna/ludan/CSonEmbedding/models/v7_classifier'
    
    print("="*70)
    print("V7模型概率分布分析")
    print("="*70)
    
    print("\n加载V7模型...")
    model, tokenizer = load_v7_model(model_path, device)
    
    # 收集各数据集的概率
    all_data = {}
    
    # 1. Everyday-Conversations (正常)
    print("\n加载 Everyday-Conversations...")
    ds = load_dataset("HuggingFaceTB/everyday-conversations-llama3.1-2k", split='train_sft')
    texts = []
    for item in ds:
        if item.get('messages'):
            for msg in item['messages']:
                if msg.get('role') == 'user' and msg.get('content'):
                    texts.append(msg['content'])
            if len(texts) >= 500:
                break
    texts = texts[:500]
    probs = predict_batch(model, tokenizer, texts, device)
    all_data['Everyday-Conv (benign)'] = probs
    
    # 2. Alpaca (正常)
    print("加载 Alpaca...")
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
    probs = predict_batch(model, tokenizer, texts, device)
    all_data['Alpaca (benign)'] = probs
    
    # 3. HarmBench (恶意)
    print("加载 HarmBench...")
    import pandas as pd
    harmbench_path = '/home/vicuna/ludan/CSonEmbedding/datasets/harmbench'
    texts = []
    for file in os.listdir(harmbench_path):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(harmbench_path, file))
            for col in ['Behavior', 'goal', 'prompt']:
                if col in df.columns:
                    texts.extend(df[col].dropna().tolist())
                    break
    texts = texts[:500]
    probs = predict_batch(model, tokenizer, texts, device)
    all_data['HarmBench (harmful)'] = probs
    
    # 分析概率分布
    print("\n" + "="*70)
    print("概率分布分析")
    print("="*70)
    
    for name, probs in all_data.items():
        print(f"\n{name}:")
        print(f"  样本数: {len(probs)}")
        print(f"  概率范围: [{probs.min():.4f}, {probs.max():.4f}]")
        print(f"  均值: {probs.mean():.4f}")
        print(f"  中位数: {np.median(probs):.4f}")
        print(f"  标准差: {probs.std():.4f}")
        
        # 分布统计
        print(f"  分布:")
        print(f"    prob < 0.3: {(probs < 0.3).sum()} ({(probs < 0.3).mean()*100:.1f}%)")
        print(f"    0.3 <= prob <= 0.5: {((probs >= 0.3) & (probs <= 0.5)).sum()} ({((probs >= 0.3) & (probs <= 0.5)).mean()*100:.1f}%)")
        print(f"    0.5 < prob <= 0.7: {((probs > 0.5) & (probs <= 0.7)).sum()} ({((probs > 0.5) & (probs <= 0.7)).mean()*100:.1f}%)")
        print(f"    prob > 0.7: {(probs > 0.7).sum()} ({(probs > 0.7).mean()*100:.1f}%)")
    
    # 寻找最优阈值
    print("\n" + "="*70)
    print("寻找最优灰色地带阈值")
    print("="*70)
    
    benign_probs = np.concatenate([all_data['Everyday-Conv (benign)'], all_data['Alpaca (benign)']])
    harmful_probs = all_data['HarmBench (harmful)']
    
    best_config = None
    best_score = -1
    
    for lower in np.arange(0.35, 0.50, 0.02):
        for upper in np.arange(0.50, 0.65, 0.02):
            if lower >= upper:
                continue
            
            # 正常样本
            benign_certain = (benign_probs < lower).sum()
            benign_uncertain = ((benign_probs >= lower) & (benign_probs <= upper)).sum()
            benign_wrong = (benign_probs > upper).sum()
            
            # 恶意样本
            harmful_certain = (harmful_probs > upper).sum()
            harmful_uncertain = ((harmful_probs >= lower) & (harmful_probs <= upper)).sum()
            harmful_wrong = (harmful_probs < lower).sum()
            
            # 确定样本准确率
            total_certain = benign_certain + harmful_certain
            total_wrong = benign_wrong + harmful_wrong
            certain_acc = (benign_certain + harmful_certain) / (total_certain + total_wrong) if (total_certain + total_wrong) > 0 else 0
            
            # 不确定比例
            total = len(benign_probs) + len(harmful_probs)
            uncertain_ratio = (benign_uncertain + harmful_uncertain) / total
            
            # 综合得分
            score = certain_acc * 0.7 - uncertain_ratio * 0.3
            
            if score > best_score:
                best_score = score
                best_config = {
                    'lower': lower,
                    'upper': upper,
                    'certain_acc': certain_acc,
                    'uncertain_ratio': uncertain_ratio,
                    'benign_correct': benign_certain,
                    'benign_uncertain': benign_uncertain,
                    'benign_wrong': benign_wrong,
                    'harmful_correct': harmful_certain,
                    'harmful_uncertain': harmful_uncertain,
                    'harmful_wrong': harmful_wrong
                }
    
    if best_config:
        print(f"\n最优配置:")
        print(f"  Lower: {best_config['lower']:.2f}")
        print(f"  Upper: {best_config['upper']:.2f}")
        print(f"  确定样本准确率: {best_config['certain_acc']*100:.2f}%")
        print(f"  不确定比例: {best_config['uncertain_ratio']*100:.1f}%")
        print(f"\n  正常样本: 正确={best_config['benign_correct']}, 不确定={best_config['benign_uncertain']}, 错误={best_config['benign_wrong']}")
        print(f"  恶意样本: 正确={best_config['harmful_correct']}, 不确定={best_config['harmful_uncertain']}, 错误={best_config['harmful_wrong']}")


if __name__ == '__main__':
    main()
