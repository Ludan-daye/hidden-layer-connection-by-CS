#!/usr/bin/env python3
"""
V7模型在真实日常对话数据集上的测试
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
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
    all_preds = []
    
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
            preds = (probs > 0.5).long()
        
        all_probs.extend(probs.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
    
    return np.array(all_preds), np.array(all_probs)


def three_class_predict(probs, lower=0.3, upper=0.7):
    """三分类预测"""
    preds = np.zeros(len(probs), dtype=int)
    preds[probs > upper] = 1  # harmful
    preds[probs < lower] = 0  # benign
    preds[(probs >= lower) & (probs <= upper)] = 2  # uncertain
    return preds


def evaluate(preds, probs, name, expected_label=0, lower=0.3, upper=0.7):
    """评估"""
    labels = np.array([expected_label] * len(preds))
    
    # 二分类
    binary_acc = (preds == labels).mean()
    fp = (preds == 1).sum()
    fpr = fp / len(preds) if expected_label == 0 else 0
    
    # 三分类
    three_preds = three_class_predict(probs, lower, upper)
    n_uncertain = (three_preds == 2).sum()
    uncertain_ratio = n_uncertain / len(probs)
    
    # 确定样本准确率
    certain_mask = three_preds != 2
    if certain_mask.sum() > 0:
        certain_preds = three_preds[certain_mask]
        certain_labels = labels[certain_mask]
        certain_acc = (certain_preds == certain_labels).mean()
    else:
        certain_acc = 0
    
    return {
        'name': name,
        'samples': len(preds),
        'binary_acc': binary_acc,
        'fpr': fpr,
        'harmful_count': int(fp),
        'uncertain_count': int(n_uncertain),
        'uncertain_ratio': uncertain_ratio,
        'certain_acc': certain_acc
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = '/home/vicuna/ludan/CSonEmbedding/models/v7_classifier'
    output_path = '/home/vicuna/ludan/CSonEmbedding/results/v7_real_conversations'
    
    os.makedirs(output_path, exist_ok=True)
    
    print("="*70)
    print("V7模型 - 真实日常对话数据集测试")
    print("="*70)
    
    print("\n加载V7模型...")
    model, tokenizer = load_v7_model(model_path, device)
    
    results = {}
    
    # 1. Everyday-Conversations
    print("\n[1] Everyday-Conversations...")
    try:
        ds = load_dataset("HuggingFaceTB/everyday-conversations-llama3.1-2k", split='train_sft')
        texts = []
        for item in ds:
            if item.get('messages'):
                for msg in item['messages']:
                    if msg.get('role') == 'user' and msg.get('content'):
                        content = msg['content']
                        if len(content) > 10:
                            texts.append(content)
                if len(texts) >= 1000:
                    break
        texts = texts[:1000]
        print(f"  样本数: {len(texts)}")
        
        preds, probs = predict_batch(model, tokenizer, texts, device)
        result = evaluate(preds, probs, "Everyday-Conversations")
        results['everyday_conv'] = result
        
        print(f"  Binary_Acc: {result['binary_acc']*100:.2f}%")
        print(f"  FPR: {result['fpr']*100:.2f}%")
        print(f"  Uncertain: {result['uncertain_ratio']*100:.1f}%")
        print(f"  Certain_Acc: {result['certain_acc']*100:.2f}%")
    except Exception as e:
        print(f"  失败: {e}")
    
    # 2. WildChat-English
    print("\n[2] WildChat-English...")
    try:
        ds = load_dataset("allenai/WildChat-1M", split='train', streaming=True)
        texts = []
        count = 0
        for item in ds:
            if item.get('language') and item['language'] != 'English':
                continue
            if item.get('conversation'):
                for turn in item['conversation']:
                    if turn.get('role') == 'user' and turn.get('content'):
                        content = turn['content']
                        if len(content) > 10 and len(content) < 1000:
                            ascii_ratio = sum(1 for c in content if ord(c) < 128) / len(content)
                            if ascii_ratio > 0.9:
                                texts.append(content)
                                count += 1
                                if count >= 1000:
                                    break
            if count >= 1000:
                break
        
        print(f"  样本数: {len(texts)}")
        
        preds, probs = predict_batch(model, tokenizer, texts, device)
        result = evaluate(preds, probs, "WildChat-English")
        results['wildchat'] = result
        
        print(f"  Binary_Acc: {result['binary_acc']*100:.2f}%")
        print(f"  FPR: {result['fpr']*100:.2f}%")
        print(f"  Uncertain: {result['uncertain_ratio']*100:.1f}%")
        print(f"  Certain_Acc: {result['certain_acc']*100:.2f}%")
    except Exception as e:
        print(f"  失败: {e}")
    
    # 3. Alpaca (对照)
    print("\n[3] Alpaca (对照)...")
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
        
        print(f"  样本数: {len(texts)}")
        
        preds, probs = predict_batch(model, tokenizer, texts, device)
        result = evaluate(preds, probs, "Alpaca")
        results['alpaca'] = result
        
        print(f"  Binary_Acc: {result['binary_acc']*100:.2f}%")
        print(f"  FPR: {result['fpr']*100:.2f}%")
        print(f"  Uncertain: {result['uncertain_ratio']*100:.1f}%")
        print(f"  Certain_Acc: {result['certain_acc']*100:.2f}%")
    except Exception as e:
        print(f"  失败: {e}")
    
    # 4. HarmBench (恶意对照)
    print("\n[4] HarmBench (恶意对照)...")
    try:
        import pandas as pd
        harmbench_path = '/home/vicuna/ludan/CSonEmbedding/datasets/harmbench'
        texts = []
        for file in os.listdir(harmbench_path):
            if file.endswith('.csv'):
                df = pd.read_csv(os.path.join(harmbench_path, file))
                for col in ['Behavior', 'goal', 'prompt']:
                    if col in df.columns:
                        texts.extend(df[col].dropna().tolist()[:300])
                        break
        texts = texts[:500]
        print(f"  样本数: {len(texts)}")
        
        preds, probs = predict_batch(model, tokenizer, texts, device)
        # 恶意数据集，expected_label=1
        labels = np.array([1] * len(preds))
        binary_acc = (preds == labels).mean()
        fn = (preds == 0).sum()
        fnr = fn / len(preds)
        
        three_preds = three_class_predict(probs)
        n_uncertain = (three_preds == 2).sum()
        
        result = {
            'name': 'HarmBench',
            'samples': len(preds),
            'binary_acc': binary_acc,
            'fnr': fnr,
            'miss_count': int(fn),
            'uncertain_count': int(n_uncertain),
            'uncertain_ratio': n_uncertain / len(probs)
        }
        results['harmbench'] = result
        
        print(f"  Binary_Acc: {result['binary_acc']*100:.2f}%")
        print(f"  FNR: {result['fnr']*100:.2f}%")
        print(f"  Uncertain: {result['uncertain_ratio']*100:.1f}%")
    except Exception as e:
        print(f"  失败: {e}")
    
    # 汇总
    print("\n" + "="*70)
    print("测试结果汇总")
    print("="*70)
    
    print(f"\n{'数据集':<25} {'样本数':<10} {'Acc':<10} {'FPR/FNR':<10} {'Uncertain':<12} {'Certain_Acc':<12}")
    print("-"*80)
    
    for name, result in results.items():
        metric = result.get('fpr', result.get('fnr', 0))
        metric_name = 'FPR' if 'fpr' in result else 'FNR'
        certain_acc = result.get('certain_acc', '-')
        if isinstance(certain_acc, float):
            certain_acc = f"{certain_acc*100:.2f}%"
        
        print(f"{result['name']:<25} {result['samples']:<10} {result['binary_acc']*100:<10.2f}% {metric*100:<10.2f}% {result['uncertain_ratio']*100:<12.1f}% {certain_acc:<12}")
    
    # 保存结果
    output_file = os.path.join(output_path, 'results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存至: {output_file}")


if __name__ == '__main__':
    main()
