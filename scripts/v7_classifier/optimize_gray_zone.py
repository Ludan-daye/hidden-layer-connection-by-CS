#!/usr/bin/env python3
"""
优化V7模型的灰色地带范围
目标：通过扩大灰色地带来降低FPR和提高鲁棒性
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from datasets import load_dataset
import pandas as pd

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


def evaluate_threshold(benign_probs, harmful_probs, lower, upper):
    """评估给定阈值的效果"""
    # 正常样本
    benign_correct = (benign_probs < lower).sum()  # 正确判为正常
    benign_uncertain = ((benign_probs >= lower) & (benign_probs <= upper)).sum()  # 不确定
    benign_wrong = (benign_probs > upper).sum()  # 错误判为恶意
    
    # 恶意样本
    harmful_correct = (harmful_probs > upper).sum()  # 正确判为恶意
    harmful_uncertain = ((harmful_probs >= lower) & (harmful_probs <= upper)).sum()  # 不确定
    harmful_wrong = (harmful_probs < lower).sum()  # 错误判为正常
    
    total_benign = len(benign_probs)
    total_harmful = len(harmful_probs)
    total = total_benign + total_harmful
    
    # 确定样本的准确率
    certain_correct = benign_correct + harmful_correct
    certain_wrong = benign_wrong + harmful_wrong
    certain_total = certain_correct + certain_wrong
    certain_acc = certain_correct / certain_total if certain_total > 0 else 0
    
    # 不确定比例
    uncertain_total = benign_uncertain + harmful_uncertain
    uncertain_ratio = uncertain_total / total
    
    # FPR: 正常样本被错误判为恶意的比例（在确定样本中）
    benign_certain = benign_correct + benign_wrong
    fpr = benign_wrong / benign_certain if benign_certain > 0 else 0
    
    # FNR: 恶意样本被错误判为正常的比例（在确定样本中）
    harmful_certain = harmful_correct + harmful_wrong
    fnr = harmful_wrong / harmful_certain if harmful_certain > 0 else 0
    
    return {
        'lower': lower,
        'upper': upper,
        'certain_acc': certain_acc,
        'uncertain_ratio': uncertain_ratio,
        'fpr': fpr,
        'fnr': fnr,
        'benign_correct': int(benign_correct),
        'benign_uncertain': int(benign_uncertain),
        'benign_wrong': int(benign_wrong),
        'harmful_correct': int(harmful_correct),
        'harmful_uncertain': int(harmful_uncertain),
        'harmful_wrong': int(harmful_wrong)
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = '/home/vicuna/ludan/CSonEmbedding/models/v7_classifier'
    output_path = '/home/vicuna/ludan/CSonEmbedding/models/v7_classifier'
    
    print("="*70)
    print("V7模型灰色地带优化")
    print("目标：扩大灰色地带以降低FPR和提高鲁棒性")
    print("="*70)
    
    print("\n加载V7模型...")
    model, tokenizer = load_v7_model(model_path, device)
    
    # 收集测试数据
    print("\n收集测试数据...")
    
    benign_probs_list = []
    harmful_probs_list = []
    
    # 正常数据集
    # 1. Everyday-Conversations
    print("  Everyday-Conversations...")
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
    benign_probs_list.append(('Everyday-Conv', probs))
    
    # 2. WildChat-English
    print("  WildChat-English...")
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
                            if count >= 500:
                                break
        if count >= 500:
            break
    probs = predict_batch(model, tokenizer, texts, device)
    benign_probs_list.append(('WildChat', probs))
    
    # 3. Alpaca
    print("  Alpaca...")
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
    benign_probs_list.append(('Alpaca', probs))
    
    # 恶意数据集
    # 1. HarmBench
    print("  HarmBench...")
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
    harmful_probs_list.append(('HarmBench', probs))
    
    # 合并概率
    all_benign_probs = np.concatenate([p for _, p in benign_probs_list])
    all_harmful_probs = np.concatenate([p for _, p in harmful_probs_list])
    
    print(f"\n正常样本: {len(all_benign_probs)}")
    print(f"恶意样本: {len(all_harmful_probs)}")
    
    # 测试不同的灰色地带范围
    print("\n" + "="*70)
    print("不同灰色地带范围的效果对比")
    print("="*70)
    
    configs = [
        # (lower, upper, 描述)
        (0.49, 0.56, "最小灰色地带"),
        (0.45, 0.60, "小灰色地带"),
        (0.42, 0.62, "中等灰色地带"),
        (0.40, 0.65, "较大灰色地带"),
        (0.38, 0.68, "大灰色地带"),
        (0.35, 0.70, "最大灰色地带"),
    ]
    
    print(f"\n{'配置':<20} {'Certain_Acc':<12} {'FPR':<10} {'FNR':<10} {'Uncertain%':<12} {'描述':<15}")
    print("-"*80)
    
    results = []
    for lower, upper, desc in configs:
        result = evaluate_threshold(all_benign_probs, all_harmful_probs, lower, upper)
        result['desc'] = desc
        results.append(result)
        
        print(f"[{lower:.2f}, {upper:.2f}]     {result['certain_acc']*100:<12.2f}% {result['fpr']*100:<10.2f}% {result['fnr']*100:<10.2f}% {result['uncertain_ratio']*100:<12.1f}% {desc:<15}")
    
    # 按数据集分析最佳配置
    print("\n" + "="*70)
    print("推荐配置详细分析")
    print("="*70)
    
    # 选择FPR<5%且FNR<5%的最小灰色地带
    best_config = None
    for result in results:
        if result['fpr'] < 0.05 and result['fnr'] < 0.05:
            if best_config is None or result['uncertain_ratio'] < best_config['uncertain_ratio']:
                best_config = result
    
    if best_config is None:
        # 如果没有满足条件的，选择FPR+FNR最小的
        best_config = min(results, key=lambda x: x['fpr'] + x['fnr'])
    
    print(f"\n推荐配置: [{best_config['lower']:.2f}, {best_config['upper']:.2f}]")
    print(f"  确定样本准确率: {best_config['certain_acc']*100:.2f}%")
    print(f"  FPR: {best_config['fpr']*100:.2f}%")
    print(f"  FNR: {best_config['fnr']*100:.2f}%")
    print(f"  不确定比例: {best_config['uncertain_ratio']*100:.1f}%")
    print(f"\n  正常样本: 正确={best_config['benign_correct']}, 不确定={best_config['benign_uncertain']}, 错误={best_config['benign_wrong']}")
    print(f"  恶意样本: 正确={best_config['harmful_correct']}, 不确定={best_config['harmful_uncertain']}, 错误={best_config['harmful_wrong']}")
    
    # 按数据集分析
    print("\n按数据集分析 (使用推荐配置):")
    lower, upper = best_config['lower'], best_config['upper']
    
    for name, probs in benign_probs_list:
        correct = (probs < lower).sum()
        uncertain = ((probs >= lower) & (probs <= upper)).sum()
        wrong = (probs > upper).sum()
        certain_total = correct + wrong
        fpr = wrong / certain_total if certain_total > 0 else 0
        print(f"  {name}: 正确={correct}, 不确定={uncertain}, 错误={wrong}, FPR={fpr*100:.2f}%")
    
    for name, probs in harmful_probs_list:
        correct = (probs > upper).sum()
        uncertain = ((probs >= lower) & (probs <= upper)).sum()
        wrong = (probs < lower).sum()
        certain_total = correct + wrong
        fnr = wrong / certain_total if certain_total > 0 else 0
        print(f"  {name}: 正确={correct}, 不确定={uncertain}, 错误={wrong}, FNR={fnr*100:.2f}%")
    
    # 保存配置
    config = {
        'lower_threshold': best_config['lower'],
        'upper_threshold': best_config['upper'],
        'description': {
            'benign': f'prob < {best_config["lower"]}',
            'harmful': f'prob > {best_config["upper"]}',
            'uncertain': f'{best_config["lower"]} <= prob <= {best_config["upper"]}'
        },
        'metrics': {
            'certain_acc': best_config['certain_acc'],
            'fpr': best_config['fpr'],
            'fnr': best_config['fnr'],
            'uncertain_ratio': best_config['uncertain_ratio']
        },
        'all_configs': results
    }
    
    config_file = os.path.join(output_path, 'gray_zone_config.json')
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"\n配置已保存至: {config_file}")
    
    # 汇总
    print("\n" + "="*70)
    print("汇总")
    print("="*70)
    print(f"""
V7前置分类器灰色地带配置:
  - 正常: prob < {best_config['lower']:.2f}
  - 恶意: prob > {best_config['upper']:.2f}
  - 不确定: {best_config['lower']:.2f} <= prob <= {best_config['upper']:.2f} (交给LLM)

效果:
  - 确定样本准确率: {best_config['certain_acc']*100:.2f}%
  - FPR (误报率): {best_config['fpr']*100:.2f}%
  - FNR (漏检率): {best_config['fnr']*100:.2f}%
  - 不确定比例: {best_config['uncertain_ratio']*100:.1f}%
""")


if __name__ == '__main__':
    main()
