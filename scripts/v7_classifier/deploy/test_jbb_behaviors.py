#!/usr/bin/env python3
"""
测试V7+LLM系统在JBB-Behaviors数据集上的效果
"""

import os
import sys
import json
import time
from datetime import datetime
from datasets import load_dataset
from tqdm import tqdm

from detector import HarmfulContentDetector


def test_jbb_behaviors():
    """测试JBB-Behaviors数据集"""
    
    print("="*70)
    print("JBB-Behaviors数据集测试")
    print("="*70)
    
    # 加载数据集
    print("\n加载JBB-Behaviors数据集...")
    ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
    
    print(f"数据集结构: {ds}")
    
    # 查看数据集内容
    if 'train' in ds:
        data = ds['train']
    elif 'test' in ds:
        data = ds['test']
    else:
        data = list(ds.values())[0]
    
    print(f"样本数: {len(data)}")
    print(f"字段: {data.column_names}")
    
    # 查看几个样本
    print("\n样本示例:")
    for i in range(min(3, len(data))):
        print(f"  [{i}] {data[i]}")
    
    # 初始化检测器
    print("\n初始化检测系统...")
    detector = HarmfulContentDetector(lazy_load_llm=True)
    
    # 准备测试数据
    # JBB-Behaviors包含harmful和benign两类
    harmful_texts = []
    benign_texts = []
    
    for item in data:
        # 根据数据集结构提取文本和标签
        if 'Goal' in item:
            text = item['Goal']
        elif 'goal' in item:
            text = item['goal']
        elif 'Behavior' in item:
            text = item['Behavior']
        elif 'behavior' in item:
            text = item['behavior']
        elif 'prompt' in item:
            text = item['prompt']
        else:
            text = str(item)
        
        # 判断是harmful还是benign
        if 'Category' in item or 'category' in item:
            category = item.get('Category', item.get('category', ''))
            if 'benign' in str(category).lower():
                benign_texts.append(text)
            else:
                harmful_texts.append(text)
        else:
            # 默认都是harmful
            harmful_texts.append(text)
    
    print(f"\n恶意样本: {len(harmful_texts)}")
    print(f"良性样本: {len(benign_texts)}")
    
    results = {
        'harmful': {'total': 0, 'correct': 0, 'v7_direct': 0, 'llm_judge': 0, 'details': []},
        'benign': {'total': 0, 'correct': 0, 'v7_direct': 0, 'llm_judge': 0, 'details': []}
    }
    
    # 测试恶意样本
    if harmful_texts:
        print(f"\n测试恶意样本 ({len(harmful_texts)}条)...")
        for text in tqdm(harmful_texts):
            result = detector.detect(text, verbose=False)
            results['harmful']['total'] += 1
            
            if result['label'] == 'harmful':
                results['harmful']['correct'] += 1
            
            if result['source'] == 'v7_classifier':
                results['harmful']['v7_direct'] += 1
            else:
                results['harmful']['llm_judge'] += 1
            
            results['harmful']['details'].append({
                'text': text[:100],
                'label': result['label'],
                'source': result['source'],
                'v7_prob': result['v7_prob']
            })
    
    # 测试良性样本
    if benign_texts:
        print(f"\n测试良性样本 ({len(benign_texts)}条)...")
        for text in tqdm(benign_texts):
            result = detector.detect(text, verbose=False)
            results['benign']['total'] += 1
            
            if result['label'] == 'benign':
                results['benign']['correct'] += 1
            
            if result['source'] == 'v7_classifier':
                results['benign']['v7_direct'] += 1
            else:
                results['benign']['llm_judge'] += 1
            
            results['benign']['details'].append({
                'text': text[:100],
                'label': result['label'],
                'source': result['source'],
                'v7_prob': result['v7_prob']
            })
    
    # 打印结果
    print("\n" + "="*70)
    print("测试结果")
    print("="*70)
    
    if results['harmful']['total'] > 0:
        h = results['harmful']
        acc = h['correct'] / h['total'] * 100
        v7_ratio = h['v7_direct'] / h['total'] * 100
        print(f"\n恶意样本:")
        print(f"  总数: {h['total']}")
        print(f"  正确检出: {h['correct']} ({acc:.2f}%)")
        print(f"  V7直接判断: {h['v7_direct']} ({v7_ratio:.1f}%)")
        print(f"  LLM判断: {h['llm_judge']} ({100-v7_ratio:.1f}%)")
    
    if results['benign']['total'] > 0:
        b = results['benign']
        acc = b['correct'] / b['total'] * 100
        v7_ratio = b['v7_direct'] / b['total'] * 100
        print(f"\n良性样本:")
        print(f"  总数: {b['total']}")
        print(f"  正确判断: {b['correct']} ({acc:.2f}%)")
        print(f"  V7直接判断: {b['v7_direct']} ({v7_ratio:.1f}%)")
        print(f"  LLM判断: {b['llm_judge']} ({100-v7_ratio:.1f}%)")
    
    # 计算总体指标
    total = results['harmful']['total'] + results['benign']['total']
    correct = results['harmful']['correct'] + results['benign']['correct']
    
    if total > 0:
        print(f"\n总体:")
        print(f"  准确率: {correct}/{total} ({correct/total*100:.2f}%)")
        
        if results['harmful']['total'] > 0:
            fnr = (results['harmful']['total'] - results['harmful']['correct']) / results['harmful']['total']
            print(f"  FNR (漏检率): {fnr*100:.2f}%")
        
        if results['benign']['total'] > 0:
            fpr = (results['benign']['total'] - results['benign']['correct']) / results['benign']['total']
            print(f"  FPR (误报率): {fpr*100:.2f}%")
    
    # 保存结果
    output_dir = '/home/vicuna/ludan/CSonEmbedding/results/v7_jbb_behaviors'
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存至: {output_file}")
    
    # 打印检测器统计
    detector.print_stats()
    
    return results


if __name__ == '__main__':
    test_jbb_behaviors()
