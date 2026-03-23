#!/usr/bin/env python3
"""
测试V7+LLM系统在多个数据集上的效果
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime
from datasets import load_dataset
from tqdm import tqdm

from detector import HarmfulContentDetector


def test_dataset(detector, texts, expected_label, name):
    """测试单个数据集"""
    results = {
        'name': name,
        'expected': expected_label,
        'total': len(texts),
        'correct': 0,
        'v7_benign': 0,
        'v7_harmful': 0,
        'llm_benign': 0,
        'llm_harmful': 0,
        'errors': []
    }
    
    for text in tqdm(texts, desc=name):
        result = detector.detect(text, verbose=False)
        
        is_correct = (result['label'] == expected_label)
        if is_correct:
            results['correct'] += 1
        else:
            results['errors'].append({
                'text': text[:150],
                'predicted': result['label'],
                'source': result['source'],
                'v7_prob': result['v7_prob']
            })
        
        if result['source'] == 'v7_classifier':
            if result['label'] == 'benign':
                results['v7_benign'] += 1
            else:
                results['v7_harmful'] += 1
        else:
            if result['label'] == 'benign':
                results['llm_benign'] += 1
            else:
                results['llm_harmful'] += 1
    
    return results


def print_results(results):
    """打印测试结果"""
    print(f"\n{'='*70}")
    print(f"数据集: {results['name']}")
    print(f"{'='*70}")
    print(f"期望标签: {results['expected']}")
    print(f"总样本数: {results['total']}")
    print(f"正确数: {results['correct']} ({results['correct']/results['total']*100:.2f}%)")
    
    v7_total = results['v7_benign'] + results['v7_harmful']
    llm_total = results['llm_benign'] + results['llm_harmful']
    
    print(f"\nV7直接判断: {v7_total} ({v7_total/results['total']*100:.1f}%)")
    print(f"  - benign: {results['v7_benign']}")
    print(f"  - harmful: {results['v7_harmful']}")
    
    print(f"\nLLM判断: {llm_total} ({llm_total/results['total']*100:.1f}%)")
    print(f"  - benign: {results['llm_benign']}")
    print(f"  - harmful: {results['llm_harmful']}")
    
    if results['errors']:
        print(f"\n错误样本 (前5个):")
        for i, err in enumerate(results['errors'][:5]):
            print(f"  [{i+1}] pred={err['predicted']}, prob={err['v7_prob']:.4f}, src={err['source']}")
            print(f"      {err['text'][:80]}...")


def main():
    print("="*70)
    print("V7+LLM系统多数据集测试")
    print("="*70)
    
    # 初始化检测器
    print("\n初始化检测系统...")
    detector = HarmfulContentDetector(lazy_load_llm=True)
    
    all_results = {}
    
    # ============ 1. JBB-Behaviors (恶意) ============
    print("\n" + "="*70)
    print("[1] JBB-Behaviors (恶意样本)")
    print("="*70)
    try:
        ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
        data = ds['train'] if 'train' in ds else list(ds.values())[0]
        
        texts = []
        for item in data:
            if 'Goal' in item:
                texts.append(item['Goal'])
            elif 'goal' in item:
                texts.append(item['goal'])
            elif 'Behavior' in item:
                texts.append(item['Behavior'])
        
        texts = texts[:100]  # 限制数量
        results = test_dataset(detector, texts, 'harmful', 'JBB-Behaviors')
        print_results(results)
        all_results['jbb_behaviors'] = results
    except Exception as e:
        print(f"加载失败: {e}")
    
    # ============ 2. HarmBench (恶意) ============
    print("\n" + "="*70)
    print("[2] HarmBench (恶意样本)")
    print("="*70)
    try:
        harmbench_path = '/home/vicuna/ludan/CSonEmbedding/datasets/harmbench'
        texts = []
        for file in os.listdir(harmbench_path):
            if file.endswith('.csv'):
                df = pd.read_csv(os.path.join(harmbench_path, file))
                for col in ['Behavior', 'goal', 'prompt']:
                    if col in df.columns:
                        texts.extend(df[col].dropna().tolist())
                        break
        
        texts = texts[:100]
        results = test_dataset(detector, texts, 'harmful', 'HarmBench')
        print_results(results)
        all_results['harmbench'] = results
    except Exception as e:
        print(f"加载失败: {e}")
    
    # ============ 3. AdvBench (恶意) ============
    print("\n" + "="*70)
    print("[3] AdvBench (恶意样本)")
    print("="*70)
    try:
        advbench_path = '/home/vicuna/ludan/CSonEmbedding/datasets/advbench'
        texts = []
        for file in os.listdir(advbench_path):
            if file.endswith('.csv'):
                df = pd.read_csv(os.path.join(advbench_path, file))
                for col in ['goal', 'prompt', 'Behavior']:
                    if col in df.columns:
                        texts.extend(df[col].dropna().tolist())
                        break
        
        texts = texts[:100]
        results = test_dataset(detector, texts, 'harmful', 'AdvBench')
        print_results(results)
        all_results['advbench'] = results
    except Exception as e:
        print(f"加载失败: {e}")
    
    # ============ 4. Alpaca (正常) ============
    print("\n" + "="*70)
    print("[4] Alpaca (正常样本)")
    print("="*70)
    try:
        ds = load_dataset("tatsu-lab/alpaca", split='train')
        texts = []
        for item in ds:
            if item.get('instruction'):
                text = item['instruction']
                if item.get('input'):
                    text += " " + item['input']
                texts.append(text)
                if len(texts) >= 100:
                    break
        
        results = test_dataset(detector, texts, 'benign', 'Alpaca')
        print_results(results)
        all_results['alpaca'] = results
    except Exception as e:
        print(f"加载失败: {e}")
    
    # ============ 5. Everyday-Conversations (正常) ============
    print("\n" + "="*70)
    print("[5] Everyday-Conversations (正常样本)")
    print("="*70)
    try:
        ds = load_dataset("HuggingFaceTB/everyday-conversations-llama3.1-2k", split='train_sft')
        texts = []
        for item in ds:
            if item.get('messages'):
                for msg in item['messages']:
                    if msg.get('role') == 'user' and msg.get('content'):
                        texts.append(msg['content'])
                if len(texts) >= 100:
                    break
        texts = texts[:100]
        
        results = test_dataset(detector, texts, 'benign', 'Everyday-Conversations')
        print_results(results)
        all_results['everyday_conv'] = results
    except Exception as e:
        print(f"加载失败: {e}")
    
    # ============ 6. WildChat-English (正常) ============
    print("\n" + "="*70)
    print("[6] WildChat-English (正常样本)")
    print("="*70)
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
                                if count >= 100:
                                    break
            if count >= 100:
                break
        
        results = test_dataset(detector, texts, 'benign', 'WildChat-English')
        print_results(results)
        all_results['wildchat'] = results
    except Exception as e:
        print(f"加载失败: {e}")
    
    # ============ 汇总 ============
    print("\n" + "="*70)
    print("汇总结果")
    print("="*70)
    
    print(f"\n{'数据集':<25} {'类型':<10} {'准确率':<12} {'V7直接':<10} {'LLM判断':<10}")
    print("-"*70)
    
    total_correct = 0
    total_samples = 0
    
    for name, res in all_results.items():
        acc = res['correct'] / res['total'] * 100
        v7_ratio = (res['v7_benign'] + res['v7_harmful']) / res['total'] * 100
        llm_ratio = (res['llm_benign'] + res['llm_harmful']) / res['total'] * 100
        
        print(f"{res['name']:<25} {res['expected']:<10} {acc:<12.2f}% {v7_ratio:<10.1f}% {llm_ratio:<10.1f}%")
        
        total_correct += res['correct']
        total_samples += res['total']
    
    print("-"*70)
    print(f"{'总体':<25} {'':<10} {total_correct/total_samples*100:<12.2f}%")
    
    # 保存结果
    output_dir = '/home/vicuna/ludan/CSonEmbedding/results/v7_all_datasets'
    os.makedirs(output_dir, exist_ok=True)
    
    # 简化结果（去掉详细错误列表）
    summary = {}
    for name, res in all_results.items():
        summary[name] = {
            'name': res['name'],
            'expected': res['expected'],
            'total': res['total'],
            'correct': res['correct'],
            'accuracy': res['correct'] / res['total'],
            'v7_benign': res['v7_benign'],
            'v7_harmful': res['v7_harmful'],
            'llm_benign': res['llm_benign'],
            'llm_harmful': res['llm_harmful'],
            'error_count': len(res['errors'])
        }
    
    output_file = os.path.join(output_dir, 'summary.json')
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存至: {output_file}")
    
    # 打印检测器统计
    detector.print_stats()


if __name__ == '__main__':
    main()
