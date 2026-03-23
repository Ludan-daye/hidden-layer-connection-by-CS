#!/usr/bin/env python3
"""
V6 模型评估脚本
在多个测试数据集上评估模型性能
"""

import os
import sys
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    accuracy_score, confusion_matrix, classification_report
)
from transformers import AutoTokenizer
from datasets import load_dataset

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent))
from model import V6HarmfulDetector


def load_model(model_path, device='cuda'):
    """加载训练好的模型"""
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
    
    print(f"模型加载完成: {model_path}")
    print(f"最佳F1: {config.get('best_f1', 'N/A')}")
    
    return model, tokenizer, config


def predict_batch(model, tokenizer, texts, device='cuda', batch_size=32):
    """批量预测"""
    all_preds = []
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
            preds = (probs > 0.5).long()
        
        all_preds.extend(preds.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())
    
    return np.array(all_preds), np.array(all_probs)


def evaluate_dataset(model, tokenizer, texts, labels, name, device='cuda'):
    """评估单个数据集"""
    print(f"\n{'='*60}")
    print(f"评估数据集: {name}")
    print(f"样本数: {len(texts)}")
    print(f"{'='*60}")
    
    preds, probs = predict_batch(model, tokenizer, texts, device)
    
    # 计算指标
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    
    # 混淆矩阵
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    results = {
        'name': name,
        'samples': len(texts),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fpr': fpr,
        'fnr': fnr,
        'confusion_matrix': {
            'tn': int(tn), 'fp': int(fp),
            'fn': int(fn), 'tp': int(tp)
        }
    }
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"FPR:       {fpr:.4f}")
    print(f"FNR:       {fnr:.4f}")
    print(f"混淆矩阵: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
    return results


def load_jailbreakhub(base_path):
    """加载JailbreakHub数据集"""
    texts, labels = [], []
    
    # 尝试从本地加载
    jb_path = os.path.join(base_path, 'jailbreak_llms')
    if os.path.exists(jb_path):
        for file in ['jailbreak_prompts.csv', 'jailbreak_prompts_2023_12_25.csv']:
            fpath = os.path.join(jb_path, file)
            if os.path.exists(fpath):
                import pandas as pd
                df = pd.read_csv(fpath)
                if 'prompt' in df.columns:
                    texts.extend(df['prompt'].dropna().tolist())
                    labels.extend([1] * len(df['prompt'].dropna()))
    
    # 从HuggingFace加载
    if len(texts) == 0:
        try:
            ds = load_dataset("rubend18/ChatGPT-Jailbreak-Prompts", split='train')
            texts = [x['text'] for x in ds if x.get('text')]
            labels = [1] * len(texts)
        except:
            pass
    
    return texts[:500], labels[:500]  # 限制数量


def load_toxicchat(base_path):
    """加载ToxicChat数据集"""
    texts, labels = [], []
    
    try:
        ds = load_dataset("lmsys/toxic-chat", "toxicchat0124", split='test')
        for item in ds:
            if item.get('user_input'):
                texts.append(item['user_input'])
                labels.append(1 if item.get('toxicity', 0) == 1 else 0)
    except Exception as e:
        print(f"加载ToxicChat失败: {e}")
    
    return texts[:1000], labels[:1000]


def load_advbench(base_path):
    """加载AdvBench数据集"""
    texts, labels = [], []
    
    advbench_path = os.path.join(base_path, 'advbench')
    if os.path.exists(advbench_path):
        for file in os.listdir(advbench_path):
            if file.endswith('.csv'):
                import pandas as pd
                df = pd.read_csv(os.path.join(advbench_path, file))
                for col in ['goal', 'prompt', 'text']:
                    if col in df.columns:
                        texts.extend(df[col].dropna().tolist())
                        labels.extend([1] * len(df[col].dropna()))
                        break
    
    # 从HuggingFace加载
    if len(texts) == 0:
        try:
            ds = load_dataset("walledai/AdvBench", split='train')
            texts = [x['prompt'] for x in ds if x.get('prompt')]
            labels = [1] * len(texts)
        except:
            pass
    
    return texts[:500], labels[:500]


def load_harmbench(base_path):
    """加载HarmBench数据集"""
    texts, labels = [], []
    
    try:
        ds = load_dataset("harmbench/harmbench", split='test')
        for item in ds:
            if item.get('prompt'):
                texts.append(item['prompt'])
                labels.append(1)
    except:
        pass
    
    # 本地备份
    if len(texts) == 0:
        harmbench_path = os.path.join(base_path, 'harmbench')
        if os.path.exists(harmbench_path):
            for file in os.listdir(harmbench_path):
                if file.endswith('.json') or file.endswith('.jsonl'):
                    with open(os.path.join(harmbench_path, file), 'r') as f:
                        for line in f:
                            try:
                                item = json.loads(line)
                                if item.get('prompt') or item.get('text'):
                                    texts.append(item.get('prompt') or item.get('text'))
                                    labels.append(1)
                            except:
                                pass
    
    return texts[:500], labels[:500]


def load_alpaca(base_path):
    """加载Alpaca数据集 (正常样本)"""
    texts, labels = [], []
    
    try:
        ds = load_dataset("tatsu-lab/alpaca", split='train')
        for item in ds:
            if item.get('instruction'):
                text = item['instruction']
                if item.get('input'):
                    text += " " + item['input']
                texts.append(text)
                labels.append(0)
                if len(texts) >= 500:
                    break
    except:
        pass
    
    return texts, labels


def load_dolly(base_path):
    """加载Dolly数据集 (正常样本)"""
    texts, labels = [], []
    
    try:
        ds = load_dataset("databricks/databricks-dolly-15k", split='train')
        for item in ds:
            if item.get('instruction'):
                text = item['instruction']
                if item.get('context'):
                    text += " " + item['context']
                texts.append(text)
                labels.append(0)
                if len(texts) >= 500:
                    break
    except:
        pass
    
    return texts, labels


def load_or_bench(base_path):
    """加载OR-Bench数据集 (过度拒绝测试)"""
    texts, labels = [], []
    
    try:
        ds = load_dataset("bench-llm/or-bench", split='test')
        for item in ds:
            if item.get('prompt'):
                texts.append(item['prompt'])
                # OR-Bench中的样本应该是安全的(被过度拒绝的)
                labels.append(0)
    except:
        pass
    
    # 本地备份
    if len(texts) == 0:
        or_path = os.path.join(base_path, 'or_bench')
        if os.path.exists(or_path):
            for file in os.listdir(or_path):
                if file.endswith('.jsonl'):
                    with open(os.path.join(or_path, file), 'r') as f:
                        for line in f:
                            try:
                                item = json.loads(line)
                                if item.get('prompt'):
                                    texts.append(item['prompt'])
                                    labels.append(0)
                            except:
                                pass
    
    return texts[:500], labels[:500]


def load_xstest(base_path):
    """加载XSTest数据集 (边界测试)"""
    safe_texts, safe_labels = [], []
    unsafe_texts, unsafe_labels = [], []
    
    try:
        ds = load_dataset("walledai/XSTest", split='test')
        for item in ds:
            if item.get('prompt'):
                if item.get('label') == 'safe':
                    safe_texts.append(item['prompt'])
                    safe_labels.append(0)
                else:
                    unsafe_texts.append(item['prompt'])
                    unsafe_labels.append(1)
    except:
        pass
    
    return (safe_texts[:250], safe_labels[:250]), (unsafe_texts[:250], unsafe_labels[:250])


def load_jailbreakbench_gcg(base_path):
    """加载JailbreakBench GCG数据集"""
    texts, labels = [], []
    
    try:
        ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split='test')
        for item in ds:
            if item.get('Goal'):
                texts.append(item['Goal'])
                labels.append(1)
    except:
        pass
    
    # 本地GCG样本
    gcg_path = os.path.join(base_path, 'jailbreakbench', 'gcg')
    if os.path.exists(gcg_path):
        for file in os.listdir(gcg_path):
            if file.endswith('.jsonl'):
                with open(os.path.join(gcg_path, file), 'r') as f:
                    for line in f:
                        try:
                            item = json.loads(line)
                            if item.get('prompt') or item.get('jailbreak'):
                                texts.append(item.get('prompt') or item.get('jailbreak'))
                                labels.append(1)
                        except:
                            pass
    
    return texts[:300], labels[:300]


def load_gray_samples(base_path):
    """加载灰色样本测试集"""
    gray_benign_texts, gray_benign_labels = [], []
    gray_harmful_texts, gray_harmful_labels = [], []
    
    # 灰色-正常样本
    gb_path = os.path.join(base_path, 'v6_training', 'gray_benign.jsonl')
    if os.path.exists(gb_path):
        with open(gb_path, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if item.get('text'):
                        gray_benign_texts.append(item['text'])
                        gray_benign_labels.append(0)  # 应该被判为正常
                except:
                    pass
    
    # 灰色-恶意样本
    gh_path = os.path.join(base_path, 'v6_training', 'gray_harmful.jsonl')
    if os.path.exists(gh_path):
        with open(gh_path, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if item.get('text'):
                        gray_harmful_texts.append(item['text'])
                        gray_harmful_labels.append(1)  # 应该被判为恶意
                except:
                    pass
    
    return (gray_benign_texts, gray_benign_labels), (gray_harmful_texts, gray_harmful_labels)


def main():
    parser = argparse.ArgumentParser(description='V6模型评估')
    parser.add_argument('--model_path', type=str, 
                        default='/home/vicuna/ludan/CSonEmbedding/models/v6_finetuned',
                        help='模型路径')
    parser.add_argument('--data_path', type=str,
                        default='/home/vicuna/ludan/CSonEmbedding/datasets',
                        help='数据集路径')
    parser.add_argument('--output', type=str,
                        default='/home/vicuna/ludan/CSonEmbedding/results/v6_finetuned',
                        help='输出路径')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备')
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 加载模型
    model, tokenizer, config = load_model(args.model_path, args.device)
    
    all_results = {
        'model_config': config,
        'datasets': {}
    }
    
    # ============ 恶意数据集测试 ============
    print("\n" + "="*70)
    print("恶意数据集测试 (应检测为harmful)")
    print("="*70)
    
    # JailbreakHub
    texts, labels = load_jailbreakhub(args.data_path)
    if texts:
        results = evaluate_dataset(model, tokenizer, texts, labels, "JailbreakHub", args.device)
        all_results['datasets']['jailbreakhub'] = results
    
    # AdvBench
    texts, labels = load_advbench(args.data_path)
    if texts:
        results = evaluate_dataset(model, tokenizer, texts, labels, "AdvBench", args.device)
        all_results['datasets']['advbench'] = results
    
    # HarmBench
    texts, labels = load_harmbench(args.data_path)
    if texts:
        results = evaluate_dataset(model, tokenizer, texts, labels, "HarmBench", args.device)
        all_results['datasets']['harmbench'] = results
    
    # JailbreakBench GCG
    texts, labels = load_jailbreakbench_gcg(args.data_path)
    if texts:
        results = evaluate_dataset(model, tokenizer, texts, labels, "JBB-GCG", args.device)
        all_results['datasets']['jbb_gcg'] = results
    
    # ============ 正常数据集测试 ============
    print("\n" + "="*70)
    print("正常数据集测试 (应检测为benign)")
    print("="*70)
    
    # Alpaca
    texts, labels = load_alpaca(args.data_path)
    if texts:
        results = evaluate_dataset(model, tokenizer, texts, labels, "Alpaca", args.device)
        all_results['datasets']['alpaca'] = results
    
    # Dolly
    texts, labels = load_dolly(args.data_path)
    if texts:
        results = evaluate_dataset(model, tokenizer, texts, labels, "Dolly", args.device)
        all_results['datasets']['dolly'] = results
    
    # OR-Bench (过度拒绝测试)
    texts, labels = load_or_bench(args.data_path)
    if texts:
        results = evaluate_dataset(model, tokenizer, texts, labels, "OR-Bench", args.device)
        all_results['datasets']['or_bench'] = results
    
    # ============ 混合数据集测试 ============
    print("\n" + "="*70)
    print("混合数据集测试")
    print("="*70)
    
    # ToxicChat
    texts, labels = load_toxicchat(args.data_path)
    if texts:
        results = evaluate_dataset(model, tokenizer, texts, labels, "ToxicChat", args.device)
        all_results['datasets']['toxicchat'] = results
    
    # XSTest
    (safe_texts, safe_labels), (unsafe_texts, unsafe_labels) = load_xstest(args.data_path)
    if safe_texts:
        results = evaluate_dataset(model, tokenizer, safe_texts, safe_labels, "XSTest-Safe", args.device)
        all_results['datasets']['xstest_safe'] = results
    if unsafe_texts:
        results = evaluate_dataset(model, tokenizer, unsafe_texts, unsafe_labels, "XSTest-Unsafe", args.device)
        all_results['datasets']['xstest_unsafe'] = results
    
    # ============ 灰色样本测试 ============
    print("\n" + "="*70)
    print("灰色样本测试")
    print("="*70)
    
    (gb_texts, gb_labels), (gh_texts, gh_labels) = load_gray_samples(args.data_path)
    if gb_texts:
        results = evaluate_dataset(model, tokenizer, gb_texts, gb_labels, "Gray-Benign", args.device)
        all_results['datasets']['gray_benign'] = results
    if gh_texts:
        results = evaluate_dataset(model, tokenizer, gh_texts, gh_labels, "Gray-Harmful", args.device)
        all_results['datasets']['gray_harmful'] = results
    
    # ============ 汇总结果 ============
    print("\n" + "="*70)
    print("测试结果汇总")
    print("="*70)
    
    print(f"\n{'数据集':<20} {'样本数':<10} {'Accuracy':<12} {'F1':<12} {'Precision':<12} {'Recall':<12} {'FPR':<10}")
    print("-" * 90)
    
    for name, result in all_results['datasets'].items():
        print(f"{result['name']:<20} {result['samples']:<10} {result['accuracy']:<12.4f} {result['f1']:<12.4f} {result['precision']:<12.4f} {result['recall']:<12.4f} {result['fpr']:<10.4f}")
    
    # 保存结果
    output_file = os.path.join(args.output, 'test_results.json')
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n结果已保存至: {output_file}")
    
    # 生成README
    readme_content = generate_readme(all_results)
    readme_file = os.path.join(args.output, 'README.md')
    with open(readme_file, 'w') as f:
        f.write(readme_content)
    print(f"README已保存至: {readme_file}")


def generate_readme(results):
    """生成README文件"""
    config = results['model_config']
    
    content = f"""# V6 Fine-tuned Embedding Model 测试结果

## 模型信息

- **基础模型**: {config.get('model_name', 'BAAI/bge-base-en-v1.5')}
- **投影维度**: {config.get('projection_dim', 128)}
- **训练最佳F1**: {config.get('best_f1', 'N/A'):.4f}
- **最佳Epoch**: {config.get('best_epoch', 'N/A')}

## 测试结果汇总

| 数据集 | 样本数 | Accuracy | F1 | Precision | Recall | FPR |
|--------|--------|----------|-----|-----------|--------|-----|
"""
    
    for name, result in results['datasets'].items():
        content += f"| {result['name']} | {result['samples']} | {result['accuracy']:.4f} | {result['f1']:.4f} | {result['precision']:.4f} | {result['recall']:.4f} | {result['fpr']:.4f} |\n"
    
    content += """
## 数据集说明

### 恶意数据集
- **JailbreakHub**: 越狱攻击提示词集合
- **AdvBench**: 对抗性攻击基准
- **HarmBench**: 有害行为基准
- **JBB-GCG**: JailbreakBench GCG攻击样本

### 正常数据集
- **Alpaca**: 指令微调数据集
- **Dolly**: Databricks开源指令数据集
- **OR-Bench**: 过度拒绝测试集 (应判为正常)

### 混合数据集
- **ToxicChat**: 真实对话毒性检测
- **XSTest-Safe**: 安全边界测试
- **XSTest-Unsafe**: 不安全边界测试

### 灰色样本
- **Gray-Benign**: 灰色-正常样本 (应判为正常)
- **Gray-Harmful**: 灰色-恶意样本 (应判为恶意)

## 关键指标解读

- **F1**: 综合精确率和召回率的调和平均
- **Precision**: 预测为恶意中真正恶意的比例
- **Recall**: 真正恶意中被检测出的比例
- **FPR**: 误报率 (正常被误判为恶意的比例)
"""
    
    return content


if __name__ == '__main__':
    main()
