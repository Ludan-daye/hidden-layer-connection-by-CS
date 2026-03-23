#!/usr/bin/env python3
"""
V6分类器 + 灰色地带检测

三分类输出:
- harmful: 概率 > upper_threshold
- benign: 概率 < lower_threshold
- uncertain: lower_threshold <= 概率 <= upper_threshold (交给LLM)

优势:
- 直接使用V6 fine-tuned分类器
- 无需CS压缩和多质心
- 端到端学习，效果更好
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from transformers import AutoTokenizer
from datasets import load_dataset
import pandas as pd

sys.path.append(str(Path(__file__).parent))
from model import V6HarmfulDetector


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


def predict_probs(model, tokenizer, texts, device='cuda', batch_size=32):
    """预测概率"""
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


def find_optimal_thresholds(probs, labels, max_uncertain_ratio=0.15):
    """
    寻找最优的双阈值
    
    目标:
    1. 确定样本准确率最高
    2. 不确定样本比例不超过max_uncertain_ratio
    """
    labels = np.array(labels)
    
    # 二分类基准
    binary_preds = (probs > 0.5).astype(int)
    binary_acc = (binary_preds == labels).mean()
    errors = (binary_preds != labels)
    
    print(f"二分类准确率: {binary_acc*100:.2f}%")
    print(f"错误数: {errors.sum()} / {len(labels)}")
    
    # 分析错误样本的概率分布
    error_probs = probs[errors]
    print(f"错误样本概率范围: [{error_probs.min():.4f}, {error_probs.max():.4f}]")
    
    best_result = None
    best_score = -1
    
    # 搜索双阈值
    for lower in np.arange(0.1, 0.5, 0.05):
        for upper in np.arange(0.5, 0.9, 0.05):
            if lower >= upper:
                continue
            
            # 三分类
            uncertain_mask = (probs >= lower) & (probs <= upper)
            uncertain_ratio = uncertain_mask.sum() / len(labels)
            
            if uncertain_ratio > max_uncertain_ratio:
                continue
            
            # 确定样本准确率
            certain_mask = ~uncertain_mask
            if certain_mask.sum() > 0:
                certain_preds = (probs[certain_mask] > upper).astype(int)
                certain_acc = (certain_preds == labels[certain_mask]).mean()
            else:
                certain_acc = 0
            
            # 错误捕获率
            errors_in_uncertain = errors[uncertain_mask].sum()
            error_capture = errors_in_uncertain / errors.sum() if errors.sum() > 0 else 0
            
            # 综合得分
            score = certain_acc * 0.6 + error_capture * 0.3 - uncertain_ratio * 0.1
            
            if score > best_score:
                best_score = score
                best_result = {
                    'lower': lower,
                    'upper': upper,
                    'certain_acc': certain_acc,
                    'uncertain_ratio': uncertain_ratio,
                    'error_capture': error_capture,
                    'score': score
                }
    
    return best_result


def evaluate_three_class(probs, labels, lower, upper, name):
    """评估三分类"""
    labels = np.array(labels)
    
    # 三分类: 0=benign, 1=harmful, 2=uncertain
    preds = np.zeros(len(probs), dtype=int)
    preds[probs > upper] = 1  # harmful
    preds[probs < lower] = 0  # benign
    preds[(probs >= lower) & (probs <= upper)] = 2  # uncertain
    
    n_benign = (preds == 0).sum()
    n_harmful = (preds == 1).sum()
    n_uncertain = (preds == 2).sum()
    
    # 确定样本准确率
    certain_mask = preds != 2
    if certain_mask.sum() > 0:
        certain_preds = preds[certain_mask]
        certain_labels = labels[certain_mask]
        certain_acc = (certain_preds == certain_labels).mean()
        
        # 详细指标
        precision = precision_score(certain_labels, certain_preds, zero_division=0)
        recall = recall_score(certain_labels, certain_preds, zero_division=0)
        f1 = f1_score(certain_labels, certain_preds, zero_division=0)
        
        cm = confusion_matrix(certain_labels, certain_preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    else:
        certain_acc = precision = recall = f1 = fpr = 0
        tn = fp = fn = tp = 0
    
    # 二分类对比
    binary_preds = (probs > 0.5).astype(int)
    binary_acc = (binary_preds == labels).mean()
    
    return {
        'name': name,
        'n_total': len(labels),
        'n_benign': int(n_benign),
        'n_harmful': int(n_harmful),
        'n_uncertain': int(n_uncertain),
        'uncertain_ratio': n_uncertain / len(labels),
        'certain_acc': certain_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fpr': fpr,
        'binary_acc': binary_acc,
        'improvement': certain_acc - binary_acc
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = '/home/vicuna/ludan/CSonEmbedding/models/v6_finetuned'
    data_path = '/home/vicuna/ludan/CSonEmbedding/datasets'
    output_path = '/home/vicuna/ludan/CSonEmbedding/models/v6_gray_zone'
    
    os.makedirs(output_path, exist_ok=True)
    
    print("="*70)
    print("V6分类器 + 灰色地带检测")
    print("="*70)
    
    # 加载模型
    print("\n加载V6模型...")
    model, tokenizer = load_v6_model(model_path, device)
    
    # 收集所有测试数据
    print("\n加载测试数据...")
    all_data = []
    
    # 恶意数据集
    print("  恶意数据集...")
    
    # HarmBench
    harmbench_path = os.path.join(data_path, 'harmbench')
    if os.path.exists(harmbench_path):
        texts = []
        for file in os.listdir(harmbench_path):
            if file.endswith('.csv'):
                try:
                    df = pd.read_csv(os.path.join(harmbench_path, file))
                    for col in ['Behavior', 'goal', 'prompt']:
                        if col in df.columns:
                            texts.extend(df[col].dropna().tolist()[:300])
                            break
                except:
                    pass
        if texts:
            probs = predict_probs(model, tokenizer, texts[:500], device)
            for text, prob in zip(texts[:500], probs):
                all_data.append({'text': text, 'prob': prob, 'label': 1, 'dataset': 'HarmBench'})
            print(f"    HarmBench: {len(texts[:500])} 样本")
    
    # 正常数据集
    print("  正常数据集...")
    
    # Alpaca
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
        probs = predict_probs(model, tokenizer, texts, device)
        for text, prob in zip(texts, probs):
            all_data.append({'text': text, 'prob': prob, 'label': 0, 'dataset': 'Alpaca'})
        print(f"    Alpaca: {len(texts)} 样本")
    except Exception as e:
        print(f"    Alpaca失败: {e}")
    
    # Dolly
    try:
        ds = load_dataset("databricks/databricks-dolly-15k", split='train')
        texts = []
        for item in ds:
            if item.get('instruction'):
                text = item['instruction']
                if item.get('context'):
                    text += " " + item['context']
                texts.append(text)
                if len(texts) >= 500:
                    break
        probs = predict_probs(model, tokenizer, texts, device)
        for text, prob in zip(texts, probs):
            all_data.append({'text': text, 'prob': prob, 'label': 0, 'dataset': 'Dolly'})
        print(f"    Dolly: {len(texts)} 样本")
    except Exception as e:
        print(f"    Dolly失败: {e}")
    
    # OpenAssistant
    try:
        ds = load_dataset("OpenAssistant/oasst1", split='train')
        texts = [x['text'] for x in ds if x.get('text') and x.get('role') == 'prompter'][:500]
        probs = predict_probs(model, tokenizer, texts, device)
        for text, prob in zip(texts, probs):
            all_data.append({'text': text, 'prob': prob, 'label': 0, 'dataset': 'OpenAssistant'})
        print(f"    OpenAssistant: {len(texts)} 样本")
    except Exception as e:
        print(f"    OpenAssistant失败: {e}")
    
    # 混合数据集
    print("  混合数据集...")
    
    # ToxicChat
    try:
        ds = load_dataset("lmsys/toxic-chat", "toxicchat0124", split='test')
        texts = [x['user_input'] for x in ds if x.get('user_input')][:1000]
        labels = [1 if x.get('toxicity', 0) == 1 else 0 for x in ds if x.get('user_input')][:1000]
        probs = predict_probs(model, tokenizer, texts, device)
        for text, prob, label in zip(texts, probs, labels):
            all_data.append({'text': text, 'prob': prob, 'label': label, 'dataset': 'ToxicChat'})
        print(f"    ToxicChat: {len(texts)} 样本")
    except Exception as e:
        print(f"    ToxicChat失败: {e}")
    
    # 灰色样本
    print("  灰色样本...")
    
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
        if texts:
            probs = predict_probs(model, tokenizer, texts, device)
            for text, prob in zip(texts, probs):
                all_data.append({'text': text, 'prob': prob, 'label': 0, 'dataset': 'Gray-Benign'})
            print(f"    Gray-Benign: {len(texts)} 样本")
    
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
        if texts:
            probs = predict_probs(model, tokenizer, texts, device)
            for text, prob in zip(texts, probs):
                all_data.append({'text': text, 'prob': prob, 'label': 1, 'dataset': 'Gray-Harmful'})
            print(f"    Gray-Harmful: {len(texts)} 样本")
    
    print(f"\n总样本数: {len(all_data)}")
    
    # 转换为数组
    all_probs = np.array([d['prob'] for d in all_data])
    all_labels = np.array([d['label'] for d in all_data])
    datasets = [d['dataset'] for d in all_data]
    
    # 寻找最优阈值
    print("\n" + "="*50)
    print("寻找最优双阈值")
    print("="*50)
    
    best = find_optimal_thresholds(all_probs, all_labels, max_uncertain_ratio=0.15)
    
    if best:
        lower = best['lower']
        upper = best['upper']
        print(f"\n最优阈值: lower={lower:.2f}, upper={upper:.2f}")
        print(f"  确定样本准确率: {best['certain_acc']*100:.2f}%")
        print(f"  不确定比例: {best['uncertain_ratio']*100:.2f}%")
        print(f"  错误捕获率: {best['error_capture']*100:.2f}%")
    else:
        lower, upper = 0.3, 0.7
        print(f"\n使用默认阈值: lower={lower}, upper={upper}")
    
    # 评估各数据集
    print("\n" + "="*50)
    print("按数据集评估")
    print("="*50)
    
    dataset_results = {}
    
    for ds_name in set(datasets):
        ds_mask = np.array([d == ds_name for d in datasets])
        ds_probs = all_probs[ds_mask]
        ds_labels = all_labels[ds_mask]
        
        result = evaluate_three_class(ds_probs, ds_labels, lower, upper, ds_name)
        dataset_results[ds_name] = result
        
        print(f"\n{ds_name}:")
        print(f"  样本数: {result['n_total']}")
        print(f"  不确定: {result['n_uncertain']} ({result['uncertain_ratio']*100:.1f}%)")
        print(f"  确定准确率: {result['certain_acc']*100:.2f}%")
        print(f"  二分类准确率: {result['binary_acc']*100:.2f}%")
        print(f"  提升: {result['improvement']*100:+.2f}%")
    
    # 总体评估
    print("\n" + "="*50)
    print("总体评估")
    print("="*50)
    
    overall = evaluate_three_class(all_probs, all_labels, lower, upper, "Overall")
    
    print(f"\n总样本: {overall['n_total']}")
    print(f"  判为正常: {overall['n_benign']} ({overall['n_benign']/overall['n_total']*100:.1f}%)")
    print(f"  判为恶意: {overall['n_harmful']} ({overall['n_harmful']/overall['n_total']*100:.1f}%)")
    print(f"  判为不确定: {overall['n_uncertain']} ({overall['n_uncertain']/overall['n_total']*100:.1f}%)")
    print(f"\n确定样本准确率: {overall['certain_acc']*100:.2f}%")
    print(f"二分类准确率: {overall['binary_acc']*100:.2f}%")
    print(f"提升: {overall['improvement']*100:+.2f}%")
    
    # 保存配置
    config = {
        'lower_threshold': lower,
        'upper_threshold': upper,
        'description': {
            'harmful': f'prob > {upper}',
            'benign': f'prob < {lower}',
            'uncertain': f'{lower} <= prob <= {upper}'
        },
        'overall_results': overall,
        'dataset_results': dataset_results
    }
    
    config_file = os.path.join(output_path, 'gray_zone_config.json')
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"\n配置已保存至: {config_file}")
    
    # 保存不确定样本
    uncertain_samples = []
    for d in all_data:
        if lower <= d['prob'] <= upper:
            uncertain_samples.append(d)
    
    uncertain_file = os.path.join(output_path, 'uncertain_samples.jsonl')
    with open(uncertain_file, 'w') as f:
        for sample in uncertain_samples:
            f.write(json.dumps({
                'text': sample['text'],
                'prob': float(sample['prob']),
                'label': int(sample['label']),
                'dataset': sample['dataset']
            }, ensure_ascii=False) + '\n')
    
    print(f"不确定样本已保存至: {uncertain_file} ({len(uncertain_samples)} 样本)")
    
    # 汇总
    print("\n" + "="*70)
    print("汇总")
    print("="*70)
    print(f"""
V6分类器灰色地带配置:
  - 恶意: prob > {upper:.2f}
  - 正常: prob < {lower:.2f}
  - 不确定: {lower:.2f} <= prob <= {upper:.2f} (交给LLM)

效果:
  - 不确定样本比例: {overall['uncertain_ratio']*100:.1f}%
  - 确定样本准确率: {overall['certain_acc']*100:.2f}%
  - 二分类准确率: {overall['binary_acc']*100:.2f}%
  - 提升: {overall['improvement']*100:+.2f}%
""")


if __name__ == '__main__':
    main()
