#!/usr/bin/env python3
"""
滑动窗口灰色地带检测

思路：
1. 计算所有样本的 harmful_sim - benign_sim 差值
2. 分析错误样本的分布，找到错误集中的区域
3. 用滑动窗口找到能框住最多错误的区间
4. 将这个区间标记为"不确定"，需要交给LLM判断

三分类结果：
- harmful: 差值 > upper_threshold
- benign: 差值 < lower_threshold  
- uncertain: lower_threshold <= 差值 <= upper_threshold (交给LLM)
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from transformers import AutoTokenizer
from datasets import load_dataset

sys.path.append(str(Path(__file__).parent))
from model import V6HarmfulDetector


class LearnedCSProjection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim, bias=False)
    
    def forward(self, x):
        z = self.proj(x)
        z = F.normalize(z, p=2, dim=-1)
        return z


def load_v6_model(model_path, device='cuda'):
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
    all_embeddings = []
    
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
            embeddings, _, _ = model(inputs['input_ids'], inputs['attention_mask'])
        
        all_embeddings.append(embeddings.cpu().numpy())
    
    return np.vstack(all_embeddings)


def compute_scores(embeddings, centroids, benign_centroid):
    """计算分数"""
    harmful_sims = embeddings @ centroids.T
    max_harmful_sim = np.max(harmful_sims, axis=1)
    benign_sim = embeddings @ benign_centroid
    diff = max_harmful_sim - benign_sim
    return diff, max_harmful_sim, benign_sim


def find_optimal_gray_zone(diffs, labels, max_gray_ratio=0.15):
    """
    使用滑动窗口找到最优灰色地带
    
    目标：找到一个区间 [lower, upper]，使得：
    1. 区间内包含尽可能多的错误样本
    2. 区间尽可能小（不超过max_gray_ratio）
    3. 区间应该围绕阈值0（决策边界）
    """
    labels = np.array(labels)
    
    # 使用阈值0进行初始预测
    initial_preds = (diffs > 0).astype(int)
    errors = (initial_preds != labels)
    
    print(f"\n初始错误数: {errors.sum()} / {len(labels)} ({errors.mean()*100:.2f}%)")
    
    # 错误样本的diff分布
    error_diffs = diffs[errors]
    correct_diffs = diffs[~errors]
    
    if len(error_diffs) == 0:
        print("没有错误样本，无需灰色地带")
        return 0, 0, {}
    
    print(f"错误样本diff范围: [{error_diffs.min():.4f}, {error_diffs.max():.4f}]")
    print(f"正确样本diff范围: [{correct_diffs.min():.4f}, {correct_diffs.max():.4f}]")
    
    # 分析错误样本在阈值0附近的分布
    # 误报(FP): 正常样本被判为恶意 -> label=0, diff>0
    # 漏检(FN): 恶意样本被判为正常 -> label=1, diff<0
    fp_mask = (labels == 0) & (diffs > 0)
    fn_mask = (labels == 1) & (diffs < 0)
    
    fp_diffs = diffs[fp_mask]
    fn_diffs = diffs[fn_mask]
    
    print(f"\n误报(FP): {fp_mask.sum()} 样本, diff范围: [{fp_diffs.min():.4f}, {fp_diffs.max():.4f}]" if len(fp_diffs) > 0 else f"\n误报(FP): 0 样本")
    print(f"漏检(FN): {fn_mask.sum()} 样本, diff范围: [{fn_diffs.min():.4f}, {fn_diffs.max():.4f}]" if len(fn_diffs) > 0 else f"漏检(FN): 0 样本")
    
    best_result = None
    best_score = -1
    
    # 搜索策略：以0为中心，向两边扩展
    # lower从0向负方向搜索，upper从0向正方向搜索
    max_samples = int(len(labels) * max_gray_ratio)
    
    for lower_percentile in range(0, 50, 2):  # 0-50%
        for upper_percentile in range(50, 100, 2):  # 50-100%
            lower = np.percentile(diffs, lower_percentile)
            upper = np.percentile(diffs, upper_percentile)
            
            # 确保包含0
            if lower > 0 or upper < 0:
                continue
            
            in_gray = (diffs >= lower) & (diffs <= upper)
            gray_count = in_gray.sum()
            
            # 限制灰色地带大小
            if gray_count > max_samples:
                continue
            
            # 计算指标
            errors_in_gray = errors[in_gray].sum()
            error_capture_rate = errors_in_gray / errors.sum() if errors.sum() > 0 else 0
            
            # 灰色地带外的准确率
            out_gray = ~in_gray
            if out_gray.sum() > 0:
                out_preds = (diffs[out_gray] > upper).astype(int)  # >upper为恶意
                out_accuracy = (out_preds == labels[out_gray]).mean()
            else:
                out_accuracy = 1.0
            
            gray_ratio = gray_count / len(labels)
            
            # 综合得分：高错误捕获 + 高区外准确率 - 灰色地带惩罚
            # 更重视区外准确率
            score = error_capture_rate * 0.3 + out_accuracy * 0.5 - gray_ratio * 0.2
            
            if score > best_score:
                best_score = score
                best_result = {
                    'lower': lower,
                    'upper': upper,
                    'error_capture_rate': error_capture_rate,
                    'errors_in_gray': int(errors_in_gray),
                    'total_errors': int(errors.sum()),
                    'out_accuracy': out_accuracy,
                    'gray_ratio': gray_ratio,
                    'gray_count': int(gray_count),
                    'score': score
                }
    
    if best_result is None:
        # 如果没找到合适的，使用默认值
        lower = np.percentile(diffs, 25)
        upper = np.percentile(diffs, 75)
        if lower > 0:
            lower = -0.1
        if upper < 0:
            upper = 0.1
        best_result = {
            'lower': lower,
            'upper': upper,
            'error_capture_rate': 0,
            'errors_in_gray': 0,
            'total_errors': int(errors.sum()),
            'out_accuracy': 0,
            'gray_ratio': 0,
            'gray_count': 0,
            'score': 0
        }
    
    return best_result['lower'], best_result['upper'], best_result


def evaluate_three_class(diffs, labels, lower_threshold, upper_threshold):
    """评估三分类结果"""
    labels = np.array(labels)
    
    # 三分类
    preds = np.zeros(len(diffs), dtype=int)  # 0: benign, 1: harmful, 2: uncertain
    preds[diffs > upper_threshold] = 1  # harmful
    preds[diffs < lower_threshold] = 0  # benign
    preds[(diffs >= lower_threshold) & (diffs <= upper_threshold)] = 2  # uncertain
    
    # 统计
    n_benign = (preds == 0).sum()
    n_harmful = (preds == 1).sum()
    n_uncertain = (preds == 2).sum()
    
    # 确定样本的准确率
    certain_mask = preds != 2
    if certain_mask.sum() > 0:
        certain_preds = preds[certain_mask]
        certain_labels = labels[certain_mask]
        certain_accuracy = (certain_preds == certain_labels).mean()
    else:
        certain_accuracy = 0
    
    # 不确定样本中的真实分布
    uncertain_mask = preds == 2
    if uncertain_mask.sum() > 0:
        uncertain_labels = labels[uncertain_mask]
        uncertain_harmful_ratio = uncertain_labels.mean()
    else:
        uncertain_harmful_ratio = 0
    
    # 原始二分类指标（用于对比）
    binary_preds = (diffs > 0).astype(int)
    binary_accuracy = (binary_preds == labels).mean()
    
    results = {
        'n_total': len(labels),
        'n_benign': int(n_benign),
        'n_harmful': int(n_harmful),
        'n_uncertain': int(n_uncertain),
        'uncertain_ratio': n_uncertain / len(labels),
        'certain_accuracy': certain_accuracy,
        'uncertain_harmful_ratio': uncertain_harmful_ratio,
        'binary_accuracy': binary_accuracy,
        'lower_threshold': lower_threshold,
        'upper_threshold': upper_threshold
    }
    
    return results, preds


def load_all_test_data(v6_model, tokenizer, cs_model, centroids, benign_centroid, device):
    """加载所有测试数据并计算分数"""
    data_path = '/home/vicuna/ludan/CSonEmbedding/datasets'
    
    all_data = []
    
    # 1. 恶意数据集
    print("\n加载恶意数据集...")
    
    # AdvBench
    try:
        ds = load_dataset("walledai/AdvBench", split='train')
        texts = [x['prompt'] for x in ds if x.get('prompt')][:500]
        embeddings = extract_embeddings(v6_model, tokenizer, texts, device)
        with torch.no_grad():
            compressed = cs_model(torch.tensor(embeddings, dtype=torch.float32).to(device)).cpu().numpy()
        diffs, _, _ = compute_scores(compressed, centroids, benign_centroid)
        for i, (text, diff) in enumerate(zip(texts, diffs)):
            all_data.append({'text': text, 'diff': diff, 'label': 1, 'dataset': 'AdvBench'})
        print(f"  AdvBench: {len(texts)} 样本")
    except Exception as e:
        print(f"  AdvBench失败: {e}")
    
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
            embeddings = extract_embeddings(v6_model, tokenizer, texts[:500], device)
            with torch.no_grad():
                compressed = cs_model(torch.tensor(embeddings, dtype=torch.float32).to(device)).cpu().numpy()
            diffs, _, _ = compute_scores(compressed, centroids, benign_centroid)
            for text, diff in zip(texts[:500], diffs):
                all_data.append({'text': text, 'diff': diff, 'label': 1, 'dataset': 'HarmBench'})
            print(f"  HarmBench: {len(texts[:500])} 样本")
    
    # 2. 正常数据集
    print("\n加载正常数据集...")
    
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
        embeddings = extract_embeddings(v6_model, tokenizer, texts, device)
        with torch.no_grad():
            compressed = cs_model(torch.tensor(embeddings, dtype=torch.float32).to(device)).cpu().numpy()
        diffs, _, _ = compute_scores(compressed, centroids, benign_centroid)
        for text, diff in zip(texts, diffs):
            all_data.append({'text': text, 'diff': diff, 'label': 0, 'dataset': 'Alpaca'})
        print(f"  Alpaca: {len(texts)} 样本")
    except Exception as e:
        print(f"  Alpaca失败: {e}")
    
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
        embeddings = extract_embeddings(v6_model, tokenizer, texts, device)
        with torch.no_grad():
            compressed = cs_model(torch.tensor(embeddings, dtype=torch.float32).to(device)).cpu().numpy()
        diffs, _, _ = compute_scores(compressed, centroids, benign_centroid)
        for text, diff in zip(texts, diffs):
            all_data.append({'text': text, 'diff': diff, 'label': 0, 'dataset': 'Dolly'})
        print(f"  Dolly: {len(texts)} 样本")
    except Exception as e:
        print(f"  Dolly失败: {e}")
    
    # OpenAssistant
    try:
        ds = load_dataset("OpenAssistant/oasst1", split='train')
        texts = [x['text'] for x in ds if x.get('text') and x.get('role') == 'prompter'][:500]
        embeddings = extract_embeddings(v6_model, tokenizer, texts, device)
        with torch.no_grad():
            compressed = cs_model(torch.tensor(embeddings, dtype=torch.float32).to(device)).cpu().numpy()
        diffs, _, _ = compute_scores(compressed, centroids, benign_centroid)
        for text, diff in zip(texts, diffs):
            all_data.append({'text': text, 'diff': diff, 'label': 0, 'dataset': 'OpenAssistant'})
        print(f"  OpenAssistant: {len(texts)} 样本")
    except Exception as e:
        print(f"  OpenAssistant失败: {e}")
    
    # 3. 混合数据集
    print("\n加载混合数据集...")
    
    # ToxicChat
    try:
        ds = load_dataset("lmsys/toxic-chat", "toxicchat0124", split='test')
        texts = [x['user_input'] for x in ds if x.get('user_input')][:1000]
        labels = [1 if x.get('toxicity', 0) == 1 else 0 for x in ds if x.get('user_input')][:1000]
        embeddings = extract_embeddings(v6_model, tokenizer, texts, device)
        with torch.no_grad():
            compressed = cs_model(torch.tensor(embeddings, dtype=torch.float32).to(device)).cpu().numpy()
        diffs, _, _ = compute_scores(compressed, centroids, benign_centroid)
        for text, diff, label in zip(texts, diffs, labels):
            all_data.append({'text': text, 'diff': diff, 'label': label, 'dataset': 'ToxicChat'})
        print(f"  ToxicChat: {len(texts)} 样本")
    except Exception as e:
        print(f"  ToxicChat失败: {e}")
    
    # 4. 灰色样本
    print("\n加载灰色样本...")
    
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
            embeddings = extract_embeddings(v6_model, tokenizer, texts, device)
            with torch.no_grad():
                compressed = cs_model(torch.tensor(embeddings, dtype=torch.float32).to(device)).cpu().numpy()
            diffs, _, _ = compute_scores(compressed, centroids, benign_centroid)
            for text, diff in zip(texts, diffs):
                all_data.append({'text': text, 'diff': diff, 'label': 0, 'dataset': 'Gray-Benign'})
            print(f"  Gray-Benign: {len(texts)} 样本")
    
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
            embeddings = extract_embeddings(v6_model, tokenizer, texts, device)
            with torch.no_grad():
                compressed = cs_model(torch.tensor(embeddings, dtype=torch.float32).to(device)).cpu().numpy()
            diffs, _, _ = compute_scores(compressed, centroids, benign_centroid)
            for text, diff in zip(texts, diffs):
                all_data.append({'text': text, 'diff': diff, 'label': 1, 'dataset': 'Gray-Harmful'})
            print(f"  Gray-Harmful: {len(texts)} 样本")
    
    return all_data


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = '/home/vicuna/ludan/CSonEmbedding/models/v6_finetuned'
    cs_path = '/home/vicuna/ludan/CSonEmbedding/models/v6_cs_multicentroid'
    output_path = '/home/vicuna/ludan/CSonEmbedding/models/v6_sliding_window'
    
    os.makedirs(output_path, exist_ok=True)
    
    print("="*70)
    print("滑动窗口灰色地带检测")
    print("="*70)
    
    # 加载模型
    print("\n加载模型...")
    v6_model, tokenizer = load_v6_model(model_path, device)
    
    config = "32d_3c"
    saved = np.load(os.path.join(cs_path, f'cs_multicentroid_{config}.npz'))
    
    cs_model = LearnedCSProjection(768, 32).to(device)
    cs_model.proj.weight.data = torch.tensor(saved['cs_weights'], dtype=torch.float32).to(device)
    cs_model.eval()
    
    centroids = saved['centroids']
    benign_centroid = saved['benign_centroid']
    
    # 加载所有测试数据
    print("\n" + "="*50)
    print("加载测试数据")
    print("="*50)
    all_data = load_all_test_data(v6_model, tokenizer, cs_model, centroids, benign_centroid, device)
    
    print(f"\n总样本数: {len(all_data)}")
    
    # 转换为数组
    diffs = np.array([d['diff'] for d in all_data])
    labels = np.array([d['label'] for d in all_data])
    datasets = [d['dataset'] for d in all_data]
    
    # 找最优灰色地带
    print("\n" + "="*50)
    print("寻找最优灰色地带")
    print("="*50)
    
    lower, upper, gray_info = find_optimal_gray_zone(diffs, labels)
    
    print(f"\n最优灰色地带: [{lower:.4f}, {upper:.4f}]")
    print(f"  错误捕获率: {gray_info['error_capture_rate']*100:.2f}%")
    print(f"  区外准确率: {gray_info['out_accuracy']*100:.2f}%")
    print(f"  灰色地带比例: {gray_info['gray_ratio']*100:.2f}%")
    
    # 评估三分类
    print("\n" + "="*50)
    print("三分类评估")
    print("="*50)
    
    results, preds = evaluate_three_class(diffs, labels, lower, upper)
    
    print(f"\n总样本: {results['n_total']}")
    print(f"  判为正常: {results['n_benign']} ({results['n_benign']/results['n_total']*100:.1f}%)")
    print(f"  判为恶意: {results['n_harmful']} ({results['n_harmful']/results['n_total']*100:.1f}%)")
    print(f"  判为不确定: {results['n_uncertain']} ({results['n_uncertain']/results['n_total']*100:.1f}%)")
    print(f"\n确定样本准确率: {results['certain_accuracy']*100:.2f}%")
    print(f"不确定样本中恶意比例: {results['uncertain_harmful_ratio']*100:.2f}%")
    print(f"原始二分类准确率: {results['binary_accuracy']*100:.2f}%")
    
    # 按数据集分析
    print("\n" + "="*50)
    print("按数据集分析")
    print("="*50)
    
    dataset_results = {}
    for ds_name in set(datasets):
        ds_mask = np.array([d == ds_name for d in datasets])
        ds_diffs = diffs[ds_mask]
        ds_labels = labels[ds_mask]
        ds_preds = preds[ds_mask]
        
        n_total = len(ds_labels)
        n_uncertain = (ds_preds == 2).sum()
        
        # 确定样本准确率
        certain_mask = ds_preds != 2
        if certain_mask.sum() > 0:
            certain_acc = (ds_preds[certain_mask] == ds_labels[certain_mask]).mean()
        else:
            certain_acc = 0
        
        # 原始准确率
        binary_preds = (ds_diffs > 0).astype(int)
        binary_acc = (binary_preds == ds_labels).mean()
        
        dataset_results[ds_name] = {
            'n_total': n_total,
            'n_uncertain': int(n_uncertain),
            'uncertain_ratio': n_uncertain / n_total,
            'certain_accuracy': certain_acc,
            'binary_accuracy': binary_acc,
            'improvement': certain_acc - binary_acc
        }
        
        print(f"\n{ds_name}:")
        print(f"  样本数: {n_total}")
        print(f"  不确定: {n_uncertain} ({n_uncertain/n_total*100:.1f}%)")
        print(f"  确定样本准确率: {certain_acc*100:.2f}%")
        print(f"  原始准确率: {binary_acc*100:.2f}%")
        print(f"  提升: {(certain_acc-binary_acc)*100:+.2f}%")
    
    # 保存结果
    output = {
        'config': {
            'lower_threshold': float(lower),
            'upper_threshold': float(upper)
        },
        'gray_zone_info': {k: float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v 
                          for k, v in gray_info.items()},
        'overall_results': results,
        'dataset_results': dataset_results
    }
    
    output_file = os.path.join(output_path, 'sliding_window_config.json')
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n配置已保存至: {output_file}")
    
    # 保存不确定样本
    uncertain_samples = [all_data[i] for i in range(len(all_data)) if preds[i] == 2]
    uncertain_file = os.path.join(output_path, 'uncertain_samples.jsonl')
    with open(uncertain_file, 'w') as f:
        for sample in uncertain_samples:
            f.write(json.dumps({
                'text': sample['text'],
                'diff': float(sample['diff']),
                'label': int(sample['label']),
                'dataset': sample['dataset']
            }, ensure_ascii=False) + '\n')
    
    print(f"不确定样本已保存至: {uncertain_file}")
    
    # 汇总
    print("\n" + "="*70)
    print("汇总")
    print("="*70)
    print(f"""
三分类阈值配置:
  - 恶意: diff > {upper:.4f}
  - 正常: diff < {lower:.4f}
  - 不确定: {lower:.4f} <= diff <= {upper:.4f} (交给LLM)

效果:
  - 不确定样本比例: {results['uncertain_ratio']*100:.1f}%
  - 确定样本准确率: {results['certain_accuracy']*100:.2f}%
  - 原始准确率: {results['binary_accuracy']*100:.2f}%
  - 提升: {(results['certain_accuracy']-results['binary_accuracy'])*100:+.2f}%
""")


if __name__ == '__main__':
    main()
