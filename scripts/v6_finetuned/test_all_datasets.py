#!/usr/bin/env python3
"""
在所有数据集上测试V6 + CS压缩 + 多质心检测
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


def predict_multicentroid(embeddings, centroids, benign_centroid, threshold=0.0):
    harmful_sims = embeddings @ centroids.T
    max_harmful_sim = np.max(harmful_sims, axis=1)
    benign_sim = embeddings @ benign_centroid
    diff = max_harmful_sim - benign_sim
    preds = (diff > threshold).astype(int)
    return preds, max_harmful_sim, benign_sim, diff


def evaluate(preds, labels, name):
    labels = np.array(labels)
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
        'samples': len(labels),
        'accuracy': float(accuracy),
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'fpr': float(fpr),
        'fnr': float(fnr),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
    }


def test_dataset(name, texts, labels, v6_model, tokenizer, cs_model, centroids, benign_centroid, device):
    """测试单个数据集"""
    if len(texts) == 0:
        return None
    
    print(f"\n测试 {name} ({len(texts)} 样本)...")
    
    embeddings = extract_embeddings(v6_model, tokenizer, texts, device)
    with torch.no_grad():
        compressed = cs_model(torch.tensor(embeddings, dtype=torch.float32).to(device)).cpu().numpy()
    
    preds, _, _, _ = predict_multicentroid(compressed, centroids, benign_centroid, threshold=0.0)
    result = evaluate(preds, labels, name)
    
    print(f"  Accuracy: {result['accuracy']:.4f}, F1: {result['f1']:.4f}, "
          f"Precision: {result['precision']:.4f}, Recall: {result['recall']:.4f}, FPR: {result['fpr']:.4f}")
    
    return result


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = '/home/vicuna/ludan/CSonEmbedding/models/v6_finetuned'
    cs_path = '/home/vicuna/ludan/CSonEmbedding/models/v6_cs_multicentroid'
    data_path = '/home/vicuna/ludan/CSonEmbedding/datasets'
    output_path = '/home/vicuna/ludan/CSonEmbedding/results/v6_cs_multicentroid'
    
    os.makedirs(output_path, exist_ok=True)
    
    print("="*70)
    print("V6 + CS压缩 + 多质心 - 全数据集测试")
    print("="*70)
    
    # 加载模型
    print("\n加载模型...")
    v6_model, tokenizer = load_v6_model(model_path, device)
    
    # 加载CS压缩模型 (32D + 3质心)
    config = "32d_3c"
    saved = np.load(os.path.join(cs_path, f'cs_multicentroid_{config}.npz'))
    
    cs_model = LearnedCSProjection(768, 32).to(device)
    cs_model.proj.weight.data = torch.tensor(saved['cs_weights'], dtype=torch.float32).to(device)
    cs_model.eval()
    
    centroids = saved['centroids']
    benign_centroid = saved['benign_centroid']
    
    results = {'config': config, 'datasets': {}}
    
    # ============ 恶意数据集 ============
    print("\n" + "="*50)
    print("恶意数据集测试 (应检测为harmful)")
    print("="*50)
    
    # 1. AdvBench
    try:
        ds = load_dataset("walledai/AdvBench", split='train')
        texts = [x['prompt'] for x in ds if x.get('prompt')][:500]
        labels = [1] * len(texts)
        result = test_dataset("AdvBench", texts, labels, v6_model, tokenizer, cs_model, centroids, benign_centroid, device)
        if result:
            results['datasets']['advbench'] = result
    except Exception as e:
        print(f"AdvBench失败: {e}")
    
    # 2. HarmBench
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
            labels = [1] * len(texts)
            result = test_dataset("HarmBench", texts[:500], labels[:500], v6_model, tokenizer, cs_model, centroids, benign_centroid, device)
            if result:
                results['datasets']['harmbench'] = result
    
    # 3. JailbreakHub
    try:
        ds = load_dataset("rubend18/ChatGPT-Jailbreak-Prompts", split='train')
        texts = [x['text'] for x in ds if x.get('text')][:500]
        labels = [1] * len(texts)
        result = test_dataset("JailbreakHub", texts, labels, v6_model, tokenizer, cs_model, centroids, benign_centroid, device)
        if result:
            results['datasets']['jailbreakhub'] = result
    except Exception as e:
        print(f"JailbreakHub失败: {e}")
    
    # 4. JailbreakBench
    jbb_path = os.path.join(data_path, 'jailbreakbench')
    if os.path.exists(jbb_path):
        texts = []
        for file in os.listdir(jbb_path):
            if file.endswith('.csv'):
                try:
                    df = pd.read_csv(os.path.join(jbb_path, file))
                    for col in ['prompt', 'goal', 'jailbreak']:
                        if col in df.columns:
                            texts.extend(df[col].dropna().tolist())
                            break
                except:
                    pass
        if texts:
            labels = [1] * len(texts)
            result = test_dataset("JailbreakBench", texts[:500], labels[:500], v6_model, tokenizer, cs_model, centroids, benign_centroid, device)
            if result:
                results['datasets']['jailbreakbench'] = result
    
    # 5. BeaverTails
    beavertails_path = os.path.join(data_path, 'beavertails')
    if os.path.exists(beavertails_path):
        texts = []
        for file in os.listdir(beavertails_path):
            if file.endswith('.csv'):
                try:
                    df = pd.read_csv(os.path.join(beavertails_path, file))
                    if 'prompt' in df.columns:
                        texts.extend(df['prompt'].dropna().tolist())
                except:
                    pass
        if texts:
            labels = [1] * len(texts)
            result = test_dataset("BeaverTails", texts[:500], labels[:500], v6_model, tokenizer, cs_model, centroids, benign_centroid, device)
            if result:
                results['datasets']['beavertails'] = result
    
    # ============ 正常数据集 ============
    print("\n" + "="*50)
    print("正常数据集测试 (应检测为benign)")
    print("="*50)
    
    # 6. Alpaca
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
        result = test_dataset("Alpaca", texts, labels, v6_model, tokenizer, cs_model, centroids, benign_centroid, device)
        if result:
            results['datasets']['alpaca'] = result
    except Exception as e:
        print(f"Alpaca失败: {e}")
    
    # 7. Dolly
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
        labels = [0] * len(texts)
        result = test_dataset("Dolly", texts, labels, v6_model, tokenizer, cs_model, centroids, benign_centroid, device)
        if result:
            results['datasets']['dolly'] = result
    except Exception as e:
        print(f"Dolly失败: {e}")
    
    # 8. OpenAssistant
    try:
        ds = load_dataset("OpenAssistant/oasst1", split='train')
        texts = [x['text'] for x in ds if x.get('text') and x.get('role') == 'prompter'][:500]
        labels = [0] * len(texts)
        result = test_dataset("OpenAssistant", texts, labels, v6_model, tokenizer, cs_model, centroids, benign_centroid, device)
        if result:
            results['datasets']['openassistant'] = result
    except Exception as e:
        print(f"OpenAssistant失败: {e}")
    
    # 9. OR-Bench (过度拒绝测试)
    or_path = os.path.join(data_path, 'or_bench')
    if os.path.exists(or_path):
        texts = []
        for file in os.listdir(or_path):
            if file.endswith('.jsonl'):
                with open(os.path.join(or_path, file), 'r') as f:
                    for line in f:
                        try:
                            item = json.loads(line)
                            if item.get('prompt'):
                                texts.append(item['prompt'])
                        except:
                            pass
        if texts:
            labels = [0] * len(texts)
            result = test_dataset("OR-Bench", texts[:500], labels[:500], v6_model, tokenizer, cs_model, centroids, benign_centroid, device)
            if result:
                results['datasets']['or_bench'] = result
    
    # ============ 混合数据集 ============
    print("\n" + "="*50)
    print("混合数据集测试")
    print("="*50)
    
    # 10. ToxicChat
    try:
        ds = load_dataset("lmsys/toxic-chat", "toxicchat0124", split='test')
        texts = [x['user_input'] for x in ds if x.get('user_input')][:1000]
        labels = [1 if x.get('toxicity', 0) == 1 else 0 for x in ds if x.get('user_input')][:1000]
        result = test_dataset("ToxicChat", texts, labels, v6_model, tokenizer, cs_model, centroids, benign_centroid, device)
        if result:
            results['datasets']['toxicchat'] = result
    except Exception as e:
        print(f"ToxicChat失败: {e}")
    
    # 11. XSTest
    try:
        ds = load_dataset("walledai/XSTest", split='test')
        safe_texts = [x['prompt'] for x in ds if x.get('prompt') and x.get('label') == 'safe'][:250]
        unsafe_texts = [x['prompt'] for x in ds if x.get('prompt') and x.get('label') != 'safe'][:250]
        
        if safe_texts:
            result = test_dataset("XSTest-Safe", safe_texts, [0]*len(safe_texts), v6_model, tokenizer, cs_model, centroids, benign_centroid, device)
            if result:
                results['datasets']['xstest_safe'] = result
        
        if unsafe_texts:
            result = test_dataset("XSTest-Unsafe", unsafe_texts, [1]*len(unsafe_texts), v6_model, tokenizer, cs_model, centroids, benign_centroid, device)
            if result:
                results['datasets']['xstest_unsafe'] = result
    except Exception as e:
        print(f"XSTest失败: {e}")
    
    # ============ 灰色样本 ============
    print("\n" + "="*50)
    print("灰色样本测试")
    print("="*50)
    
    # 12. Gray-Benign
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
            labels = [0] * len(texts)
            result = test_dataset("Gray-Benign", texts, labels, v6_model, tokenizer, cs_model, centroids, benign_centroid, device)
            if result:
                results['datasets']['gray_benign'] = result
    
    # 13. Gray-Harmful
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
            labels = [1] * len(texts)
            result = test_dataset("Gray-Harmful", texts, labels, v6_model, tokenizer, cs_model, centroids, benign_centroid, device)
            if result:
                results['datasets']['gray_harmful'] = result
    
    # ============ 汇总 ============
    print("\n" + "="*70)
    print("测试结果汇总")
    print("="*70)
    
    print(f"\n{'数据集':<20} {'样本数':<8} {'Accuracy':<10} {'F1':<10} {'Precision':<10} {'Recall':<10} {'FPR':<8}")
    print("-"*86)
    
    for name, result in results['datasets'].items():
        print(f"{result['name']:<20} {result['samples']:<8} {result['accuracy']:<10.4f} {result['f1']:<10.4f} {result['precision']:<10.4f} {result['recall']:<10.4f} {result['fpr']:<8.4f}")
    
    # 保存结果
    output_file = os.path.join(output_path, 'all_datasets_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存至: {output_file}")
    
    # 生成README
    readme = generate_readme(results)
    readme_file = os.path.join(output_path, 'README.md')
    with open(readme_file, 'w') as f:
        f.write(readme)
    print(f"README已保存至: {readme_file}")


def generate_readme(results):
    content = """# V6 + CS压缩 + 多质心检测 测试结果

## 配置

- **压缩维度**: 32D (768D -> 32D, 24x压缩)
- **质心数量**: 3个恶意质心 + 1个正常质心
- **基础模型**: V6 Fine-tuned BGE-base-en-v1.5

## 测试结果

| 数据集 | 样本数 | Accuracy | F1 | Precision | Recall | FPR |
|--------|--------|----------|-----|-----------|--------|-----|
"""
    
    for name, result in results['datasets'].items():
        content += f"| {result['name']} | {result['samples']} | {result['accuracy']:.4f} | {result['f1']:.4f} | {result['precision']:.4f} | {result['recall']:.4f} | {result['fpr']:.4f} |\n"
    
    content += """
## 数据集分类

### 恶意数据集 (应检测为harmful)
- AdvBench, HarmBench, JailbreakHub, JailbreakBench, BeaverTails

### 正常数据集 (应检测为benign)
- Alpaca, Dolly, OpenAssistant, OR-Bench

### 混合数据集
- ToxicChat, XSTest-Safe, XSTest-Unsafe

### 灰色样本
- Gray-Benign (应判为正常), Gray-Harmful (应判为恶意)

## 关键指标

- **Recall**: 恶意样本被正确检测的比例 (越高越好)
- **FPR**: 正常样本被误判为恶意的比例 (越低越好)
- **F1**: 综合精确率和召回率的调和平均
"""
    
    return content


if __name__ == '__main__':
    main()
