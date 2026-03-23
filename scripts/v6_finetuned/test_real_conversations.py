#!/usr/bin/env python3
"""
在真实日常对话数据集上测试V6模型
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
from transformers import AutoTokenizer
from datasets import load_dataset

sys.path.append(str(Path(__file__).parent))
from model import V6HarmfulDetector


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


def evaluate(preds, name, expected_label=0):
    """评估 - 日常对话应该都是正常的(label=0)"""
    labels = np.array([expected_label] * len(preds))
    
    accuracy = accuracy_score(labels, preds)
    
    if expected_label == 0:
        # 正常数据集，计算FPR
        fp = (preds == 1).sum()
        fpr = fp / len(preds)
        metric_name = "FPR"
        metric_value = fpr
    else:
        # 恶意数据集，计算FNR
        fn = (preds == 0).sum()
        fnr = fn / len(preds)
        metric_name = "FNR"
        metric_value = fnr
    
    return {
        'name': name,
        'samples': len(preds),
        'accuracy': accuracy,
        metric_name.lower(): metric_value,
        'harmful_count': int((preds == 1).sum()),
        'benign_count': int((preds == 0).sum())
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = '/home/vicuna/ludan/CSonEmbedding/models/v6_finetuned'
    output_path = '/home/vicuna/ludan/CSonEmbedding/results/v6_real_conversations'
    
    os.makedirs(output_path, exist_ok=True)
    
    print("="*70)
    print("V6模型 - 真实日常对话数据集测试")
    print("="*70)
    
    # 加载模型
    print("\n加载V6模型...")
    model, tokenizer = load_v6_model(model_path, device)
    
    results = {}
    false_positives = []  # 收集误报样本
    
    # ============ 1. DailyDialog ============
    print("\n[1] DailyDialog (日常对话)...")
    try:
        ds = load_dataset("li2017dailydialog/daily_dialog", split='test')
        texts = []
        for item in ds:
            if item.get('dialog'):
                # 取对话中的每句话
                for utterance in item['dialog'][:3]:  # 每个对话取前3句
                    if utterance and len(utterance) > 10:
                        texts.append(utterance)
                if len(texts) >= 1000:
                    break
        
        texts = texts[:1000]
        print(f"  样本数: {len(texts)}")
        
        preds, probs = predict_batch(model, tokenizer, texts, device)
        result = evaluate(preds, "DailyDialog")
        results['dailydialog'] = result
        
        print(f"  Accuracy: {result['accuracy']*100:.2f}%")
        print(f"  FPR: {result['fpr']*100:.2f}%")
        print(f"  误报数: {result['harmful_count']}")
        
        # 收集误报
        for i, (text, pred, prob) in enumerate(zip(texts, preds, probs)):
            if pred == 1:
                false_positives.append({
                    'text': text,
                    'prob': float(prob),
                    'dataset': 'DailyDialog'
                })
    except Exception as e:
        print(f"  失败: {e}")
    
    # ============ 2. LMSYS-Chat-1M ============
    print("\n[2] LMSYS-Chat-1M (真实LLM对话)...")
    try:
        ds = load_dataset("lmsys/lmsys-chat-1m", split='train', streaming=True)
        texts = []
        count = 0
        for item in ds:
            if item.get('conversation'):
                for turn in item['conversation']:
                    if turn.get('role') == 'user' and turn.get('content'):
                        content = turn['content']
                        if len(content) > 10 and len(content) < 1000:
                            texts.append(content)
                            count += 1
                            if count >= 1000:
                                break
            if count >= 1000:
                break
        
        print(f"  样本数: {len(texts)}")
        
        preds, probs = predict_batch(model, tokenizer, texts, device)
        result = evaluate(preds, "LMSYS-Chat-1M")
        results['lmsys_chat'] = result
        
        print(f"  Accuracy: {result['accuracy']*100:.2f}%")
        print(f"  FPR: {result['fpr']*100:.2f}%")
        print(f"  误报数: {result['harmful_count']}")
        
        for i, (text, pred, prob) in enumerate(zip(texts, preds, probs)):
            if pred == 1:
                false_positives.append({
                    'text': text[:500],  # 截断长文本
                    'prob': float(prob),
                    'dataset': 'LMSYS-Chat-1M'
                })
    except Exception as e:
        print(f"  失败: {e}")
    
    # ============ 3. Everyday Conversations ============
    print("\n[3] Everyday-Conversations (日常对话)...")
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
        result = evaluate(preds, "Everyday-Conversations")
        results['everyday_conv'] = result
        
        print(f"  Accuracy: {result['accuracy']*100:.2f}%")
        print(f"  FPR: {result['fpr']*100:.2f}%")
        print(f"  误报数: {result['harmful_count']}")
        
        for i, (text, pred, prob) in enumerate(zip(texts, preds, probs)):
            if pred == 1:
                false_positives.append({
                    'text': text,
                    'prob': float(prob),
                    'dataset': 'Everyday-Conversations'
                })
    except Exception as e:
        print(f"  失败: {e}")
    
    # ============ 4. Topical-Chat ============
    print("\n[4] Topical-Chat (话题对话)...")
    try:
        ds = load_dataset("Conversational-Reasoning/Topical-Chat", split='train')
        texts = []
        for item in ds:
            if item.get('message'):
                msg = item['message']
                if len(msg) > 10:
                    texts.append(msg)
                if len(texts) >= 1000:
                    break
        
        texts = texts[:1000]
        print(f"  样本数: {len(texts)}")
        
        preds, probs = predict_batch(model, tokenizer, texts, device)
        result = evaluate(preds, "Topical-Chat")
        results['topical_chat'] = result
        
        print(f"  Accuracy: {result['accuracy']*100:.2f}%")
        print(f"  FPR: {result['fpr']*100:.2f}%")
        print(f"  误报数: {result['harmful_count']}")
        
        for i, (text, pred, prob) in enumerate(zip(texts, preds, probs)):
            if pred == 1:
                false_positives.append({
                    'text': text,
                    'prob': float(prob),
                    'dataset': 'Topical-Chat'
                })
    except Exception as e:
        print(f"  失败: {e}")
    
    # ============ 5. ShareGPT ============
    print("\n[5] ShareGPT (真实用户对话)...")
    try:
        ds = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", split='train')
        texts = []
        for item in ds:
            if item.get('conversations'):
                for conv in item['conversations']:
                    if conv.get('from') == 'human' and conv.get('value'):
                        content = conv['value']
                        if len(content) > 10 and len(content) < 1000:
                            texts.append(content)
                if len(texts) >= 1000:
                    break
        
        texts = texts[:1000]
        print(f"  样本数: {len(texts)}")
        
        preds, probs = predict_batch(model, tokenizer, texts, device)
        result = evaluate(preds, "ShareGPT")
        results['sharegpt'] = result
        
        print(f"  Accuracy: {result['accuracy']*100:.2f}%")
        print(f"  FPR: {result['fpr']*100:.2f}%")
        print(f"  误报数: {result['harmful_count']}")
        
        for i, (text, pred, prob) in enumerate(zip(texts, preds, probs)):
            if pred == 1:
                false_positives.append({
                    'text': text[:500],
                    'prob': float(prob),
                    'dataset': 'ShareGPT'
                })
    except Exception as e:
        print(f"  失败: {e}")
    
    # ============ 6. WildChat (仅英文) ============
    print("\n[6] WildChat-English (野外真实对话-仅英文)...")
    try:
        ds = load_dataset("allenai/WildChat-1M", split='train', streaming=True)
        texts = []
        count = 0
        for item in ds:
            # 只取英文对话
            if item.get('language') and item['language'] != 'English':
                continue
            if item.get('conversation'):
                for turn in item['conversation']:
                    if turn.get('role') == 'user' and turn.get('content'):
                        content = turn['content']
                        # 简单过滤非英文
                        if len(content) > 10 and len(content) < 1000:
                            # 检查是否主要是ASCII字符（英文）
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
        result = evaluate(preds, "WildChat-English")
        results['wildchat'] = result
        
        print(f"  Accuracy: {result['accuracy']*100:.2f}%")
        print(f"  FPR: {result['fpr']*100:.2f}%")
        print(f"  误报数: {result['harmful_count']}")
        
        for i, (text, pred, prob) in enumerate(zip(texts, preds, probs)):
            if pred == 1:
                false_positives.append({
                    'text': text[:500],
                    'prob': float(prob),
                    'dataset': 'WildChat-English'
                })
    except Exception as e:
        print(f"  失败: {e}")
    
    # ============ 汇总 ============
    print("\n" + "="*70)
    print("测试结果汇总")
    print("="*70)
    
    print(f"\n{'数据集':<25} {'样本数':<10} {'Accuracy':<12} {'FPR':<10} {'误报数':<10}")
    print("-"*70)
    
    total_samples = 0
    total_fp = 0
    
    for name, result in results.items():
        print(f"{result['name']:<25} {result['samples']:<10} {result['accuracy']*100:<12.2f}% {result['fpr']*100:<10.2f}% {result['harmful_count']:<10}")
        total_samples += result['samples']
        total_fp += result['harmful_count']
    
    print("-"*70)
    print(f"{'总计':<25} {total_samples:<10} {(1-total_fp/total_samples)*100:<12.2f}% {total_fp/total_samples*100:<10.2f}% {total_fp:<10}")
    
    # 保存结果
    output_file = os.path.join(output_path, 'real_conversations_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存至: {output_file}")
    
    # 保存误报样本
    fp_file = os.path.join(output_path, 'false_positives.jsonl')
    with open(fp_file, 'w') as f:
        for fp in false_positives:
            f.write(json.dumps(fp, ensure_ascii=False) + '\n')
    
    print(f"误报样本已保存至: {fp_file} ({len(false_positives)} 样本)")
    
    # 显示一些误报样本
    if false_positives:
        print("\n" + "="*70)
        print("部分误报样本示例")
        print("="*70)
        for fp in false_positives[:10]:
            print(f"\n[{fp['dataset']}] prob={fp['prob']:.4f}")
            print(f"  {fp['text'][:200]}...")


if __name__ == '__main__':
    main()
