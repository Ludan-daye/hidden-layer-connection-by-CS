#!/usr/bin/env python3
"""
V7前置分类器数据集准备

目标：训练一个高准确率的前置分类器
- 确定判定的样本准确率要高
- 允许较大的灰色地带（不确定样本交给LLM）

数据集组成：
1. 恶意样本：AdvBench, HarmBench, JailbreakHub等
2. 正常样本：Alpaca, Dolly + DailyDialog, WildChat等真实对话
3. 灰色样本：Gray-Benign, Gray-Harmful
"""

import os
import json
import random
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

def load_harmful_data(base_path, max_samples=2000):
    """加载恶意数据"""
    texts = []
    
    # 1. HarmBench
    harmbench_path = os.path.join(base_path, 'harmbench')
    if os.path.exists(harmbench_path):
        for file in os.listdir(harmbench_path):
            if file.endswith('.csv'):
                try:
                    df = pd.read_csv(os.path.join(harmbench_path, file))
                    for col in ['Behavior', 'goal', 'prompt']:
                        if col in df.columns:
                            texts.extend(df[col].dropna().tolist())
                            break
                except:
                    pass
        print(f"  HarmBench: {len(texts)} 样本")
    
    # 2. JailbreakHub
    try:
        ds = load_dataset("rubend18/ChatGPT-Jailbreak-Prompts", split='train')
        jailbreak_texts = [x['text'] for x in ds if x.get('text')]
        texts.extend(jailbreak_texts)
        print(f"  JailbreakHub: {len(jailbreak_texts)} 样本")
    except Exception as e:
        print(f"  JailbreakHub失败: {e}")
    
    # 3. JailbreakBench
    jbb_path = os.path.join(base_path, 'jailbreakbench')
    if os.path.exists(jbb_path):
        jbb_count = 0
        for file in os.listdir(jbb_path):
            if file.endswith('.csv'):
                try:
                    df = pd.read_csv(os.path.join(jbb_path, file))
                    for col in ['prompt', 'goal', 'jailbreak']:
                        if col in df.columns:
                            texts.extend(df[col].dropna().tolist())
                            jbb_count += len(df[col].dropna())
                            break
                except:
                    pass
        print(f"  JailbreakBench: {jbb_count} 样本")
    
    # 4. BeaverTails
    beavertails_path = os.path.join(base_path, 'beavertails')
    if os.path.exists(beavertails_path):
        bt_count = 0
        for file in os.listdir(beavertails_path):
            if file.endswith('.csv'):
                try:
                    df = pd.read_csv(os.path.join(beavertails_path, file))
                    if 'prompt' in df.columns:
                        texts.extend(df['prompt'].dropna().tolist())
                        bt_count += len(df['prompt'].dropna())
                except:
                    pass
        print(f"  BeaverTails: {bt_count} 样本")
    
    # 去重和过滤
    texts = list(set([t for t in texts if isinstance(t, str) and 10 < len(t) < 2000]))
    random.shuffle(texts)
    
    return texts[:max_samples]


def load_benign_data(base_path, max_samples=4000):
    """加载正常数据 - 包含指令数据和真实日常对话"""
    texts = []
    
    # ========== 指令数据 ==========
    # 1. Alpaca
    try:
        ds = load_dataset("tatsu-lab/alpaca", split='train')
        for item in ds:
            if item.get('instruction'):
                text = item['instruction']
                if item.get('input'):
                    text += " " + item['input']
                texts.append(text)
                if len(texts) >= 1000:
                    break
        print(f"  Alpaca: 1000 样本")
    except Exception as e:
        print(f"  Alpaca失败: {e}")
    
    # 2. Dolly
    try:
        ds = load_dataset("databricks/databricks-dolly-15k", split='train')
        dolly_count = 0
        for item in ds:
            if item.get('instruction'):
                text = item['instruction']
                if item.get('context'):
                    text += " " + item['context']
                texts.append(text)
                dolly_count += 1
                if dolly_count >= 1000:
                    break
        print(f"  Dolly: {dolly_count} 样本")
    except Exception as e:
        print(f"  Dolly失败: {e}")
    
    # ========== 真实日常对话 ==========
    # 3. DailyDialog
    try:
        ds = load_dataset("li2017dailydialog/daily_dialog", split='train')
        daily_count = 0
        for item in ds:
            if item.get('dialog'):
                for utterance in item['dialog']:
                    if utterance and len(utterance) > 10:
                        texts.append(utterance)
                        daily_count += 1
                        if daily_count >= 1000:
                            break
            if daily_count >= 1000:
                break
        print(f"  DailyDialog: {daily_count} 样本")
    except Exception as e:
        print(f"  DailyDialog失败: {e}")
    
    # 4. Everyday-Conversations
    try:
        ds = load_dataset("HuggingFaceTB/everyday-conversations-llama3.1-2k", split='train_sft')
        everyday_count = 0
        for item in ds:
            if item.get('messages'):
                for msg in item['messages']:
                    if msg.get('role') == 'user' and msg.get('content'):
                        content = msg['content']
                        if len(content) > 10:
                            texts.append(content)
                            everyday_count += 1
                            if everyday_count >= 500:
                                break
            if everyday_count >= 500:
                break
        print(f"  Everyday-Conversations: {everyday_count} 样本")
    except Exception as e:
        print(f"  Everyday-Conversations失败: {e}")
    
    # 5. WildChat (仅英文)
    try:
        ds = load_dataset("allenai/WildChat-1M", split='train', streaming=True)
        wildchat_count = 0
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
                                wildchat_count += 1
                                if wildchat_count >= 500:
                                    break
            if wildchat_count >= 500:
                break
        print(f"  WildChat-English: {wildchat_count} 样本")
    except Exception as e:
        print(f"  WildChat失败: {e}")
    
    # 去重和过滤
    texts = list(set([t for t in texts if isinstance(t, str) and 10 < len(t) < 2000]))
    random.shuffle(texts)
    
    return texts[:max_samples]


def load_gray_data(base_path):
    """加载灰色样本"""
    gray_benign = []
    gray_harmful = []
    
    # Gray-Benign
    gray_benign_path = os.path.join(base_path, 'v6_training', 'gray_benign.jsonl')
    if os.path.exists(gray_benign_path):
        with open(gray_benign_path, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if item.get('text'):
                        gray_benign.append(item['text'])
                except:
                    pass
        print(f"  Gray-Benign: {len(gray_benign)} 样本")
    
    # Gray-Harmful
    gray_harmful_path = os.path.join(base_path, 'v6_training', 'gray_harmful.jsonl')
    if os.path.exists(gray_harmful_path):
        with open(gray_harmful_path, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if item.get('text'):
                        gray_harmful.append(item['text'])
                except:
                    pass
        print(f"  Gray-Harmful: {len(gray_harmful)} 样本")
    
    return gray_benign, gray_harmful


def main():
    base_path = '/home/vicuna/ludan/CSonEmbedding/datasets'
    output_path = '/home/vicuna/ludan/CSonEmbedding/datasets/v7_training'
    
    os.makedirs(output_path, exist_ok=True)
    
    print("="*70)
    print("V7前置分类器数据集准备")
    print("="*70)
    
    # 加载数据
    print("\n[1/4] 加载恶意数据...")
    harmful_texts = load_harmful_data(base_path, max_samples=2000)
    print(f"  总计: {len(harmful_texts)} 恶意样本")
    
    print("\n[2/4] 加载正常数据...")
    benign_texts = load_benign_data(base_path, max_samples=4000)
    print(f"  总计: {len(benign_texts)} 正常样本")
    
    print("\n[3/4] 加载灰色样本...")
    gray_benign, gray_harmful = load_gray_data(base_path)
    
    # 构建数据集
    print("\n[4/4] 构建数据集...")
    
    # 标签: 0=benign, 1=harmful, 2=gray_benign, 3=gray_harmful
    dataset = []
    
    for text in harmful_texts:
        dataset.append({'text': text, 'label': 1, 'category': 'harmful'})
    
    for text in benign_texts:
        dataset.append({'text': text, 'label': 0, 'category': 'benign'})
    
    for text in gray_benign:
        dataset.append({'text': text, 'label': 2, 'category': 'gray_benign'})
    
    for text in gray_harmful:
        dataset.append({'text': text, 'label': 3, 'category': 'gray_harmful'})
    
    random.shuffle(dataset)
    
    # 统计
    print(f"\n数据集统计:")
    print(f"  恶意样本: {len(harmful_texts)}")
    print(f"  正常样本: {len(benign_texts)}")
    print(f"  灰色正常: {len(gray_benign)}")
    print(f"  灰色恶意: {len(gray_harmful)}")
    print(f"  总计: {len(dataset)}")
    
    # 划分训练集和验证集
    val_ratio = 0.1
    val_size = int(len(dataset) * val_ratio)
    
    val_data = dataset[:val_size]
    train_data = dataset[val_size:]
    
    print(f"\n划分:")
    print(f"  训练集: {len(train_data)}")
    print(f"  验证集: {len(val_data)}")
    
    # 保存
    train_file = os.path.join(output_path, 'train.jsonl')
    with open(train_file, 'w') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    val_file = os.path.join(output_path, 'val.jsonl')
    with open(val_file, 'w') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 保存配置
    config = {
        'harmful_count': len(harmful_texts),
        'benign_count': len(benign_texts),
        'gray_benign_count': len(gray_benign),
        'gray_harmful_count': len(gray_harmful),
        'total': len(dataset),
        'train_size': len(train_data),
        'val_size': len(val_data),
        'sources': {
            'harmful': ['HarmBench', 'JailbreakHub', 'JailbreakBench', 'BeaverTails'],
            'benign': ['Alpaca', 'Dolly', 'DailyDialog', 'Everyday-Conversations', 'WildChat-English'],
            'gray': ['Gray-Benign', 'Gray-Harmful']
        }
    }
    
    config_file = os.path.join(output_path, 'config.json')
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"\n数据集已保存至: {output_path}")
    print(f"  训练集: {train_file}")
    print(f"  验证集: {val_file}")
    print(f"  配置: {config_file}")


if __name__ == '__main__':
    main()
