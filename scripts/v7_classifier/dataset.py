#!/usr/bin/env python3
"""
V7数据集处理
"""

import os
import json
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import numpy as np


class V7Dataset(Dataset):
    """V7数据集"""
    
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        label = item['label']  # 0=benign, 1=harmful, 2=gray_benign, 3=gray_harmful
        
        # 二分类标签: benign/gray_benign -> 0, harmful/gray_harmful -> 1
        binary_label = 0 if label in [0, 2] else 1
        
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long),
            'binary_label': torch.tensor(binary_label, dtype=torch.long)
        }


def load_dataset_from_file(file_path):
    """从jsonl文件加载数据"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
                data.append(item)
            except:
                pass
    return data


def prepare_dataset(data_path, tokenizer, max_length=512):
    """准备训练和验证数据集"""
    train_file = os.path.join(data_path, 'train.jsonl')
    val_file = os.path.join(data_path, 'val.jsonl')
    
    train_data = load_dataset_from_file(train_file)
    val_data = load_dataset_from_file(val_file)
    
    train_dataset = V7Dataset(train_data, tokenizer, max_length)
    val_dataset = V7Dataset(val_data, tokenizer, max_length)
    
    return train_dataset, val_dataset


def get_balanced_sampler(dataset):
    """获取平衡采样器"""
    labels = [item['label'] for item in dataset.data]
    
    # 统计每个类别的数量
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    # 计算权重
    weights = []
    for label in labels:
        weights.append(1.0 / label_counts[label])
    
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )
    
    return sampler
