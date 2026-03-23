#!/usr/bin/env python3
"""
V7分类器封装模块
提供有害内容检测的前置分类功能
"""

import os
import sys
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append('/home/vicuna/ludan/CSonEmbedding/scripts/v7_classifier')
from model import V6HarmfulDetector


class V7Classifier:
    """
    V7有害内容分类器
    
    根据概率阈值将输入分为三类:
    - benign: prob < lower_threshold
    - harmful: prob > upper_threshold  
    - uncertain: lower_threshold <= prob <= upper_threshold
    """
    
    def __init__(
        self,
        model_path: str = None,
        device: str = None,
        lower_threshold: float = 0.45,
        upper_threshold: float = 0.60
    ):
        if model_path is None:
            model_path = '/home/vicuna/ludan/CSonEmbedding/models/v7_classifier'
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        self.model_path = model_path
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold
        
        # 加载配置
        config_path = os.path.join(model_path, 'config.json')
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # 加载灰色地带配置（如果存在）
        gray_config_path = os.path.join(model_path, 'gray_zone_config.json')
        if os.path.exists(gray_config_path):
            with open(gray_config_path, 'r') as f:
                gray_config = json.load(f)
                self.lower_threshold = gray_config.get('lower_threshold', lower_threshold)
                self.upper_threshold = gray_config.get('upper_threshold', upper_threshold)
        
        # 加载模型
        self.model = V6HarmfulDetector(
            model_name=self.config['model_name'],
            projection_dim=self.config['projection_dim']
        )
        
        weights_path = os.path.join(model_path, 'best_model.pt')
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
        
        print(f"V7分类器已加载")
        print(f"  设备: {device}")
        print(f"  阈值: benign < {self.lower_threshold} < uncertain < {self.upper_threshold} < harmful")
    
    def predict(self, text: str) -> tuple:
        """
        预测单条文本
        
        Args:
            text: 输入文本
            
        Returns:
            (label, prob): label为'benign'/'harmful'/'uncertain', prob为有害概率
        """
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            prob = self.model.predict(inputs['input_ids'], inputs['attention_mask'])
            prob = prob.item()
        
        if prob < self.lower_threshold:
            label = 'benign'
        elif prob > self.upper_threshold:
            label = 'harmful'
        else:
            label = 'uncertain'
        
        return label, prob
    
    def predict_batch(self, texts: list, batch_size: int = 32) -> list:
        """
        批量预测
        
        Args:
            texts: 文本列表
            batch_size: 批大小
            
        Returns:
            [(label, prob), ...]: 预测结果列表
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            inputs = self.tokenizer(
                batch_texts,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                probs = self.model.predict(inputs['input_ids'], inputs['attention_mask'])
                probs = probs.cpu().numpy()
            
            for prob in probs:
                if prob < self.lower_threshold:
                    label = 'benign'
                elif prob > self.upper_threshold:
                    label = 'harmful'
                else:
                    label = 'uncertain'
                results.append((label, float(prob)))
        
        return results
    
    def get_thresholds(self) -> dict:
        """获取当前阈值配置"""
        return {
            'lower_threshold': self.lower_threshold,
            'upper_threshold': self.upper_threshold
        }
    
    def set_thresholds(self, lower: float = None, upper: float = None):
        """设置阈值"""
        if lower is not None:
            self.lower_threshold = lower
        if upper is not None:
            self.upper_threshold = upper


if __name__ == '__main__':
    print("="*60)
    print("V7分类器测试")
    print("="*60)
    
    classifier = V7Classifier()
    
    test_texts = [
        "How do I make a chocolate cake?",
        "What's the weather like today?",
        "How to hack into someone's computer?",
        "Write a poem about nature",
        "How to make a bomb?",
        "Tell me about machine learning",
    ]
    
    print("\n测试结果:")
    print("-"*60)
    for text in test_texts:
        label, prob = classifier.predict(text)
        print(f"[{label:10s}] prob={prob:.4f} | {text[:50]}")
