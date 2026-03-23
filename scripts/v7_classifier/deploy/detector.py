#!/usr/bin/env python3
"""
完整有害内容检测系统
V7分类器 + Llama-2-7B判断
"""

import time
from typing import Optional

from v7_classifier import V7Classifier
from llm_judge import LLMJudge


class HarmfulContentDetector:
    """
    完整有害内容检测系统
    
    流程:
    1. V7分类器预测概率
    2. 根据阈值判断:
       - prob < lower: 直接返回benign
       - prob > upper: 直接返回harmful
       - otherwise: 交给LLM判断
    """
    
    def __init__(
        self,
        v7_model_path: str = None,
        llm_model_path: str = None,
        device: str = None,
        lower_threshold: float = 0.45,
        upper_threshold: float = 0.60,
        lazy_load_llm: bool = True
    ):
        """
        初始化检测系统
        
        Args:
            v7_model_path: V7模型路径
            llm_model_path: Llama-2-7B模型路径
            device: 设备
            lower_threshold: 下阈值
            upper_threshold: 上阈值
            lazy_load_llm: 是否延迟加载LLM（首次需要时加载）
        """
        self.llm_model_path = llm_model_path
        self.device = device
        self.lazy_load_llm = lazy_load_llm
        
        # 加载V7分类器
        print("="*60)
        print("初始化有害内容检测系统")
        print("="*60)
        
        self.v7_classifier = V7Classifier(
            model_path=v7_model_path,
            device=device,
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold
        )
        
        # LLM判断模块
        self.llm_judge = None
        if not lazy_load_llm:
            self._load_llm()
        else:
            print("LLM判断模块: 延迟加载（首次需要时加载）")
        
        # 统计信息
        self.stats = {
            'total': 0,
            'v7_benign': 0,
            'v7_harmful': 0,
            'llm_benign': 0,
            'llm_harmful': 0
        }
        
        print("="*60)
        print("检测系统初始化完成")
        print("="*60)
    
    def _load_llm(self):
        """加载LLM模块"""
        if self.llm_judge is None:
            print("\n加载LLM判断模块...")
            self.llm_judge = LLMJudge(
                model_path=self.llm_model_path,
                device=self.device
            )
    
    def detect(self, text: str, verbose: bool = False) -> dict:
        """
        检测输入是否有害
        
        Args:
            text: 输入文本
            verbose: 是否打印详细信息
            
        Returns:
            {
                'label': 'benign' or 'harmful',
                'source': 'v7_classifier' or 'llm_judge',
                'v7_prob': float,
                'v7_label': str,
                'llm_response': str or None,
                'time_ms': float
            }
        """
        start_time = time.time()
        
        # V7分类器预测
        v7_label, v7_prob = self.v7_classifier.predict(text)
        
        result = {
            'label': None,
            'source': None,
            'v7_prob': v7_prob,
            'v7_label': v7_label,
            'llm_response': None,
            'time_ms': None
        }
        
        self.stats['total'] += 1
        
        if v7_label == 'benign':
            result['label'] = 'benign'
            result['source'] = 'v7_classifier'
            self.stats['v7_benign'] += 1
            
        elif v7_label == 'harmful':
            result['label'] = 'harmful'
            result['source'] = 'v7_classifier'
            self.stats['v7_harmful'] += 1
            
        else:  # uncertain
            # 加载LLM（如果需要）
            self._load_llm()
            
            # LLM判断
            llm_label, llm_response = self.llm_judge.judge(text)
            result['label'] = llm_label
            result['source'] = 'llm_judge'
            result['llm_response'] = llm_response
            
            if llm_label == 'benign':
                self.stats['llm_benign'] += 1
            else:
                self.stats['llm_harmful'] += 1
        
        result['time_ms'] = (time.time() - start_time) * 1000
        
        if verbose:
            self._print_result(text, result)
        
        return result
    
    def _print_result(self, text: str, result: dict):
        """打印检测结果"""
        print(f"\n{'='*60}")
        print(f"输入: {text[:80]}{'...' if len(text) > 80 else ''}")
        print(f"-"*60)
        print(f"V7概率: {result['v7_prob']:.4f} ({result['v7_label']})")
        
        if result['source'] == 'llm_judge':
            print(f"LLM判断: {result['llm_response']}")
        
        print(f"-"*60)
        label_color = '\033[92m' if result['label'] == 'benign' else '\033[91m'
        print(f"最终结果: {label_color}{result['label']}\033[0m (by {result['source']})")
        print(f"耗时: {result['time_ms']:.1f}ms")
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        stats = self.stats.copy()
        if stats['total'] > 0:
            stats['v7_ratio'] = (stats['v7_benign'] + stats['v7_harmful']) / stats['total']
            stats['llm_ratio'] = (stats['llm_benign'] + stats['llm_harmful']) / stats['total']
        else:
            stats['v7_ratio'] = 0
            stats['llm_ratio'] = 0
        return stats
    
    def print_stats(self):
        """打印统计信息"""
        stats = self.get_stats()
        print(f"\n{'='*60}")
        print("检测统计")
        print(f"{'='*60}")
        print(f"总检测数: {stats['total']}")
        print(f"V7直接判断: {stats['v7_benign'] + stats['v7_harmful']} ({stats['v7_ratio']*100:.1f}%)")
        print(f"  - benign: {stats['v7_benign']}")
        print(f"  - harmful: {stats['v7_harmful']}")
        print(f"LLM判断: {stats['llm_benign'] + stats['llm_harmful']} ({stats['llm_ratio']*100:.1f}%)")
        print(f"  - benign: {stats['llm_benign']}")
        print(f"  - harmful: {stats['llm_harmful']}")
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total': 0,
            'v7_benign': 0,
            'v7_harmful': 0,
            'llm_benign': 0,
            'llm_harmful': 0
        }


if __name__ == '__main__':
    print("完整检测系统测试")
    
    detector = HarmfulContentDetector(lazy_load_llm=True)
    
    test_texts = [
        "How do I make a chocolate cake?",
        "What's the weather like today?",
        "How to hack into someone's computer?",
        "Write a poem about nature",
        "How to make a bomb?",
        "Tell me about machine learning",
        "What are some good programming practices?",
    ]
    
    print("\n开始测试...")
    for text in test_texts:
        result = detector.detect(text, verbose=True)
    
    detector.print_stats()
