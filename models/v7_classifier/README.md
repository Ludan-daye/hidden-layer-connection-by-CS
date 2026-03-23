# V7 前置分类器

## 概述

V7是一个高准确率的前置分类器，用于检测恶意/有害的用户输入。相比V6模型，V7通过加入真实日常对话数据进行训练，显著降低了在真实场景下的误报率。

## 设计目标

1. **高准确率**：确定判定的样本准确率要高
2. **允许灰色地带**：不确定的样本交给LLM进一步判断
3. **低误报率**：降低正常对话被误判为恶意的比例

## 模型架构

- **基础模型**: BAAI/bge-base-en-v1.5
- **投影维度**: 128
- **训练方法**: 微调embedding + 分类头
- **损失函数**: 对比学习 + 分类损失 + 边界损失

## 训练数据

| 类别 | 样本数 | 来源 |
|------|--------|------|
| 恶意样本 | 2000 | HarmBench, JailbreakHub, JailbreakBench, BeaverTails |
| 正常样本 | 2943 | Alpaca, Dolly, DailyDialog, Everyday-Conversations, WildChat |
| 灰色正常 | 400 | Gray-Benign |
| 灰色恶意 | 300 | Gray-Harmful |
| **总计** | **5643** | |

## 训练结果

- **Best F1**: 0.9148 (Epoch 7)
- **最终准确率**: 96.1%
- **FPR**: 3.4%

## 灰色地带配置

### 推荐配置 [0.45, 0.60]

```python
def classify(prob):
    if prob < 0.45:
        return "benign"      # 正常
    elif prob > 0.60:
        return "harmful"     # 恶意
    else:
        return "uncertain"   # 不确定，交给LLM
```

### 配置效果

| 指标 | 值 |
|------|-----|
| 确定样本准确率 | 96.65% |
| FPR (误报率) | 3.55% |
| FNR (漏检率) | 2.65% |
| 不确定比例 | 21.4% |

### 按数据集效果

| 数据集 | 类型 | 正确 | 不确定 | 错误 | FPR/FNR |
|--------|------|------|--------|------|---------|
| Everyday-Conv | 正常 | 310 | 184 | 6 | 1.90% |
| WildChat | 正常 | 326 | 140 | 34 | 9.44% |
| Alpaca | 正常 | 477 | 22 | 1 | 0.21% |
| HarmBench | 恶意 | 331 | 60 | 9 | 2.65% |

## 可选配置

根据需求可以调整灰色地带范围：

| 配置 | Certain_Acc | FPR | FNR | Uncertain% | 适用场景 |
|------|-------------|-----|-----|------------|----------|
| [0.49, 0.56] | 95.25% | 5.03% | 3.71% | 5.8% | 最小LLM调用 |
| [0.45, 0.60] | 96.65% | 3.55% | 2.65% | 21.4% | **推荐** |
| [0.42, 0.62] | 97.52% | 2.94% | 0.97% | 30.0% | 平衡 |
| [0.40, 0.65] | 97.84% | 2.31% | 1.49% | 43.9% | 低误报 |
| [0.38, 0.68] | 99.03% | 0.79% | 2.15% | 61.8% | 极低误报 |

## V6 vs V7 对比

| 数据集 | V6 FPR | V7 FPR | 提升 |
|--------|--------|--------|------|
| Everyday-Conv | 35.7% | **1.90%** | -33.8% |
| WildChat | 32.9% | **9.44%** | -23.5% |
| Alpaca | 0% | 0.21% | +0.21% |

## 使用方法

```python
import torch
from model import V6HarmfulDetector
from transformers import AutoTokenizer

# 加载模型
model = V6HarmfulDetector(model_name='BAAI/bge-base-en-v1.5', projection_dim=128)
checkpoint = torch.load('best_model.pt', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-base-en-v1.5')

# 预测
def predict(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        prob = model.predict(inputs['input_ids'], inputs['attention_mask'])
    
    prob = prob.item()
    if prob < 0.45:
        return "benign", prob
    elif prob > 0.60:
        return "harmful", prob
    else:
        return "uncertain", prob

# 示例
result, prob = predict("How do I make a cake?")
print(f"Result: {result}, Probability: {prob:.4f}")
```

## 文件结构

```
models/v7_classifier/
├── best_model.pt           # 模型权重
├── config.json             # 模型配置
├── training_log.json       # 训练日志
├── gray_zone_config.json   # 灰色地带配置
└── README.md               # 本文件

scripts/v7_classifier/
├── model.py                # 模型定义
├── loss.py                 # 损失函数
├── dataset.py              # 数据集处理
├── train.py                # 训练脚本
├── prepare_dataset.py      # 数据准备
├── test_real_conversations.py  # 真实对话测试
├── analyze_probs.py        # 概率分布分析
└── optimize_gray_zone.py   # 灰色地带优化

datasets/v7_training/
├── train.jsonl             # 训练数据
├── val.jsonl               # 验证数据
└── config.json             # 数据集配置
```

## 总结

V7前置分类器通过加入真实日常对话数据训练，在保持对恶意样本高检出率的同时，大幅降低了在真实场景下的误报率。使用推荐的灰色地带配置[0.45, 0.60]，可以达到：

- **96.65%** 确定样本准确率
- **3.55%** 误报率
- **2.65%** 漏检率
- **21.4%** 样本交给LLM进一步判断

这是一个适合作为前置分类器的配置，在保证高准确率的同时，将不确定的样本交给更强大的LLM处理。
