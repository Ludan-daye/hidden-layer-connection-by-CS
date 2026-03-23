# V6 + CS压缩 + 多质心检测 测试结果

## 配置

- **压缩维度**: 32D (768D -> 32D, 24x压缩)
- **质心数量**: 3个恶意质心 + 1个正常质心
- **基础模型**: V6 Fine-tuned BGE-base-en-v1.5

## 测试结果

| 数据集 | 样本数 | Accuracy | F1 | Precision | Recall | FPR |
|--------|--------|----------|-----|-----------|--------|-----|
| HarmBench | 300 | 0.9967 | 0.9983 | 1.0000 | 0.9967 | 0.0000 |
| JailbreakBench | 486 | 0.8025 | 0.8904 | 1.0000 | 0.8025 | 0.0000 |
| BeaverTails | 500 | 0.9940 | 0.9970 | 1.0000 | 0.9940 | 0.0000 |
| Alpaca | 500 | 0.9980 | 0.0000 | 0.0000 | 0.0000 | 0.0020 |
| Dolly | 500 | 0.9700 | 0.0000 | 0.0000 | 0.0000 | 0.0300 |
| OpenAssistant | 500 | 0.5480 | 0.0000 | 0.0000 | 0.0000 | 0.4520 |
| ToxicChat | 1000 | 0.5170 | 0.4030 | 0.2782 | 0.7309 | 0.5444 |
| Gray-Benign | 400 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Gray-Harmful | 300 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

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
