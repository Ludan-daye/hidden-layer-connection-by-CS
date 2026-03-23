# 模型训练结果目录

按模型版本分开存放训练结果、测试结果、数据集信息等。

## 目录结构

```
results/
├── v1_baseline/          # V1 基础版
│   ├── README.md         # 模型说明
│   ├── *.npy             # 模型文件
│   └── test_results.json # 测试结果
├── v2_fix_tech_fp/       # V2 修正技术误报版
│   ├── README.md
│   └── ...
├── v3_multi_centroid/    # V3 多质心版
│   ├── README.md
│   └── ...
├── v4_max_separation/    # V4 最大分离版
│   ├── README.md
│   └── ...
└── v4_enhanced/          # V4 增强版 (最新)
    ├── README.md
    ├── learned_projection_weights_19d_v4_enhanced.npy
    ├── centroid_19d_malicious_enhanced.npy
    ├── centroid_19d_normal_enhanced.npy
    ├── v4_enhanced_config.json
    ├── v4_adaptive_threshold_config.json
    └── test_results.json
```

## 版本演进

| 版本 | 改进点 | 恶意检出率 | 正常正确率 |
|------|--------|-----------|-----------|
| V1 | 基础单质心 | ~90% | ~85% |
| V2 | 修正技术误报 | ~90% | ~88% |
| V3 | 多质心设计 | ~92% | ~92% |
| V4 | 最大分离 | ~95% | ~83% |
| V4增强 | 更多数据+自适应阈值 | **94.9%** | **82.7%** |

## 最新模型 (V4增强版)

- **恶意检出率**: 94.9%
- **正常正确率**: 82.7%
- **有效准确率**: 89.1%
- **质心分离度**: -0.9985

### 阈值配置

```
相似度 > 0.95  → 恶意
相似度 < 0.85  → 正常
0.85 ~ 0.95    → 灰色地带（需人工审核）
```

## 使用方法

```python
import numpy as np
import json

# 加载最新模型
weights = np.load('results/v4_enhanced/learned_projection_weights_19d_v4_enhanced.npy')
with open('results/v4_enhanced/v4_adaptive_threshold_config.json') as f:
    config = json.load(f)

lower = config['lower_threshold']  # 0.85
upper = config['upper_threshold']  # 0.95
centroid = np.array(config['malicious_centroid'])
```
