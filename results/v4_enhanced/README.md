# V4 增强版模型 (最新)

## 模型信息

- **版本**: v4_enhanced
- **训练日期**: 2025-01-28
- **目标**: 最大化正常/恶意方向分离 + 类内聚集 + 自适应阈值

## 训练配置

- **Embedding模型**: BAAI/bge-small-en-v1.5 (384维)
- **投影维度**: 19维
- **训练轮数**: 300
- **优化器**: Adam (lr=0.01)
- **损失函数**: 质心分离损失 + 类内聚集损失

## 训练数据集

### 恶意数据集 (共3,624条)

| 数据集 | 样本数 | 说明 |
|-------|-------|------|
| AdvBench | 520 | 直接恶意指令 |
| HarmBench | 400 | 多样化恶意行为 |
| BeaverTails | 1,000 | 道德伤害 |
| JBB-GCG | 200 | GCG后缀攻击 |
| JBB-PAIR | 86 | PAIR迭代攻击 |
| ToxicChat | 384 | 有毒对话 |
| MaliciousInstruct | 100 | 恶意指令 |
| JailbreakHub+ | 1,000 | 越狱模板+Do-Not-Answer+HH-RLHF+PKU-SafeRLHF |

### 正常数据集 (共4,369条)

| 数据集 | 样本数 | 说明 |
|-------|-------|------|
| Normal Embeddings | 500 | 原始正常样本 |
| XSTest | 250 | 安全边界测试 |
| OR-Bench | 1,319 | 边界问题 |
| Alpaca | 2,000 | 通用指令 |
| 安全教育样本 | 300 | 网络安全技术问题 |

## 自适应阈值配置

| 参数 | 值 | 说明 |
|-----|-----|------|
| 下界 | 0.85 | 低于此值判为正常 |
| 上界 | 0.95 | 高于此值判为恶意 |
| 灰色地带 | [0.85, 0.95] | 需人工审核 |

## 测试结果

### 恶意数据集测试

| 数据集 | 总数 | 检出率 |
|-------|------|-------|
| AdvBench | 300 | 96.3% |
| HarmBench | 300 | 92.0% |
| BeaverTails | 300 | 96.7% |
| JBB-GCG | 200 | 94.5% |
| **平均** | - | **94.9%** |

### 正常数据集测试

| 数据集 | 总数 | 正确率 | 误报率 |
|-------|------|-------|-------|
| Alpaca | 5,000 | 95.1% | 4.7% |
| OpenAssistant | 3,000 | 70.8% | 28.4% |
| Dolly | 3,000 | 66.5% | 32.8% |
| OR-Bench | 500 | 90.6% | - |

### 总体指标

| 指标 | 结果 |
|------|------|
| 恶意检出率 (TPR) | 94.9% |
| 正常正确率 (TNR) | 82.7% |
| 有效准确率 | 89.1% |
| 质心分离度 | -0.9985 |

## 模型文件

- `learned_projection_weights_19d_v4_enhanced.npy` - 投影矩阵
- `centroid_19d_malicious_enhanced.npy` - 恶意质心
- `centroid_19d_normal_enhanced.npy` - 正常质心
- `v4_enhanced_config.json` - 完整配置
- `v4_adaptive_threshold_config.json` - 自适应阈值配置

## 训练脚本

- `scripts/v4_max_separation/train_v4_enhanced.py`
- `scripts/v4_max_separation/train_adaptive_threshold.py`

## 改进点

1. 加入更多恶意数据集（JailbreakHub、Do-Not-Answer、HH-RLHF、PKU-SafeRLHF）
2. 质心分离度达到-0.9985（接近完美分离）
3. 自适应灰色地带阈值设计
4. 三分类：恶意/灰色地带/正常

## 已知问题

- 非英语内容误报率较高
- 简短回复可能被误判
- OpenAssistant/Dolly等复杂对话误报率较高
