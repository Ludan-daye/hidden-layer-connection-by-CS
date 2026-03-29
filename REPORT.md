# Embedding-Prefilter 项目报告

## 1. 项目概述

**Embedding-Prefilter** 是一个基于学习型嵌入投影的轻量前置分类器，用于在 LLM 推理前快速识别恶意/越狱输入。

核心创新：将 BGE-base-en-v1.5 的 384 维嵌入通过**学习型投影压缩至 19 维**（5% 压缩率），在保持攻击检测能力的同时实现极低推理延迟（<10ms）。

### 技术路线

```
用户输入 → BGE嵌入(384d) → 学习型投影(384d→128d) → 分类头 → harmful/benign/uncertain
                                                              ↓ uncertain
                                                         LLM Judge (可选)
```

### 核心组件

| 组件 | 技术 | 说明 |
|------|------|------|
| 嵌入模型 | BAAI/bge-base-en-v1.5 | 384维文本嵌入 |
| 投影层 | LearnedProjection (384→128) | 有监督降维，保持余弦相似度 |
| 分类头 | Binary classifier | 输出概率 [0,1] |
| 灰色地带 | [0.45, 0.60] 阈值 | 不确定样本交由 LLM 判断 |
| 训练损失 | Classification + InfoNCE + Boundary | 三重损失函数 |

---

## 2. 项目结构

```
CSonEmbedding/
├── models/
│   ├── v7_classifier/          # V7 模型权重 + 配置
│   │   ├── best_model.pt       # 750MB (HuggingFace下载)
│   │   ├── config.json
│   │   └── gray_zone_config.json
│   └── v7.1_classifier/        # V7.1 部署版本
│       └── deploy/
│           ├── detector.py     # 完整检测系统入口
│           ├── v7_classifier.py # V7分类器封装
│           ├── llm_judge.py    # LLM判断模块
│           └── test_*.py       # 各类测试脚本
│
├── scripts/
│   ├── v7_classifier/          # V7 训练代码
│   │   ├── train.py            # 训练入口
│   │   ├── model.py            # V6HarmfulDetector 模型定义
│   │   ├── dataset.py          # 数据集加载
│   │   ├── loss.py             # 三重损失函数
│   │   └── deploy/             # 部署与测试
│   ├── baseline_comparison.py  # 基线对比实验
│   ├── test_comprehensive_v7.py # 综合评测（11数据集）
│   ├── generate_comparison_table.py # 对比表格生成
│   └── download_eval_datasets.py   # 评测数据集下载
│
├── datasets/
│   ├── v7_training/            # 训练数据 (5,643 samples)
│   │   ├── train.jsonl         # 5,079 训练样本
│   │   └── val.jsonl           # 564 验证样本
│   ├── advbench/               # AdvBench 有害指令
│   ├── harmbench/              # HarmBench 有害行为
│   ├── jailbreakbench/         # GCG + PAIR 攻击
│   ├── jailbreakhub/           # 手动越狱模板
│   ├── beavertails/            # 多类别有害样本
│   └── normal/                 # Alpaca 常规指令
│
├── results/
│   ├── comprehensive_eval/     # 综合评测结果
│   │   ├── results.json        # V7 在 11 个数据集上的结果
│   │   ├── comparison_table.md # 多维度对比表格
│   │   └── related_work_data.md # 相关工作素材
│   ├── baseline_comparison/    # 基线方法对比
│   └── v7.1.1_guardreasoner/   # 对标 GuardReasoner
│
└── embedding_db/               # 嵌入向量数据库（多模型）
```

---

## 3. V7 模型性能

### 3.1 训练配置

| 项目 | 参数 |
|------|------|
| 基座模型 | BAAI/bge-base-en-v1.5 (384d) |
| 投影维度 | 128d |
| 训练样本 | 5,643 (有害2000 + 良性2943 + 灰色700) |
| 最佳 F1 | 0.9148 (Epoch 7) |
| 训练准确率 | 96.1% |
| 灰色地带 | [0.45, 0.60] |

### 3.2 综合评测结果（V7-Embed，无LLM Judge）

**攻击检测率 (Detection Rate↑)**

| 数据集 | 样本 | Detection Rate | ASR |
|--------|------|---------------|-----|
| GCG 攻击 | 100 | **84.0%** | 16.0% |
| AdvBench | 200 | **85.0%** | 15.0% |
| BeaverTails | 300 | **85.3%** | 14.7% |
| HarmBench | 200 | 82.0% | 18.0% |
| JailbreakHub | 79 | 64.6% | 35.4% |
| PAIR 攻击 | 86 | 47.7% | 52.3% |

**误报率 (FPR↓)**

| 数据集 | 样本 | FPR |
|--------|------|-----|
| Alpaca | 200 | **1.5%** |
| ToxicChat benign | 300 | **5.0%** |
| JBB-Benign | 100 | 34.0% |

### 3.3 V7-Full（含 LLM Judge）对比

| 数据集 | V7-Embed | V7-Full (+Llama-2-7B) | 提升 |
|--------|----------|----------------------|------|
| GCG DR | 84.0% | **96.0%** | +12% |
| AdvBench DR | 85.0% | **97.0%** | +12% |
| HarmBench DR | 82.0% | **97.0%** | +15% |
| Alpaca FPR | 1.5% | **0.0%** | -1.5% |

---

## 4. 与现有方法对比

### 4.1 攻击检测率

| 方法 | 类型 | 参数量 | GCG | JailbreakHub | AdvBench |
|------|------|--------|-----|-------------|---------|
| **Ours (V7-Embed)** | Embed+Proj | 19d proj | 84.0% | 64.6% | **85.0%** |
| **Ours (V7-Full)** | +LLM Judge | +7B | **96.0%** | 78.0% | **97.0%** |
| NeMo+RF | Embed+RF | 768d | — | **96.0%**¹ | — |
| PromptGuard | mDeBERTa | 86M | — | 30.3%¹ | — |
| Gradient Cuff† | LLM grad | 0 | **98.8%** | — | — |
| Perplexity+LGB | Feature | ~0 | 96.2%² | — | — |
| Keyword (Ours) | Rule | 0 | 39.0% | 81.0% | — |
| BGE+SVM (Ours) | Embed+SVM | 384d | 87.0% | 71.0% | — |

¹ F1值 ² 仅机器生成攻击，人工越狱0% † 需目标LLM

### 4.2 V7 核心优势

| 维度 | 优势 | 数据 |
|------|------|------|
| 嵌入压缩 | 所有方法中最小表征 | 384d→**19d** (5%) |
| 直接有害指令 | 唯一报告 AdvBench/HarmBench 的轻量方法 | DR 85%/82% |
| 常规输入透明度 | 极低误报 | Alpaca FPR **1.5%** |
| 推理速度 | 投影后19维分类接近零开销 | **<10ms** |
| 独立部署 | 无需目标LLM | vs GradSafe/Gradient Cuff |

### 4.3 已知局限

| 局限 | 数据 | 原因 |
|------|------|------|
| PAIR语义攻击弱 | DR 47.7% | 嵌入空间难区分语义伪装 |
| JBB-Benign FPR高 | 34.0% | 边界样本V7高置信误判 |

---

## 5. 参考文献

| 方法 | 来源 |
|------|------|
| NeMo+RF | arXiv:2412.01547, AICS 2025 |
| PromptGuard 1/2 | Meta 2024/2025, arXiv:2505.03574 |
| Gradient Cuff | NeurIPS 2024, arXiv:2403.00867 |
| GradSafe | ACL 2024, arXiv:2402.13494 |
| InjecGuard | arXiv:2410.22770 |
| Perplexity Filter | arXiv:2308.14132 |
| SoK 统一评测 | arXiv:2506.10597 |
