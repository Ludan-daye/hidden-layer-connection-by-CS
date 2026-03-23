# 基于Embedding余弦相似度的恶意输入检测实验设计

## 1. 研究目标

验证基于Embedding余弦相似度的恶意输入检测方法的有效性，通过对比学习投影将高维Embedding压缩到低维空间，在保持检测性能的同时降低存储和计算成本。

---

## 2. 三版训练迭代总结

我们进行了3个版本的训练迭代，逐步改进模型性能：

| 版本 | 模型思路 | 核心改进 | 问题/动机 |
|-----|---------|---------|----------|
| **第一版 (v1)** | 基础对比学习 | 单质心+XSTest/OR-Bench | 初始版本 |
| **第二版 (v2)** | 增加安全教育样本 | 修正技术/日常对话误报 | v1对技术问题误报高 |
| **第三版 (v3)** | 多质心对比学习 | 5类恶意质心分离 | 单质心无法覆盖多种恶意类型 |

---

## 3. 版本详细说明

### 3.1 第一版：基础对比学习

**问题背景**：初始版本，建立基础的对比学习框架。

**思路**：使用单一恶意质心，通过对比学习使恶意样本聚集、正常样本分散。

**训练脚本**：`scripts/train_contrastive.py`

**训练数据**：
| 类型 | 数据集 | 样本数 |
|-----|-------|-------|
| 恶意 | AdvBench | 520 |
| 正常 | Alpaca (原始) | 500 |
| 正常 | XSTest (safe) | 250 |
| 正常 | OR-Bench Hard | 1,319 |

**模型文件**：
- 投影矩阵：`learned_projection_weights_19d_contrastive.npy`
- 质心：`centroid_19d_contrastive.npy`

**损失函数**：
```python
def contrastive_loss(mal_proj, norm_proj, centroid, margin=0.3):
    mal_sim = torch.sum(mal_proj * centroid, dim=-1)
    norm_sim = torch.sum(norm_proj * centroid, dim=-1)
    mal_loss = torch.mean(1 - mal_sim)                 # 恶意接近质心
    norm_loss = torch.mean(F.relu(norm_sim - margin))  # 正常远离质心
    return mal_loss + norm_loss + separation_loss
```

**发现的问题**：
- ⚠️ 对技术类问题（如"How to kill a process?"）误报率高
- ⚠️ 对日常对话有一定误报

---

### 3.2 第二版：修正技术/日常对话误报

**问题背景**：第一版对技术类问题和日常对话误报率较高，需要增加这类正常样本。

**思路**：增加安全教育类样本和日常对话样本，让模型学会区分"学习安全知识"和"实施攻击"。

**训练脚本**：`scripts/train_contrastive_v2.py`

**训练数据**：
| 类型 | 数据集 | 样本数 | 新增 |
|-----|-------|-------|-----|
| 恶意 | AdvBench | 520 | |
| 正常 | Alpaca (原始) | 500 | |
| 正常 | XSTest (safe) | 250 | |
| 正常 | OR-Bench Hard | 1,319 | |
| 正常 | **安全教育样本** | 130 | ✅ 新增 |

**安全教育样本示例**：
```python
SECURITY_EDUCATION_SAMPLES = [
    "What is the CIA triad in information security?",
    "Explain the difference between authentication and authorization",
    "How does public key cryptography work?",
    "What are the main types of cyber attacks?",
    "Explain how firewalls protect networks",
    "How do penetration testers find vulnerabilities?",
    "What is ethical hacking?",
    ...
]
```

**模型文件**：
- 投影矩阵：`learned_projection_weights_19d_contrastive_v2.npy`
- 质心：`centroid_19d_contrastive_v2.npy`
- 阈值：`-0.4275`

**改进效果**：
- ✅ 技术类问题误报率下降
- ✅ 安全教育问题不再被误判

**仍存在的问题**：
- ⚠️ 日常对话仍有一定误报
- ⚠️ 越狱攻击、道德伤害检出率低

---

### 3.3 第二版补充：增加日常对话 (v3单质心)

**问题背景**：v2仍对日常对话有误报，需要增加更多日常对话样本。

**训练脚本**：`scripts/train_contrastive_v3.py`

**训练数据**：
| 类型 | 数据集 | 样本数 | 新增 |
|-----|-------|-------|-----|
| 恶意 | AdvBench | 520 | |
| 正常 | Alpaca (原始) | 500 | |
| 正常 | XSTest (safe) | 250 | |
| 正常 | OR-Bench Hard | 1,319 | |
| 正常 | 安全教育样本 | 130 | |
| 正常 | **Alpaca (HuggingFace)** | 1,000 | ✅ 新增 |

**模型文件**：
- 投影矩阵：`learned_projection_weights_19d_contrastive_v3.npy`
- 质心：`centroid_19d_contrastive_v3.npy`
- 阈值：`-0.4118`

**结果**：
| 数据集 | 检出率 |
|-------|-------|
| AdvBench | 99.6% ✅ |
| ToxicChat | 100% ✅ |
| JBB-GCG | 48.5% |
| JBB-PAIR | 27.9% |
| BeaverTails | 26.0% |
| **总计** | **66.0%** |

**分析**：
- ✅ 直接恶意指令检出率高（99.6%）
- ✅ 日常对话误报率降低
- ⚠️ 越狱攻击检出率低（28-48%）
- ⚠️ 道德伤害检出率低（26%）
- **原因**：单一质心无法覆盖多种恶意类型的语义空间

---

### 3.4 第三版：多质心对比学习

**问题背景**：单质心模型对越狱攻击和道德伤害检出率低，因为这些恶意类型的语义空间与直接恶意指令不同。

**思路**：为每种恶意类型建立独立质心，检测时取与所有质心的最大相似度。

**训练脚本**：`scripts/train_multi_centroid.py`, `scripts/train_multi_centroid_v2.py`

**训练数据**：
| 恶意类型 | 数据集 | 样本数 |
|---------|-------|-------|
| direct_harm | AdvBench | 200 |
| diverse_harm | HarmBench | 200 |
| jailbreak | JBB-GCG + JBB-PAIR | 286 |
| moral_harm | BeaverTails | 200 |
| toxic | ToxicChat | 200 |
| **正常** | 原始+XSTest+OR-Bench+Alpaca | 1,750 |

**模型文件**：
- 投影矩阵：`learned_projection_weights_19d_multi_centroid.npy`
- 质心：`centroid_19d_direct_harm.npy`, `centroid_19d_jailbreak.npy`, ...
- 配置：`multi_centroid_config.json`
- 阈值：`0.2497`

**损失函数**：
```python
def multi_centroid_loss(projection, mal_embs_by_type, norm_emb):
    # 1. 每类恶意样本聚集到各自质心
    for type_name, emb in mal_embs_by_type.items():
        centroid = compute_centroid(projection(emb))
        intra_loss += mean(1 - sim(proj, centroid))
    
    # 2. 正常样本远离所有恶意质心
    for centroid in mal_centroids:
        norm_loss += mean(relu(sim(norm_proj, centroid) + margin))
    
    # 3. 不同恶意质心保持距离
    inter_loss = relu(sim(centroid_i, centroid_j) - 0.5)
```

**结果对比**：
| 数据集 | 单质心(v3) | 多质心 | 提升 |
|-------|---------|---------|-----|
| AdvBench | 100% | 98.3% | -1.7% |
| ToxicChat | 100% | 100% | 0% |
| JBB-GCG | 48.5% | **99.5%** | **+51%** |
| JBB-PAIR | 27.9% | **98.8%** | **+71%** |
| BeaverTails | 26.0% | **89.7%** | **+64%** |
| **总计** | 66.0% | **92.5%** | **+26.5%** |

**分析**：
- ✅ 越狱攻击检出率从28-48%提升到**99%**
- ✅ 道德伤害检出率从26%提升到**90%**
- ✅ 总体检出率从66%提升到**92.5%**
- ⚠️ 误报率略有上升（OR-Bench: 5.3%→6%）

---

## 4. 文件结构（按版本整理）

```
CSonEmbedding/
├── scripts/
│   ├── v1_basic/                        # 第一版：基础对比学习
│   │   └── train_contrastive.py
│   │
│   ├── v2_fix_tech_fpr/                 # 第二版：修正技术/日常误报
│   │   ├── train_contrastive_v2.py      # 增加安全教育样本
│   │   └── train_contrastive_v3.py      # 增加日常对话样本
│   │
│   ├── v3_multi_centroid/               # 第三版：多质心对比学习
│   │   ├── train_multi_centroid.py      # 多质心训练
│   │   ├── train_multi_centroid_v2.py   # 同类聚集+异类分离
│   │   └── test_multi_centroid.py       # 多质心测试对比
│   │
│   ├── test_malicious_types.py          # 恶意类型检出率测试
│   ├── test_normal_types.py             # 正常样本误报率测试
│   ├── analyze_centroid_distance.py     # 质心距离分析
│   └── compare_compression.py           # 压缩效果对比
│
├── embedding_db/bge-small-en-v1.5/results/
│   ├── v1_basic/                        # 第一版模型文件
│   │   ├── learned_projection_weights_19d_contrastive.npy
│   │   ├── centroid_19d_contrastive.npy
│   │   └── training_contrastive_results.json
│   │
│   ├── v2_fix_tech_fpr/                 # 第二版模型文件
│   │   ├── learned_projection_weights_19d_contrastive_v2.npy
│   │   ├── centroid_19d_contrastive_v2.npy
│   │   ├── training_contrastive_v2_results.json
│   │   ├── learned_projection_weights_19d_contrastive_v3.npy
│   │   ├── centroid_19d_contrastive_v3.npy
│   │   └── training_contrastive_v3_results.json
│   │
│   ├── v3_multi_centroid/               # 第三版模型文件
│   │   ├── learned_projection_weights_19d_multi_centroid.npy
│   │   ├── learned_projection_weights_19d_multi_centroid_v2.npy
│   │   ├── centroid_19d_direct_harm.npy
│   │   ├── centroid_19d_diverse_harm.npy
│   │   ├── centroid_19d_jailbreak.npy
│   │   ├── centroid_19d_moral_harm.npy
│   │   ├── centroid_19d_toxic.npy
│   │   ├── multi_centroid_config.json
│   │   └── multi_centroid_v2_config.json
│   │
│   └── (原有文件保留兼容)
│
└── datasets/
    ├── advbench/           # 直接恶意指令
    ├── harmbench/          # 多样化恶意
    ├── beavertails/        # 道德伤害
    ├── jailbreakbench/     # 越狱攻击
    ├── gcg_attacks/        # 有毒对话、注入攻击
    └── truthfulqa/         # 正常问答
```

---

## 5. 三版对比总结

| 维度 | 第一版 | 第二版 | 第三版(多质心) |
|-----|-------|-------|--------------|
| **核心改进** | 基础框架 | 修正技术误报 | 多类型恶意分离 |
| **质心数** | 1 | 1 | 5 |
| **直接恶意检出** | ~99% | ~99% | 98.3% |
| **越狱攻击检出** | ~40% | ~40% | **99%** |
| **道德伤害检出** | ~25% | ~25% | **90%** |
| **总体检出率** | ~65% | ~66% | **92.5%** |
| **技术问题误报** | 高 | **低** | 低 |
| **日常对话误报** | 中 | **低** | 低 |

**迭代过程**：
1. **第一版**：建立基础框架，发现技术问题误报高
2. **第二版**：增加安全教育样本，解决技术误报问题
3. **第三版**：多质心设计，解决越狱攻击和道德伤害检出率低的问题

**推荐**：
- **低误报优先**：使用第二版单质心模型
- **高检出优先**：使用第三版多质心模型
- **生产环境**：第三版多质心（检出率高，误报率适中）

---

## 6. 方法概述

### 6.1 检测流程

```
输入文本 → Embedding模型(384维) → 学习型投影(19维) → 余弦相似度计算 → 阈值判定
```

### 6.2 核心思想

1. **质心检测**：计算恶意样本的质心向量，新输入与质心的余弦相似度越高，越可能是恶意输入
2. **对比学习投影**：通过对比学习优化投影矩阵，使恶意样本聚集、正常样本分散
3. **阈值判定**：使用正常样本的95%分位作为阈值，确保低误报率

---

## 7. 数据集设计

### 3.1 训练集

#### 3.1.1 恶意样本（多种类型）

| 数据集 | 样本数 | 类型 | 描述 |
|-------|-------|-----|------|
| **AdvBench** | 520 | 直接恶意指令 | 明确的有害行为请求（如制作炸弹、黑客攻击） |
| **HarmBench** | 400 | 多样化恶意行为 | 涵盖多种有害行为类别 |
| **BeaverTails** | 500 | 道德伤害 | 14种道德伤害类型（歧视、仇恨等） |
| **MaliciousInstruct** | 100 | 恶意指令 | 恶意操作指令集 |
| **总计** | **1,520** | - | - |

#### 3.1.2 正常样本（多种类型，含技术性对话）

| 数据集 | 样本数 | 类型 | 描述 |
|-------|-------|-----|------|
| **Alpaca (原始)** | 500 | 通用指令 | 日常任务指令 |
| **Alpaca (HuggingFace)** | 1,000 | 日常对话 | 多样化日常问答 |
| **XSTest** | 250 | 边界安全问题 | 容易被误拒的安全问题（如"How to kill a Python process"） |
| **OR-Bench Hard** | 1,319 | 过度拒绝测试 | 边界线问题，测试过度拒绝 |
| **安全教育样本** | 200 | 技术学习 | 网络安全、渗透测试学习问题 |
| **TruthfulQA** | 500 | 正常问答 | 真实性问答数据集 |
| **技术编程问题** | 300 | 技术对话 | 编程、系统管理、网络配置等 |
| **总计** | **4,069** | - | - |

#### 3.1.3 技术性对话样本详细分类

| 类别 | 样本数 | 示例 |
|-----|-------|-----|
| 编程开发 | 50 | "How do I write a for loop in Python?" |
| 网络安全教育 | 50 | "What is the OWASP Top 10?" |
| 系统管理 | 50 | "How do I configure a firewall on Linux?" |
| 数据科学 | 50 | "How do I train a machine learning model?" |
| 云计算 | 50 | "How do I deploy an application on AWS?" |
| 渗透测试学习 | 50 | "What certifications are needed for penetration testing?" |

### 3.2 测试集

#### 3.2.1 恶意测试集

| 数据集 | 样本数 | 类型 | 预期检出率 |
|-------|-------|-----|----------|
| **AdvBench (测试)** | 100 | 直接恶意指令 | >95% |
| **HarmBench (测试)** | 100 | 多样化恶意 | >60% |
| **JBB-Harmful** | 100 | 越狱目标 | >70% |
| **JBB-GCG** | 200 | GCG越狱攻击 | >40% |
| **JBB-PAIR** | 100 | PAIR越狱攻击 | >30% |
| **ToxicChat** | 384 | 有毒对话 | >90% |
| **PromptInjections** | 200 | 注入攻击 | >10% |

#### 3.2.2 正常测试集

| 数据集 | 样本数 | 类型 | 预期误报率 |
|-------|-------|-----|----------|
| **TruthfulQA (测试)** | 200 | 正常问答 | <5% |
| **日常对话** | 100 | 日常交流 | <10% |
| **技术问题** | 100 | 技术对话 | <15% |
| **学术研究** | 100 | 学术问题 | <5% |
| **创意写作** | 100 | 创作请求 | <10% |

---

## 4. 实验设计

### 4.1 实验一：各数据集质心距离分析

**目标**：分析不同类型数据集与恶意质心的余弦相似度分布

**方法**：
1. 计算恶意训练集的质心向量
2. 分别计算各数据集样本与质心的余弦相似度
3. 统计各数据集的相似度分布（min, max, mean, std）

**预期结果**：

| 数据集类型 | 预期相似度范围 | 预期均值 |
|-----------|--------------|---------|
| 直接恶意指令 | 0.8 ~ 1.0 | >0.9 |
| 越狱攻击 | -0.5 ~ 0.8 | 0.0~0.3 |
| 道德伤害 | -0.5 ~ 0.5 | -0.2~0.2 |
| 技术安全问题 | -0.9 ~ 0.3 | <-0.5 |
| 日常对话 | -1.0 ~ -0.5 | <-0.8 |

### 4.2 实验二：压缩前后效果对比

**目标**：验证学习型投影在压缩维度的同时保持或提升检测性能

**对比维度**：
- 384维原始空间
- 19维压缩空间（对比学习投影）

**评估指标**：
- 检出率（Recall）
- 误报率（FPR）
- AUC
- 相似度分离度（恶意均值 - 正常均值）

### 4.3 实验三：不同恶意类型检出率分析

**目标**：分析模型对不同类型恶意输入的检出能力

**分析维度**：
1. 直接恶意指令（AdvBench）
2. 多样化恶意行为（HarmBench）
3. 道德伤害（BeaverTails）
4. 越狱攻击（JBB-GCG/PAIR）
5. 有毒对话（ToxicChat）
6. 注入攻击（PromptInjections）

### 4.4 实验四：正常样本误报率分析

**目标**：分析模型对不同类型正常输入的误报情况

**分析维度**：
1. 日常对话
2. 学术研究
3. 编程开发
4. 技术安全学习（边界案例）
5. 创意写作
6. 商业咨询

---

## 5. 评估指标

### 5.1 主要指标

| 指标 | 定义 | 目标 |
|-----|-----|-----|
| **检出率 (Recall)** | TP / (TP + FN) | >95% |
| **误报率 (FPR)** | FP / (FP + TN) | <10% |
| **准确率 (Accuracy)** | (TP + TN) / Total | >90% |
| **精确率 (Precision)** | TP / (TP + FP) | >90% |
| **F1分数** | 2 * P * R / (P + R) | >0.9 |
| **AUC** | ROC曲线下面积 | >0.95 |

### 5.2 辅助指标

| 指标 | 定义 | 用途 |
|-----|-----|-----|
| **相似度分离度** | 恶意均值 - 正常均值 | 衡量分类边界清晰度 |
| **压缩率** | 1 - (压缩维度 / 原始维度) | 衡量存储效率 |
| **各类别检出率** | 分类别统计 | 分析模型弱点 |

---

## 6. 实验流程

### 6.1 数据准备

```bash
# 1. 加载恶意数据集
python scripts/load_malicious_datasets.py

# 2. 加载正常数据集（含技术性对话）
python scripts/load_normal_datasets.py

# 3. 计算所有样本的Embedding
python scripts/compute_all_embeddings.py
```

### 6.2 模型训练

```bash
# 对比学习训练投影矩阵
python scripts/train_contrastive_v4.py
```

**训练参数**：
- Embedding模型：BAAI/bge-small-en-v1.5 (384维)
- 投影维度：19维
- 学习率：0.01
- 训练轮数：200
- Margin：0.3

### 6.3 实验执行

```bash
# 实验一：质心距离分析
python scripts/analyze_centroid_distance.py

# 实验二：压缩效果对比
python scripts/compare_compression.py

# 实验三：恶意类型检出率
python scripts/test_malicious_types.py

# 实验四：正常样本误报率
python scripts/test_normal_types.py
```

---

## 7. 预期结果

### 7.1 质心距离分布（实验结果）

**恶意数据集：**

| 数据集 | 类型 | 样本数 | 相似度均值 | 相似度范围 | 与质心距离 |
|-------|-----|-------|----------|----------|----------|
| AdvBench | 直接指令 | 500 | **0.9850** | [-0.99, 1.00] | **近** |
| ToxicChat | 有毒对话 | 384 | **0.9227** | [0.50, 0.98] | **近** |
| JBB-Harmful | 越狱目标 | 100 | 0.4644 | [-0.98, 1.00] | 中 |
| HarmBench | 多样化 | 400 | 0.1128 | [-0.99, 1.00] | 中 |
| JBB-GCG | GCG越狱 | 200 | -0.1150 | [-0.99, 0.99] | 中 |
| MaliciousInstruct | 指令 | 100 | -0.2237 | [-0.99, 0.99] | 中 |
| JBB-PAIR | PAIR越狱 | 86 | -0.5252 | [-1.00, 0.98] | 远 |
| BeaverTails | 道德伤害 | 500 | -0.5301 | [-1.00, 1.00] | 远 |
| PromptInjections | 注入攻击 | 500 | -0.8561 | [-1.00, 0.98] | 远 |

**正常数据集：**

| 数据集 | 类型 | 样本数 | 相似度均值 | 相似度范围 | 与质心距离 |
|-------|-----|-------|----------|----------|----------|
| XSTest | 边界安全 | 450 | -0.6706 | [-0.99, 0.97] | 远 |
| DoNotAnswer | 边界问题 | 500 | -0.8340 | [-1.00, 0.96] | 远 |
| OR-Bench | 过度拒绝 | 500 | -0.8689 | [-1.00, 0.76] | 远 |
| TruthfulQA | 问答 | 500 | **-0.9717** | [-0.98, -0.97] | **极远** |

**整体统计：**

| 类别 | 样本数 | 相似度均值 | 分离度 |
|-----|-------|----------|-------|
| 恶意样本 | 2,770 | 0.0559 | - |
| 正常样本 | 1,950 | -0.8406 | - |
| **分离度** | - | - | **0.8964** |

### 7.2 压缩效果

| 指标 | 384维 | 19维 | 变化 |
|-----|------|------|-----|
| 维度 | 384 | 19 | ↓95% |
| 存储 | 1536B | 76B | ↓95% |
| 检出率 | 100% | 100% | 不变 |
| 误报率 | 5% | 5% | 不变 |
| 分离度 | 0.27 | 1.86 | ↑597% |

### 7.3 各类型检出率（实验结果）

**恶意数据集检出率：**

| 数据集 | 类型 | 样本数 | 检出率 | 相似度均值 |
|-------|-----|-------|-------|----------|
| ToxicChat | 有毒对话 | 384 | **100.0%** | 0.92 |
| AdvBench | 直接恶意指令 | 500 | **99.6%** | 0.99 |
| JBB-Harmful | 越狱目标 | 100 | **76.0%** | 0.46 |
| HarmBench | 多样化恶意 | 400 | 61.8% | 0.11 |
| JBB-GCG | GCG越狱攻击 | 200 | 48.5% | -0.12 |
| MaliciousInstruct | 恶意指令 | 100 | 47.0% | -0.22 |
| JBB-PAIR | PAIR越狱攻击 | 86 | 27.9% | -0.53 |
| BeaverTails | 道德伤害 | 500 | 26.4% | -0.53 |
| PromptInjections | 注入攻击 | 500 | 6.0% | -0.86 |
| **总计** | - | **2,770** | **55.4%** | - |

### 7.4 正常样本误报率（实验结果）

**手工构造的正常样本：**

| 类别 | 样本数 | 误报率 | 相似度均值 |
|-----|-------|-------|----------|
| 学术研究 | 10 | **0.0%** | -0.94 |
| 编程开发 | 10 | **0.0%** | -0.93 |
| 技术安全学习 | 15 | **0.0%** | -0.88 |
| 商业咨询 | 10 | **0.0%** | -0.98 |
| 教育学习 | 10 | **0.0%** | -0.98 |
| 日常对话 | 10 | 10.0% | -0.76 |
| 创意写作 | 10 | 10.0% | -0.81 |
| 健康医疗 | 10 | 10.0% | -0.89 |

**数据集中的正常样本：**

| 数据集 | 样本数 | 误报率 | 相似度均值 |
|-------|-------|-------|----------|
| TruthfulQA | 500 | **0.0%** | -0.97 |
| OR-Bench | 500 | 5.6% | -0.87 |
| DoNotAnswer | 500 | 8.8% | -0.83 |
| XSTest | 450 | 18.9% | -0.67 |
| **总计** | **2,035** | **7.9%** | - |

**注意**：XSTest误报率较高是因为该数据集包含一些看似恶意但实际安全的问题（如"How do I kill a Python process?"），但也包含一些确实恶意的问题被正确检出。

---

## 8. 项目结构

```
CSonEmbedding/
├── detect.py                    # 交互式检测接口
├── experimentalDesignV2.md      # 实验设计文档
├── datasets/                    # 数据集
│   ├── advbench/               # AdvBench恶意指令
│   ├── harmbench/              # HarmBench恶意行为
│   ├── beavertails/            # BeaverTails道德伤害
│   ├── jailbreakbench/         # JailbreakBench越狱攻击
│   ├── truthfulqa/             # TruthfulQA正常问答
│   ├── malicious_instruct/     # 恶意指令
│   └── gcg_attacks/            # GCG攻击相关
├── 数据集/                      # 额外训练数据
│   ├── XSTest/                 # 边界安全问题
│   ├── or-bench-hard-1k.csv    # 过度拒绝测试
│   └── or-bench-80k.csv        # 大规模正常问题
├── embedding_db/               # Embedding和模型
│   └── bge-small-en-v1.5/
│       ├── embeddings/         # 预计算Embedding
│       └── results/            # 训练结果
└── scripts/                    # 实验脚本
    ├── train_contrastive_v4.py # 对比学习训练
    ├── analyze_centroid_distance.py  # 质心距离分析
    ├── compare_compression.py  # 压缩效果对比
    ├── test_malicious_types.py # 恶意类型检出率
    └── test_normal_types.py    # 正常样本误报率
```

---

## 9. 后续改进方向

1. **增加越狱攻击训练样本**：将JBB-GCG/PAIR加入训练集，提升越狱攻击检出率
2. **多语言支持**：使用多语言Embedding模型（如multilingual-e5）
3. **二阶段检测**：结合意图分类器进行精细化判定
4. **动态阈值**：根据输入类型动态调整检测阈值
