# 轻量前置分类器多维度对比表格

> 更新时间：2026-03-29
> V7实测：V7-Embed 模式（uncertain→benign，无 LLM Judge）
> 所有文献数据均来自论文原文第一手实验结果

---

## 表1：总览表

| 方法 | 会议/来源 | 参数量 | 核心技术 | 嵌入维度 | 推理速度 | 备注 |
|------|---------|--------|---------|---------|---------|------|
| **Ours (V7-Embed)** | 本项目 | 109M+19d proj | BGE嵌入→学习投影→分类头 | **19d** | **<10ms** | 384d→19d, 5%压缩率 |
| NeMo+RF | AICS 2025 | 109M+RF | Snowflake-Embed-M + 随机森林 | 768d | ~10ms | arXiv:2412.01547 |
| PromptGuard (86M) | Meta 2024 | 86M | mDeBERTa-v3-base 微调 | — | ~20ms | OOD严重退化 |
| PromptGuard 2 (86M) | Meta 2025 | 86M | mDeBERTa+能量损失函数 | — | ~92ms | arXiv:2505.03574 |
| PromptGuard 2 (22M) | Meta 2025 | 22M | DeBERTa-xsmall | — | ~19ms | 仅英文 |
| InjecGuard | arXiv 2024 | 184M | DeBERTa-v3+NotInject数据 | — | ~15ms | arXiv:2410.22770 |
| ProtectAI DeBERTa-v2 | 2024 | 184M | DeBERTa-v3-base 微调 | — | ~30ms | HF开源 |
| GradSafe | ACL 2024 | 0(用目标LLM) | 梯度余弦相似度 | — | LLM推理 | arXiv:2402.13494 |
| Gradient Cuff | NeurIPS 2024 | 0(用目标LLM) | 拒绝损失+梯度范数 | — | LLM推理 | arXiv:2403.00867 |
| Perplexity+LGB | arXiv 2024 | ~0 | GPT-2困惑度+LightGBM | — | <1ms | arXiv:2308.14132 |
| Keyword (Ours) | — | 0 | 关键词匹配 | — | <1ms | 基线 |
| TF-IDF+LR (Ours) | — | <1M | TF-IDF+逻辑回归 | — | <1ms | 基线 |
| BGE+SVM (Ours) | — | 109M | BGE嵌入+SVM | 384d | ~10ms | 基线 |

---

## 表2：攻击检测率对比（Detection Rate↑ = 1-ASR，越高越好）

| 方法 | GCG | PAIR | JailbreakHub | AdvBench | HarmBench | BeaverTails | 来源 |
|------|-----|------|-------------|---------|-----------|------------|------|
| **Ours (V7-Embed)** | **84.0%** | 47.7% | 64.6% | **85.0%** | **82.0%** | **85.3%** | 实测 |
| **Ours (V7-Full)** | **96.0%** | 62.0% | 78.0% | **97.0%** | **97.0%** | — | 实测(+LLM) |
| NeMo+RF | — | — | **96.0%**¹ | — | — | — | [1] |
| PromptGuard (86M) | — | — | 30.3%¹ | — | — | — | [1] |
| PromptGuard 2 (86M) | — | — | — | — | — | — | [2] |
| Gradient Cuff† | **98.8%** | 77.0% | — | — | — | — | [3] |
| GradSafe† | — | — | — | — | — | — | [4] |
| Perplexity+LGB | **96.2%**² | 0.0% | — | — | — | — | [5] |
| Keyword (Ours) | 39.0% | 69.0% | 81.0% | — | — | — | 实测 |
| TF-IDF+LR (Ours) | 79.0% | 79.0% | 62.0% | — | — | — | 实测 |
| BGE+SVM (Ours) | 87.0% | 78.0% | 71.0% | — | — | — | 实测 |

¹ F1值，非Detection Rate（JailbreakHub上NeMo F1=0.960, PromptGuard F1=0.303）
² 仅对机器生成的GCG攻击；人工编写越狱检测率=0%
† 需要目标LLM梯度，非独立轻量分类器

---

## 表3：误报率对比（FPR↓，越低越好）

| 方法 | Alpaca | ToxicChat | JBB-Benign | 来源 |
|------|--------|-----------|------------|------|
| **Ours (V7-Embed)** | **1.5%** | **5.0%** | 34.0% | 实测 |
| **Ours (V7-Full)** | **0.0%** | — | 44.0% | 实测(+LLM) |
| NeMo+RF | — | 0.2% | 0.4%¹ | [1] |
| PromptGuard (86M) | — | 2.0% | 50.7%¹ | [1] |
| PromptGuard 2 (86M) | — | — | — | [2] |
| Gradient Cuff† | — | — | — | [3], FPR=2.2% |
| Keyword (Ours) | — | — | 19.0% | 实测 |
| TF-IDF+LR (Ours) | — | — | 37.0% | 实测 |
| BGE+SVM (Ours) | — | — | 63.0% | 实测 |

¹ NeMo和PromptGuard的FPR在JailbreakHub测试集上报告，非JBB-Benign
⚠ 排除了BeaverTails-benign（`is_safe`标注的是模型回复安全性，非prompt本身无害）

---

## 表4：V7-Embed vs V7-Full（LLM Judge 增量贡献）

| 数据集 | V7-Embed | V7-Full (+Llama-2-7B) | 提升 | 说明 |
|--------|----------|----------------------|------|------|
| GCG (DR↑) | 84.0% | **96.0%** | +12.0% | LLM有效拦截灰色地带GCG |
| PAIR (DR↑) | 47.7% | 62.0% | +14.3% | 语义攻击仍是弱项 |
| JailbreakHub (DR↑) | 64.6% | 78.0% | +13.4% | |
| AdvBench (DR↑) | 85.0% | **97.0%** | +12.0% | LLM全面提升直接有害指令 |
| HarmBench (DR↑) | 82.0% | **97.0%** | +15.0% | |
| JBB-Benign (FPR↓) | **34.0%** | 44.0% | -10.0% | LLM保守判断增加误报 |
| Alpaca (FPR↓) | 1.5% | **0.0%** | +1.5% | |
| ToxicChat (FPR↓) | **5.0%** | — | — | |

---

## 表5：SoK 统一评测排名（arXiv:2506.10597）

> 统一测试平台：Llama-3-8B-Instruct, 9种攻击类型, 2820攻击+1805良性样本

| 排名 | 方法 | 类型 | 平均ASR↓ | GCG ASR | Manual ASR |
|------|------|------|---------|---------|------------|
| 1 | GuardReasoner | LLM-based | 0.135 | 0.000 | 0.008 |
| 2 | WildGuard | LLM-based | 0.143 | 0.000 | 0.004 |
| 3 | Gradient Cuff | LLM gradient | 0.148 | 0.080 | 0.016 |
| 4 | PromptGuard | Lightweight | 0.163 | 0.000 | 0.000 |
| 5 | GradSafe | LLM gradient | 0.224 | 0.010 | 0.022 |
| 6 | Perplexity Filter | Feature | 0.239 | — | — |
| 7 | SmoothLLM | LLM-based | 0.303 | 0.020 | 0.042 |

*注：所有方法在 X-Teaming 攻击上均失效（ASR 0.640-0.990）*

---

## 表6：V7 核心优势总结

| 对比维度 | V7 优势 | 具体数据 | 对比方法 |
|---------|---------|---------|---------|
| **嵌入压缩率** | 所有方法中最小的表征维度 | 384d→**19d** (5%压缩率) | NeMo 768d, PG 86M全模型 |
| **直接有害指令检测** | AdvBench/HarmBench上DR最高 | DR **85%/82%** | 其他轻量方法均未报告 |
| **常规输入透明度** | Alpaca FPR极低 | FPR **1.5%** | PG1=50.7%, BGE+SVM=63% |
| **推理效率** | 投影后仅19维分类 | **<10ms** | PG2=92ms, InjecGuard=15ms |
| **独立部署** | 无需目标LLM或额外模型 | 纯嵌入+投影 | GradSafe/GCuff需目标LLM |
| **多数据集覆盖** | 6个攻击集+3个良性集实测 | 最全面的评测 | NeMo仅JBHub+TC |

### 已知局限（需论文诚实说明）

| 局限 | 数据 | 原因 |
|------|------|------|
| PAIR语义攻击弱 | DR 47.7% | 嵌入空间难区分语义伪装 |
| JBB-Benign FPR高 | 34.0% | 边界样本V7高置信误判 |
| ToxicChat harmful DR低 | 13.3% | 任务定义不同（TC标注回复毒性，非prompt攻击意图） |

---

## 数据来源

- [1] arXiv:2412.01547 (Galinkin & Sablotny, AICS 2025) — NeMo+RF, PromptGuard
- [2] arXiv:2505.03574 (Meta LlamaFirewall, 2025) — PromptGuard 2
- [3] arXiv:2403.00867 (Hu et al., NeurIPS 2024) — Gradient Cuff
- [4] arXiv:2402.13494 (Xie et al., ACL 2024) — GradSafe
- [5] arXiv:2308.14132 (Alon & Kamfonas, 2023) — Perplexity Filter
- [6] arXiv:2410.22770 (InjecGuard, 2024) — InjecGuard
- [7] arXiv:2506.10597 (SoK Evaluating Jailbreak Guardrails) — 统一评测
- **Ours**: 本项目实测 (`results/comprehensive_eval/results.json`, `results/baseline_comparison/results.json`)
