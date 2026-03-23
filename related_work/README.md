# LLM输入防御相关工作

本目录整理了2024-2025年LLM输入防御(Jailbreak Detection)领域的主要研究工作。

---

## 一、核心论文列表

### 1.1 防御方法论文

| # | 论文 | 会议/来源 | 年份 | 方法类型 | 链接 |
|---|------|-----------|------|----------|------|
| 1 | **GradSafe: Detecting Jailbreak Prompts for LLMs via Safety-Critical Gradient Analysis** | ACL 2024 | 2024 | 梯度分析 | [ACL](https://aclanthology.org/2024.acl-long.30/) / [GitHub](https://github.com/xyq7/GradSafe) |
| 2 | **SmoothLLM: Defending Large Language Models Against Jailbreaking Attacks** | ICLR 2024 | 2024 | 输入扰动 | [OpenReview](https://openreview.net/forum?id=xq7h9nfdY2) / [arXiv](https://arxiv.org/abs/2310.03684) |
| 3 | **Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations** | Meta | 2024 | LLM分类器 | [arXiv](https://arxiv.org/abs/2312.06674) / [HuggingFace](https://huggingface.co/meta-llama/LlamaGuard-7b) |
| 4 | **WildGuard: Open One-Stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs** | Allen AI | 2024 | LLM分类器 | [arXiv](https://arxiv.org/abs/2406.18495) / [HuggingFace](https://huggingface.co/allenai/wildguard) |
| 5 | **GuardReasoner: Towards Reasoning-based LLM Safeguards** | - | 2025 | LLM推理 | [arXiv](https://arxiv.org/abs/2501.18492) |
| 6 | **Gradient Cuff: Detecting Jailbreak Attacks on Large Language Models by Exploring Refusal Loss Landscapes** | - | 2024 | 梯度分析 | [arXiv](https://arxiv.org/abs/2403.00867) |
| 7 | **SelfDefend: LLMs Can Defend Themselves against Jailbreaking in a Practical Manner** | - | 2024 | 自我防护 | [arXiv](https://arxiv.org/abs/2406.05498) |
| 8 | **Detecting LLM-Generated Text in Computing Education via Perplexity** | - | 2023 | 困惑度过滤 | [arXiv](https://arxiv.org/abs/2308.14132) |
| 9 | **PromptGuard** | Meta | 2024 | 轻量分类器 | [HuggingFace](https://huggingface.co/meta-llama/Prompt-Guard-86M) |
| 10 | **Token Highlighter: Inspecting and Mitigating Jailbreak Prompts for Large Language Models** | - | 2024 | Token分析 | [arXiv](https://arxiv.org/abs/2405.02492) |

### 1.2 评测框架论文

| # | 论文 | 会议/来源 | 年份 | 说明 | 链接 |
|---|------|-----------|------|------|------|
| 1 | **JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models** | NeurIPS 2024 | 2024 | 标准化评测 | [arXiv](https://arxiv.org/abs/2404.01318) / [GitHub](https://github.com/JailbreakBench/jailbreakbench) |
| 2 | **HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal** | ICML 2024 | 2024 | 有害行为基准 | [arXiv](https://arxiv.org/abs/2402.04249) / [GitHub](https://github.com/centerforaisafety/HarmBench) |
| 3 | **SoK: Evaluating Jailbreak Guardrails for Large Language Models** | - | 2025 | 综合评测 | [arXiv](https://arxiv.org/html/2506.10597v2) |
| 4 | **JBShield: Defending Large Language Models from Jailbreak Attacks through Activated Concept Analysis and Manipulation** | USENIX Security 2025 | 2025 | 综合防御 | [arXiv](https://arxiv.org/abs/2412.02765) |

### 1.3 攻击方法论文

| # | 论文 | 会议/来源 | 年份 | 攻击类型 | 链接 |
|---|------|-----------|------|----------|------|
| 1 | **Universal and Transferable Adversarial Attacks on Aligned Language Models (GCG)** | - | 2023 | 梯度优化 | [arXiv](https://arxiv.org/abs/2307.15043) / [GitHub](https://github.com/llm-attacks/llm-attacks) |
| 2 | **AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned Large Language Models** | ICLR 2024 | 2024 | 遗传算法 | [arXiv](https://arxiv.org/abs/2310.04451) |
| 3 | **Tree of Attacks: Jailbreaking Black-Box LLMs Automatically (TAP)** | - | 2023 | 树形搜索 | [arXiv](https://arxiv.org/abs/2312.02119) |
| 4 | **FuzzLLM: A Novel and Universal Fuzzing Framework for Proactively Discovering Jailbreak Vulnerabilities** | - | 2023 | 模糊测试 | [arXiv](https://arxiv.org/abs/2309.05274) |
| 5 | **DrAttack: Prompt Decomposition and Reconstruction Makes Powerful LLM Jailbreakers** | - | 2024 | 隐式攻击 | [arXiv](https://arxiv.org/abs/2402.16914) |
| 6 | **ActorAttack: Black-box Adversarial Attack on LLMs with Multi-Turn Dialogues** | - | 2024 | 多轮攻击 | [arXiv](https://arxiv.org/abs/2402.05733) |

---

## 二、防御方法详细总结

### 2.1 GradSafe (ACL 2024)

**核心方法**：
- 分析LLM对输入的梯度响应
- 计算安全关键参数的梯度与不安全参考梯度的相似度
- 恶意提示会触发与不安全样本相似的梯度模式

**效果**：
- 在Llama-2上无需额外训练，优于Llama Guard
- 需要白盒访问（不适用于闭源API）

---

### 2.2 SmoothLLM (ICLR 2024)

**核心方法**：
- 对输入进行字符级随机扰动（交换、插入、删除）
- 多次采样后投票判断
- 对抗后缀对扰动敏感，正常输入对扰动鲁棒

**效果**：
- 对GCG攻击有效：ASR从~90%降至~10%
- **局限**：仅对优化类攻击有效，Mean-ASR=0.303（最弱）
- FPR较高

---

### 2.3 Llama Guard (Meta 2024)

**核心方法**：
- 微调Llama-2-7B作为安全分类器
- 输出6类有害类别

**效果**：
- 平衡性好，低延迟，低FPR
- 对复杂jailbreak检测能力有限

---

### 2.4 GuardReasoner (2025)

**核心方法**：
- 使用LLM进行深度推理分析输入意图
- 通过推理理解真实意图，而非表面特征

**效果**：
- **Mean-ASR=0.135（最强防御）**
- 延迟高，GPU显存占用大

---

### 2.5 PromptGuard (Meta 2024)

**核心方法**：
- 微调DeBERTa (86M参数) 作为输入安全分类器
- 轻量级分类器快速过滤

**效果**：
- **综合评分最高**，高效低延迟，低FPR
- 检测鲁棒性一般，对复杂攻击效果有限

---

### 2.6 GradientCuff (2024)

**核心方法**：
- 计算refusal loss对输入的梯度范数
- jailbreak提示会导致异常高的梯度范数

**效果**：
- FPR=0.083 (AlpacaEval)
- 需要白盒访问，计算开销较大

---

### 2.7 SelfDefend (2024)

**核心方法**：
- 让目标LLM自己分析输入意图
- 利用LLM自身能力进行自我防护

**效果**：
- 平衡性较好
- FPR较高 (OR-Bench: 0.221)

---

### 2.8 Perplexity Filter (2023)

**核心方法**：
- 计算输入的困惑度(PPL)
- GCG等攻击生成的后缀PPL异常高

**效果**：
- 对GCG有效，效率极高
- **仅对不可读对抗后缀有效**，对语义攻击无效

---

## 三、效果对比总结

### 3.1 ASR (Attack Success Rate)

| 方法 | Mean-ASR | 排名 |
|------|----------|------|
| **GuardReasoner (Pre)** | **0.135** | 1 |
| GuardReasoner (Post) | 0.141 | 2 |
| SelfDefend (Intent) | ~0.15 | 3 |
| WildGuard | ~0.18 | 4 |
| Llama Guard | ~0.20 | 5 |
| PromptGuard | ~0.25 | 6 |
| Perplexity Filter | ~0.28 | 7 |
| **SmoothLLM** | **0.303** | 8 |

### 3.2 综合评价

| 方法 | ASR↓ | FPR↓ | 效率 | 适用场景 |
|------|------|------|------|----------|
| GuardReasoner | ⭐⭐⭐ | ⭐⭐ | ⭐ | 高安全要求 |
| PromptGuard | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | 生产部署 |
| Llama Guard | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | 通用场景 |
| GradSafe | ⭐⭐ | ⭐⭐ | ⭐⭐ | 白盒场景 |
| SmoothLLM | ⭐ | ⭐ | ⭐⭐ | 仅GCG攻击 |

---

## 四、数据集汇总

### 4.1 攻击数据集

| 数据集 | 样本数 | 说明 | 链接 |
|--------|--------|------|------|
| **JailbreakBench** | 100 | 100个恶意行为 | [GitHub](https://github.com/JailbreakBench/jailbreakbench) |
| **AdvBench** | 520 | 经典对抗攻击 | [GitHub](https://github.com/llm-attacks/llm-attacks) |
| **HarmBench** | 400+ | 有害行为基准 | [GitHub](https://github.com/centerforaisafety/HarmBench) |
| **JailbreakHub** | 1000+ | 野外jailbreak | [GitHub](https://github.com/verazuo/jailbreak_llms) |
| **SafeMTData** | 600 | 多轮攻击 | [HuggingFace](https://huggingface.co/datasets/SafeMTData) |
| **MultiJail** | 315 | 多语言攻击 | [GitHub](https://github.com/DAMO-NLP-SG/MultiJail) |

### 4.2 正常数据集

| 数据集 | 样本数 | 说明 | 链接 |
|--------|--------|------|------|
| **AlpacaEval** | 805 | 指令跟随 | [GitHub](https://github.com/tatsu-lab/alpaca_eval) |
| **OR-Bench** | 80,000 | 过度拒绝测试 | [HuggingFace](https://huggingface.co/datasets/allenai/or-bench) |
| **XSTest** | 450 | 安全边界 | - |

---

## 五、目录结构

```
related_work/
├── README.md                      # 本文件
├── defense_methods_comparison.md  # 防御方法详细对比
└── papers/                        # 论文PDF (可选)
```

---

*最后更新: 2026-02-05*
