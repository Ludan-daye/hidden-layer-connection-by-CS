# LLM输入防御Baseline方法整理

整理2024-2025年顶刊/顶会LLM输入防御方法的数据集、测试模型和效果。

---

## 一、Baseline方法汇总

### 1.1 预处理防御 (Pre-processing)

| 方法 | 会议/来源 | 年份 | 原理 | 论文链接 |
|------|-----------|------|------|----------|
| **GradSafe** | ACL 2024 | 2024 | 分析安全关键梯度检测jailbreak | [ACL Anthology](https://aclanthology.org/2024.acl-long.30/) / [GitHub](https://github.com/xyq7/GradSafe) |
| **Llama Guard** | Meta | 2024 | 微调Llama-2-7B做毒性分类 | [arXiv](https://arxiv.org/abs/2312.06674) / [HuggingFace](https://huggingface.co/meta-llama/LlamaGuard-7b) |
| **PromptGuard** | Meta | 2024 | 微调预训练分类器评估输入安全 | [HuggingFace](https://huggingface.co/meta-llama/Prompt-Guard-86M) |
| **WildGuard** | Allen AI | 2024 | 针对野外对话的防护 | [arXiv](https://arxiv.org/abs/2406.18495) / [HuggingFace](https://huggingface.co/allenai/wildguard) |
| **GuardReasoner** | - | 2025 | LLM推理分析输入意图 | [arXiv](https://arxiv.org/abs/2501.18492) |
| **Perplexity Filter** | - | 2023 | 计算输入困惑度检测对抗后缀 | [arXiv](https://arxiv.org/abs/2308.14132) |
| **OpenAI Moderation** | OpenAI | 2023 | API形式的内容审核 | [OpenAI API](https://platform.openai.com/docs/guides/moderation) |

### 1.2 处理中防御 (Intra-processing)

| 方法 | 会议/来源 | 年份 | 原理 | 论文链接 |
|------|-----------|------|------|----------|
| **GradientCuff** | - | 2024 | 比较refusal loss梯度范数 | [arXiv](https://arxiv.org/abs/2403.00867) |
| **Token Highlighter** | - | 2024 | 定位jailbreak关键token | [arXiv](https://arxiv.org/abs/2405.02492) |

### 1.3 动态防御 (Dynamic)

| 方法 | 会议/来源 | 年份 | 原理 | 论文链接 |
|------|-----------|------|------|----------|
| **SmoothLLM** | ICLR 2024 | 2024 | 字符级扰动防御 | [OpenReview](https://openreview.net/forum?id=xq7h9nfdY2) / [arXiv](https://arxiv.org/abs/2310.03684) |
| **SelfDefend** | - | 2024 | 让LLM自己判断输入意图 | [arXiv](https://arxiv.org/abs/2406.05498) |

### 1.4 综合框架与评测

| 方法 | 会议/来源 | 年份 | 说明 | 论文链接 |
|------|-----------|------|------|----------|
| **JBShield** | USENIX Security 2025 | 2025 | 综合防御框架 | [arXiv](https://arxiv.org/abs/2412.02765) |
| **JailbreakBench** | NeurIPS 2024 | 2024 | 标准化评测框架 | [arXiv](https://arxiv.org/abs/2404.01318) / [GitHub](https://github.com/JailbreakBench/jailbreakbench) |
| **HarmBench** | ICML 2024 | 2024 | 有害行为基准 | [arXiv](https://arxiv.org/abs/2402.04249) / [GitHub](https://github.com/centerforaisafety/HarmBench) |
| **SoK: Evaluating Jailbreak Guardrails** | - | 2025 | 综合评测论文 | [arXiv](https://arxiv.org/html/2506.10597v2) |

---

## 二、Baseline使用的数据集

### 2.1 攻击数据集 (Harmful)

| 数据集 | 样本数 | 来源 | 说明 | 链接 |
|--------|--------|------|------|------|
| **JailbreakBench** | 100 | JailbreakBench | 100个恶意行为 | [GitHub](https://github.com/JailbreakBench/jailbreakbench) |
| **AdvBench** | 520 | llm-attacks | 经典对抗攻击数据集 | [GitHub](https://github.com/llm-attacks/llm-attacks) |
| **HarmBench** | 400+ | HarmBench | 有害行为基准 | [GitHub](https://github.com/centerforaisafety/HarmBench) |
| **JailbreakHub (IJP)** | 1000+ | JailbreakHub | 野外jailbreak提示 | [GitHub](https://github.com/verazuo/jailbreak_llms) |
| **SafeMTData** | 600 | ActorAttack | 多轮jailbreak攻击 | [HuggingFace](https://huggingface.co/datasets/SafeMTData) |
| **MultiJail** | 315 | MultiJail | 多语言jailbreak | [GitHub](https://github.com/DAMO-NLP-SG/MultiJail) |

### 2.2 正常数据集 (Benign)

| 数据集 | 样本数 | 来源 | 说明 | 链接 |
|--------|--------|------|------|------|
| **AlpacaEval** | 805 | AlpacaEval | 指令跟随评测 | [GitHub](https://github.com/tatsu-lab/alpaca_eval) |
| **OR-Bench** | 80,000 | OR-Bench | 看似有害但良性的提示 | [HuggingFace](https://huggingface.co/datasets/allenai/or-bench) |

### 2.3 攻击方法分类

| 类型 | 攻击方法 | 说明 | 论文链接 |
|------|----------|------|----------|
| 优化攻击 | GCG | 基于梯度优化对抗后缀 | [arXiv](https://arxiv.org/abs/2307.15043) |
| 优化攻击 | AutoDAN | 自动生成对抗提示 | [arXiv](https://arxiv.org/abs/2310.04451) |
| 生成攻击 | TAP | 树形攻击提示 | [arXiv](https://arxiv.org/abs/2312.02119) |
| 生成攻击 | LLM-Fuzzer | 使用LLM模糊测试 | [arXiv](https://arxiv.org/abs/2309.05274) |
| 隐式攻击 | DrAttack | 隐藏恶意意图 | [arXiv](https://arxiv.org/abs/2402.16914) |
| 多轮攻击 | ActorAttack | 多轮对话攻击 | [arXiv](https://arxiv.org/abs/2402.05733) |

---

## 三、Baseline测试模型

### 3.1 目标LLM

| 模型 | 参数量 | 类型 | 链接 |
|------|--------|------|------|
| **Llama-3-8B-Instruct** | 8B | 开源 | [HuggingFace](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) |
| **Vicuna-13b-v1.5** | 13B | 开源 | [HuggingFace](https://huggingface.co/lmsys/vicuna-13b-v1.5) |
| **GPT-4-0125-Preview** | - | 闭源 | [OpenAI](https://platform.openai.com/docs/models/gpt-4) |
| **Llama-2-7B-chat** | 7B | 开源 | [HuggingFace](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) |

### 3.2 防御模型

| 方法 | 基础模型 | 参数量 | 链接 |
|------|----------|--------|------|
| Llama Guard | Llama-2-7B | 7B | [HuggingFace](https://huggingface.co/meta-llama/LlamaGuard-7b) |
| GradSafe | Llama-2 | 7B/13B | [GitHub](https://github.com/xyq7/GradSafe) |
| PromptGuard | DeBERTa | 86M | [HuggingFace](https://huggingface.co/meta-llama/Prompt-Guard-86M) |

---

## 四、Baseline效果对比

> 数据来源: [SoK: Evaluating Jailbreak Guardrails for Large Language Models](https://arxiv.org/html/2506.10597v2)

### 4.1 ASR (Attack Success Rate) - 越低越好

| 方法 | Mean-ASR | 说明 |
|------|----------|------|
| **GuardReasoner (Pre)** | **0.135** | 最强防御 |
| GuardReasoner (Post) | 0.141 | |
| SelfDefend (Intent) | ~0.15 | |
| Llama Guard | ~0.20 | |
| PromptGuard | ~0.25 | |
| **SmoothLLM** | **0.303** | 最弱防御，仅对GCG有效 |

### 4.2 FPR (False Positive Rate) - 越低越好

| 方法 | FPR (AlpacaEval) | FPR (OR-Bench) |
|------|------------------|----------------|
| PromptGuard | 低 | 低 |
| Llama Guard | 低 | 低 |
| GradientCuff | **0.083** | - |
| SelfDefend (Direct) | - | **0.221** |
| SmoothLLM | 高 | 高 |

### 4.3 效率对比

| 方法 | 延迟 | GPU显存 | 说明 |
|------|------|---------|------|
| Perplexity Filter | 极低 | 极低 | 最高效 |
| Llama Guard | 低 | 中 | 平衡 |
| PromptGuard | 低 | 低 | 高效 |
| SmoothLLM | 中 | 极低 | |
| GuardReasoner | **高** | **高** | 推理开销大 |
| GradientCuff | 高 | 低 | 需要梯度计算 |

### 4.4 综合排名 (Composite Score)

| 排名 | 方法 | 特点 |
|------|------|------|
| 1 | PromptGuard | 高效但检测鲁棒性一般 |
| 2 | Llama Guard | 平衡 |
| 3 | SelfDefend (Intent) | 平衡但FPR较高 |
| 4 | GuardReasoner (Pre) | 最强防御但效率低 |

---

## 五、关键参考文献

1. **GradSafe** - Xie et al., "GradSafe: Detecting Jailbreak Prompts for LLMs via Safety-Critical Gradient Analysis", ACL 2024
2. **SmoothLLM** - Robey et al., "SmoothLLM: Defending Large Language Models Against Jailbreaking Attacks", ICLR 2024
3. **Llama Guard** - Inan et al., "Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations", Meta 2024
4. **JailbreakBench** - Chao et al., "JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models", NeurIPS 2024
5. **HarmBench** - Mazeika et al., "HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal", ICML 2024
6. **GCG Attack** - Zou et al., "Universal and Transferable Adversarial Attacks on Aligned Language Models", 2023
7. **SoK Paper** - "SoK: Evaluating Jailbreak Guardrails for Large Language Models", 2025

---

*文档生成日期: 2026-02-05*
