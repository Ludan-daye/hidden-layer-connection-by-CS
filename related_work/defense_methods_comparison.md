# LLM越狱防御方法对比研究

## 1. 现有防御方法分类

### 1.1 输入过滤类 (Input Filtering)

| 方法 | 原理 | 优点 | 缺点 | 论文链接 |
|-----|-----|-----|-----|----------|
| **Perplexity Filter** | 计算输入困惑度，高困惑度视为攻击 | 延迟低，无需额外模型 | 对低困惑度攻击无效 | [arXiv](https://arxiv.org/abs/2308.14132) |
| **PromptGuard** | 基于DeBERTa的分类器检测 | 轻量级(86M参数)，延迟低 | 检测鲁棒性有限 | [HuggingFace](https://huggingface.co/meta-llama/Prompt-Guard-86M) |
| **SmoothLLM** | 对输入进行字符级扰动 | 对GCG等后缀攻击有效 | 对其他攻击类型效果差 | [ICLR 2024](https://openreview.net/forum?id=xq7h9nfdY2) |

### 1.2 LLM-based检测器

| 方法 | 原理 | 优点 | 缺点 | 论文链接 |
|-----|-----|-----|-----|----------|
| **LlamaGuard** | 基于Llama的安全分类器 | 多任务支持 | 延迟高，资源消耗大 | [arXiv](https://arxiv.org/abs/2312.06674) / [HuggingFace](https://huggingface.co/meta-llama/LlamaGuard-7b) |
| **WildGuard** | 7B参数安全审核模型 | F1分数高，超越GPT-4 | 需要大模型推理 | [arXiv](https://arxiv.org/abs/2406.18495) |
| **GuardReasoner** | 带推理的安全判断 | ASR最低(0.135) | 延迟最高，GPU占用大 | [arXiv](https://arxiv.org/abs/2501.18492) |

### 1.3 梯度分析类

| 方法 | 原理 | 优点 | 缺点 | 论文链接 |
|-----|-----|-----|-----|----------|
| **GradSafe** | 安全关键梯度相似度分析 | 无需额外训练 | 需要白盒访问 | [ACL 2024](https://aclanthology.org/2024.acl-long.30/) / [GitHub](https://github.com/xyq7/GradSafe) |
| **GradientCuff** | Refusal loss梯度范数检测 | FPR低(0.083) | 计算开销大 | [arXiv](https://arxiv.org/abs/2403.00867) |

### 1.4 Embedding-based检测器

| 方法 | 原理 | 优点 | 缺点 |
|-----|-----|-----|-----|
| **NeMo Guard** | 随机森林+预训练Embedding | 轻量级 | 需要训练数据 |
| **Embedding+RF** | Snowflake Embedding+随机森林 | F1=0.96，FPR<1% | 需要特定Embedding |
| **我们的方法(V7.1)** | BGE微调+LLM灰色地带判断 | 准确率94.83%，延迟低 | 需要微调Embedding |

---

## 2. 防御效果数据统计

### 2.1 JailbreakBench基准测试 (SoK论文)

| 防御方法 | ASR↓ | FPR↓ | 延迟 | GPU内存 |
|---------|------|------|-----|--------|
| GuardReasoner (Pre) | **0.135** | 0.08 | 最高 | 最高 |
| GuardReasoner (Post) | 0.14 | 0.05 | 高 | 高 |
| WildGuard (Pre) | 0.18 | 0.07 | 中 | 中 |
| Llama Guard | 0.22 | 0.03 | 低 | 中 |
| PromptGuard | 0.25 | **0.02** | **最低** | **最低** |
| SmoothLLM | 0.303 | 0.06 | 低 | 低 |
| Perplexity Filter | 0.28 | 0.04 | 最低 | 最低 |

*ASR = Attack Success Rate (越低越好)*
*FPR = False Positive Rate (越低越好)*

### 2.2 Embedding-based检测器对比 (arxiv:2412.01547)

| 方法 | F1 Score | FPR | 数据集 |
|-----|----------|-----|-------|
| **RF+Snowflake Embedding** | **0.9601** | **<1%** | JailbreakHub |
| PromptGuard | 0.3029 | >50% | JailbreakHub |
| gpt-3.5-turbo | 0.25 | >50% | JailbreakHub |
| gelectra-base-injection | 0.18 | >33% | ToxicChat |
| deberta-v3-base-injection | 0.15 | >33% | ToxicChat |

### 2.3 WildGuard评估结果 (arxiv:2406.18495)

| 任务 | WildGuard | GPT-4 | LlamaGuard2 | 最佳开源 |
|-----|-----------|-------|-------------|---------|
| Prompt分类 (F1) | **0.89** | 0.87 | 0.78 | 0.78 |
| Response分类 (F1) | 0.85 | 0.86 | 0.80 | 0.82 |
| Refusal检测 (F1) | **0.91** | 0.95 | - | 0.65 |

---

## 3. 我们的方法 vs 现有方法

### 3.1 性能对比

| 指标 | 我们的V4 | PromptGuard | WildGuard | GuardReasoner |
|-----|---------|-------------|-----------|---------------|
| **检出率** | **95.9%** | ~75% | ~89% | ~86% |
| **误报率** | **2.2%** | 2% | 7% | 8% |
| **延迟** | **<1ms** | <10ms | ~500ms | >1s |
| **模型大小** | **19维投影** | 86M | 7B | 8B |
| **GPU需求** | **无** | 低 | 高 | 最高 |

### 3.2 优势分析

1. **极低延迟**：仅需Embedding计算+矩阵乘法，无需LLM推理
2. **低资源消耗**：投影矩阵仅384×19=7,296参数
3. **高检出率**：95.9%总体检出率，越狱攻击97%
4. **低误报率**：2.2%，优于大多数LLM-based方法
5. **可解释性**：基于余弦相似度，阈值可调

### 3.3 劣势分析

1. 需要训练数据构建投影矩阵
2. 对未见过的攻击类型泛化能力待验证
3. 无法提供细粒度的风险分类

---

## 4. 关键论文

### 4.1 防御方法

| # | 论文 | 会议 | 链接 |
|---|------|------|------|
| 1 | GradSafe: Detecting Jailbreak Prompts for LLMs via Safety-Critical Gradient Analysis | ACL 2024 | [ACL](https://aclanthology.org/2024.acl-long.30/) |
| 2 | SmoothLLM: Defending Large Language Models Against Jailbreaking Attacks | ICLR 2024 | [OpenReview](https://openreview.net/forum?id=xq7h9nfdY2) |
| 3 | Llama Guard: LLM-based Input-Output Safeguard | Meta 2024 | [arXiv](https://arxiv.org/abs/2312.06674) |
| 4 | WildGuard: Open One-Stop Moderation Tools | Allen AI 2024 | [arXiv](https://arxiv.org/abs/2406.18495) |
| 5 | GuardReasoner: Towards Reasoning-based LLM Safeguards | 2025 | [arXiv](https://arxiv.org/abs/2501.18492) |
| 6 | Gradient Cuff: Detecting Jailbreak Attacks | 2024 | [arXiv](https://arxiv.org/abs/2403.00867) |
| 7 | SelfDefend: LLMs Can Defend Themselves | 2024 | [arXiv](https://arxiv.org/abs/2406.05498) |

### 4.2 评测框架

| # | 论文 | 会议 | 链接 |
|---|------|------|------|
| 1 | JailbreakBench: An Open Robustness Benchmark | NeurIPS 2024 | [arXiv](https://arxiv.org/abs/2404.01318) / [GitHub](https://github.com/JailbreakBench/jailbreakbench) |
| 2 | HarmBench: A Standardized Evaluation Framework | ICML 2024 | [arXiv](https://arxiv.org/abs/2402.04249) / [GitHub](https://github.com/centerforaisafety/HarmBench) |
| 3 | SoK: Evaluating Jailbreak Guardrails for LLMs | 2025 | [arXiv](https://arxiv.org/html/2506.10597v2) |
| 4 | JBShield: Defending LLMs from Jailbreak Attacks | USENIX Sec 2025 | [arXiv](https://arxiv.org/abs/2412.02765) |

### 4.3 攻击方法

| # | 论文 | 攻击类型 | 链接 |
|---|------|----------|------|
| 1 | Universal and Transferable Adversarial Attacks (GCG) | 梯度优化 | [arXiv](https://arxiv.org/abs/2307.15043) |
| 2 | AutoDAN: Generating Stealthy Jailbreak Prompts | 遗传算法 | [arXiv](https://arxiv.org/abs/2310.04451) |
| 3 | Tree of Attacks (TAP) | 树形搜索 | [arXiv](https://arxiv.org/abs/2312.02119) |
| 4 | DrAttack: Prompt Decomposition | 隐式攻击 | [arXiv](https://arxiv.org/abs/2402.16914) |
| 5 | ActorAttack: Multi-Turn Dialogues | 多轮攻击 | [arXiv](https://arxiv.org/abs/2402.05733) |

---

## 5. 测试数据集统计

### 5.1 恶意/攻击数据集

| 数据集 | 样本数 | 类型 | 使用论文 |
|-------|-------|-----|---------|
| **AdvBench** | 520 | 直接恶意指令 | JailbreakBench, WildGuard, SoK |
| **HarmBench** | 400 | 多样化恶意行为 | JailbreakBench, WildGuard |
| **JailbreakHub** | 1,405 | 野外越狱攻击 | Embedding Detection |
| **ToxicChat** | 10,164 | 有毒对话 | WildGuard, Embedding Detection |
| **JBB-Behaviors** | 100 | 越狱行为 | JailbreakBench |
| **GCG Attacks** | - | 对抗后缀攻击 | SoK, FJD |
| **AutoDAN** | - | 自动化越狱 | SoK, FJD |
| **PAIR** | - | 迭代越狱 | JailbreakBench |
| **Cipher** | - | 编码越狱 | FJD |
| **Hand-crafted** | 28种 | 手工越狱模板 | FJD |

### 5.2 正常/良性数据集

| 数据集 | 样本数 | 类型 | 使用论文 |
|-------|-------|-----|---------|
| **AlpacaEval** | 805 | 通用指令 | SoK, WildGuard |
| **OR-Bench** | 1,000+ | 边界问题 | SoK |
| **OpenPlatypus** | - | 良性对话 | FJD |
| **SuperGLUE** | - | NLU任务 | FJD |
| **PureDove** | - | 良性对话 | FJD |
| **XSTest** | 450 | 安全边界测试 | WildGuard |

### 5.3 各论文使用的数据集

| 论文 | 恶意数据集 | 正常数据集 |
|-----|-----------|-----------|
| **JailbreakBench** | AdvBench, HarmBench, JBB-Behaviors | - |
| **WildGuard** | ToxicChat, HarmBench, AdvBench | AlpacaEval, XSTest |
| **SoK** | GCG, AutoDAN, PAIR, TAP, DeepInception | AlpacaEval, OR-Bench |
| **Embedding Detection** | JailbreakHub, ToxicChat | 自定义良性集 |
| **FJD** | AdvBench, AutoDAN, Cipher, Hand-crafted | PureDove, OpenPlatypus, SuperGLUE |

### 5.4 我们使用的数据集

| 类型 | 数据集 | 样本数 |
|-----|-------|-------|
| **恶意** | AdvBench | 500 |
| **恶意** | HarmBench | 400 |
| **恶意** | BeaverTails | 500 |
| **恶意** | JBB-GCG | 200 |
| **恶意** | JBB-PAIR | 86 |
| **恶意** | ToxicChat | 384 |
| **恶意** | MaliciousInstruct | 100 |
| **正常** | Alpaca | 1,500 |
| **正常** | XSTest | 250 |
| **正常** | OR-Bench | 1,319 |
| **正常** | TruthfulQA | 500 |
| **正常** | DoNotAnswer | 500 |
| **正常** | 安全教育样本 | 150 |

**总计**：恶意2,170条，正常4,219条

---

## 6. 结论

我们的Embedding-based方法在**检出率、误报率、延迟、资源消耗**四个维度均表现优异：

| 维度 | 排名 | 说明 |
|-----|-----|-----|
| 检出率 | **第1** | 95.9% vs WildGuard 89% |
| 误报率 | **第2** | 2.2% vs PromptGuard 2% |
| 延迟 | **第1** | <1ms vs PromptGuard 10ms |
| 资源 | **第1** | 无GPU vs PromptGuard 86M |

**推荐场景**：
- 高吞吐量API网关
- 边缘设备部署
- 成本敏感场景
- 需要实时检测的应用
