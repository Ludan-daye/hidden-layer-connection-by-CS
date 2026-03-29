# Related Work 素材：轻量前置分类器研究整理

> 供论文 Related Work / Experimental Comparison / Discussion 直接引用

---

## 一、各方法详细档案

### 1. NeMo Guard JailbreakDetect (NVIDIA)

- **论文**: "Improved Large Language Model Jailbreak Detection via Pretrained Embeddings", Galinkin & Sablotny, AICS 2025
- **arXiv**: 2412.01547
- **架构**: Snowflake Arctic Embed M Long (768d) → Random Forest
- **参数**: ~109M (嵌入模型) + RF (轻量)
- **训练数据**: 17,085 样本 (1,580 jailbreaks + 15,505 benign)
- **测试数据集及结果**:
  | 数据集 | F1 | FPR | FNR |
  |--------|-----|-----|-----|
  | JailbreakHub | **0.960** | 0.004 | 0.044 |
  | ToxicChat | 0.283 | 0.002 | — |
  | 自建测试集 | 0.833 | 0.009 | 0.219 |
  | 5-fold CV | 0.974 (avg) | — | — |
- **对比V7**: NeMo在JailbreakHub上F1远优于V7(DR=64.6%)，但其768d嵌入远大于V7的19d；NeMo未在AdvBench/HarmBench上评测

---

### 2. PromptGuard (Meta)

- **来源**: PurpleLlama项目，2024年7月开源
- **架构**: mDeBERTa-v3-base 微调
- **参数**: 86M (backbone) + 192M (词嵌入)
- **Meta自报基准**:
  | 评测集 | TPR | FPR | AUC |
  |--------|-----|-----|-----|
  | 越狱 (内分布) | 99.9% | 0.4% | 0.997 |
  | 注入 (内分布) | 99.5% | 0.8% | 1.000 |
  | OOD 越狱 | 97.5% | 3.9% | 0.975 |
- **独立评测 (arXiv:2412.01547)**:
  | 数据集 | F1 | FPR |
  |--------|-----|-----|
  | JailbreakHub | **0.303** | **50.7%** |
  | ToxicChat | 0.529 | 2.0% |
- **已知漏洞**: 99.8% 白字符间距绕过率 (Robust Intelligence, 2024)
- **对比V7**: PG1在JailbreakHub上FPR=50.7%远差于V7(FPR=34.0% on JBB-Benign)；内分布AUC=0.997但OOD退化严重

---

### 3. PromptGuard 2 (Meta, LlamaFirewall)

- **论文**: "LlamaFirewall: An Open Source Guardrail System for Building Secure AI Agents", arXiv:2505.03574, 2025
- **架构**: mDeBERTa-base (86M) / DeBERTa-xsmall (22M)
- **改进**: 二分类 (benign/malicious)，能量损失函数，对抗分词修复
- **基准**:
  | 指标 | PG2-86M | PG2-22M |
  |------|---------|---------|
  | AUC (英文) | 0.980 | **0.995** |
  | Recall @ 1% FPR | **97.5%** | 88.7% |
  | 延迟 | 92.4ms | **19.3ms** |
- **AgentDojo**: ASR 17.63% → 7.53% (PG2单独) → 1.75% (PG2+AlignmentCheck)
- **对比V7**: PG2是V7最强竞品；PG2-22M的19ms速度接近V7；但V7的19d投影维度更小，且不需要微调完整transformer

---

### 4. InjecGuard

- **论文**: "InjecGuard: Benchmarking and Mitigating Over-defense in Prompt Injection Guardrail Models", arXiv:2410.22770, 2024
- **架构**: DeBERTa-v3-base + NotInject训练数据
- **参数**: 184M
- **推理**: 15.34ms, 60.45 GFLOPs
- **基准**:
  | 数据集 | 准确率 |
  |--------|--------|
  | NotInject (1词) | 91.15% |
  | NotInject (2词) | 89.38% |
  | NotInject (3词) | 81.42% |
  | WildGuard (benign) | 76.11% |
  | PINT (injection) | 95.36% |
  | PINT (benign) | 86.43% |
  | BIPIA | 90.90% |
  | **平均** | **83.48%** |
- **对比V7**: InjecGuard聚焦prompt injection（非越狱），184M参数是V7投影层的数千倍；两者侧重点不同

---

### 5. GradSafe (ACL 2024)

- **论文**: "GradSafe: Detecting Jailbreak Prompts for LLMs via Safety-Critical Gradient Analysis", ACL 2024
- **arXiv**: 2402.13494
- **方法**: 计算目标LLM对"Sure."的梯度 → 与安全集梯度的余弦相似度 → 零样本检测
- **参数**: 0（无额外模型，使用目标LLM自身梯度）
- **基准**:
  | 数据集 | Precision | Recall | F1 | AUPRC |
  |--------|-----------|--------|-----|-------|
  | ToxicChat | 0.753 | 0.667 | **0.707** | 0.755 |
  | XSTest | 0.856 | 0.950 | **0.900** | 0.936 |
- **注意**: 常被误引为"F1=0.98 on GCG"，实际论文Table 3中无此数据
- **SoK评测**: 平均ASR=0.224（排第5，不如PromptGuard的0.163）
- **已知失效**: SAA, DrAttack, Zulu, Base64攻击完全失效 (JBShield, USENIX 2025)
- **对比V7**: GradSafe需要目标LLM梯度（非独立部署），V7是独立嵌入分类器；GradSafe在XSTest上F1=0.90更优，但无法脱离LLM运行

---

### 6. Gradient Cuff (NeurIPS 2024, IBM)

- **论文**: "Gradient Cuff: Detecting Jailbreak Attacks on LLMs by Exploring Refusal Loss Landscapes", NeurIPS 2024
- **arXiv**: 2403.00867
- **方法**: (1) 拒绝损失阈值 (2) 梯度范数过滤，两步检测
- **参数**: 0（无额外模型）
- **基准 (LLaMA-2-7B-Chat)**:
  | 攻击 | ASR (防御后) |
  |------|------------|
  | GCG | **1.2%** |
  | AutoDAN | 15.8% |
  | PAIR | 23.0% |
  | TAP | 5.0% |
  | Base64 | 19.8% |
  | LRL | 5.4% |
  | **平均** | **11.7%** |
- FPR: LLaMA=2.2%, Vicuna=3.4%
- **SoK评测**: 平均ASR=0.148（轻量方法中最优）
- **对比V7**: GCuff在GCG上ASR=1.2%远优于V7(16%)，但完全依赖目标LLM梯度无法独立部署

---

### 7. Perplexity Filter + LightGBM

- **论文**: "Detecting Language Model Attacks with Perplexity", arXiv:2308.14132, 2023
- **方法**: GPT-2计算困惑度 + token长度 → LightGBM二分类
- **参数**: ~0（仅使用现有GPT-2）
- **基准**:
  - 机器生成攻击TP: 96.2% (355/369), FP=0
  - 人工编写越狱: **0% detection** (0/23)
  - F2-score: 94.2% (全部), 99.1% (排除人工)
- **SoK评测**: 平均ASR=0.239（几乎等同无防御）
- **对比V7**: 困惑度仅对GCG后缀有效，语义攻击完全失效；V7虽然PAIR弱(DR=48%)但远优于困惑度的0%

---

## 二、标准数据集跨论文使用情况

| 数据集 | 样本量 | 使用论文 | 类型 |
|--------|--------|---------|------|
| **AdvBench** | 520 | GradSafe, Perplexity, HarmBench | 直接有害指令 |
| **ToxicChat** | 10,166 | NeMo, PG, GradSafe, SoK | 真实用户对话 |
| **JailbreakHub** | 1,405 | NeMo, PG, SoK | 手动越狱模板 |
| **XSTest** | 450 (250安全+200不安全) | GradSafe, SoK | 过度拒绝测试 |
| **HarmBench** | 1,528 behaviors | HarmBench, SoK | 有害行为分类 |
| **JailbreakBench** | 200 (100害+100良) | SoK, 本项目 | GCG/PAIR攻击 |
| **BeaverTails** | 3,820 | 本项目 | 多类别有害 |
| **Alpaca** | 52,002 | 本项目 | 常规指令 |
| **PINT** | — | InjecGuard, ProtectAI | 注入检测 |
| **BIPIA** | — | InjecGuard | 后门注入 |
| **AgentDojo** | — | PG2, LlamaFirewall | Agent攻击 |

---

## 三、V7 论文定位建议

### Related Work 段落框架

> 现有轻量前置分类方法可分为三类：
>
> (1) **嵌入+传统ML**: NeMo Guard [1] 使用768维嵌入+随机森林，在JailbreakHub上达到F1=0.960，但嵌入维度高，存储开销大。
>
> (2) **微调Transformer**: PromptGuard [Meta 2024] 微调mDeBERTa-v3-base (86M参数)，内分布AUC=0.997但OOD退化至F1=0.303 [1]。PromptGuard 2 [2] 引入能量损失函数改进OOD泛化，Recall@1%FPR达97.5%。InjecGuard [6] 在DeBERTa基础上解决过度防御问题。
>
> (3) **目标LLM信号**: GradSafe [4] 和Gradient Cuff [3] 利用目标LLM自身梯度/损失做检测，GCuff在SoK统一评测中取得轻量方法最低ASR(0.148) [7]，但完全依赖目标LLM，无法独立部署。
>
> 本工作提出学习型投影方法，将384维BGE嵌入压缩至19维，在保持直接有害指令检测率(AdvBench DR=85%, HarmBench DR=82%)的同时，实现了所有方法中最小的表征空间和最低的推理延迟(<10ms)。

### Discussion 对比要点

1. **压缩率优势**: V7使用19维投影（5%的原始维度），而NeMo使用完整768维，PromptGuard需要86M参数的完整transformer推理。V7的极致压缩证明了安全分类信息可以在极低维空间中表达。

2. **直接有害指令**: V7在AdvBench(85%)和HarmBench(82%)上的表现是轻量方法中唯一报告的baseline。这填补了该领域直接有害指令检测的评测空白。

3. **Alpaca误报率**: V7的1.5% FPR优于PromptGuard的50.7%(JailbreakHub), 说明V7对正常用户请求几乎透明。

4. **诚实说明局限**: PAIR攻击(DR=47.7%)和JBB-Benign(FPR=34%)是已知弱点。这与所有嵌入方法面临的共性挑战一致——语义伪装攻击在嵌入空间中与良性请求难以区分。

---

## 四、参考文献

1. Galinkin & Sablotny, "Improved Large Language Model Jailbreak Detection via Pretrained Embeddings", AICS 2025, arXiv:2412.01547
2. Meta, "LlamaFirewall: An Open Source Guardrail System for Building Secure AI Agents", arXiv:2505.03574, 2025
3. Hu, Chen & Ho, "Gradient Cuff: Detecting Jailbreak Attacks on LLMs by Exploring Refusal Loss Landscapes", NeurIPS 2024, arXiv:2403.00867
4. Xie et al., "GradSafe: Detecting Jailbreak Prompts for LLMs via Safety-Critical Gradient Analysis", ACL 2024, arXiv:2402.13494
5. Alon & Kamfonas, "Detecting Language Model Attacks with Perplexity", arXiv:2308.14132, 2023
6. InjecGuard, "Benchmarking and Mitigating Over-defense in Prompt Injection Guardrail Models", arXiv:2410.22770, 2024
7. SoK, "Evaluating Jailbreak Guardrails in Large Language Models", arXiv:2506.10597, 2025
