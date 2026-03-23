# V5 模型数据集设计

## 目标

构建一个覆盖尽可能多攻击方式和正常交流方式的数据集，用于训练更鲁棒的恶意检测模型。

---

## 一、恶意数据集

### 1. 直接恶意指令

| 数据集 | 来源 | 样本数 | 说明 |
|--------|------|--------|------|
| AdvBench | Zou et al. 2023 | 520 | 直接有害行为指令 |
| HarmBench | Mazeika et al. 2024 | 400 | 多样化恶意行为 |
| MaliciousInstruct | Huang et al. 2023 | 100 | 恶意指令集 |
| Do-Not-Answer | Wang et al. 2023 | 939 | 不应回答的问题 |

### 2. 越狱攻击 (Jailbreak)

| 数据集 | 来源 | 样本数 | 攻击类型 |
|--------|------|--------|----------|
| JailbreakHub | Shen et al. 2024 | 1,405 | 78种越狱模板 |
| JailbreakBench-GCG | Chao et al. 2024 | 200 | GCG后缀攻击 |
| JailbreakBench-PAIR | Chao et al. 2024 | 100 | PAIR迭代攻击 |
| JailbreakBench-TAP | Chao et al. 2024 | 100 | TAP树攻击 |
| WildJailbreak | Jiang et al. 2024 | 262K | 野外越狱样本 |
| In-The-Wild Jailbreak | Shen et al. 2023 | 666 | 真实越狱提示 |
| ChatGPT-Jailbreak-Prompts | GitHub | 79 | 社区越狱模板 |

### 3. 对抗攻击 (Adversarial)

| 数据集 | 来源 | 样本数 | 攻击类型 |
|--------|------|--------|----------|
| GCG Attacks | Zou et al. 2023 | 500+ | 梯度优化后缀 |
| AutoDAN | Liu et al. 2023 | 100+ | 自动化越狱 |
| GPTFUZZER | Yu et al. 2023 | 1,000+ | 模糊测试攻击 |
| DeepInception | Li et al. 2023 | 100+ | 深度嵌套攻击 |
| ReNeLLM | Ding et al. 2023 | 100+ | 重写攻击 |

### 4. 有毒/不安全内容

| 数据集 | 来源 | 样本数 | 说明 |
|--------|------|--------|------|
| ToxicChat | Lin et al. 2023 | 10,166 | 真实有毒对话 |
| BeaverTails | Ji et al. 2023 | 330K | 道德伤害内容 |
| PKU-SafeRLHF | Ji et al. 2024 | 44K | 安全RLHF数据 |
| HH-RLHF (red-team) | Anthropic | 40K | 红队攻击数据 |
| RealToxicityPrompts | Gehman et al. 2020 | 100K | 有毒提示 |
| CivilComments | Borkan et al. 2019 | 2M | 有毒评论 |

### 5. 特殊攻击类型

| 数据集 | 来源 | 样本数 | 攻击类型 |
|--------|------|--------|----------|
| Prompt Injection | Perez et al. 2022 | 500+ | 提示注入 |
| Indirect Injection | Greshake et al. 2023 | 200+ | 间接注入 |
| Code Injection | 自建 | 200+ | 代码注入 |
| Role-play Attacks | 自建 | 300+ | 角色扮演攻击 |
| Base64/Cipher Attacks | 自建 | 200+ | 编码绕过 |

---

## 二、正常数据集

### 1. 通用指令/对话

| 数据集 | 来源 | 样本数 | 说明 |
|--------|------|--------|------|
| Alpaca | Stanford | 52K | 通用指令 |
| Dolly | Databricks | 15K | 多样化指令 |
| OpenAssistant | LAION | 161K | 助手对话 |
| ShareGPT | 社区 | 90K | 真实对话 |
| WizardLM | Microsoft | 70K | 复杂指令 |
| LIMA | Meta | 1K | 高质量对话 |

### 2. 安全边界测试

| 数据集 | 来源 | 样本数 | 说明 |
|--------|------|--------|------|
| XSTest | Röttger et al. 2023 | 450 | 安全边界测试 |
| OR-Bench | Cui et al. 2024 | 1,319 | 过度拒绝测试 |
| SafetyBench | Zhang et al. 2023 | 11K | 安全基准 |

### 3. 技术/专业问题

| 数据集 | 来源 | 样本数 | 说明 |
|--------|------|--------|------|
| 安全教育样本 | 自建 | 500+ | 网络安全、加密等 |
| StackOverflow | 社区 | 10K+ | 技术问答 |
| 医学问答 | 自建 | 500+ | 医学咨询 |
| 法律问答 | 自建 | 500+ | 法律咨询 |

### 4. 日常对话

| 数据集 | 来源 | 样本数 | 说明 |
|--------|------|--------|------|
| DailyDialog | Li et al. 2017 | 13K | 日常对话 |
| PersonaChat | Zhang et al. 2018 | 10K | 人格对话 |
| EmpatheticDialogues | Rashkin et al. 2019 | 25K | 共情对话 |
| BlendedSkillTalk | Smith et al. 2020 | 5K | 混合技能对话 |

---

## 三、数据集统计

### 恶意数据集总计

| 类别 | 数据集数 | 预估样本数 |
|------|---------|-----------|
| 直接恶意指令 | 4 | ~2,000 |
| 越狱攻击 | 7 | ~5,000 |
| 对抗攻击 | 5 | ~2,000 |
| 有毒内容 | 6 | ~50,000 (采样) |
| 特殊攻击 | 5 | ~1,500 |
| **总计** | **27** | **~60,500** |

### 正常数据集总计

| 类别 | 数据集数 | 预估样本数 |
|------|---------|-----------|
| 通用指令/对话 | 6 | ~50,000 (采样) |
| 安全边界测试 | 3 | ~13,000 |
| 技术/专业问题 | 4 | ~12,000 |
| 日常对话 | 4 | ~50,000 (采样) |
| **总计** | **17** | **~125,000** |

---

## 四、数据集获取方式

### 可直接下载 (HuggingFace)

```python
# 恶意数据集
"lmsys/toxic-chat"
"PKU-Alignment/BeaverTails"
"PKU-Alignment/PKU-SafeRLHF"
"Anthropic/hh-rlhf"
"allenai/real-toxicity-prompts"

# 正常数据集
"tatsu-lab/alpaca"
"databricks/databricks-dolly-15k"
"OpenAssistant/oasst1"
"li2017dailydialog/daily_dialog"
```

### 需要申请/下载

- JailbreakHub: https://github.com/verazuo/jailbreak_llms
- JailbreakBench: https://jailbreakbench.github.io/
- WildJailbreak: 需要申请
- HarmBench: https://github.com/centerforaisafety/HarmBench

### 需要自建

- 角色扮演攻击
- 编码绕过攻击
- 技术/专业问题

---

## 五、数据处理策略

### 1. 去重

- 基于文本哈希去重
- 基于Embedding相似度去重 (阈值0.95)

### 2. 平衡

- 恶意:正常 ≈ 1:2
- 各攻击类型均衡采样

### 3. 质量过滤

- 过滤过短文本 (<10字符)
- 过滤过长文本 (>2000字符)
- 过滤乱码/无意义文本

---

## 六、下一步

1. 下载所有可获取的数据集
2. 整理和清洗数据
3. 构建统一格式的训练集
4. 设计V5模型架构
5. 训练和评估
