# V7.1.3 Prompt调优测试报告

**测试日期**: 2026-02-05 15:11:28

## Prompt配置对比

| Prompt | 描述 | 准确率 | ASR↓ | FPR↓ |
|--------|------|--------|------|------|
| baseline | 原始平衡版Prompt | 70.00% | 0.253 | 0.440 |
| jailbreak_aware | Jailbreak敏感版 | 75.50% | 0.080 | 0.740 |
| strict_security | 严格安全版 | 66.50% | 0.360 | 0.260 |
| pattern_focused | 模式识别版 | 67.00% | 0.353 | 0.260 |

## 推荐配置: **pattern_focused** (ASR+FPR最小)
