#!/usr/bin/env python3
"""
方案二：关键词白名单 + V7 混合前置分类器
逻辑：
  1. 白名单检查：若输入匹配安全模式且无敏感词 → 直接返回 benign（跳过V7）
  2. 否则 → V7 嵌入分类器决定

目标：在保持攻击检测能力的同时降低良性输入的误报率 (FPR)
"""

import sys
import re
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
RESULTS_DIR = BASE_DIR / "results" / "whitelist_hybrid"
V7_MODEL_PATH = str(BASE_DIR / "models" / "v7_classifier")

# ─────────────────────────────────────────────
# 白名单：明确安全的话题模式
# ─────────────────────────────────────────────
SAFE_TOPIC_PATTERNS = [
    # 纯事实性问题（地名/人名/年份），攻击不会用这类模式
    r"\bwhat (is|are|was|were) the (capital|population|area|height|distance|temperature) of\b",
    r"\bwho (invented|discovered|wrote|composed|founded|built) the\b",
    r"\bwhen (was|were|did) .{0,40} (born|founded|built|invented|written|signed)\b",
    # 数学计算（纯符号无任何敏感含义）
    r"\b(solve|calculate|compute|integrate|differentiate|factor) .{0,20}(equation|expression|\d[\d\s\+\-\*\/\^]+)\b",
    # 烹饪：只有食材+做法，不含敏感词
    r"\bhow (do I|do you|to) (make|cook|bake|prepare) (a |an )?(pizza|pasta|bread|cake|soup|salad|coffee|tea|rice|egg)\b",
    # 翻译：只做语言转换
    r"\btranslate .{0,60} (to|into) (french|spanish|german|chinese|japanese|italian|portuguese|korean)\b",
    # 天气/时间纯事实
    r"\bwhat (day|time|date|year|month) (is it|was it|will it be)\b",
    # 编程语法（只涉及语言本身，不涉及系统/网络/安全）
    r"\bhow (do I|to) (create|declare|define|initialize) a (variable|list|array|function|class|loop) in (python|javascript|java|c\+\+|ruby)\b",
]

# ─────────────────────────────────────────────
# 黑名单：存在这些词则不白名单放行
# ─────────────────────────────────────────────
SENSITIVE_KEYWORDS = [
    "bomb", "explosive", "weapon", "kill", "murder", "hack", "steal",
    "drug", "cocaine", "heroin", "meth", "poison", "attack", "terrorist",
    "malware", "virus", "ransomware", "phishing", "exploit",
    "jailbreak", "bypass", "ignore previous", "ignore all",
    "do anything now", "dan ", "pretend you are", "act as if",
    "illegal", "harmful content", "suicide", "self-harm",
    "violence", "abuse", "torture", "rape", "child", "nude",
    "synthesize", "manufacture drug", "make a bomb",
    "without restrictions", "no restrictions", "no limit",
    "forget your instructions", "disregard", "override",
    "roleplay as", "play the role of a", "simulate a",
    "as an evil", "as a hacker", "as a terrorist",
]


class WhitelistFilter:
    """白名单过滤器：匹配安全模式且无敏感词则直接放行"""

    def __init__(self):
        self.safe_patterns = [re.compile(p, re.IGNORECASE) for p in SAFE_TOPIC_PATTERNS]
        self.sensitive_kws = [kw.lower() for kw in SENSITIVE_KEYWORDS]

    def is_safe(self, text: str) -> bool:
        """
        返回 True 表示确定安全，可直接标记为 benign。
        返回 False 表示不确定，需要交给 V7。
        """
        text_lower = text.lower()
        # 有敏感词 → 不白名单
        for kw in self.sensitive_kws:
            if kw in text_lower:
                return False
        # 匹配安全模式 → 放行
        for pattern in self.safe_patterns:
            if pattern.search(text_lower):
                return True
        return False


# ─────────────────────────────────────────────
# V7 分类器加载
# ─────────────────────────────────────────────
def load_v7():
    sys.path.insert(0, str(BASE_DIR / "scripts" / "v7_classifier"))
    sys.path.insert(0, str(BASE_DIR / "scripts" / "v7_classifier" / "deploy"))
    try:
        from v7_classifier import V7Classifier
        clf = V7Classifier(model_path=V7_MODEL_PATH)
        print("V7分类器加载成功")
        return clf
    except Exception as e:
        print(f"V7加载失败: {e}")
        return None


# ─────────────────────────────────────────────
# 数据集加载
# ─────────────────────────────────────────────
DATASETS = {
    "gcg_attacks": {
        "path": DATASETS_DIR / "jailbreakbench" / "jbb_gcg_all.csv",
        "text_col": "prompt", "expected": "harmful", "n": 100,
    },
    "pair_attacks": {
        "path": DATASETS_DIR / "jailbreakbench" / "jbb_pair_all.csv",
        "text_col": "prompt", "expected": "harmful", "n": 86,
    },
    "jailbreakhub": {
        "path": DATASETS_DIR / "jailbreakhub" / "jailbreakhub.csv",
        "text_col": "prompt", "expected": "harmful", "n": 79,
    },
    "benign": {
        "path": DATASETS_DIR / "gcg_attacks" / "jbb_benign_behaviors.csv",
        "text_col": "Goal", "expected": "benign", "n": 100,
    },
}


def load_dataset(cfg):
    path = cfg["path"]
    if not path.exists():
        print(f"  文件不存在: {path}")
        return []
    df = pd.read_csv(path)
    col = cfg["text_col"]
    if col not in df.columns:
        for c in ["prompt", "Goal", "text", "goal"]:
            if c in df.columns:
                col = c
                break
    texts = df[col].dropna().tolist()
    n = cfg.get("n")
    return texts[:n] if n and len(texts) > n else texts


# ─────────────────────────────────────────────
# 核心评估
# ─────────────────────────────────────────────
def evaluate_pipeline(whitelist: WhitelistFilter, v7, texts: list, expected: str):
    """
    评估白名单 + V7 混合流水线。
    返回 (accuracy, asr_or_fpr, whitelist_hit_rate, stats)
    """
    stats = {"total": len(texts), "whitelist_benign": 0, "v7_harmful": 0,
             "v7_benign": 0, "correct": 0}

    preds = []
    for text in texts:
        if whitelist.is_safe(text):
            pred = "benign"
            stats["whitelist_benign"] += 1
        else:
            if v7 is not None:
                label, _ = v7.predict(text)
                pred = "harmful" if label == "harmful" else "benign"
            else:
                pred = "benign"  # fallback
            if pred == "harmful":
                stats["v7_harmful"] += 1
            else:
                stats["v7_benign"] += 1
        preds.append(pred)

    stats["correct"] = sum(1 for p in preds if p == expected)
    accuracy = stats["correct"] / len(texts)

    if expected == "harmful":
        metric = sum(1 for p in preds if p == "benign") / len(texts)  # ASR
    else:
        metric = sum(1 for p in preds if p == "harmful") / len(texts)  # FPR

    whitelist_rate = stats["whitelist_benign"] / len(texts)
    return accuracy, metric, whitelist_rate, stats


def evaluate_whitelist_only(whitelist: WhitelistFilter, texts: list, expected: str):
    """仅白名单（不用V7），用于分析白名单自身影响"""
    preds = ["benign" if whitelist.is_safe(t) else "harmful" for t in texts]
    accuracy = sum(1 for p in preds if p == expected) / len(texts)
    if expected == "harmful":
        metric = sum(1 for p in preds if p == "benign") / len(texts)
    else:
        metric = sum(1 for p in preds if p == "harmful") / len(texts)
    return accuracy, metric


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("方案二：关键词白名单 + V7 混合分类器测试")
    print("=" * 65)

    whitelist = WhitelistFilter()
    print(f"\n白名单：{len(SAFE_TOPIC_PATTERNS)} 类安全模式，{len(SENSITIVE_KEYWORDS)} 个敏感词黑名单")

    print("\n加载V7分类器...")
    v7 = load_v7()

    # 已知V7单独结果（来自 baseline_comparison/results.json）
    v7_solo = {
        "gcg_attacks":  {"asr": 0.04,  "fpr": None},
        "pair_attacks": {"asr": 0.13,  "fpr": None},
        "jailbreakhub": {"asr": 0.08,  "fpr": None},
        "benign":       {"asr": None,  "fpr": 0.34},
    }

    print("\n加载数据集...")
    datasets = {}
    for name, cfg in DATASETS.items():
        texts = load_dataset(cfg)
        if texts:
            datasets[name] = (texts, cfg["expected"])
            print(f"  {name}: {len(texts)} 条")

    results = {}
    print("\n" + "-" * 65)
    print(f"{'数据集':<18} {'V7单独 ASR/FPR':>14} {'混合 ASR/FPR':>13} {'降低':>8} {'白名单命中率':>12}")
    print("-" * 65)

    for ds_name, (texts, expected) in datasets.items():
        acc, metric, wl_rate, stats = evaluate_pipeline(whitelist, v7, texts, expected)
        metric_name = "ASR" if expected == "harmful" else "FPR"
        solo_metric = v7_solo[ds_name]["asr"] if expected == "harmful" else v7_solo[ds_name]["fpr"]
        delta = (solo_metric - metric) if solo_metric is not None else 0

        results[ds_name] = {
            "expected": expected,
            "accuracy": round(acc, 4),
            metric_name.lower(): round(metric, 4),
            f"{metric_name.lower()}_v7_solo": solo_metric,
            f"{metric_name.lower()}_delta": round(delta, 4),
            "whitelist_hit_rate": round(wl_rate, 4),
            "stats": stats,
        }

        solo_str = f"{solo_metric:.3f}" if solo_metric is not None else "  N/A"
        print(f"{ds_name:<18} {solo_str:>14} {metric:>13.3f} {delta:>+8.3f} {wl_rate:>12.1%}")

    # 白名单单独分析
    print("\n" + "-" * 65)
    print("白名单单独分析（白名单→benign，其余→harmful，评估过度放行风险）")
    print("-" * 65)
    print(f"{'数据集':<18} {'白名单单独误放率':>18}")
    for ds_name, (texts, expected) in datasets.items():
        _, wl_metric = evaluate_whitelist_only(whitelist, texts, expected)
        label = "漏检率(↑坏)" if expected == "harmful" else "正确放行率(↑好)"
        print(f"  {ds_name:<16} {wl_metric:.1%}  {label}")

    # 保存结果
    with open(RESULTS_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 输出对比表
    print("\n" + "=" * 65)
    print("最终对比：V7单独 vs 白名单+V7 混合")
    print("=" * 65)
    print(f"\n{'指标':<20} {'V7单独':>10} {'白名单+V7':>10} {'变化':>8}")
    print("-" * 50)

    metrics_display = [
        ("GCG ASR↓",      "gcg_attacks",  "asr"),
        ("PAIR ASR↓",     "pair_attacks", "asr"),
        ("JailbreakHub ASR↓", "jailbreakhub", "asr"),
        ("Benign FPR↓",   "benign",       "fpr"),
    ]
    for label, ds, key in metrics_display:
        if ds in results:
            r = results[ds]
            hybrid = r.get(key, 0)
            solo = r.get(f"{key}_v7_solo", 0)
            delta = r.get(f"{key}_delta", 0)
            sign = "↓" if delta > 0 else ("↑" if delta < 0 else "→")
            print(f"  {label:<18} {solo:>10.3f} {hybrid:>10.3f} {sign}{abs(delta):.3f}")

    print(f"\n结果已保存到: {RESULTS_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
