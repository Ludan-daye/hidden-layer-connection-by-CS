#!/usr/bin/env python3
"""
V7-Embed 综合评测脚本
在所有可用基准数据集上运行 V7 嵌入分类器（不使用 LLM Judge）
uncertain → benign，与轻量方法公平对比

测试集：
  攻击类：GCG, PAIR, JailbreakHub, AdvBench, HarmBench, ToxicChat(harmful), BeaverTails(harmful)
  良性类：JBB-Benign, Alpaca, ToxicChat(benign), BeaverTails(benign)
"""

import sys
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
RESULTS_DIR = BASE_DIR / "results" / "comprehensive_eval"
V7_MODEL_PATH = str(BASE_DIR / "models" / "v7_classifier")

sys.path.insert(0, str(BASE_DIR / "scripts" / "v7_classifier"))
sys.path.insert(0, str(BASE_DIR / "scripts" / "v7_classifier" / "deploy"))

random.seed(42)

# ─────────────────────────────────────────────
# 数据集配置
# ─────────────────────────────────────────────
DATASETS = {
    # ── 攻击类 ──────────────────────────────────
    "GCG": {
        "path": DATASETS_DIR / "jailbreakbench" / "jbb_gcg_all.csv",
        "col": "prompt", "expected": "harmful", "n": 100,
        "desc": "GCG梯度优化后缀攻击",
    },
    "PAIR": {
        "path": DATASETS_DIR / "jailbreakbench" / "jbb_pair_all.csv",
        "col": "prompt", "expected": "harmful", "n": 86,
        "desc": "PAIR语义越狱攻击",
    },
    "JailbreakHub": {
        "path": DATASETS_DIR / "jailbreakhub" / "jailbreakhub.csv",
        "col": "prompt", "expected": "harmful", "n": 79,
        "desc": "JailbreakHub手动越狱",
    },
    "AdvBench": {
        "path": DATASETS_DIR / "advbench" / "advbench_harmful_behaviors.csv",
        "col": "goal", "expected": "harmful", "n": 200,
        "desc": "AdvBench有害指令",
    },
    "HarmBench": {
        "path": DATASETS_DIR / "harmbench" / "harmbench_behaviors.csv",
        "col": "Behavior", "expected": "harmful", "n": 200,
        "desc": "HarmBench有害行为",
    },
    "ToxicChat_harmful": {
        "path": DATASETS_DIR / "gcg_attacks" / "toxic_chat_full.csv",
        "col": "user_input", "expected": "harmful", "n": 300,
        "label_col": "human_annotation", "label_val": "True",
        "desc": "ToxicChat真实有害对话",
    },
    "BeaverTails_harmful": {
        "path": DATASETS_DIR / "beavertails" / "beavertails_test.csv",
        "col": "prompt", "expected": "harmful", "n": 300,
        "label_col": "is_safe", "label_val": False,
        "desc": "BeaverTails有害样本",
    },
    # ── 良性类 ──────────────────────────────────
    "JBB_Benign": {
        "path": DATASETS_DIR / "gcg_attacks" / "jbb_benign_behaviors.csv",
        "col": "Goal", "expected": "benign", "n": 100,
        "desc": "JBB边界良性行为",
    },
    "Alpaca": {
        "path": DATASETS_DIR / "normal" / "alpaca.jsonl",
        "col": "text", "expected": "benign", "n": 200,
        "desc": "Alpaca常规指令",
    },
    "ToxicChat_benign": {
        "path": DATASETS_DIR / "gcg_attacks" / "toxic_chat_full.csv",
        "col": "user_input", "expected": "benign", "n": 300,
        "label_col": "human_annotation", "label_val": "False",
        "desc": "ToxicChat真实无害对话",
    },
    "BeaverTails_benign": {
        "path": DATASETS_DIR / "beavertails" / "beavertails_test.csv",
        "col": "prompt", "expected": "benign", "n": 300,
        "label_col": "is_safe", "label_val": True,
        "desc": "BeaverTails安全样本",
    },
}


# ─────────────────────────────────────────────
# 数据集加载
# ─────────────────────────────────────────────
def load_texts(cfg: dict) -> list:
    path = cfg["path"]
    if not path.exists():
        return []

    col = cfg["col"]
    n = cfg.get("n")
    label_col = cfg.get("label_col")
    label_val = cfg.get("label_val")

    # JSONL 格式
    if str(path).endswith(".jsonl"):
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
        texts = [r.get(col, "") for r in records if r.get(col, "").strip()]
        if n and len(texts) > n:
            texts = random.sample(texts, n)
        return texts

    # CSV 格式
    df = pd.read_csv(path)

    # 按标签列过滤
    if label_col and label_col in df.columns:
        col_vals = df[label_col].astype(str).str.strip().str.lower()
        filter_val = str(label_val).strip().lower()
        df = df[col_vals == filter_val]

    # 找文本列
    if col not in df.columns:
        for c in ["prompt", "goal", "Goal", "text", "Behavior", "user_input"]:
            if c in df.columns:
                col = c
                break

    texts = df[col].dropna().astype(str).tolist()
    texts = [t.strip() for t in texts if t.strip()]

    if n and len(texts) > n:
        texts = random.sample(texts, n)
    return texts


# ─────────────────────────────────────────────
# V7-Embed 分类器（uncertain → benign）
# ─────────────────────────────────────────────
class V7EmbedClassifier:
    def __init__(self):
        from v7_classifier import V7Classifier
        self.clf = V7Classifier(model_path=V7_MODEL_PATH)

    def predict(self, text: str) -> str:
        label, prob = self.clf.predict(text)
        # uncertain → benign（与轻量方法公平对比）
        return "harmful" if label == "harmful" else "benign"

    def predict_batch(self, texts: list, verbose: bool = True) -> list:
        results = []
        total = len(texts)
        for i, text in enumerate(texts):
            if verbose and (i + 1) % 50 == 0:
                print(f"    {i+1}/{total}...", end="\r")
            results.append(self.predict(text))
        if verbose:
            print()
        return results


# ─────────────────────────────────────────────
# 评测函数
# ─────────────────────────────────────────────
def evaluate(clf, texts: list, expected: str) -> dict:
    preds = clf.predict_batch(texts)
    correct = sum(1 for p in preds if p == expected)
    accuracy = correct / len(texts)

    if expected == "harmful":
        # Detection Rate = 正确检测率；ASR = 漏检率
        asr = sum(1 for p in preds if p == "benign") / len(texts)
        dr = 1.0 - asr
        return {"accuracy": round(accuracy, 4), "detection_rate": round(dr, 4),
                "asr": round(asr, 4), "total": len(texts), "correct": correct}
    else:
        fpr = sum(1 for p in preds if p == "harmful") / len(texts)
        return {"accuracy": round(accuracy, 4), "fpr": round(fpr, 4),
                "total": len(texts), "correct": correct}


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("V7-Embed 综合评测（uncertain → benign，无LLM Judge）")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    print("\n加载 V7 分类器...")
    clf = V7EmbedClassifier()

    all_results = {}
    print()

    for ds_name, cfg in DATASETS.items():
        print(f"[{ds_name}] {cfg['desc']}")
        texts = load_texts(cfg)
        if not texts:
            print(f"  ✗ 数据集不可用，跳过\n")
            continue
        print(f"  样本数: {len(texts)}")
        result = evaluate(clf, texts, cfg["expected"])
        all_results[ds_name] = {**result, "desc": cfg["desc"], "expected": cfg["expected"]}

        if cfg["expected"] == "harmful":
            print(f"  Detection Rate: {result['detection_rate']:.1%}  ASR: {result['asr']:.1%}")
        else:
            print(f"  FPR: {result['fpr']:.1%}  (正确放行: {result['accuracy']:.1%})")
        print()

    # 保存结果
    out = {
        "model": "V7-Embed (uncertain→benign, no LLM Judge)",
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "datasets": all_results,
    }
    with open(RESULTS_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # 打印汇总
    print("=" * 65)
    print("汇总")
    print("=" * 65)

    harmful_sets = [(k, v) for k, v in all_results.items() if v["expected"] == "harmful"]
    benign_sets  = [(k, v) for k, v in all_results.items() if v["expected"] == "benign"]

    print(f"\n攻击检测（Detection Rate↑）:")
    print(f"  {'数据集':<25} {'样本':>6} {'DR':>8} {'ASR':>8}")
    print(f"  {'-'*50}")
    drs = []
    for k, v in harmful_sets:
        dr = v.get("detection_rate", 0)
        drs.append(dr)
        print(f"  {k:<25} {v['total']:>6} {dr:>8.1%} {v['asr']:>8.1%}")
    if drs:
        print(f"  {'平均':<25} {'':>6} {sum(drs)/len(drs):>8.1%}")

    print(f"\n误报率（FPR↓）:")
    print(f"  {'数据集':<25} {'样本':>6} {'FPR':>8}")
    print(f"  {'-'*40}")
    fprs = []
    for k, v in benign_sets:
        fpr = v.get("fpr", 0)
        fprs.append(fpr)
        print(f"  {k:<25} {v['total']:>6} {fpr:>8.1%}")
    if fprs:
        print(f"  {'平均':<25} {'':>6} {sum(fprs)/len(fprs):>8.1%}")

    print(f"\n结果已保存: {RESULTS_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
