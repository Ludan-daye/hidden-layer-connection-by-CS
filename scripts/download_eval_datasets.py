#!/usr/bin/env python3
"""
下载综合评测所需的缺失数据集
- WildGuardTest (allenai/wildguardmix)
- XSTest (paul-rottger/xstest)
- Alpaca (tatsu-lab/alpaca)
"""

import json
import csv
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATASETS_DIR = BASE_DIR / "datasets"


def download_wildguardtest():
    """下载 WildGuardTest — WildGuard论文标准测试集"""
    out_dir = DATASETS_DIR / "wildguardtest"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "wildguardtest.jsonl"

    if out_file.exists():
        lines = sum(1 for _ in open(out_file))
        print(f"  WildGuardTest 已存在 ({lines} 条)，跳过")
        return True

    print("  下载 WildGuardTest (allenai/wildguardmix)...")
    try:
        from datasets import load_dataset
        # wildguardmix 是 gated dataset，需要申请权限；尝试 wildguardtest split
        ds = load_dataset("allenai/wildguardmix", split="wildguardtest")
        records = []
        for item in ds:
            records.append({
                "text": item.get("prompt", ""),
                "prompt_harm_label": item.get("prompt_harm_label", ""),
                "response": item.get("response", ""),
                "response_harm_label": item.get("response_harm_label", ""),
            })
        with open(out_file, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  WildGuardTest 保存完成：{len(records)} 条 → {out_file}")
        return True
    except Exception as e:
        print(f"  WildGuardTest 下载失败 (需申请权限): {e}")
        print("  → 跳过，将用 ToxicChat 替代作为真实对话测试集")
        return False


def download_xstest():
    """下载 XSTest — 过度拒绝测试集（全部为良性）"""
    out_dir = DATASETS_DIR / "xstest"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "xstest.jsonl"

    if out_file.exists():
        lines = sum(1 for _ in open(out_file))
        print(f"  XSTest 已存在 ({lines} 条)，跳过")
        return True

    print("  下载 XSTest (HuggingFaceFW/xstest-v2)...")
    try:
        from datasets import load_dataset
        # 尝试多个可能的 HF 路径
        for hf_id in ["HuggingFaceFW/xstest-v2", "Aeala/xstest", "xstest"]:
            try:
                ds = load_dataset(hf_id, split="test")
                break
            except Exception:
                ds = None
        if ds is None:
            raise ValueError("所有路径均失败")
        records = []
        for item in ds:
            records.append({
                "text": item.get("prompt", item.get("text", "")),
                "type": item.get("type", ""),
                "label": "benign" if str(item.get("type", "")).startswith("safe") else "harmful",
            })
        with open(out_file, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  XSTest 保存完成：{len(records)} 条 → {out_file}")
        return True
    except Exception as e:
        # 回退：从 GitHub 直接下载 CSV
        print(f"  HF 失败 ({e})，尝试 GitHub 直接下载...")
        try:
            import urllib.request
            url = "https://raw.githubusercontent.com/paul-rottger/exaggerated-safety/main/data/xstest_v2_prompts.csv"
            tmp = out_dir / "xstest_raw.csv"
            urllib.request.urlretrieve(url, tmp)
            records = []
            with open(tmp, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    records.append({
                        "text": row.get("prompt", ""),
                        "type": row.get("type", ""),
                        "label": "benign" if str(row.get("type", "")).startswith("safe") else "harmful",
                    })
            with open(out_file, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  XSTest 保存完成：{len(records)} 条 → {out_file}")
            return True
        except Exception as e2:
            print(f"  XSTest 下载失败: {e2}")
            return False


def download_alpaca():
    """下载 Alpaca 并本地化保存"""
    out_dir = DATASETS_DIR / "normal"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "alpaca.jsonl"

    if out_file.exists():
        lines = sum(1 for _ in open(out_file))
        print(f"  Alpaca 已存在 ({lines} 条)，跳过")
        return True

    print("  下载 Alpaca (tatsu-lab/alpaca)...")
    try:
        from datasets import load_dataset
        ds = load_dataset("tatsu-lab/alpaca", split="train", trust_remote_code=True)
        records = []
        for item in ds:
            text = item.get("instruction", "")
            if item.get("input"):
                text += "\n" + item["input"]
            if text.strip():
                records.append({"text": text.strip(), "label": "benign"})
        with open(out_file, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  Alpaca 保存完成：{len(records)} 条 → {out_file}")
        return True
    except Exception as e:
        print(f"  Alpaca 下载失败: {e}")
        return False


def check_existing():
    """检查已有数据集"""
    print("\n已有数据集：")
    checks = [
        ("AdvBench", DATASETS_DIR / "advbench" / "advbench_harmful_behaviors.csv"),
        ("HarmBench", DATASETS_DIR / "harmbench" / "harmbench_behaviors.csv"),
        ("JailbreakBench GCG", DATASETS_DIR / "jailbreakbench" / "jbb_gcg_all.csv"),
        ("JailbreakBench PAIR", DATASETS_DIR / "jailbreakbench" / "jbb_pair_all.csv"),
        ("JailbreakHub", DATASETS_DIR / "jailbreakhub" / "jailbreakhub.csv"),
        ("ToxicChat", DATASETS_DIR / "gcg_attacks" / "toxic_chat.csv"),
        ("BeaverTails", DATASETS_DIR / "beavertails" / "beavertails_test.csv"),
        ("JBB-Benign", DATASETS_DIR / "gcg_attacks" / "jbb_benign_behaviors.csv"),
    ]
    for name, path in checks:
        status = f"✓ ({path.stat().st_size // 1024}KB)" if path.exists() else "✗ 缺失"
        print(f"  {name:<25} {status}")


def main():
    print("=" * 55)
    print("综合评测数据集下载工具")
    print("=" * 55)

    check_existing()

    print("\n下载缺失数据集：")
    results = {
        "WildGuardTest": download_wildguardtest(),
        "XSTest":        download_xstest(),
        "Alpaca":        download_alpaca(),
    }

    print("\n下载结果：")
    for name, ok in results.items():
        print(f"  {name}: {'✓ 成功' if ok else '✗ 失败（网络问题，将跳过该集）'}")


if __name__ == "__main__":
    main()
