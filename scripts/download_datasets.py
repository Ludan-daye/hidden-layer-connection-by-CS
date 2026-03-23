"""
下载多个恶意行为数据集用于测试
"""

import os
import urllib.request
import json

DATA_DIR = "data/datasets"
os.makedirs(DATA_DIR, exist_ok=True)

datasets = {
    # HarmBench - 更全面的有害行为数据集
    "harmbench": {
        "url": "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv",
        "filename": "harmbench_behaviors.csv",
        "description": "HarmBench有害行为数据集"
    },
    # JailbreakBench - 越狱攻击数据集
    "jailbreakbench": {
        "url": "https://raw.githubusercontent.com/JailbreakBench/jailbreakbench/main/src/jailbreakbench/data/behaviors.csv",
        "filename": "jailbreakbench_behaviors.csv",
        "description": "JailbreakBench越狱攻击数据集"
    },
    # AdvBench strings (另一个版本)
    "advbench_strings": {
        "url": "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_strings.csv",
        "filename": "advbench_strings.csv",
        "description": "AdvBench有害字符串数据集"
    },
}

print("=" * 70)
print("下载恶意行为数据集")
print("=" * 70)

for name, info in datasets.items():
    output_path = os.path.join(DATA_DIR, info["filename"])
    print(f"\n[{name}] {info['description']}")
    print(f"  URL: {info['url']}")
    
    try:
        urllib.request.urlretrieve(info["url"], output_path)
        
        # 统计行数
        with open(output_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        print(f"  ✅ 下载成功: {output_path}")
        print(f"  行数: {len(lines)}")
        
    except Exception as e:
        print(f"  ❌ 下载失败: {e}")

print("\n" + "=" * 70)
print("下载完成")
print("=" * 70)
