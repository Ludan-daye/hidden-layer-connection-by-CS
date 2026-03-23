"""
下载更多恶意行为数据集
"""

import os
import urllib.request
import json

DATA_DIR = "data/datasets"
os.makedirs(DATA_DIR, exist_ok=True)

datasets = {
    # ToxiGen - 有毒内容生成数据集
    "toxigen": {
        "url": "https://raw.githubusercontent.com/microsoft/TOXIGEN/main/prompts/hate_gens.txt",
        "filename": "toxigen_hate.txt",
        "description": "ToxiGen有毒内容数据集 (Microsoft)"
    },
    # Do-Not-Answer - 不应回答的问题
    "do_not_answer": {
        "url": "https://raw.githubusercontent.com/Libr-AI/do-not-answer/main/data/do_not_answer_en.csv",
        "filename": "do_not_answer.csv",
        "description": "Do-Not-Answer不应回答问题数据集"
    },
    # MaliciousInstruct
    "malicious_instruct": {
        "url": "https://raw.githubusercontent.com/Princeton-SysML/Jailbreak_LLM/main/data/MaliciousInstruct.txt",
        "filename": "malicious_instruct.txt",
        "description": "MaliciousInstruct恶意指令数据集"
    },
    # SafetyBench (中文+英文)
    "safety_bench": {
        "url": "https://raw.githubusercontent.com/thu-coai/SafetyBench/main/data/en_test.json",
        "filename": "safetybench_en.json",
        "description": "SafetyBench安全测试数据集 (清华)"
    },
    # BeaverTails - 有害内容分类
    "beavertails": {
        "url": "https://huggingface.co/datasets/PKU-Alignment/BeaverTails/raw/main/round0_330k_test.jsonl",
        "filename": "beavertails_test.jsonl",
        "description": "BeaverTails有害内容数据集 (北大)"
    },
}

print("=" * 70)
print("下载更多恶意行为数据集")
print("=" * 70)

successful = []
failed = []

for name, info in datasets.items():
    output_path = os.path.join(DATA_DIR, info["filename"])
    print(f"\n[{name}] {info['description']}")
    print(f"  URL: {info['url'][:60]}...")
    
    try:
        # 设置超时
        urllib.request.urlretrieve(info["url"], output_path)
        
        # 统计
        with open(output_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            if info["filename"].endswith('.json'):
                data = json.loads(content)
                count = len(data) if isinstance(data, list) else 1
            elif info["filename"].endswith('.jsonl'):
                count = len(content.strip().split('\n'))
            else:
                count = len(content.strip().split('\n'))
        
        print(f"  ✅ 下载成功: {output_path}")
        print(f"  样本数: {count}")
        successful.append(name)
        
    except Exception as e:
        print(f"  ❌ 下载失败: {e}")
        failed.append(name)

print("\n" + "=" * 70)
print("下载完成")
print("=" * 70)
print(f"成功: {len(successful)} 个 - {successful}")
print(f"失败: {len(failed)} 个 - {failed}")
