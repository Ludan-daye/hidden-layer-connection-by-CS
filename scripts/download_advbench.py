"""下载AdvBench数据集"""

import os
import urllib.request

# AdvBench harmful behaviors数据集URL
ADVBENCH_URL = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"

output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "advbench_harmful_behaviors.csv")

print(f"下载AdvBench数据集...")
print(f"URL: {ADVBENCH_URL}")

try:
    urllib.request.urlretrieve(ADVBENCH_URL, output_path)
    print(f"✅ 下载成功: {output_path}")
    
    # 统计行数
    with open(output_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    print(f"   总行数: {len(lines)} (含表头)")
except Exception as e:
    print(f"❌ 下载失败: {e}")
