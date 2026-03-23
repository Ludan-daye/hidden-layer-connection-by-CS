"""收集正常样本数据作为对照组"""

import urllib.request
import json
import os

output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

# 使用Alpaca数据集的一部分作为正常样本
ALPACA_URL = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"

output_path = os.path.join(output_dir, "alpaca_normal_samples.json")

print("下载Alpaca数据集作为正常样本...")
print(f"URL: {ALPACA_URL}")

try:
    urllib.request.urlretrieve(ALPACA_URL, output_path)
    
    with open(output_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ 下载成功: {output_path}")
    print(f"   总样本数: {len(data)}")
    
    # 提取前520条instruction作为正常样本（与AdvBench数量匹配）
    normal_texts = [item['instruction'] for item in data[:520]]
    
    # 保存为简单文本格式
    normal_path = os.path.join(output_dir, "normal_samples.txt")
    with open(normal_path, 'w', encoding='utf-8') as f:
        for text in normal_texts:
            f.write(text + '\n')
    
    print(f"   提取正常样本: {len(normal_texts)} 条")
    print(f"   保存到: {normal_path}")
    
except Exception as e:
    print(f"❌ 下载失败: {e}")
