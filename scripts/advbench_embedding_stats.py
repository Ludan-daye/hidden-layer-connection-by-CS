"""
AdvBench Embedding维度统计
输入: AdvBench harmful behaviors数据集
输出: 每个输入对应的embedding矩阵，保存维度统计信息
"""

import csv
import numpy as np
from fastembed import TextEmbedding
from pathlib import Path
import json

# 配置
DATA_PATH = "datasets/advbench/advbench_harmful_behaviors.csv"
OUTPUT_DIR = "embedding_db/bge-small-en-v1.5/embeddings"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

def load_advbench(csv_path: str) -> list[dict]:
    """加载AdvBench数据集"""
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                "goal": row.get("goal", ""),
                "target": row.get("target", "")
            })
    return data

def main():
    # 创建输出目录
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print(f"加载AdvBench数据集: {DATA_PATH}")
    advbench_data = load_advbench(DATA_PATH)
    print(f"  样本数: {len(advbench_data)}")
    
    # 提取所有goal文本作为输入
    inputs = [item["goal"] for item in advbench_data]
    
    # 初始化embedding模型
    print(f"\n加载Embedding模型: {MODEL_NAME}")
    model = TextEmbedding(MODEL_NAME)
    
    # 计算所有embeddings
    print(f"\n计算Embeddings...")
    embeddings_list = list(model.embed(inputs))
    
    # 转换为numpy矩阵 (N, D)
    embeddings_matrix = np.array(embeddings_list)
    
    # 统计维度信息
    print("\n" + "=" * 60)
    print("Embedding 维度统计")
    print("=" * 60)
    
    stats = {
        "model": MODEL_NAME,
        "num_samples": embeddings_matrix.shape[0],
        "embedding_dim": embeddings_matrix.shape[1],
        "matrix_shape": list(embeddings_matrix.shape),
        "dtype": str(embeddings_matrix.dtype),
        "total_size_bytes": embeddings_matrix.nbytes,
        "total_size_mb": round(embeddings_matrix.nbytes / (1024 * 1024), 4),
        "per_vector_bytes": embeddings_matrix.shape[1] * 4,  # float32
    }
    
    print(f"  样本数量 (N): {stats['num_samples']}")
    print(f"  Embedding维度 (D): {stats['embedding_dim']}")
    print(f"  矩阵形状: {stats['matrix_shape']} (N x D)")
    print(f"  数据类型: {stats['dtype']}")
    print(f"  总存储大小: {stats['total_size_bytes']} bytes ({stats['total_size_mb']} MB)")
    print(f"  每向量大小: {stats['per_vector_bytes']} bytes")
    
    # 向量统计
    print("\n" + "=" * 60)
    print("向量数值统计")
    print("=" * 60)
    
    vector_stats = {
        "mean": float(np.mean(embeddings_matrix)),
        "std": float(np.std(embeddings_matrix)),
        "min": float(np.min(embeddings_matrix)),
        "max": float(np.max(embeddings_matrix)),
        "norm_mean": float(np.mean(np.linalg.norm(embeddings_matrix, axis=1))),
        "norm_std": float(np.std(np.linalg.norm(embeddings_matrix, axis=1))),
    }
    stats["vector_stats"] = vector_stats
    
    print(f"  均值: {vector_stats['mean']:.6f}")
    print(f"  标准差: {vector_stats['std']:.6f}")
    print(f"  最小值: {vector_stats['min']:.6f}")
    print(f"  最大值: {vector_stats['max']:.6f}")
    print(f"  向量L2范数均值: {vector_stats['norm_mean']:.6f}")
    print(f"  向量L2范数标准差: {vector_stats['norm_std']:.6f}")
    
    # 输入文本长度统计
    print("\n" + "=" * 60)
    print("输入文本长度统计")
    print("=" * 60)
    
    input_lengths = [len(text) for text in inputs]
    input_stats = {
        "min_chars": min(input_lengths),
        "max_chars": max(input_lengths),
        "mean_chars": round(np.mean(input_lengths), 2),
        "std_chars": round(np.std(input_lengths), 2),
    }
    stats["input_stats"] = input_stats
    
    print(f"  最短输入: {input_stats['min_chars']} 字符")
    print(f"  最长输入: {input_stats['max_chars']} 字符")
    print(f"  平均长度: {input_stats['mean_chars']} 字符")
    print(f"  长度标准差: {input_stats['std_chars']} 字符")
    
    # 保存结果
    print("\n" + "=" * 60)
    print("保存结果")
    print("=" * 60)
    
    # 保存embedding矩阵 (.npy)
    matrix_path = f"{OUTPUT_DIR}/advbench_embeddings.npy"
    np.save(matrix_path, embeddings_matrix)
    print(f"  ✅ Embedding矩阵: {matrix_path}")
    
    # 保存统计信息 (.json)
    stats_path = f"{OUTPUT_DIR}/advbench_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  ✅ 统计信息: {stats_path}")
    
    # 保存详细数据 (每个输入对应的embedding)
    detailed_path = f"{OUTPUT_DIR}/advbench_detailed.npz"
    np.savez(
        detailed_path,
        inputs=np.array(inputs, dtype=object),
        embeddings=embeddings_matrix,
    )
    print(f"  ✅ 详细数据: {detailed_path}")
    
    # 打印前5个样本示例
    print("\n" + "=" * 60)
    print("样本示例 (前5个)")
    print("=" * 60)
    for i in range(min(5, len(inputs))):
        print(f"\n[{i}] 输入: \"{inputs[i][:60]}{'...' if len(inputs[i]) > 60 else ''}\"")
        print(f"    输入长度: {len(inputs[i])} 字符")
        print(f"    Embedding形状: {embeddings_list[i].shape}")
        print(f"    Embedding前5维: {embeddings_list[i][:5]}")

if __name__ == "__main__":
    main()
