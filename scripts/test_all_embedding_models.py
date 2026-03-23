"""
测试所有支持的Embedding模型
训练: AdvBench (80%)
测试: AdvBench Test (20%) + HarmBench (Zero-shot)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from fastembed import TextEmbedding
import json
import os

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ============================================================
# 学习型投影模型
# ============================================================

class LearnedProjection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim, bias=False)
        nn.init.normal_(self.proj.weight, mean=0, std=1/np.sqrt(output_dim))
    
    def forward(self, x):
        z = self.proj(x)
        z = F.normalize(z, p=2, dim=-1)
        return z

def train_learned_projection(train_embeddings, target_dim, epochs=300, lr=0.01):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embeddings_tensor = torch.tensor(train_embeddings, dtype=torch.float32).to(device)
    embeddings_norm = F.normalize(embeddings_tensor, p=2, dim=-1)
    
    model = LearnedProjection(train_embeddings.shape[1], target_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        original_sim = embeddings_norm @ embeddings_norm.T
        compressed = model(embeddings_tensor)
        compressed_sim = compressed @ compressed.T
        loss = F.mse_loss(compressed_sim, original_sim)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return model

def apply_projection(model, embeddings):
    device = next(model.parameters()).device
    with torch.no_grad():
        tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
        compressed = model(tensor).cpu().numpy()
    return compressed

def compute_detection_with_fixed_centroid(mal_emb, norm_emb, centroid_norm):
    mal_sims = np.array([np.dot(e/np.linalg.norm(e), centroid_norm) for e in mal_emb])
    norm_sims = np.array([np.dot(e/np.linalg.norm(e), centroid_norm) for e in norm_emb])
    
    threshold = (mal_sims.mean() + norm_sims.mean()) / 2
    detection_rate = (mal_sims > threshold).sum() / len(mal_sims)
    false_positive_rate = (norm_sims > threshold).sum() / len(norm_sims)
    
    y_true = np.concatenate([np.ones(len(mal_sims)), np.zeros(len(norm_sims))])
    y_scores = np.concatenate([mal_sims, norm_sims])
    auc = roc_auc_score(y_true, y_scores)
    
    return {
        'detection_rate': float(detection_rate),
        'false_positive_rate': float(false_positive_rate),
        'auc': float(auc),
    }

def test_embedding_model(model_name, advbench_texts, harmbench_texts, normal_texts):
    """测试单个embedding模型"""
    
    model_short_name = model_name.replace("/", "_").replace("-", "_")
    
    print(f"\n{'='*70}")
    print(f"测试: {model_name}")
    print("=" * 70)
    
    # 创建目录
    EMBEDDINGS_DIR = f"embedding_db/{model_short_name}/embeddings"
    RESULTS_DIR = f"embedding_db/{model_short_name}/results"
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    try:
        # 加载embedding模型
        embed_model = TextEmbedding(model_name)
        
        # 测试维度
        test_emb = list(embed_model.embed(["test"]))[0]
        embed_dim = len(test_emb)
        print(f"  维度: {embed_dim}")
        
        # 计算Embeddings
        print("  计算Embeddings...")
        advbench_emb = np.array(list(embed_model.embed(advbench_texts)))
        harmbench_emb = np.array(list(embed_model.embed(harmbench_texts)))
        normal_emb = np.array(list(embed_model.embed(normal_texts)))
        
        # 保存
        np.save(f"{EMBEDDINGS_DIR}/advbench_embeddings.npy", advbench_emb)
        np.save(f"{EMBEDDINGS_DIR}/harmbench_embeddings.npy", harmbench_emb)
        np.save(f"{EMBEDDINGS_DIR}/normal_embeddings.npy", normal_emb)
        
        # 划分训练/测试集
        advbench_train, advbench_test = train_test_split(advbench_emb, test_size=0.2, random_state=RANDOM_SEED)
        normal_train, normal_test = train_test_split(normal_emb, test_size=0.2, random_state=RANDOM_SEED)
        
        # 计算训练集质心
        centroid_train = advbench_train.mean(axis=0)
        centroid_train_norm = centroid_train / np.linalg.norm(centroid_train)
        
        # 训练投影矩阵
        train_emb = np.vstack([advbench_train, normal_train])
        
        target_dim_10pct = max(int(embed_dim * 0.1), 1)
        target_dim_5pct = max(int(embed_dim * 0.05), 1)
        
        model_10pct = train_learned_projection(train_emb, target_dim_10pct, epochs=300)
        model_5pct = train_learned_projection(train_emb, target_dim_5pct, epochs=300)
        
        # 计算压缩后的质心
        advbench_train_10pct = apply_projection(model_10pct, advbench_train)
        centroid_train_10pct = advbench_train_10pct.mean(axis=0)
        centroid_train_10pct_norm = centroid_train_10pct / np.linalg.norm(centroid_train_10pct)
        
        advbench_train_5pct = apply_projection(model_5pct, advbench_train)
        centroid_train_5pct = advbench_train_5pct.mean(axis=0)
        centroid_train_5pct_norm = centroid_train_5pct / np.linalg.norm(centroid_train_5pct)
        
        # 测试
        results = {
            'model_name': model_name,
            'embed_dim': embed_dim,
            'target_dim_10pct': target_dim_10pct,
            'target_dim_5pct': target_dim_5pct,
            'train_samples': {
                'advbench': len(advbench_train),
                'normal': len(normal_train),
            },
            'datasets': {}
        }
        
        test_datasets = {
            'AdvBench_Test': advbench_test,
            'HarmBench': harmbench_emb,
        }
        
        for dataset_name, mal_emb in test_datasets.items():
            n = min(len(mal_emb), len(normal_test))
            mal_subset = mal_emb[:n]
            norm_subset = normal_test[:n]
            
            # 原始空间
            orig = compute_detection_with_fixed_centroid(mal_subset, norm_subset, centroid_train_norm)
            
            # 学习型投影 (10%)
            mal_10pct = apply_projection(model_10pct, mal_subset)
            norm_10pct = apply_projection(model_10pct, norm_subset)
            l10 = compute_detection_with_fixed_centroid(mal_10pct, norm_10pct, centroid_train_10pct_norm)
            
            # 学习型投影 (5%)
            mal_5pct = apply_projection(model_5pct, mal_subset)
            norm_5pct = apply_projection(model_5pct, norm_subset)
            l5 = compute_detection_with_fixed_centroid(mal_5pct, norm_5pct, centroid_train_5pct_norm)
            
            results['datasets'][dataset_name] = {
                'original': orig,
                'learned_10pct': l10,
                'learned_5pct': l5,
            }
        
        # 打印结果
        harm_res = results['datasets']['HarmBench']
        print(f"  HarmBench: 原始AUC={harm_res['original']['auc']:.4f}, 10%={harm_res['learned_10pct']['auc']:.4f}, 5%={harm_res['learned_5pct']['auc']:.4f}")
        
        # 保存结果
        with open(f"{RESULTS_DIR}/detection_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        return results
        
    except Exception as e:
        print(f"  错误: {e}")
        return None

# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    
    # 加载数据集文本
    print("加载数据集...")
    
    # AdvBench
    advbench_df = pd.read_csv("datasets/advbench/advbench_harmful_behaviors.csv")
    advbench_texts = advbench_df['goal'].dropna().tolist()
    print(f"  AdvBench: {len(advbench_texts)} 条")
    
    # HarmBench
    harmbench_df = pd.read_csv("datasets/harmbench/harmbench_behaviors.csv")
    harmbench_texts = harmbench_df['Behavior'].dropna().tolist()
    print(f"  HarmBench: {len(harmbench_texts)} 条")
    
    # 正常样本
    templates = [
        'Can you explain {}?', 'What is the meaning of {}?', 'How does {} work?',
        'Tell me about {}.', 'What are the benefits of {}?', 'Describe {} in simple terms.',
        'What is the history of {}?', 'How can I learn about {}?',
    ]
    topics = [
        'mathematics', 'science', 'history', 'geography', 'literature',
        'music', 'art', 'cooking', 'gardening', 'photography',
        'programming', 'design', 'economics', 'psychology', 'philosophy',
        'biology', 'chemistry', 'physics', 'astronomy', 'medicine',
        'education', 'sports', 'travel', 'technology', 'engineering',
    ]
    normal_texts = []
    for topic in topics:
        for template in templates:
            normal_texts.append(template.format(topic))
            if len(normal_texts) >= 500:
                break
        if len(normal_texts) >= 500:
            break
    print(f"  正常样本: {len(normal_texts)} 条")
    
    # 要测试的模型
    models_to_test = [
        "BAAI/bge-small-en-v1.5",
        "BAAI/bge-base-en-v1.5",
        "BAAI/bge-large-en-v1.5",
        "sentence-transformers/all-MiniLM-L6-v2",
        "thenlper/gte-base",
        "thenlper/gte-large",
        "nomic-ai/nomic-embed-text-v1.5",
        "jinaai/jina-embeddings-v2-small-en",
        "jinaai/jina-embeddings-v2-base-en",
    ]
    
    all_results = {}
    
    for model_name in models_to_test:
        results = test_embedding_model(model_name, advbench_texts, harmbench_texts, normal_texts)
        if results:
            all_results[model_name] = results
    
    # 保存汇总结果
    with open("embedding_db/all_models_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # 打印汇总表格
    print("\n" + "=" * 90)
    print("所有模型对比总结 (HarmBench Zero-shot Transfer)")
    print("=" * 90)
    print(f"{'模型':<40} {'维度':<8} {'原始AUC':<10} {'10%压缩':<10} {'5%压缩':<10}")
    print("-" * 90)
    
    for model_name, res in all_results.items():
        embed_dim = res['embed_dim']
        harm_res = res['datasets']['HarmBench']
        orig_auc = harm_res['original']['auc']
        l10_auc = harm_res['learned_10pct']['auc']
        l5_auc = harm_res['learned_5pct']['auc']
        
        print(f"{model_name:<40} {embed_dim:<8} {orig_auc:<10.4f} {l10_auc:<10.4f} {l5_auc:<10.4f}")
    
    print("-" * 90)
    print("\n结果已保存到: embedding_db/all_models_summary.json")
