"""
测试新的Embedding模型
- bge-base-en-v1.5 (768维)
- all-MiniLM-L6-v2 (384维)
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

def test_embedding_model(model_name, model_short_name):
    """测试单个embedding模型"""
    
    print("\n" + "=" * 70)
    print(f"测试 Embedding 模型: {model_name}")
    print("=" * 70)
    
    # 创建目录
    EMBEDDINGS_DIR = f"embedding_db/{model_short_name}/embeddings"
    RESULTS_DIR = f"embedding_db/{model_short_name}/results"
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 加载embedding模型
    print("\n[1] 加载Embedding模型...")
    embed_model = TextEmbedding(model_name)
    
    # 测试维度
    test_emb = list(embed_model.embed(["test"]))[0]
    embed_dim = len(test_emb)
    print(f"  模型: {model_name}")
    print(f"  维度: {embed_dim}")
    
    # 加载数据集文本
    print("\n[2] 加载数据集...")
    
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
    ]
    topics = [
        'mathematics', 'science', 'history', 'geography', 'literature',
        'music', 'art', 'cooking', 'gardening', 'photography',
        'programming', 'design', 'economics', 'psychology', 'philosophy',
        'biology', 'chemistry', 'physics', 'astronomy', 'medicine',
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
    
    # 计算Embeddings
    print("\n[3] 计算Embeddings...")
    
    advbench_emb = np.array(list(embed_model.embed(advbench_texts)))
    np.save(f"{EMBEDDINGS_DIR}/advbench_embeddings.npy", advbench_emb)
    print(f"  AdvBench: {advbench_emb.shape}")
    
    harmbench_emb = np.array(list(embed_model.embed(harmbench_texts)))
    np.save(f"{EMBEDDINGS_DIR}/harmbench_embeddings.npy", harmbench_emb)
    print(f"  HarmBench: {harmbench_emb.shape}")
    
    normal_emb = np.array(list(embed_model.embed(normal_texts)))
    np.save(f"{EMBEDDINGS_DIR}/normal_embeddings.npy", normal_emb)
    print(f"  正常样本: {normal_emb.shape}")
    
    # 划分训练/测试集
    print("\n[4] 划分训练/测试集 (80%/20%)...")
    advbench_train, advbench_test = train_test_split(advbench_emb, test_size=0.2, random_state=RANDOM_SEED)
    normal_train, normal_test = train_test_split(normal_emb, test_size=0.2, random_state=RANDOM_SEED)
    
    print(f"  AdvBench Train: {advbench_train.shape[0]} 条")
    print(f"  AdvBench Test: {advbench_test.shape[0]} 条")
    print(f"  正常样本 Test: {normal_test.shape[0]} 条")
    
    # 计算训练集质心
    centroid_train = advbench_train.mean(axis=0)
    centroid_train_norm = centroid_train / np.linalg.norm(centroid_train)
    
    # 训练投影矩阵
    print("\n[5] 训练投影矩阵...")
    train_emb = np.vstack([advbench_train, normal_train])
    
    # 压缩到10%和5%
    target_dim_10pct = max(int(embed_dim * 0.1), 1)
    target_dim_5pct = max(int(embed_dim * 0.05), 1)
    
    model_10pct = train_learned_projection(train_emb, target_dim_10pct, epochs=300)
    print(f"  {target_dim_10pct}维投影矩阵训练完成 (10%)")
    
    model_5pct = train_learned_projection(train_emb, target_dim_5pct, epochs=300)
    print(f"  {target_dim_5pct}维投影矩阵训练完成 (5%)")
    
    # 计算压缩后的质心
    advbench_train_10pct = apply_projection(model_10pct, advbench_train)
    centroid_train_10pct = advbench_train_10pct.mean(axis=0)
    centroid_train_10pct_norm = centroid_train_10pct / np.linalg.norm(centroid_train_10pct)
    
    advbench_train_5pct = apply_projection(model_5pct, advbench_train)
    centroid_train_5pct = advbench_train_5pct.mean(axis=0)
    centroid_train_5pct_norm = centroid_train_5pct / np.linalg.norm(centroid_train_5pct)
    
    # 测试
    print("\n[6] 测试各数据集...")
    
    test_datasets = {
        'AdvBench_Test': advbench_test,
        'HarmBench': harmbench_emb,
    }
    
    results = {
        'model_name': model_name,
        'embed_dim': embed_dim,
        'target_dim_10pct': target_dim_10pct,
        'target_dim_5pct': target_dim_5pct,
        'datasets': {}
    }
    
    for dataset_name, mal_emb in test_datasets.items():
        print(f"\n--- {dataset_name} ({mal_emb.shape[0]} 条) ---")
        
        n = min(len(mal_emb), len(normal_test))
        mal_subset = mal_emb[:n]
        norm_subset = normal_test[:n]
        
        dataset_results = {}
        
        # 原始空间
        orig = compute_detection_with_fixed_centroid(mal_subset, norm_subset, centroid_train_norm)
        dataset_results[f'original_{embed_dim}d'] = orig
        print(f"  原始({embed_dim}维): 检出率={orig['detection_rate']*100:.1f}%, 误报率={orig['false_positive_rate']*100:.1f}%, AUC={orig['auc']:.4f}")
        
        # 学习型投影 (10%)
        mal_10pct = apply_projection(model_10pct, mal_subset)
        norm_10pct = apply_projection(model_10pct, norm_subset)
        l10 = compute_detection_with_fixed_centroid(mal_10pct, norm_10pct, centroid_train_10pct_norm)
        dataset_results[f'learned_{target_dim_10pct}d'] = l10
        print(f"  学习型({target_dim_10pct}维,10%): 检出率={l10['detection_rate']*100:.1f}%, 误报率={l10['false_positive_rate']*100:.1f}%, AUC={l10['auc']:.4f}")
        
        # 学习型投影 (5%)
        mal_5pct = apply_projection(model_5pct, mal_subset)
        norm_5pct = apply_projection(model_5pct, norm_subset)
        l5 = compute_detection_with_fixed_centroid(mal_5pct, norm_5pct, centroid_train_5pct_norm)
        dataset_results[f'learned_{target_dim_5pct}d'] = l5
        print(f"  学习型({target_dim_5pct}维,5%): 检出率={l5['detection_rate']*100:.1f}%, 误报率={l5['false_positive_rate']*100:.1f}%, AUC={l5['auc']:.4f}")
        
        results['datasets'][dataset_name] = dataset_results
    
    # 保存结果
    with open(f"{RESULTS_DIR}/detection_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n结果已保存到: {RESULTS_DIR}/detection_results.json")
    
    return results

# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    models_to_test = [
        ("BAAI/bge-base-en-v1.5", "bge-base-en-v1.5"),
        ("sentence-transformers/all-MiniLM-L6-v2", "all-MiniLM-L6-v2"),
    ]
    
    all_results = {}
    
    for model_name, model_short_name in models_to_test:
        results = test_embedding_model(model_name, model_short_name)
        all_results[model_short_name] = results
    
    # 总结对比
    print("\n" + "=" * 70)
    print("所有模型对比总结")
    print("=" * 70)
    
    print("\nHarmBench Zero-shot Transfer 效果:")
    print("-" * 80)
    print(f"{'模型':<25} {'维度':<10} {'原始AUC':<12} {'10%压缩AUC':<12} {'5%压缩AUC':<12}")
    print("-" * 80)
    
    for model_name, res in all_results.items():
        embed_dim = res['embed_dim']
        dim_10 = res['target_dim_10pct']
        dim_5 = res['target_dim_5pct']
        
        harm_res = res['datasets']['HarmBench']
        orig_auc = harm_res[f'original_{embed_dim}d']['auc']
        l10_auc = harm_res[f'learned_{dim_10}d']['auc']
        l5_auc = harm_res[f'learned_{dim_5}d']['auc']
        
        print(f"{model_name:<25} {embed_dim:<10} {orig_auc:<12.4f} {l10_auc:<12.4f} {l5_auc:<12.4f}")
    
    # 加上之前的bge-small结果
    print(f"{'bge-small-en-v1.5':<25} {'384':<10} {'0.9568':<12} {'0.9774':<12} {'0.9949':<12}")
    
    print("-" * 80)
    print("完成!")
