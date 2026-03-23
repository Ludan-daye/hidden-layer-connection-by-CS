"""
使用JailbreakBench的GCG攻击数据测试防御效果

GCG (Greedy Coordinate Gradient) 是一种自动生成对抗性后缀的攻击方法，
通过优化后缀来绕过LLM的安全防护。

本脚本：
1. 从JailbreakBench下载GCG攻击数据
2. 使用我们的Embedding检测方法测试防御效果
3. 对比原始请求和带对抗性后缀的请求的检测率
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from fastembed import TextEmbedding
import jailbreakbench as jbb
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

# ============================================================
# 主程序
# ============================================================

print("=" * 70)
print("JailbreakBench GCG攻击防御测试")
print("=" * 70)

# 1. 下载GCG攻击数据
print("\n[1] 从JailbreakBench下载GCG攻击数据...")

# 获取多个模型的GCG攻击结果
models_to_test = ['vicuna-13b-v1.5', 'llama-2-7b-chat-hf']
all_gcg_data = []

for model_name in models_to_test:
    try:
        artifact = jbb.read_artifact(method='GCG', model_name=model_name)
        print(f"  {model_name}: {len(artifact.jailbreaks)} 条攻击")
        for jb in artifact.jailbreaks:
            all_gcg_data.append({
                'model': model_name,
                'goal': jb.goal,
                'prompt': jb.prompt,
                'jailbroken': jb.jailbroken
            })
    except Exception as e:
        print(f"  {model_name}: 获取失败 - {e}")

if not all_gcg_data:
    print("错误: 无法获取GCG攻击数据")
    exit(1)

gcg_df = pd.DataFrame(all_gcg_data)
print(f"\n  总计: {len(gcg_df)} 条GCG攻击")
print(f"  成功越狱: {gcg_df['jailbroken'].sum()} 条 ({gcg_df['jailbroken'].mean()*100:.1f}%)")

# 保存GCG数据
os.makedirs("datasets/gcg_attacks", exist_ok=True)
gcg_df.to_csv("datasets/gcg_attacks/jbb_gcg_all.csv", index=False)

# 2. 加载Embedding模型和训练数据
print("\n[2] 加载Embedding模型和训练数据...")
embed_model = TextEmbedding('BAAI/bge-small-en-v1.5')

advbench_emb = np.load("embedding_db/bge-small-en-v1.5/embeddings/advbench_embeddings.npy")
normal_emb = np.load("embedding_db/bge-small-en-v1.5/embeddings/normal_embeddings.npy")

# 划分训练/测试集
advbench_train, advbench_test = train_test_split(advbench_emb, test_size=0.2, random_state=RANDOM_SEED)
normal_train, normal_test = train_test_split(normal_emb, test_size=0.2, random_state=RANDOM_SEED)

print(f"  AdvBench Train: {advbench_train.shape[0]} 条")
print(f"  正常样本 Train: {normal_train.shape[0]} 条")

# 3. 训练投影矩阵
print("\n[3] 训练投影矩阵...")
train_emb = np.vstack([advbench_train, normal_train])
model_19d = train_learned_projection(train_emb, 19, epochs=300)
print("  19维投影矩阵训练完成")

# 4. 计算质心和阈值
print("\n[4] 计算质心和阈值...")

# 原始空间
centroid_orig = advbench_train.mean(axis=0)
centroid_orig_norm = centroid_orig / np.linalg.norm(centroid_orig)

train_mal_sims = np.array([np.dot(e/np.linalg.norm(e), centroid_orig_norm) for e in advbench_train])
train_norm_sims = np.array([np.dot(e/np.linalg.norm(e), centroid_orig_norm) for e in normal_train])
threshold_orig = (train_mal_sims.mean() + train_norm_sims.mean()) / 2

# 压缩空间
advbench_train_19d = apply_projection(model_19d, advbench_train)
normal_train_19d = apply_projection(model_19d, normal_train)
centroid_19d = advbench_train_19d.mean(axis=0)
centroid_19d_norm = centroid_19d / np.linalg.norm(centroid_19d)

train_mal_sims_19d = np.array([np.dot(e/np.linalg.norm(e), centroid_19d_norm) for e in advbench_train_19d])
train_norm_sims_19d = np.array([np.dot(e/np.linalg.norm(e), centroid_19d_norm) for e in normal_train_19d])
threshold_19d = (train_mal_sims_19d.mean() + train_norm_sims_19d.mean()) / 2

print(f"  原始空间阈值: {threshold_orig:.4f}")
print(f"  19维空间阈值: {threshold_19d:.4f}")

# 5. 测试GCG攻击
print("\n" + "=" * 70)
print("GCG攻击防御测试结果")
print("=" * 70)

# 提取原始goal和带后缀的prompt
goals = gcg_df['goal'].unique().tolist()
prompts = gcg_df['prompt'].tolist()

print(f"\n测试样本: {len(goals)} 个原始目标, {len(prompts)} 个GCG攻击prompt")

# 计算原始goal的embedding
print("\n计算Embedding...")
goal_embs = np.array(list(embed_model.embed(goals)))
prompt_embs = np.array(list(embed_model.embed(prompts)))

# 原始空间检测
print("\n[原始空间 (384维)]")
goal_sims_orig = np.array([np.dot(e/np.linalg.norm(e), centroid_orig_norm) for e in goal_embs])
prompt_sims_orig = np.array([np.dot(e/np.linalg.norm(e), centroid_orig_norm) for e in prompt_embs])

goal_detection_orig = (goal_sims_orig > threshold_orig).mean()
prompt_detection_orig = (prompt_sims_orig > threshold_orig).mean()

print(f"  原始目标 (goal) 检出率: {goal_detection_orig*100:.1f}%")
print(f"  GCG攻击 (prompt) 检出率: {prompt_detection_orig*100:.1f}%")
print(f"  相似度变化: {goal_sims_orig.mean():.4f} -> {prompt_sims_orig.mean():.4f} ({(prompt_sims_orig.mean()-goal_sims_orig.mean())*100:+.2f}%)")

# 压缩空间检测
print("\n[学习型投影 (19维)]")
goal_embs_19d = apply_projection(model_19d, goal_embs)
prompt_embs_19d = apply_projection(model_19d, prompt_embs)

goal_sims_19d = np.array([np.dot(e/np.linalg.norm(e), centroid_19d_norm) for e in goal_embs_19d])
prompt_sims_19d = np.array([np.dot(e/np.linalg.norm(e), centroid_19d_norm) for e in prompt_embs_19d])

goal_detection_19d = (goal_sims_19d > threshold_19d).mean()
prompt_detection_19d = (prompt_sims_19d > threshold_19d).mean()

print(f"  原始目标 (goal) 检出率: {goal_detection_19d*100:.1f}%")
print(f"  GCG攻击 (prompt) 检出率: {prompt_detection_19d*100:.1f}%")
print(f"  相似度变化: {goal_sims_19d.mean():.4f} -> {prompt_sims_19d.mean():.4f} ({(prompt_sims_19d.mean()-goal_sims_19d.mean())*100:+.2f}%)")

# 6. 分析成功越狱的样本
print("\n" + "=" * 70)
print("成功越狱样本分析")
print("=" * 70)

jailbroken_prompts = gcg_df[gcg_df['jailbroken'] == True]['prompt'].tolist()
if jailbroken_prompts:
    jailbroken_embs = np.array(list(embed_model.embed(jailbroken_prompts)))
    jailbroken_sims_orig = np.array([np.dot(e/np.linalg.norm(e), centroid_orig_norm) for e in jailbroken_embs])
    jailbroken_detection_orig = (jailbroken_sims_orig > threshold_orig).mean()
    
    jailbroken_embs_19d = apply_projection(model_19d, jailbroken_embs)
    jailbroken_sims_19d = np.array([np.dot(e/np.linalg.norm(e), centroid_19d_norm) for e in jailbroken_embs_19d])
    jailbroken_detection_19d = (jailbroken_sims_19d > threshold_19d).mean()
    
    print(f"\n成功越狱的GCG攻击: {len(jailbroken_prompts)} 条")
    print(f"  原始空间检出率: {jailbroken_detection_orig*100:.1f}%")
    print(f"  19维空间检出率: {jailbroken_detection_19d*100:.1f}%")
else:
    print("\n没有成功越狱的样本")

# 7. 汇总结果
print("\n" + "=" * 70)
print("汇总对比")
print("=" * 70)

print(f"\n{'样本类型':<25} {'原始384维':<15} {'投影19维':<15}")
print("-" * 55)
print(f"{'原始目标 (goal)':<25} {goal_detection_orig*100:.1f}%          {goal_detection_19d*100:.1f}%")
print(f"{'GCG攻击 (prompt)':<25} {prompt_detection_orig*100:.1f}%          {prompt_detection_19d*100:.1f}%")
if jailbroken_prompts:
    print(f"{'成功越狱的GCG攻击':<25} {jailbroken_detection_orig*100:.1f}%          {jailbroken_detection_19d*100:.1f}%")

# 保存结果
results = {
    'gcg_attack_count': len(prompts),
    'jailbroken_count': int(gcg_df['jailbroken'].sum()),
    'original_space': {
        'goal_detection_rate': float(goal_detection_orig),
        'prompt_detection_rate': float(prompt_detection_orig),
        'goal_sim_mean': float(goal_sims_orig.mean()),
        'prompt_sim_mean': float(prompt_sims_orig.mean()),
    },
    'learned_projection_19d': {
        'goal_detection_rate': float(goal_detection_19d),
        'prompt_detection_rate': float(prompt_detection_19d),
        'goal_sim_mean': float(goal_sims_19d.mean()),
        'prompt_sim_mean': float(prompt_sims_19d.mean()),
    }
}

if jailbroken_prompts:
    results['jailbroken_detection'] = {
        'original_space': float(jailbroken_detection_orig),
        'learned_projection_19d': float(jailbroken_detection_19d),
    }

os.makedirs("embedding_db/bge-small-en-v1.5/results/gcg_attacks", exist_ok=True)
with open("embedding_db/bge-small-en-v1.5/results/gcg_attacks/jbb_gcg_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n结果已保存到: embedding_db/bge-small-en-v1.5/results/gcg_attacks/jbb_gcg_results.json")
