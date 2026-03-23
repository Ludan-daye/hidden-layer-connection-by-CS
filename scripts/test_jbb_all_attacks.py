"""
使用JailbreakBench的所有攻击方法测试防御效果

攻击方法：
- GCG: Greedy Coordinate Gradient (对抗性后缀)
- PAIR: Prompt Automatic Iterative Refinement (自动迭代优化)
- JBC: Jailbreak Chat (越狱对话)
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
print("JailbreakBench 全攻击方法防御测试")
print("=" * 70)

# 1. 加载Embedding模型和训练数据
print("\n[1] 加载Embedding模型和训练数据...")
embed_model = TextEmbedding('BAAI/bge-small-en-v1.5')

advbench_emb = np.load("embedding_db/bge-small-en-v1.5/embeddings/advbench_embeddings.npy")
normal_emb = np.load("embedding_db/bge-small-en-v1.5/embeddings/normal_embeddings.npy")

advbench_train, _ = train_test_split(advbench_emb, test_size=0.2, random_state=RANDOM_SEED)
normal_train, _ = train_test_split(normal_emb, test_size=0.2, random_state=RANDOM_SEED)

print(f"  AdvBench Train: {advbench_train.shape[0]} 条")
print(f"  正常样本 Train: {normal_train.shape[0]} 条")

# 2. 训练投影矩阵
print("\n[2] 训练投影矩阵...")
train_emb = np.vstack([advbench_train, normal_train])
model_19d = train_learned_projection(train_emb, 19, epochs=300)
print("  19维投影矩阵训练完成")

# 3. 计算质心和阈值
centroid_orig = advbench_train.mean(axis=0)
centroid_orig_norm = centroid_orig / np.linalg.norm(centroid_orig)

train_mal_sims = np.array([np.dot(e/np.linalg.norm(e), centroid_orig_norm) for e in advbench_train])
train_norm_sims = np.array([np.dot(e/np.linalg.norm(e), centroid_orig_norm) for e in normal_train])
threshold_orig = (train_mal_sims.mean() + train_norm_sims.mean()) / 2

advbench_train_19d = apply_projection(model_19d, advbench_train)
normal_train_19d = apply_projection(model_19d, normal_train)
centroid_19d = advbench_train_19d.mean(axis=0)
centroid_19d_norm = centroid_19d / np.linalg.norm(centroid_19d)

train_mal_sims_19d = np.array([np.dot(e/np.linalg.norm(e), centroid_19d_norm) for e in advbench_train_19d])
train_norm_sims_19d = np.array([np.dot(e/np.linalg.norm(e), centroid_19d_norm) for e in normal_train_19d])
threshold_19d = (train_mal_sims_19d.mean() + train_norm_sims_19d.mean()) / 2

print(f"  原始空间阈值: {threshold_orig:.4f}")
print(f"  19维空间阈值: {threshold_19d:.4f}")

# 4. 下载并测试所有攻击方法
print("\n[3] 下载JailbreakBench攻击数据...")

attack_methods = ['GCG', 'PAIR', 'JBC']
target_models = ['vicuna-13b-v1.5', 'llama-2-7b-chat-hf']

all_results = {}

for method in attack_methods:
    print(f"\n{'='*70}")
    print(f"攻击方法: {method}")
    print(f"{'='*70}")
    
    all_data = []
    for model_name in target_models:
        try:
            artifact = jbb.read_artifact(method=method, model_name=model_name)
            print(f"  {model_name}: {len(artifact.jailbreaks)} 条攻击")
            for jb in artifact.jailbreaks:
                all_data.append({
                    'model': model_name,
                    'goal': jb.goal,
                    'prompt': jb.prompt,
                    'jailbroken': jb.jailbroken
                })
        except Exception as e:
            print(f"  {model_name}: 获取失败 - {e}")
    
    if not all_data:
        print(f"  跳过 {method}")
        continue
    
    df = pd.DataFrame(all_data)
    print(f"  总计: {len(df)} 条攻击, 成功越狱: {df['jailbroken'].sum()} 条 ({df['jailbroken'].mean()*100:.1f}%)")
    
    # 保存数据
    os.makedirs("datasets/jailbreakbench", exist_ok=True)
    df.to_csv(f"datasets/jailbreakbench/jbb_{method.lower()}_all.csv", index=False)
    
    # 计算Embedding (过滤None值)
    goals = [g for g in df['goal'].unique().tolist() if g is not None and isinstance(g, str)]
    prompts = [p for p in df['prompt'].tolist() if p is not None and isinstance(p, str)]
    
    if not prompts:
        print(f"  跳过 {method}: 没有有效的prompt")
        continue
    
    print(f"\n  计算Embedding...")
    goal_embs = np.array(list(embed_model.embed(goals)))
    prompt_embs = np.array(list(embed_model.embed(prompts)))
    
    # 原始空间检测
    goal_sims_orig = np.array([np.dot(e/np.linalg.norm(e), centroid_orig_norm) for e in goal_embs])
    prompt_sims_orig = np.array([np.dot(e/np.linalg.norm(e), centroid_orig_norm) for e in prompt_embs])
    
    goal_detection_orig = (goal_sims_orig > threshold_orig).mean()
    prompt_detection_orig = (prompt_sims_orig > threshold_orig).mean()
    
    # 压缩空间检测
    goal_embs_19d = apply_projection(model_19d, goal_embs)
    prompt_embs_19d = apply_projection(model_19d, prompt_embs)
    
    goal_sims_19d = np.array([np.dot(e/np.linalg.norm(e), centroid_19d_norm) for e in goal_embs_19d])
    prompt_sims_19d = np.array([np.dot(e/np.linalg.norm(e), centroid_19d_norm) for e in prompt_embs_19d])
    
    goal_detection_19d = (goal_sims_19d > threshold_19d).mean()
    prompt_detection_19d = (prompt_sims_19d > threshold_19d).mean()
    
    # 成功越狱样本分析
    jailbroken_prompts = df[df['jailbroken'] == True]['prompt'].tolist()
    if jailbroken_prompts:
        jailbroken_embs = np.array(list(embed_model.embed(jailbroken_prompts)))
        jailbroken_sims_orig = np.array([np.dot(e/np.linalg.norm(e), centroid_orig_norm) for e in jailbroken_embs])
        jailbroken_detection_orig = (jailbroken_sims_orig > threshold_orig).mean()
        
        jailbroken_embs_19d = apply_projection(model_19d, jailbroken_embs)
        jailbroken_sims_19d = np.array([np.dot(e/np.linalg.norm(e), centroid_19d_norm) for e in jailbroken_embs_19d])
        jailbroken_detection_19d = (jailbroken_sims_19d > threshold_19d).mean()
    else:
        jailbroken_detection_orig = 0
        jailbroken_detection_19d = 0
    
    # 打印结果
    print(f"\n  检测结果:")
    print(f"  {'样本类型':<25} {'原始384维':<15} {'投影19维':<15}")
    print(f"  {'-'*55}")
    print(f"  {'原始目标 (goal)':<25} {goal_detection_orig*100:.1f}%          {goal_detection_19d*100:.1f}%")
    print(f"  {'攻击prompt':<25} {prompt_detection_orig*100:.1f}%          {prompt_detection_19d*100:.1f}%")
    if jailbroken_prompts:
        print(f"  {'成功越狱的攻击':<25} {jailbroken_detection_orig*100:.1f}%          {jailbroken_detection_19d*100:.1f}%")
    
    # 保存结果
    all_results[method] = {
        'attack_count': len(df),
        'jailbroken_count': int(df['jailbroken'].sum()),
        'jailbroken_rate': float(df['jailbroken'].mean()),
        'original_space': {
            'goal_detection': float(goal_detection_orig),
            'prompt_detection': float(prompt_detection_orig),
            'jailbroken_detection': float(jailbroken_detection_orig) if jailbroken_prompts else None,
        },
        'learned_projection_19d': {
            'goal_detection': float(goal_detection_19d),
            'prompt_detection': float(prompt_detection_19d),
            'jailbroken_detection': float(jailbroken_detection_19d) if jailbroken_prompts else None,
        }
    }

# 5. 汇总结果
print("\n" + "=" * 70)
print("汇总对比")
print("=" * 70)

print(f"\n{'攻击方法':<10} {'攻击数':<8} {'越狱率':<10} {'原始检出':<12} {'19维检出':<12} {'越狱检出(原始)':<15} {'越狱检出(19维)':<15}")
print("-" * 90)

for method, result in all_results.items():
    orig_det = result['original_space']['prompt_detection'] * 100
    proj_det = result['learned_projection_19d']['prompt_detection'] * 100
    jb_orig = result['original_space']['jailbroken_detection']
    jb_proj = result['learned_projection_19d']['jailbroken_detection']
    jb_orig_str = f"{jb_orig*100:.1f}%" if jb_orig else "N/A"
    jb_proj_str = f"{jb_proj*100:.1f}%" if jb_proj else "N/A"
    
    print(f"{method:<10} {result['attack_count']:<8} {result['jailbroken_rate']*100:.1f}%      {orig_det:.1f}%        {proj_det:.1f}%        {jb_orig_str:<15} {jb_proj_str:<15}")

# 保存结果
os.makedirs("embedding_db/bge-small-en-v1.5/results/jailbreakbench", exist_ok=True)
with open("embedding_db/bge-small-en-v1.5/results/jailbreakbench/all_attacks_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n结果已保存到: embedding_db/bge-small-en-v1.5/results/jailbreakbench/all_attacks_results.json")
