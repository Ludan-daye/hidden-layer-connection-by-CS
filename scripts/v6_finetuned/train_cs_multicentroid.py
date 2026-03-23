#!/usr/bin/env python3
"""
V6 Fine-tuned Embedding + 学习型CS压缩 + 多质心检测

结合三种技术:
1. V6 fine-tuned embedding: 768维语义向量
2. 学习型CS压缩: 768维 -> 低维(如32/64维)，保持相似度
3. 多质心检测: 不同恶意类型有不同质心

优势:
- 压缩后向量更小，计算更快
- 多质心能捕捉不同类型的恶意模式
- 结合fine-tuned embedding的语义理解能力
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from transformers import AutoTokenizer
from datasets import load_dataset

sys.path.append(str(Path(__file__).parent))
from model import V6HarmfulDetector


class LearnedCSProjection(nn.Module):
    """学习型CS压缩投影"""
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim, bias=False)
        nn.init.normal_(self.proj.weight, mean=0, std=1/np.sqrt(output_dim))
    
    def forward(self, x):
        z = self.proj(x)
        z = F.normalize(z, p=2, dim=-1)
        return z


class MultiCentroidDetector:
    """多质心检测器"""
    
    def __init__(self, n_centroids=5, threshold=0.5):
        self.n_centroids = n_centroids
        self.threshold = threshold
        self.centroids = None  # 恶意质心
        self.benign_centroid = None  # 正常质心
    
    def fit(self, harmful_embeddings, benign_embeddings):
        """训练多质心"""
        # 对恶意样本进行聚类
        kmeans = KMeans(n_clusters=self.n_centroids, random_state=42, n_init=10)
        kmeans.fit(harmful_embeddings)
        self.centroids = kmeans.cluster_centers_
        # L2归一化
        self.centroids = self.centroids / np.linalg.norm(self.centroids, axis=1, keepdims=True)
        
        # 正常样本质心
        self.benign_centroid = np.mean(benign_embeddings, axis=0)
        self.benign_centroid = self.benign_centroid / np.linalg.norm(self.benign_centroid)
        
        print(f"训练完成: {self.n_centroids}个恶意质心, 1个正常质心")
    
    def predict(self, embeddings):
        """预测"""
        # 计算与所有恶意质心的最大相似度
        harmful_sims = embeddings @ self.centroids.T  # [N, n_centroids]
        max_harmful_sim = np.max(harmful_sims, axis=1)  # [N]
        
        # 计算与正常质心的相似度
        benign_sim = embeddings @ self.benign_centroid  # [N]
        
        # 判定: 与恶意质心更近则为恶意
        preds = (max_harmful_sim > benign_sim + self.threshold).astype(int)
        
        return preds, max_harmful_sim, benign_sim


def load_v6_model(model_path, device='cuda'):
    """加载V6模型"""
    config_path = os.path.join(model_path, 'config.json')
    weights_path = os.path.join(model_path, 'best_model.pt')
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model = V6HarmfulDetector(
        model_name=config['model_name'],
        projection_dim=config['projection_dim']
    )
    
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    
    return model, tokenizer


def extract_embeddings(model, tokenizer, texts, device='cuda', batch_size=32):
    """提取V6 fine-tuned embeddings"""
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="提取embeddings"):
        batch_texts = texts[i:i+batch_size]
        
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            embeddings, _, _ = model(inputs['input_ids'], inputs['attention_mask'])
        
        all_embeddings.append(embeddings.cpu().numpy())
    
    return np.vstack(all_embeddings)


def train_cs_projection(embeddings, target_dim, epochs=200, lr=0.01, batch_size=256):
    """训练学习型CS压缩"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
    n_samples = embeddings_tensor.shape[0]
    input_dim = embeddings_tensor.shape[1]
    
    # 归一化
    embeddings_norm = F.normalize(embeddings_tensor, p=2, dim=-1)
    
    # 模型
    model = LearnedCSProjection(input_dim, target_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print(f"\n训练CS压缩: {input_dim}D -> {target_dim}D")
    
    for epoch in range(epochs):
        # 随机采样
        indices = torch.randperm(n_samples)[:batch_size]
        batch = embeddings_tensor[indices]
        batch_norm = embeddings_norm[indices]
        
        # 原始相似度
        original_sim = batch_norm @ batch_norm.T
        
        # 压缩后相似度
        compressed = model(batch)
        compressed_sim = compressed @ compressed.T
        
        # MSE损失
        loss = F.mse_loss(compressed_sim, original_sim)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            with torch.no_grad():
                orig_flat = original_sim.flatten().cpu().numpy()
                comp_flat = compressed_sim.flatten().cpu().numpy()
                corr = np.corrcoef(orig_flat, comp_flat)[0, 1]
            print(f"  Epoch {epoch+1}/{epochs} | Loss: {loss.item():.6f} | SimCorr: {corr:.4f}")
    
    return model


def load_harmful_data(base_path):
    """加载恶意数据"""
    texts = []
    
    # AdvBench
    try:
        ds = load_dataset("walledai/AdvBench", split='train')
        texts.extend([x['prompt'] for x in ds if x.get('prompt')][:500])
    except:
        pass
    
    # HarmBench
    harmbench_path = os.path.join(base_path, 'harmbench')
    if os.path.exists(harmbench_path):
        import pandas as pd
        for file in os.listdir(harmbench_path):
            if file.endswith('.csv'):
                try:
                    df = pd.read_csv(os.path.join(harmbench_path, file))
                    for col in ['Behavior', 'goal', 'prompt']:
                        if col in df.columns:
                            texts.extend(df[col].dropna().tolist()[:300])
                            break
                except:
                    pass
    
    # JailbreakHub
    try:
        ds = load_dataset("rubend18/ChatGPT-Jailbreak-Prompts", split='train')
        texts.extend([x['text'] for x in ds if x.get('text')][:300])
    except:
        pass
    
    # 灰色恶意样本
    gray_path = os.path.join(base_path, 'v6_training', 'gray_harmful.jsonl')
    if os.path.exists(gray_path):
        with open(gray_path, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if item.get('text'):
                        texts.append(item['text'])
                except:
                    pass
    
    return texts


def load_benign_data(base_path):
    """加载正常数据"""
    texts = []
    
    # Alpaca
    try:
        ds = load_dataset("tatsu-lab/alpaca", split='train')
        for item in ds:
            if item.get('instruction'):
                text = item['instruction']
                if item.get('input'):
                    text += " " + item['input']
                texts.append(text)
                if len(texts) >= 500:
                    break
    except:
        pass
    
    # Dolly
    try:
        ds = load_dataset("databricks/databricks-dolly-15k", split='train')
        for item in ds:
            if item.get('instruction'):
                text = item['instruction']
                if item.get('context'):
                    text += " " + item['context']
                texts.append(text)
                if len(texts) >= 1000:
                    break
    except:
        pass
    
    # 灰色正常样本
    gray_path = os.path.join(base_path, 'v6_training', 'gray_benign.jsonl')
    if os.path.exists(gray_path):
        with open(gray_path, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if item.get('text'):
                        texts.append(item['text'])
                except:
                    pass
    
    return texts


def evaluate(detector, embeddings, labels, name):
    """评估"""
    preds, harmful_sims, benign_sims = detector.predict(embeddings)
    
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    print(f"\n{name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1: {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  FPR: {fpr:.4f}")
    print(f"  混淆矩阵: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
    return {
        'name': name,
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'fpr': fpr
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = '/home/vicuna/ludan/CSonEmbedding/models/v6_finetuned'
    data_path = '/home/vicuna/ludan/CSonEmbedding/datasets'
    output_path = '/home/vicuna/ludan/CSonEmbedding/models/v6_cs_multicentroid'
    
    os.makedirs(output_path, exist_ok=True)
    
    print("="*70)
    print("V6 Fine-tuned + 学习型CS压缩 + 多质心检测")
    print("="*70)
    
    # 1. 加载V6模型
    print("\n[1/6] 加载V6 fine-tuned模型...")
    v6_model, tokenizer = load_v6_model(model_path, device)
    
    # 2. 加载训练数据
    print("\n[2/6] 加载训练数据...")
    harmful_texts = load_harmful_data(data_path)
    benign_texts = load_benign_data(data_path)
    print(f"  恶意样本: {len(harmful_texts)}")
    print(f"  正常样本: {len(benign_texts)}")
    
    # 3. 提取embeddings
    print("\n[3/6] 提取V6 fine-tuned embeddings...")
    harmful_embeddings = extract_embeddings(v6_model, tokenizer, harmful_texts, device)
    benign_embeddings = extract_embeddings(v6_model, tokenizer, benign_texts, device)
    print(f"  恶意embeddings: {harmful_embeddings.shape}")
    print(f"  正常embeddings: {benign_embeddings.shape}")
    
    # 合并用于训练CS压缩
    all_embeddings = np.vstack([harmful_embeddings, benign_embeddings])
    
    # 4. 训练CS压缩
    print("\n[4/6] 训练学习型CS压缩...")
    target_dims = [32, 64, 128]
    
    results = {}
    
    for target_dim in target_dims:
        print(f"\n{'='*50}")
        print(f"目标维度: {target_dim}D")
        print(f"{'='*50}")
        
        # 训练CS压缩
        cs_model = train_cs_projection(all_embeddings, target_dim, epochs=200)
        
        # 压缩embeddings
        with torch.no_grad():
            harmful_compressed = cs_model(
                torch.tensor(harmful_embeddings, dtype=torch.float32).to(device)
            ).cpu().numpy()
            benign_compressed = cs_model(
                torch.tensor(benign_embeddings, dtype=torch.float32).to(device)
            ).cpu().numpy()
        
        print(f"\n压缩后维度: {harmful_compressed.shape[1]}D")
        
        # 5. 训练多质心检测器
        print("\n[5/6] 训练多质心检测器...")
        
        for n_centroids in [3, 5, 7]:
            print(f"\n--- {n_centroids}个质心 ---")
            
            detector = MultiCentroidDetector(n_centroids=n_centroids, threshold=0.0)
            detector.fit(harmful_compressed, benign_compressed)
            
            # 在训练集上评估
            train_embeddings = np.vstack([benign_compressed, harmful_compressed])
            train_labels = np.array([0]*len(benign_compressed) + [1]*len(harmful_compressed))
            
            result = evaluate(detector, train_embeddings, train_labels, f"训练集 ({target_dim}D, {n_centroids}质心)")
            
            key = f"{target_dim}d_{n_centroids}c"
            results[key] = {
                'target_dim': target_dim,
                'n_centroids': n_centroids,
                'train_result': result
            }
            
            # 保存模型
            model_file = os.path.join(output_path, f'cs_multicentroid_{key}.npz')
            np.savez(
                model_file,
                cs_weights=cs_model.proj.weight.cpu().detach().numpy(),
                centroids=detector.centroids,
                benign_centroid=detector.benign_centroid,
                threshold=detector.threshold
            )
    
    # 6. 测试最佳配置
    print("\n[6/6] 测试最佳配置...")
    
    # 选择最佳配置 (64D, 5质心)
    best_dim = 64
    best_n_centroids = 5
    
    # 重新加载
    cs_model = LearnedCSProjection(768, best_dim).to(device)
    saved = np.load(os.path.join(output_path, f'cs_multicentroid_{best_dim}d_{best_n_centroids}c.npz'))
    cs_model.proj.weight.data = torch.tensor(saved['cs_weights'], dtype=torch.float32).to(device)
    
    detector = MultiCentroidDetector(n_centroids=best_n_centroids)
    detector.centroids = saved['centroids']
    detector.benign_centroid = saved['benign_centroid']
    detector.threshold = float(saved['threshold'])
    
    # 测试ToxicChat
    print("\n测试ToxicChat...")
    try:
        ds = load_dataset("lmsys/toxic-chat", "toxicchat0124", split='test')
        test_texts = [x['user_input'] for x in ds if x.get('user_input')][:1000]
        test_labels = [1 if x.get('toxicity', 0) == 1 else 0 for x in ds if x.get('user_input')][:1000]
        
        test_embeddings = extract_embeddings(v6_model, tokenizer, test_texts, device)
        with torch.no_grad():
            test_compressed = cs_model(
                torch.tensor(test_embeddings, dtype=torch.float32).to(device)
            ).cpu().numpy()
        
        result = evaluate(detector, test_compressed, test_labels, "ToxicChat")
        results['toxicchat'] = result
    except Exception as e:
        print(f"ToxicChat测试失败: {e}")
    
    # 保存结果
    results_file = os.path.join(output_path, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n结果已保存至: {results_file}")
    print("\n" + "="*70)
    print("完成!")
    print("="*70)


if __name__ == '__main__':
    main()
