#!/usr/bin/env python3
"""
Baseline方法对比测试
对比V7与简单分类方法的效果
"""

import sys
import os
import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

# 路径配置
DATASETS_DIR = Path("/home/vicuna/ludan/CSonEmbedding/datasets")
RESULTS_DIR = Path("/home/vicuna/ludan/CSonEmbedding/results/baseline_comparison")
V7_MODEL_PATH = "/home/vicuna/ludan/CSonEmbedding/models/v7_classifier"

# 测试数据集
TEST_DATASETS = {
    "gcg_attacks": {
        "path": DATASETS_DIR / "jailbreakbench" / "jbb_gcg_all.csv",
        "text_col": "prompt", "expected": "harmful", "sample_size": 100,
    },
    "pair_attacks": {
        "path": DATASETS_DIR / "jailbreakbench" / "jbb_pair_all.csv",
        "text_col": "prompt", "expected": "harmful", "sample_size": 86,
    },
    "jailbreakhub": {
        "path": DATASETS_DIR / "jailbreakhub" / "jailbreakhub.csv",
        "text_col": "prompt", "expected": "harmful", "sample_size": 79,
    },
    "benign": {
        "path": DATASETS_DIR / "gcg_attacks" / "jbb_benign_behaviors.csv",
        "text_col": "Goal", "expected": "benign", "sample_size": 100,
    },
}

# 有害关键词列表
HARMFUL_KEYWORDS = [
    "bomb", "explosive", "weapon", "kill", "murder", "hack", "steal",
    "drug", "cocaine", "heroin", "meth", "poison", "attack", "terrorist",
    "malware", "virus", "ransomware", "phishing", "exploit", "vulnerability",
    "jailbreak", "bypass", "ignore previous", "dan", "do anything now",
    "pretend", "roleplay", "hypothetical", "illegal", "harmful",
    "suicide", "self-harm", "violence", "abuse", "torture"
]


def load_dataset(config: dict) -> list:
    """加载数据集"""
    path, text_col = config["path"], config["text_col"]
    sample_size = config.get("sample_size")
    
    if not path.exists():
        return []
    
    df = pd.read_csv(path)
    if text_col not in df.columns:
        for col in ["prompt", "text", "Goal", "goal"]:
            if col in df.columns:
                text_col = col
                break
        else:
            return []
    
    texts = df[text_col].dropna().tolist()
    if sample_size and len(texts) > sample_size:
        texts = texts[:sample_size]
    return texts


def load_training_data():
    """加载训练数据用于TF-IDF和SVM"""
    train_file = DATASETS_DIR / "v7_training" / "train.jsonl"
    texts, labels = [], []
    
    with open(train_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            texts.append(item['text'])
            # 0,2 -> benign, 1,3 -> harmful
            labels.append(1 if item['label'] in [1, 3] else 0)
    
    return texts, labels


# ============ Baseline 1: Keyword Matching ============
class KeywordMatcher:
    """关键词匹配分类器"""
    
    def __init__(self, keywords=HARMFUL_KEYWORDS):
        self.keywords = [k.lower() for k in keywords]
    
    def predict(self, text: str) -> str:
        text_lower = text.lower()
        for keyword in self.keywords:
            if keyword in text_lower:
                return "harmful"
        return "benign"
    
    def predict_batch(self, texts: list) -> list:
        return [self.predict(t) for t in texts]


# ============ Baseline 2: TF-IDF + Logistic Regression ============
class TfidfLRClassifier:
    """TF-IDF + 逻辑回归分类器"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.classifier = LogisticRegression(max_iter=1000)
        self.trained = False
    
    def train(self, texts: list, labels: list):
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)
        self.trained = True
    
    def predict(self, text: str) -> str:
        if not self.trained:
            return "benign"
        X = self.vectorizer.transform([text])
        pred = self.classifier.predict(X)[0]
        return "harmful" if pred == 1 else "benign"
    
    def predict_batch(self, texts: list) -> list:
        if not self.trained:
            return ["benign"] * len(texts)
        X = self.vectorizer.transform(texts)
        preds = self.classifier.predict(X)
        return ["harmful" if p == 1 else "benign" for p in preds]


# ============ Baseline 3: BGE Embedding + Cosine ============
class BGECosineClassifier:
    """BGE Embedding + 余弦相似度分类器"""
    
    def __init__(self, model_name="BAAI/bge-base-en-v1.5"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.harmful_center = None
        self.benign_center = None
    
    def encode(self, texts: list) -> np.ndarray:
        embeddings = []
        batch_size = 32
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, 
                                   truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Mean pooling
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                emb = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embeddings.append(emb.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def train(self, texts: list, labels: list):
        embeddings = self.encode(texts)
        labels = np.array(labels)
        
        self.harmful_center = embeddings[labels == 1].mean(axis=0)
        self.benign_center = embeddings[labels == 0].mean(axis=0)
    
    def predict(self, text: str) -> str:
        emb = self.encode([text])[0]
        
        # 余弦相似度
        harm_sim = np.dot(emb, self.harmful_center) / (np.linalg.norm(emb) * np.linalg.norm(self.harmful_center))
        benign_sim = np.dot(emb, self.benign_center) / (np.linalg.norm(emb) * np.linalg.norm(self.benign_center))
        
        return "harmful" if harm_sim > benign_sim else "benign"
    
    def predict_batch(self, texts: list) -> list:
        embeddings = self.encode(texts)
        results = []
        
        for emb in embeddings:
            harm_sim = np.dot(emb, self.harmful_center) / (np.linalg.norm(emb) * np.linalg.norm(self.harmful_center))
            benign_sim = np.dot(emb, self.benign_center) / (np.linalg.norm(emb) * np.linalg.norm(self.benign_center))
            results.append("harmful" if harm_sim > benign_sim else "benign")
        
        return results


# ============ Baseline 4: BGE Embedding + SVM ============
class BGESVMClassifier:
    """BGE Embedding + SVM分类器"""
    
    def __init__(self, model_name="BAAI/bge-base-en-v1.5"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.svm = SVC(kernel='rbf', C=1.0)
        self.trained = False
    
    def encode(self, texts: list) -> np.ndarray:
        embeddings = []
        batch_size = 32
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True,
                                   truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                emb = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embeddings.append(emb.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def train(self, texts: list, labels: list):
        print("  编码训练数据...")
        embeddings = self.encode(texts)
        print("  训练SVM...")
        self.svm.fit(embeddings, labels)
        self.trained = True
    
    def predict_batch(self, texts: list) -> list:
        if not self.trained:
            return ["benign"] * len(texts)
        embeddings = self.encode(texts)
        preds = self.svm.predict(embeddings)
        return ["harmful" if p == 1 else "benign" for p in preds]


# ============ Baseline 5: PromptGuard ============
class PromptGuardClassifier:
    """Meta PromptGuard分类器"""
    
    def __init__(self, model_name="meta-llama/Prompt-Guard-86M"):
        print(f"  加载PromptGuard模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
    
    def predict_batch(self, texts: list) -> list:
        results = []
        batch_size = 16
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True,
                                   truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # PromptGuard: 0=benign, 1=jailbreak, 2=injection
                preds = outputs.logits.argmax(dim=1).cpu().numpy()
                for p in preds:
                    results.append("harmful" if p > 0 else "benign")
        
        return results


# ============ V7 Classifier ============
class V7ClassifierWrapper:
    """V7分类器封装"""
    
    def __init__(self):
        sys.path.append(str(Path(__file__).parent / "v7_classifier" / "deploy"))
        from v7_classifier import V7Classifier
        self.classifier = V7Classifier()
    
    def predict_batch(self, texts: list) -> list:
        results = self.classifier.predict_batch(texts)
        return [r[0] for r in results]  # 只返回label


def evaluate_classifier(classifier, datasets: dict) -> dict:
    """评估分类器"""
    results = {}
    
    for ds_name, ds_config in datasets.items():
        texts = load_dataset(ds_config)
        if not texts:
            continue
        
        expected = ds_config["expected"]
        preds = classifier.predict_batch(texts)
        
        correct = sum(1 for p in preds if p == expected)
        accuracy = correct / len(texts)
        
        if expected == "harmful":
            # ASR = 攻击成功率 = 被误判为benign的比例
            asr = sum(1 for p in preds if p == "benign") / len(texts)
            results[ds_name] = {"accuracy": accuracy, "asr": asr, "total": len(texts)}
        else:
            # FPR = 误报率 = 被误判为harmful的比例
            fpr = sum(1 for p in preds if p == "harmful") / len(texts)
            results[ds_name] = {"accuracy": accuracy, "fpr": fpr, "total": len(texts)}
    
    return results


def main():
    print("=" * 70)
    print("Baseline方法对比测试")
    print("=" * 70)
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 加载训练数据
    print("\n[1/7] 加载训练数据...")
    train_texts, train_labels = load_training_data()
    print(f"  训练样本: {len(train_texts)}")
    
    # 加载测试数据集
    print("\n[2/7] 加载测试数据集...")
    datasets = {}
    for name, config in TEST_DATASETS.items():
        texts = load_dataset(config)
        if texts:
            datasets[name] = config
            print(f"  {name}: {len(texts)} 条")
    
    all_results = {}
    
    # Baseline 1: Keyword Matching
    print("\n[3/7] 测试 Keyword Matching...")
    keyword_clf = KeywordMatcher()
    all_results["Keyword"] = evaluate_classifier(keyword_clf, datasets)
    
    # Baseline 2: TF-IDF + LR
    print("\n[4/7] 测试 TF-IDF + LogisticRegression...")
    tfidf_clf = TfidfLRClassifier()
    tfidf_clf.train(train_texts, train_labels)
    all_results["TF-IDF+LR"] = evaluate_classifier(tfidf_clf, datasets)
    
    # Baseline 3: BGE + Cosine
    print("\n[5/7] 测试 BGE + Cosine...")
    bge_cos_clf = BGECosineClassifier()
    bge_cos_clf.train(train_texts, train_labels)
    all_results["BGE+Cosine"] = evaluate_classifier(bge_cos_clf, datasets)
    
    # Baseline 4: BGE + SVM
    print("\n[6/7] 测试 BGE + SVM...")
    bge_svm_clf = BGESVMClassifier()
    bge_svm_clf.train(train_texts[:2000], train_labels[:2000])  # SVM太慢，用部分数据
    all_results["BGE+SVM"] = evaluate_classifier(bge_svm_clf, datasets)
    
    # Baseline 5: PromptGuard
    print("\n[7/7] 测试 PromptGuard...")
    try:
        pg_clf = PromptGuardClassifier()
        all_results["PromptGuard"] = evaluate_classifier(pg_clf, datasets)
    except Exception as e:
        print(f"  PromptGuard加载失败: {e}")
        all_results["PromptGuard"] = {"error": str(e)}
    
    # V7
    print("\n[8/7] 测试 V7...")
    v7_clf = V7ClassifierWrapper()
    all_results["V7"] = evaluate_classifier(v7_clf, datasets)
    
    # 计算总体指标
    print("\n" + "=" * 70)
    print("结果对比")
    print("=" * 70)
    
    print(f"\n{'方法':<15} {'GCG ASR':<10} {'PAIR ASR':<10} {'JBHub ASR':<10} {'FPR':<10} {'Avg ASR':<10}")
    print("-" * 70)
    
    for method, results in all_results.items():
        if "error" in results:
            print(f"{method:<15} ERROR: {results['error'][:40]}")
            continue
        
        gcg_asr = results.get("gcg_attacks", {}).get("asr", -1)
        pair_asr = results.get("pair_attacks", {}).get("asr", -1)
        jbhub_asr = results.get("jailbreakhub", {}).get("asr", -1)
        fpr = results.get("benign", {}).get("fpr", -1)
        
        asrs = [a for a in [gcg_asr, pair_asr, jbhub_asr] if a >= 0]
        avg_asr = sum(asrs) / len(asrs) if asrs else -1
        
        print(f"{method:<15} {gcg_asr:<10.3f} {pair_asr:<10.3f} {jbhub_asr:<10.3f} {fpr:<10.3f} {avg_asr:<10.3f}")
    
    # 保存结果
    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # 生成报告
    report = f"""# Baseline方法对比报告

**测试日期**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 方法对比

| 方法 | GCG ASR↓ | PAIR ASR↓ | JBHub ASR↓ | FPR↓ | Avg ASR↓ |
|------|----------|-----------|------------|------|----------|
"""
    
    for method, results in all_results.items():
        if "error" in results:
            report += f"| {method} | ERROR | - | - | - | - |\n"
            continue
        
        gcg_asr = results.get("gcg_attacks", {}).get("asr", -1)
        pair_asr = results.get("pair_attacks", {}).get("asr", -1)
        jbhub_asr = results.get("jailbreakhub", {}).get("asr", -1)
        fpr = results.get("benign", {}).get("fpr", -1)
        
        asrs = [a for a in [gcg_asr, pair_asr, jbhub_asr] if a >= 0]
        avg_asr = sum(asrs) / len(asrs) if asrs else -1
        
        report += f"| {method} | {gcg_asr:.3f} | {pair_asr:.3f} | {jbhub_asr:.3f} | {fpr:.3f} | {avg_asr:.3f} |\n"
    
    report += """
## 说明

- **ASR (Attack Success Rate)**: 攻击绕过率，越低越好
- **FPR (False Positive Rate)**: 误报率，越低越好
- **Avg ASR**: 三种攻击的平均ASR
"""
    
    with open(RESULTS_DIR / "test_report.md", "w") as f:
        f.write(report)
    
    print(f"\n结果已保存到: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
