#!/usr/bin/env python3
"""
V7前置分类器训练脚本

目标：高准确率的前置分类器，允许较大灰色地带
- 微调BGE embedding + 分类头
- 渐进式解冻策略
- 对比学习 + 分类损失
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np

from model import V6HarmfulDetector
from dataset import prepare_dataset, get_balanced_sampler
from loss import TripleCategoryLoss


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_unfreezing_schedule(epoch, total_epochs):
    """渐进式解冻策略"""
    if epoch <= total_epochs // 3:
        return 'heads_only'
    elif epoch <= 2 * total_epochs // 3:
        return 'last_6_layers'
    else:
        return 'all_layers'


def apply_unfreezing(model, stage):
    """应用解冻策略"""
    if stage == 'heads_only':
        model.freeze_encoder()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    elif stage == 'last_6_layers':
        model.unfreeze_encoder_layers(6)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        model.unfreeze_all()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return trainable


def evaluate(model, dataloader, criterion, device):
    """评估"""
    model.eval()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    all_binary_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            binary_labels = batch['binary_label'].to(device)
            
            embeddings, projections, logits = model(input_ids, attention_mask)
            
            loss, _ = criterion(embeddings, projections, logits, labels)
            total_loss += loss.item()
            
            probs = F.softmax(logits, dim=1)[:, 1]
            preds = (probs > 0.5).long()
            
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_binary_labels.extend(binary_labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
    
    all_preds = np.array(all_preds)
    all_binary_labels = np.array(all_binary_labels)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # 二分类指标
    binary_acc = (all_preds == all_binary_labels).mean()
    
    # 分类指标
    tp = ((all_preds == 1) & (all_binary_labels == 1)).sum()
    fp = ((all_preds == 1) & (all_binary_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_binary_labels == 1)).sum()
    tn = ((all_preds == 0) & (all_binary_labels == 0)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # 灰色样本准确率
    gray_mask = (all_labels == 2) | (all_labels == 3)
    if gray_mask.sum() > 0:
        gray_acc = (all_preds[gray_mask] == all_binary_labels[gray_mask]).mean()
    else:
        gray_acc = 0
    
    # 灰色地带分析（prob在0.3-0.7之间）
    gray_zone_mask = (all_probs >= 0.3) & (all_probs <= 0.7)
    gray_zone_ratio = gray_zone_mask.sum() / len(all_probs)
    
    # 确定样本准确率（prob<0.3或prob>0.7）
    certain_mask = ~gray_zone_mask
    if certain_mask.sum() > 0:
        certain_preds = (all_probs[certain_mask] > 0.5).astype(int)
        certain_acc = (certain_preds == all_binary_labels[certain_mask]).mean()
    else:
        certain_acc = 0
    
    metrics = {
        'loss': total_loss / len(dataloader),
        'binary_acc': binary_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fpr': fpr,
        'gray_acc': gray_acc,
        'gray_zone_ratio': gray_zone_ratio,
        'certain_acc': certain_acc
    }
    
    return metrics


def train(args):
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print("V7前置分类器训练")
    print("="*70)
    
    # 加载tokenizer和数据
    print("\n[1/4] 加载数据...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset, val_dataset = prepare_dataset(args.data_path, tokenizer, args.max_length)
    
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  验证集: {len(val_dataset)} 样本")
    
    # 数据加载器
    train_sampler = get_balanced_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 模型
    print("\n[2/4] 初始化模型...")
    model = V6HarmfulDetector(
        model_name=args.model_name,
        projection_dim=args.projection_dim
    )
    model.to(device)
    
    # 损失函数
    criterion = TripleCategoryLoss(
        margin=args.margin,
        temperature=args.temperature,
        cls_weight=args.cls_weight,
        con_weight=args.con_weight,
        margin_weight=args.margin_weight
    )
    
    # 训练
    print("\n[3/4] 开始训练...")
    
    best_f1 = 0
    best_certain_acc = 0
    best_epoch = 0
    training_log = []
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    global_step = 0
    
    for epoch in range(1, args.epochs + 1):
        # 渐进式解冻
        stage = get_unfreezing_schedule(epoch, args.epochs)
        trainable_params = apply_unfreezing(model, stage)
        
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{args.epochs} | Stage: {stage} | Trainable: {trainable_params:,}")
        print(f"{'='*50}")
        
        # 优化器
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        # 学习率调度
        total_steps = len(train_loader)
        warmup_steps = int(total_steps * 0.1)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            binary_labels = batch['binary_label'].to(device)
            
            optimizer.zero_grad()
            
            embeddings, projections, logits = model(input_ids, attention_mask)
            loss, _ = criterion(embeddings, projections, logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # 定期评估
            if global_step > 0 and global_step % args.eval_steps == 0:
                metrics = evaluate(model, val_loader, criterion, device)
                
                print(f"\n  Step {global_step} | Loss: {metrics['loss']:.4f} | "
                      f"F1: {metrics['f1']:.4f} | Certain_Acc: {metrics['certain_acc']:.4f} | "
                      f"Gray_Zone: {metrics['gray_zone_ratio']*100:.1f}%")
                
                # 保存最佳模型（以certain_acc为主要指标）
                if metrics['certain_acc'] > best_certain_acc:
                    best_certain_acc = metrics['certain_acc']
                    best_f1 = metrics['f1']
                    best_epoch = epoch
                    
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'metrics': metrics
                    }, os.path.join(args.output_dir, 'best_model.pt'))
                    
                    print(f"  *** 新最佳模型! Certain_Acc: {best_certain_acc:.4f} ***")
                
                model.train()
        
        # Epoch结束评估
        metrics = evaluate(model, val_loader, criterion, device)
        training_log.append({
            'epoch': epoch,
            'stage': stage,
            'metrics': metrics
        })
        
        print(f"\nEpoch {epoch} 完成:")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Binary_Acc: {metrics['binary_acc']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  FPR: {metrics['fpr']:.4f}")
        print(f"  Gray_Acc: {metrics['gray_acc']:.4f}")
        print(f"  Gray_Zone_Ratio: {metrics['gray_zone_ratio']*100:.1f}%")
        print(f"  Certain_Acc: {metrics['certain_acc']:.4f}")
    
    # 保存训练日志和配置
    print("\n[4/4] 保存模型...")
    
    config = {
        'model_name': args.model_name,
        'projection_dim': args.projection_dim,
        'best_f1': best_f1,
        'best_certain_acc': best_certain_acc,
        'best_epoch': best_epoch,
        'metrics': training_log[-1]['metrics'] if training_log else {}
    }
    
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    with open(os.path.join(args.output_dir, 'training_log.json'), 'w') as f:
        json.dump(training_log, f, indent=2)
    
    print(f"\n训练完成!")
    print(f"  最佳Certain_Acc: {best_certain_acc:.4f} (Epoch {best_epoch})")
    print(f"  最佳F1: {best_f1:.4f}")
    print(f"  模型保存至: {args.output_dir}")


def main():
    parser = argparse.ArgumentParser()
    
    # 数据
    parser.add_argument('--data_path', type=str, 
                        default='/home/vicuna/ludan/CSonEmbedding/datasets/v7_training')
    parser.add_argument('--output_dir', type=str,
                        default='/home/vicuna/ludan/CSonEmbedding/models/v7_classifier')
    
    # 模型
    parser.add_argument('--model_name', type=str, default='BAAI/bge-base-en-v1.5')
    parser.add_argument('--projection_dim', type=int, default=128)
    parser.add_argument('--max_length', type=int, default=512)
    
    # 训练
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--eval_steps', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    
    # 损失函数
    parser.add_argument('--margin', type=float, default=0.3)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--cls_weight', type=float, default=1.0)
    parser.add_argument('--con_weight', type=float, default=0.5)
    parser.add_argument('--margin_weight', type=float, default=0.3)
    
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
