"""
V6 训练脚本：微调Embedding + 三类样本训练
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from pathlib import Path
from datetime import datetime

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import V6HarmfulDetector, load_tokenizer
from dataset import prepare_dataset, get_balanced_sampler
from loss import TripleCategoryLoss


def parse_args():
    parser = argparse.ArgumentParser(description='V6 Model Training')
    parser.add_argument('--model_name', type=str, default='BAAI/bge-base-en-v1.5')
    parser.add_argument('--data_path', type=str, default='/home/vicuna/ludan/CSonEmbedding/datasets')
    parser.add_argument('--output_dir', type=str, default='/home/vicuna/ludan/CSonEmbedding/models/v6_finetuned')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--projection_dim', type=int, default=128)
    parser.add_argument('--cls_weight', type=float, default=1.0)
    parser.add_argument('--con_weight', type=float, default=0.5)
    parser.add_argument('--margin_weight', type=float, default=0.3)
    parser.add_argument('--margin', type=float, default=0.3)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_steps', type=int, default=500)
    parser.add_argument('--eval_steps', type=int, default=200)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_unfreezing_schedule(epoch, total_epochs=15):
    """
    渐进式解冻策略
    - 阶段1 (1-5): 只训练head
    - 阶段2 (6-10): 解冻后6层
    - 阶段3 (11-15): 全部解冻
    """
    if epoch <= 5:
        return 'head_only'
    elif epoch <= 10:
        return 'last_6'
    else:
        return 'all'


def apply_unfreezing(model, stage):
    """应用解冻策略"""
    if stage == 'head_only':
        model.freeze_encoder()
    elif stage == 'last_6':
        model.unfreeze_encoder_layers(6)
    else:
        model.unfreeze_all()
    return model.get_trainable_params()


def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_binary_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            binary_labels = batch['binary_label'].to(device)
            
            embeddings, proj, logits = model(input_ids, attention_mask)
            loss, _ = criterion(embeddings, proj, logits, labels)
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_binary_labels.extend(binary_labels.cpu().numpy())
    
    # 计算指标
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_binary_labels = np.array(all_binary_labels)
    
    # 二分类准确率
    binary_acc = (all_preds == all_binary_labels).mean()
    
    # 各类准确率
    metrics = {
        'loss': total_loss / len(dataloader),
        'binary_acc': binary_acc,
    }
    
    # 计算F1, Precision, Recall (针对harmful类)
    tp = ((all_preds == 1) & (all_binary_labels == 1)).sum()
    fp = ((all_preds == 1) & (all_binary_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_binary_labels == 1)).sum()
    tn = ((all_preds == 0) & (all_binary_labels == 0)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1'] = f1
    metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # 灰色样本准确率
    gray_mask = (all_labels == 2) | (all_labels == 3)
    if gray_mask.sum() > 0:
        gray_preds = all_preds[gray_mask]
        gray_binary = all_binary_labels[gray_mask]
        metrics['gray_acc'] = (gray_preds == gray_binary).mean()
    
    return metrics


def train(args):
    """主训练函数"""
    print("=" * 60)
    print("V6 模型训练")
    print("=" * 60)
    print(f"设备: {args.device}")
    print(f"模型: {args.model_name}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    
    set_seed(args.seed)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载tokenizer和数据
    print("\n加载数据...")
    tokenizer = load_tokenizer(args.model_name)
    train_dataset, val_dataset = prepare_dataset(
        args.data_path, tokenizer, args.max_length
    )
    
    # 创建DataLoader
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
    
    # 创建模型
    print("\n创建模型...")
    model = V6HarmfulDetector(
        model_name=args.model_name,
        projection_dim=args.projection_dim
    )
    model = model.to(args.device)
    
    print(f"总参数: {model.get_total_params():,}")
    
    # 损失函数
    criterion = TripleCategoryLoss(
        margin=args.margin,
        temperature=args.temperature,
        cls_weight=args.cls_weight,
        con_weight=args.con_weight,
        margin_weight=args.margin_weight
    )
    
    # 训练循环
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    best_f1 = 0
    global_step = 0
    training_log = []
    
    for epoch in range(1, args.epochs + 1):
        # 应用解冻策略
        stage = get_unfreezing_schedule(epoch, args.epochs)
        trainable_params = apply_unfreezing(model, stage)
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs} - 解冻策略: {stage}")
        print(f"可训练参数: {trainable_params:,}")
        print(f"{'='*60}")
        
        # 重新创建优化器 (因为参数变化)
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=0.01
        )
        
        # 学习率调度
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps // args.epochs,
            num_training_steps=len(train_loader)
        )
        
        model.train()
        epoch_loss = 0
        epoch_losses = {'cls': 0, 'con': 0, 'margin': 0}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            labels = batch['label'].to(args.device)
            
            # 前向传播
            embeddings, proj, logits = model(input_ids, attention_mask)
            loss, loss_dict = criterion(embeddings, proj, logits, labels)
            
            # 反向传播
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            
            epoch_loss += loss.item() * args.gradient_accumulation_steps
            for k, v in loss_dict.items():
                if k in epoch_losses:
                    epoch_losses[k] += v
            
            pbar.set_postfix({
                'loss': f"{loss.item()*args.gradient_accumulation_steps:.4f}",
                'cls': f"{loss_dict['cls']:.4f}",
                'con': f"{loss_dict.get('con', 0):.4f}"
            })
            
            # 定期评估
            if global_step > 0 and global_step % args.eval_steps == 0:
                metrics = evaluate(model, val_loader, criterion, args.device)
                print(f"\n  Step {global_step}: F1={metrics['f1']:.4f}, "
                      f"Prec={metrics['precision']:.4f}, Rec={metrics['recall']:.4f}, "
                      f"FPR={metrics['fpr']:.4f}")
                model.train()
        
        # Epoch结束评估
        avg_loss = epoch_loss / len(train_loader)
        metrics = evaluate(model, val_loader, criterion, args.device)
        
        log_entry = {
            'epoch': epoch,
            'train_loss': avg_loss,
            'val_loss': metrics['loss'],
            'f1': metrics['f1'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'fpr': metrics['fpr'],
            'binary_acc': metrics['binary_acc'],
            'gray_acc': metrics.get('gray_acc', 0),
            'stage': stage
        }
        training_log.append(log_entry)
        
        print(f"\nEpoch {epoch} 结果:")
        print(f"  Train Loss: {avg_loss:.4f}")
        print(f"  Val Loss: {metrics['loss']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  FPR: {metrics['fpr']:.4f}")
        print(f"  Binary Acc: {metrics['binary_acc']:.4f}")
        if 'gray_acc' in metrics:
            print(f"  Gray Acc: {metrics['gray_acc']:.4f}")
        
        # 保存最佳模型
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            print(f"  ✓ 新最佳模型! F1={best_f1:.4f}")
            
            # 保存模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1': best_f1,
                'metrics': metrics,
            }, output_dir / 'best_model.pt')
            
            # 保存配置
            config = {
                'model_name': args.model_name,
                'projection_dim': args.projection_dim,
                'best_f1': best_f1,
                'best_epoch': epoch,
                'metrics': metrics,
            }
            with open(output_dir / 'config.json', 'w') as f:
                json.dump(config, f, indent=2)
    
    # 保存训练日志
    with open(output_dir / 'training_log.json', 'w') as f:
        json.dump(training_log, f, indent=2)
    
    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"最佳F1: {best_f1:.4f}")
    print(f"模型保存至: {output_dir}")
    print("=" * 60)
    
    return best_f1


if __name__ == "__main__":
    args = parse_args()
    train(args)
