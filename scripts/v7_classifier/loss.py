"""
V6 损失函数：三类样本联合训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripleCategoryLoss(nn.Module):
    """
    三类样本联合损失函数
    
    包含:
    - 分类损失: 二分类 (benign/gray_benign vs harmful/gray_harmful)
    - 对比损失: 同类拉近，异类推远
    - 边界损失: 灰色样本的特殊处理
    
    标签:
    - 0: benign (正常)
    - 1: harmful (恶意)
    - 2: gray_benign (灰色-正常)
    - 3: gray_harmful (灰色-恶意)
    """
    
    def __init__(self, margin=0.3, temperature=0.07, 
                 cls_weight=1.0, con_weight=0.5, margin_weight=0.3):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
        self.cls_weight = cls_weight
        self.con_weight = con_weight
        self.margin_weight = margin_weight
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, embeddings, proj, logits, labels):
        """
        计算总损失
        
        Args:
            embeddings: 原始embedding (batch, hidden_size)
            proj: 投影后的embedding (batch, projection_dim)
            logits: 分类logits (batch, 2)
            labels: 四分类标签 (batch,) - 0,1,2,3
        
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的字典
        """
        # 1. 分类损失 (二分类: 0,2→正常(0), 1,3→恶意(1))
        binary_labels = ((labels == 1) | (labels == 3)).long()
        L_cls = self.ce_loss(logits, binary_labels)
        
        # 2. 对比损失
        L_con = self.contrastive_loss(proj, labels)
        
        # 3. 边界损失 (灰色样本特殊处理)
        L_margin = self.margin_loss(embeddings, labels)
        
        total = (self.cls_weight * L_cls + 
                 self.con_weight * L_con + 
                 self.margin_weight * L_margin)
        
        return total, {
            'cls': L_cls.item(),
            'con': L_con.item(),
            'margin': L_margin.item(),
            'total': total.item()
        }
    
    def contrastive_loss(self, proj, labels):
        """
        InfoNCE对比损失
        
        同类为正样本: (0,2)同类, (1,3)同类
        """
        batch_size = proj.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=proj.device)
        
        # 计算相似度矩阵
        sim_matrix = torch.mm(proj, proj.t()) / self.temperature
        
        # 构建正样本mask
        # benign类: 0, 2
        # harmful类: 1, 3
        is_benign = (labels == 0) | (labels == 2)
        is_harmful = (labels == 1) | (labels == 3)
        
        # 同类为正样本
        positive_mask = (is_benign.unsqueeze(1) & is_benign.unsqueeze(0)) | \
                        (is_harmful.unsqueeze(1) & is_harmful.unsqueeze(0))
        
        # 排除自身
        positive_mask.fill_diagonal_(False)
        
        # 检查是否有正样本
        positive_count = positive_mask.sum(dim=1).float()
        valid = positive_count > 0
        
        if valid.sum() == 0:
            return torch.tensor(0.0, device=proj.device)
        
        # InfoNCE loss
        # 对于每个样本，计算 log(exp(sim_pos) / sum(exp(sim_all)))
        exp_sim = torch.exp(sim_matrix)
        
        # 排除自身
        mask_self = torch.eye(batch_size, dtype=torch.bool, device=proj.device)
        exp_sim = exp_sim.masked_fill(mask_self, 0)
        
        # 分母: 所有非自身样本的exp(sim)之和
        denominator = exp_sim.sum(dim=1, keepdim=True)
        
        # 分子: 正样本的exp(sim)
        pos_exp_sim = exp_sim * positive_mask.float()
        
        # log(正样本 / 所有)
        log_prob = torch.log(pos_exp_sim / (denominator + 1e-8) + 1e-8)
        
        # 对每个样本，取正样本的平均
        loss = -(log_prob * positive_mask.float()).sum(dim=1) / positive_count.clamp(min=1)
        
        return loss[valid].mean()
    
    def margin_loss(self, embeddings, labels):
        """
        边界损失: 确保灰色样本在正确一侧
        
        - gray_benign (2) 应该更接近 benign (0) 的中心
        - gray_harmful (3) 应该更接近 harmful (1) 的中心
        """
        # 获取各类mask
        benign_mask = labels == 0
        harmful_mask = labels == 1
        gray_benign_mask = labels == 2
        gray_harmful_mask = labels == 3
        
        # 需要有足够的样本计算中心
        if benign_mask.sum() < 2 or harmful_mask.sum() < 2:
            return torch.tensor(0.0, device=embeddings.device)
        
        # 计算类中心
        benign_center = embeddings[benign_mask].mean(dim=0)
        harmful_center = embeddings[harmful_mask].mean(dim=0)
        
        loss = torch.tensor(0.0, device=embeddings.device)
        count = 0
        
        # 灰色-正常应该更接近正常中心
        if gray_benign_mask.sum() > 0:
            gray_benign = embeddings[gray_benign_mask]
            sim_to_benign = F.cosine_similarity(gray_benign, benign_center.unsqueeze(0), dim=1)
            sim_to_harmful = F.cosine_similarity(gray_benign, harmful_center.unsqueeze(0), dim=1)
            # 希望 sim_to_benign > sim_to_harmful + margin
            margin_loss = F.relu(sim_to_harmful - sim_to_benign + self.margin)
            loss = loss + margin_loss.mean()
            count += 1
        
        # 灰色-恶意应该更接近恶意中心
        if gray_harmful_mask.sum() > 0:
            gray_harmful = embeddings[gray_harmful_mask]
            sim_to_benign = F.cosine_similarity(gray_harmful, benign_center.unsqueeze(0), dim=1)
            sim_to_harmful = F.cosine_similarity(gray_harmful, harmful_center.unsqueeze(0), dim=1)
            # 希望 sim_to_harmful > sim_to_benign + margin
            margin_loss = F.relu(sim_to_benign - sim_to_harmful + self.margin)
            loss = loss + margin_loss.mean()
            count += 1
        
        return loss / max(count, 1)


class SimpleBinaryLoss(nn.Module):
    """简单二分类损失 (用于对比测试)"""
    
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, embeddings, proj, logits, labels):
        binary_labels = ((labels == 1) | (labels == 3)).long()
        loss = self.ce_loss(logits, binary_labels)
        return loss, {'cls': loss.item(), 'total': loss.item()}


if __name__ == "__main__":
    # 测试损失函数
    print("测试损失函数...")
    
    batch_size = 8
    hidden_size = 768
    proj_dim = 128
    
    # 模拟数据
    embeddings = F.normalize(torch.randn(batch_size, hidden_size), dim=1)
    proj = F.normalize(torch.randn(batch_size, proj_dim), dim=1)
    logits = torch.randn(batch_size, 2)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])  # 每类2个
    
    # 测试TripleCategoryLoss
    criterion = TripleCategoryLoss()
    total_loss, loss_dict = criterion(embeddings, proj, logits, labels)
    
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Loss breakdown: {loss_dict}")
    
    # 测试梯度
    total_loss.backward()
    print("梯度计算成功!")
