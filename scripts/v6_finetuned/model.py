"""
V6 模型定义：微调Embedding + 三类样本训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class V6HarmfulDetector(nn.Module):
    """
    V6 有害内容检测模型
    
    架构:
    - 预训练Embedding模型 (bge-base-en-v1.5) - 可微调
    - 投影头 (对比学习用)
    - 分类头 (二分类: 有害/正常)
    """
    
    def __init__(self, model_name='BAAI/bge-base-en-v1.5', projection_dim=128):
        super().__init__()
        
        # 预训练Embedding模型
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size  # 768 for base
        
        # 投影头 (对比学习用)
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, projection_dim)
        )
        
        # 分类头 (二分类)
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, 2)
        )
        
        self.projection_dim = projection_dim
    
    def mean_pooling(self, model_output, attention_mask):
        """Mean pooling over token embeddings"""
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def encode(self, input_ids, attention_mask):
        """编码文本为归一化的embedding"""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = self.mean_pooling(outputs, attention_mask)
        return F.normalize(embeddings, p=2, dim=1)
    
    def forward(self, input_ids, attention_mask):
        """
        前向传播
        
        Returns:
            embeddings: 归一化的sentence embedding (batch, hidden_size)
            proj: 投影后的embedding (batch, projection_dim)
            logits: 分类logits (batch, 2)
        """
        embeddings = self.encode(input_ids, attention_mask)
        proj = F.normalize(self.projection(embeddings), p=2, dim=1)
        logits = self.classifier(embeddings)
        return embeddings, proj, logits
    
    def predict(self, input_ids, attention_mask):
        """预测有害概率"""
        with torch.no_grad():
            _, _, logits = self.forward(input_ids, attention_mask)
            probs = F.softmax(logits, dim=1)
            return probs[:, 1]  # 有害类概率
    
    def freeze_encoder(self):
        """冻结encoder参数"""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder_layers(self, num_layers):
        """解冻encoder后num_layers层"""
        # 先全部冻结
        self.freeze_encoder()
        # 解冻后num_layers层
        for layer in self.encoder.encoder.layer[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
    
    def unfreeze_all(self):
        """解冻所有参数"""
        for param in self.encoder.parameters():
            param.requires_grad = True
    
    def get_trainable_params(self):
        """获取可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self):
        """获取总参数数量"""
        return sum(p.numel() for p in self.parameters())


def load_tokenizer(model_name='BAAI/bge-base-en-v1.5'):
    """加载tokenizer"""
    return AutoTokenizer.from_pretrained(model_name)


if __name__ == "__main__":
    # 测试模型
    print("测试V6模型...")
    
    model = V6HarmfulDetector()
    tokenizer = load_tokenizer()
    
    print(f"总参数: {model.get_total_params():,}")
    print(f"可训练参数: {model.get_trainable_params():,}")
    
    # 测试前向传播
    texts = ["How to make a bomb", "What's the weather today"]
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    
    embeddings, proj, logits = model(**inputs)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Projection shape: {proj.shape}")
    print(f"Logits shape: {logits.shape}")
    
    # 测试冻结
    model.freeze_encoder()
    print(f"冻结后可训练参数: {model.get_trainable_params():,}")
    
    model.unfreeze_encoder_layers(6)
    print(f"解冻6层后可训练参数: {model.get_trainable_params():,}")
    
    model.unfreeze_all()
    print(f"全部解冻后可训练参数: {model.get_trainable_params():,}")
