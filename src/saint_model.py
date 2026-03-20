import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, h, -1).transpose(1, 2), qkv)

        dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, h, 1, 1)
            dots.masked_fill_(~mask, torch.finfo(dots.dtype).min)

        attn = dots.softmax(dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)


class ColumnAttentionBlock(nn.Module):
    """
    列间自注意力 (Intrasample / Feature-wise Attention)
    在单条样本内部，不同特征之间通过注意力交互
    """
    def __init__(self, dim, heads, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = MultiHeadAttention(dim, heads=heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        # x shape: [batch_size, num_features, dim]
        x = self.norm1(x + self.attn(x, mask))
        x = self.norm2(x + self.ff(x))
        return x


class RowAttentionBlock(nn.Module):
    """
    行间注意力 (Intersample / Row-wise Attention)
    在小批量内，不同样本之间做注意力
    """
    def __init__(self, dim, heads, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = MultiHeadAttention(dim, heads=heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        # x shape: [batch_size, num_features, dim]
        # 转换为 [num_features, batch_size, dim] 来做行间注意力
        x = x.transpose(0, 1)  # [num_features, batch_size, dim]
        x = self.norm1(x + self.attn(x, mask))
        x = self.norm2(x + self.ff(x))
        x = x.transpose(0, 1)  # 转换回 [batch_size, num_features, dim]
        return x


class SAINTEncoderBlock(nn.Module):
    """
    SAINT 编码器块：列注意力 + 行注意力
    """
    def __init__(self, dim, heads, mlp_dim, dropout=0.0):
        super().__init__()
        self.column_attn = ColumnAttentionBlock(dim, heads, mlp_dim, dropout)
        self.row_attn = RowAttentionBlock(dim, heads, mlp_dim, dropout)

    def forward(self, x):
        x = self.column_attn(x)
        x = self.row_attn(x)
        return x


class NumericalEmbedding(nn.Module):
    """
    改进的连续特征嵌入方式
    不是简单标准化后直接输入，而是通过可学习的参数更好地与离散特征统一
    """
    def __init__(self, num_features, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_features, dim))
        self.bias = nn.Parameter(torch.randn(num_features, dim))
        nn.init.normal_(self.weight, std=0.01)
        nn.init.normal_(self.bias, std=0.01)

    def forward(self, x):
        # x shape: [batch_size, num_numerical_features]
        x = x.unsqueeze(-1)  # [batch_size, num_numerical_features, 1]
        weight = self.weight.unsqueeze(0)  # [1, num_numerical_features, dim]
        bias = self.bias.unsqueeze(0)  # [1, num_numerical_features, dim]
        return x * weight + bias


class SAINT(nn.Module):
    """
    SAINT: Self-Attention and Intersample Attention Transformer
    
    基于论文: SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training
    """
    def __init__(
        self,
        num_categorical_features,
        num_numerical_features,
        num_boolean_features,
        cardinalities,
        embedding_dim=32,
        depth=6,
        heads=8,
        mlp_dim=128,
        dropout=0.1,
    ):
        super().__init__()
        
        self.num_categorical = num_categorical_features
        self.num_numerical = num_numerical_features
        self.num_boolean = num_boolean_features
        self.total_features = (
            num_categorical_features + num_numerical_features + num_boolean_features
        )
        
        # 类别特征嵌入
        self.categorical_embeddings = nn.ModuleList()
        for card in cardinalities:
            self.categorical_embeddings.append(nn.Embedding(card, embedding_dim))
        
        # 改进的数值特征嵌入
        if num_numerical_features > 0:
            self.numerical_embeddings = NumericalEmbedding(num_numerical_features, embedding_dim)
        else:
            self.numerical_embeddings = None
        
        # 布尔特征嵌入 ('true'->0, 'false'->1, 'null'->2)
        self.boolean_embeddings = nn.ModuleList()
        for _ in range(num_boolean_features):
            self.boolean_embeddings.append(nn.Embedding(3, embedding_dim))
        
        # 特征位置编码
        self.feature_pos_emb = nn.Parameter(torch.randn(1, self.total_features, embedding_dim))
        
        # SAINT 编码器（列注意力 + 行注意力）
        self.transformer = nn.ModuleList([
            SAINTEncoderBlock(embedding_dim, heads, mlp_dim, dropout)
            for _ in range(depth)
        ])
        
        # 投影头 (用于对比学习)
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            GELU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def _encode_features(self, categorical_data, numerical_data, boolean_data):
        embeddings = []
        
        # 编码类别特征
        for i, emb_layer in enumerate(self.categorical_embeddings):
            if categorical_data is not None and i < categorical_data.size(1):
                emb = emb_layer(categorical_data[:, i].long())
                embeddings.append(emb)
        
        # 编码数值特征（改进的嵌入方式）
        if self.numerical_embeddings is not None and numerical_data is not None:
            num_emb = self.numerical_embeddings(numerical_data)  # [batch_size, num_numerical, dim]
            for i in range(num_emb.size(1)):
                embeddings.append(num_emb[:, i, :])
        
        # 编码布尔特征
        for i, emb_layer in enumerate(self.boolean_embeddings):
            if boolean_data is not None and i < boolean_data.size(1):
                emb = emb_layer(boolean_data[:, i].long())
                embeddings.append(emb)
        
        # 拼接所有特征嵌入
        if len(embeddings) == 0:
            raise ValueError("No features provided")
        
        x = torch.stack(embeddings, dim=1)  # [batch_size, num_features, embedding_dim]
        return x

    def forward(
        self,
        categorical_data=None,
        numerical_data=None,
        boolean_data=None,
        return_embeddings=False
    ):
        # 编码特征
        x = self._encode_features(categorical_data, numerical_data, boolean_data)
        
        # 添加位置编码
        x = x + self.feature_pos_emb
        x = self.dropout(x)
        
        # SAINT 编码器（列注意力 + 行注意力）
        for block in self.transformer:
            x = block(x)
        
        # 池化：均值池化所有特征
        x = x.mean(dim=1)  # [batch_size, embedding_dim]
        x = self.norm(x)
        
        if return_embeddings:
            return x
        
        # 对比学习投影
        z = self.projection_head(x)
        return x, z


def augment_batch_with_mask(batch, mask_prob=0.3, replace_prob=0.3):
    """
    对表格样本进行扰动，构造同一行的不同视图（用于对比学习）
    包括：随机掩码、值替换等
    """
    batch1 = {k: v.clone() for k, v in batch.items()}
    batch2 = {k: v.clone() for k, v in batch.items()}
    
    def mask_and_replace(data):
        if data.numel() == 0:
            return data
        
        device = data.device
        mask = torch.rand(data.shape, device=device) < mask_prob
        replace = torch.rand(data.shape, device=device) < replace_prob
        
        # 随机掩码（用 0 或随机值）
        if len(data.shape) == 2:  # [batch, features]
            data[mask] = 0
            # 随机替换为同列的其他值
            if replace.any():
                batch_size = data.shape[0]
                for col in range(data.shape[1]):
                    col_replace = replace[:, col]
                    if col_replace.any():
                        rand_indices = torch.randint(0, batch_size, (col_replace.sum(),), device=device)
                        data[col_replace, col] = data[rand_indices, col]
        
        return data
    
    # 对两种视图应用不同的增强
    for key in ["categorical", "numerical", "boolean"]:
        if key in batch1 and batch1[key].numel() > 0:
            batch1[key] = mask_and_replace(batch1[key])
        if key in batch2 and batch2[key].numel() > 0:
            batch2[key] = mask_and_replace(batch2[key])
    
    return batch1, batch2


class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss for contrastive learning"""
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)
        
        # 归一化
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # 拼接
        z = torch.cat([z_i, z_j], dim=0)
        
        # 相似度矩阵
        sim = torch.matmul(z, z.T) / self.temperature
        
        # 掩码：排除自身
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(mask, -1e9)
        
        # 标签
        labels = torch.arange(batch_size, device=z.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)
        
        loss = self.cross_entropy(sim, labels) / (2 * batch_size)
        return loss
