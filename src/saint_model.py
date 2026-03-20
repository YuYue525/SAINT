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


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = MultiHeadAttention(dim, heads=heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.attn(x, mask))
        x = self.norm2(x + self.ff(x))
        return x


class SAINT(nn.Module):
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
        
        # 数值特征嵌入
        self.numerical_embeddings = nn.ModuleList()
        for _ in range(num_numerical_features):
            self.numerical_embeddings.append(nn.Linear(1, embedding_dim))
        
        # 布尔特征嵌入 ('true', 'false', 'null' -> 3个类别)
        self.boolean_embeddings = nn.ModuleList()
        for _ in range(num_boolean_features):
            self.boolean_embeddings.append(nn.Embedding(3, embedding_dim))
        
        # 特征位置编码
        self.feature_pos_emb = nn.Parameter(torch.randn(1, self.total_features, embedding_dim))
        
        # Transformer 编码器
        self.transformer = nn.ModuleList([
            TransformerBlock(embedding_dim, heads, mlp_dim, dropout)
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
        
        # 编码数值特征
        for i, emb_layer in enumerate(self.numerical_embeddings):
            if numerical_data is not None and i < numerical_data.size(1):
                emb = emb_layer(numerical_data[:, i:i+1])
                embeddings.append(emb)
        
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
        
        # Transformer 编码
        for block in self.transformer:
            x = block(x)
        
        # 池化：使用 <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> token 或者均值池化
        x = x.mean(dim=1)  # [batch_size, embedding_dim]
        x = self.norm(x)
        
        if return_embeddings:
            return x
        
        # 对比学习投影
        z = self.projection_head(x)
        return x, z


class NTXentLoss(nn.Module):
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
