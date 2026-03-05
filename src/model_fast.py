#!/usr/bin/env python3
"""
DIN-V2 with manual Transformer implementation (avoids slow PyTorch CPU TransformerEncoder).
"""
import math, torch, torch.nn as nn, torch.nn.functional as F

class ManualMultiHeadAttention(nn.Module):
    """Simple manual multi-head attention without PyTorch's slow CPU path."""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, causal=True):
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, H, S, D
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if causal:
            mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
            attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, S, D)
        return self.out_proj(out)

class ManualTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = ManualMultiHeadAttention(embed_dim, num_heads, dropout)
        self.ff = nn.Sequential(nn.Linear(embed_dim, ff_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_dim, embed_dim))
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, causal=True):
        x = x + self.dropout(self.attn(self.ln1(x), causal=causal))
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x

class FastTransformerEncoder(nn.Module):
    """Fast CPU-friendly Transformer with causal mask and learnable position encoding."""
    def __init__(self, embed_dim, num_heads=4, num_layers=2, max_seq_len=30, dropout=0.1, ff_dim=128):
        super().__init__()
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.blocks = nn.ModuleList([ManualTransformerBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, padding_mask=None):
        B, S, D = x.shape
        if padding_mask is not None:
            x = x * padding_mask.unsqueeze(-1)
        pos = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.position_embedding(pos)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, causal=True)
        x = self.final_norm(x)
        if padding_mask is not None:
            x = x * padding_mask.unsqueeze(-1)
        return x

class TargetAttention(nn.Module):
    def __init__(self, embed_dim, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(embed_dim*4, hidden_dim), nn.PReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.PReLU(), nn.Linear(hidden_dim, 1))
    
    def forward(self, query, keys, mask=None):
        S = keys.size(1)
        q = query.unsqueeze(1).expand(-1, S, -1)
        att_in = torch.cat([q, keys, q-keys, q*keys], dim=-1)
        scores = self.mlp(att_in).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=-1)
        return torch.bmm(weights.unsqueeze(1), keys).squeeze(1)

class DINV2Fast(nn.Module):
    """DIN-V2 with fast manual Transformer for CPU."""
    def __init__(self, num_items, num_categories, num_behaviors=4, embed_dim=32,
                 num_heads=4, num_transformer_layers=2, max_seq_len=30,
                 hidden_dims=[256,128,64], dropout=0.1, ff_dim=128, **kw):
        super().__init__()
        self.embed_dim = embed_dim
        self.model_name = "DIN-V2"
        self.item_embedding = nn.Embedding(num_items, embed_dim, padding_idx=0)
        self.category_embedding = nn.Embedding(num_categories, embed_dim, padding_idx=0)
        self.behavior_embedding = nn.Embedding(num_behaviors, embed_dim)
        self.transformer_encoder = FastTransformerEncoder(embed_dim, num_heads, num_transformer_layers, max_seq_len, dropout, ff_dim)
        self.target_attention = TargetAttention(embed_dim, 64)
        
        layers = []
        prev = embed_dim * 3
        for d in hidden_dims:
            layers.extend([nn.Linear(prev, d), nn.BatchNorm1d(d), nn.PReLU(), nn.Dropout(dropout)])
            prev = d
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)
        self._init()
    
    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
                if m.padding_idx is not None: nn.init.zeros_(m.weight[m.padding_idx])
    
    def forward(self, hist_items, hist_behaviors, hist_categories, mask, target_item, target_category, **kw):
        hi = self.item_embedding(hist_items) + self.behavior_embedding(hist_behaviors)
        enhanced = self.transformer_encoder(hi, padding_mask=mask)
        te = self.item_embedding(target_item.squeeze(1))
        tc = self.category_embedding(target_category.squeeze(1))
        user = self.target_attention(te, enhanced, mask)
        return self.mlp(torch.cat([user, te, tc], dim=-1))
    
    def get_model_info(self):
        return {'model_name': self.model_name, 'total_params': sum(p.numel() for p in self.parameters())}
