#!/usr/bin/env python3
"""
DIN-V2: Enhanced Deep Interest Network with Behavior-Type Embedding + Transformer Encoder.

Key upgrades over V1:
1. Behavior Type Embedding: Independent embedding for each behavior type (click, browse, 
   inquiry, add-to-cart), added to item embedding so same item has different representations 
   under different behavior contexts.

2. Transformer Encoder: 2-layer, 4-head Transformer with causal mask and learnable position 
   encoding, placed before Target Attention to model intra-sequence dependencies.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalTransformerEncoder(nn.Module):
    """
    Transformer Encoder with causal mask and learnable position encoding.
    Uses is_causal=True for optimized attention computation.
    Padding is handled by zeroing out padded positions before and after transformer.
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 4, num_layers: int = 2,
                 max_seq_len: int = 50, dropout: float = 0.1, ff_dim: int = 256):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Learnable position encoding
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # Transformer Encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            enable_nested_tensor=False
        )
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, padding_mask=None):
        """
        Args:
            x: (batch, seq_len, embed_dim) - input sequence embeddings
            padding_mask: (batch, seq_len) - 1=valid, 0=padding
        Returns:
            output: (batch, seq_len, embed_dim) - enhanced sequence representations
        """
        batch_size, seq_len, _ = x.shape
        
        # Zero out padded positions
        if padding_mask is not None:
            x = x * padding_mask.unsqueeze(-1)
        
        # Add learnable positional encoding
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)
        x = x + pos_emb
        x = self.dropout(x)
        
        # Generate causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
        
        # Apply Transformer with optimized causal path
        output = self.transformer(x, mask=causal_mask, is_causal=True)
        output = self.layer_norm(output)
        
        # Zero out padded positions in output
        if padding_mask is not None:
            output = output * padding_mask.unsqueeze(-1)
        
        return output


class TargetAttentionV2(nn.Module):
    """Enhanced Target Attention for DIN-V2."""
    
    def __init__(self, embed_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.attention_mlp = nn.Sequential(
            nn.Linear(embed_dim * 4, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, query, keys, mask=None):
        seq_len = keys.size(1)
        query = query.unsqueeze(1).expand(-1, seq_len, -1)
        
        att_input = torch.cat([query, keys, query - keys, query * keys], dim=-1)
        att_scores = self.attention_mlp(att_input).squeeze(-1)
        
        if mask is not None:
            att_scores = att_scores.masked_fill(mask == 0, -1e9)
        
        att_weights = F.softmax(att_scores, dim=-1)
        output = torch.bmm(att_weights.unsqueeze(1), keys).squeeze(1)
        return output


class DINV2(nn.Module):
    """
    DIN-V2: Behavior-Type-Aware Deep Interest Network with Transformer Encoder.
    
    Architecture:
    1. Item Embedding + Behavior Type Embedding -> composite representation
    2. Transformer Encoder (2 layers, 4 heads, causal mask, learnable positions)
    3. Target Attention -> compute user interest w.r.t. candidate item
    4. MLP -> CTR prediction
    """
    
    def __init__(self, num_items: int, num_categories: int,
                 num_behaviors: int = 4, embed_dim: int = 64,
                 num_heads: int = 4, num_transformer_layers: int = 2,
                 max_seq_len: int = 50, hidden_dims: list = [256, 128, 64],
                 dropout: float = 0.1, ff_dim: int = 256, **kwargs):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.model_name = "DIN-V2"
        
        self.item_embedding = nn.Embedding(num_items, embed_dim, padding_idx=0)
        self.category_embedding = nn.Embedding(num_categories, embed_dim, padding_idx=0)
        self.behavior_embedding = nn.Embedding(num_behaviors, embed_dim)
        
        self.transformer_encoder = CausalTransformerEncoder(
            embed_dim=embed_dim, num_heads=num_heads, num_layers=num_transformer_layers,
            max_seq_len=max_seq_len, dropout=dropout, ff_dim=ff_dim,
        )
        
        self.target_attention = TargetAttentionV2(embed_dim, hidden_dim=64)
        
        mlp_input_dim = embed_dim * 3
        layers = []
        prev_dim = mlp_input_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, dim), nn.BatchNorm1d(dim), nn.PReLU(), nn.Dropout(dropout)])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)
        
        self._init_weights()
    
    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None: nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.01)
                if module.padding_idx is not None: nn.init.zeros_(module.weight[module.padding_idx])
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight); nn.init.zeros_(module.bias)
    
    def forward(self, hist_items, hist_behaviors, hist_categories, mask,
                target_item, target_category, **kwargs):
        # Step 1: Composite Embedding (Item + Behavior Type)
        hist_item_emb = self.item_embedding(hist_items)
        hist_behavior_emb = self.behavior_embedding(hist_behaviors)
        hist_composite = hist_item_emb + hist_behavior_emb
        
        target_item_emb = self.item_embedding(target_item.squeeze(1))
        target_cat_emb = self.category_embedding(target_category.squeeze(1))
        
        # Step 2: Transformer Encoder
        enhanced_seq = self.transformer_encoder(hist_composite, padding_mask=mask)
        
        # Step 3: Target Attention
        user_interest = self.target_attention(target_item_emb, enhanced_seq, mask)
        
        # Step 4: MLP Prediction
        mlp_input = torch.cat([user_interest, target_item_emb, target_cat_emb], dim=-1)
        return self.mlp(mlp_input)
    
    def get_model_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'model_name': self.model_name, 'total_params': total_params,
                'trainable_params': trainable_params, 'embed_dim': self.embed_dim}
