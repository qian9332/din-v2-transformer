#!/usr/bin/env python3
"""
DIN-V1: Baseline Deep Interest Network.
Standard DIN without behavior type embedding and without Transformer encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TargetAttention(nn.Module):
    """Standard DIN Target Attention mechanism."""
    
    def __init__(self, embed_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.attention_mlp = nn.Sequential(
            nn.Linear(embed_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, query, keys, mask=None):
        """
        Args:
            query: (batch, embed_dim) - target item embedding
            keys: (batch, seq_len, embed_dim) - history sequence embeddings
            mask: (batch, seq_len) - 1 for real, 0 for padding
        Returns:
            output: (batch, embed_dim) - weighted sum of history
        """
        seq_len = keys.size(1)
        query = query.unsqueeze(1).expand(-1, seq_len, -1)  # (B, S, D)
        
        # Concat features for attention
        att_input = torch.cat([
            query, keys, query - keys, query * keys
        ], dim=-1)  # (B, S, 4D)
        
        att_scores = self.attention_mlp(att_input).squeeze(-1)  # (B, S)
        
        if mask is not None:
            att_scores = att_scores.masked_fill(mask == 0, -1e9)
        
        att_weights = F.softmax(att_scores, dim=-1)  # (B, S)
        
        output = torch.bmm(att_weights.unsqueeze(1), keys).squeeze(1)  # (B, D)
        
        return output


class DINV1(nn.Module):
    """
    DIN-V1: Standard Deep Interest Network.
    
    - Does NOT distinguish behavior types
    - Does NOT model sequential dependencies
    - Uses standard Target Attention only
    """
    
    def __init__(self, num_items: int, num_categories: int,
                 embed_dim: int = 64, hidden_dims: list = [256, 128, 64],
                 dropout: float = 0.1, **kwargs):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.model_name = "DIN-V1"
        
        # Embeddings
        self.item_embedding = nn.Embedding(num_items, embed_dim, padding_idx=0)
        self.category_embedding = nn.Embedding(num_categories, embed_dim, padding_idx=0)
        
        # Target Attention
        self.attention = TargetAttention(embed_dim, hidden_dim=64)
        
        # MLP predictor
        # Input: attention_output + target_item + target_category
        mlp_input_dim = embed_dim * 3
        
        layers = []
        prev_dim = mlp_input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
                if m.padding_idx is not None:
                    nn.init.zeros_(m.weight[m.padding_idx])
    
    def forward(self, hist_items, hist_behaviors, hist_categories, mask,
                target_item, target_category, **kwargs):
        """
        Args:
            hist_items: (B, S) historical item IDs
            hist_behaviors: (B, S) historical behavior type IDs (IGNORED in V1)
            hist_categories: (B, S) historical category IDs
            mask: (B, S) attention mask
            target_item: (B, 1) target item ID
            target_category: (B, 1) target category ID
        Returns:
            logits: (B, 1) predicted CTR logits
        """
        # Embeddings
        hist_item_emb = self.item_embedding(hist_items)  # (B, S, D)
        target_item_emb = self.item_embedding(target_item.squeeze(1))  # (B, D)
        target_cat_emb = self.category_embedding(target_category.squeeze(1))  # (B, D)
        
        # V1: Only use item embeddings, ignore behavior types
        hist_emb = hist_item_emb  # (B, S, D)
        
        # Target Attention
        user_interest = self.attention(target_item_emb, hist_emb, mask)  # (B, D)
        
        # Concat and predict
        mlp_input = torch.cat([user_interest, target_item_emb, target_cat_emb], dim=-1)
        logits = self.mlp(mlp_input)
        
        return logits
