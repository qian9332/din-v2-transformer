#!/usr/bin/env python3
"""
Dataset module for DIN-V2.
Handles loading, preprocessing, and creating PyTorch datasets from UserBehavior data.
"""

import os
import random
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader


# Behavior type mapping
BEHAVIOR_MAP = {
    'pv': 0,    # 点击
    'fav': 1,   # 长停留浏览/收藏
    'cart': 2,  # 询盘/加购
    'buy': 3,   # 购买
}
NUM_BEHAVIOR_TYPES = len(BEHAVIOR_MAP)


class UserBehaviorDataset(Dataset):
    """
    PyTorch Dataset for DIN-V2 training.
    
    Each sample consists of:
    - User behavior sequence (item_ids + behavior_types)
    - Target item (candidate)
    - Label (1 if user interacted, 0 otherwise)
    """
    
    def __init__(self, samples: List[dict], max_seq_len: int = 50):
        """
        Args:
            samples: List of dicts with keys:
                - hist_items: List[int] - historical item IDs
                - hist_behaviors: List[int] - historical behavior type IDs
                - hist_categories: List[int] - historical category IDs
                - target_item: int - candidate item ID
                - target_category: int - candidate category ID
                - label: int - 0 or 1
            max_seq_len: Maximum sequence length (truncate/pad)
        """
        self.samples = samples
        self.max_seq_len = max_seq_len
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        hist_items = sample['hist_items'][-self.max_seq_len:]
        hist_behaviors = sample['hist_behaviors'][-self.max_seq_len:]
        hist_categories = sample['hist_categories'][-self.max_seq_len:]
        
        seq_len = len(hist_items)
        
        # Pad sequences
        pad_len = self.max_seq_len - seq_len
        hist_items = hist_items + [0] * pad_len
        hist_behaviors = hist_behaviors + [0] * pad_len
        hist_categories = hist_categories + [0] * pad_len
        
        # Create attention mask (1 for real tokens, 0 for padding)
        mask = [1] * seq_len + [0] * pad_len
        
        return {
            'hist_items': torch.LongTensor(hist_items),
            'hist_behaviors': torch.LongTensor(hist_behaviors),
            'hist_categories': torch.LongTensor(hist_categories),
            'mask': torch.FloatTensor(mask),
            'seq_len': torch.LongTensor([seq_len]),
            'target_item': torch.LongTensor([sample['target_item']]),
            'target_category': torch.LongTensor([sample['target_category']]),
            'label': torch.FloatTensor([sample['label']]),
        }


def preprocess_data(data_path: str, 
                    output_dir: str,
                    max_seq_len: int = 50,
                    min_hist_len: int = 5,
                    neg_ratio: int = 1,
                    seed: int = 42) -> Tuple[Dict, List, List, List]:
    """
    Preprocess UserBehavior data into train/val/test samples.
    
    Strategy:
    - Sort each user's behaviors by timestamp
    - Use the last behavior as test, second-to-last as validation
    - All prior behaviors form the history sequence
    - Generate negative samples by random item sampling
    
    Args:
        data_path: Path to UserBehavior.csv
        output_dir: Directory to save processed data
        max_seq_len: Maximum sequence length
        min_hist_len: Minimum history length to keep a user
        neg_ratio: Number of negative samples per positive
        seed: Random seed
        
    Returns:
        feature_dims: Dict with vocabulary sizes
        train_samples, val_samples, test_samples: Lists of sample dicts
    """
    random.seed(seed)
    np.random.seed(seed)
    
    cache_path = os.path.join(output_dir, 'processed_data.pkl')
    if os.path.exists(cache_path):
        print(f"Loading cached processed data from {cache_path}")
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        return data['feature_dims'], data['train'], data['val'], data['test']
    
    print(f"Loading raw data from {data_path}...")
    
    # Read data in chunks for memory efficiency
    chunks = []
    chunk_size = 5_000_000
    reader = pd.read_csv(
        data_path,
        names=['user_id', 'item_id', 'category_id', 'behavior_type', 'timestamp'],
        chunksize=chunk_size,
        dtype={
            'user_id': 'int32',
            'item_id': 'int32',
            'category_id': 'int32',
            'behavior_type': 'str',
            'timestamp': 'int32'
        }
    )
    
    total_rows = 0
    for i, chunk in enumerate(reader):
        chunks.append(chunk)
        total_rows += len(chunk)
        print(f"\r  Loaded {total_rows:,} rows...", end='', flush=True)
    
    df = pd.concat(chunks, ignore_index=True)
    print(f"\n  Total records: {len(df):,}")
    
    # Map behavior types to IDs
    df['behavior_id'] = df['behavior_type'].map(BEHAVIOR_MAP)
    df = df.dropna(subset=['behavior_id'])
    df['behavior_id'] = df['behavior_id'].astype('int32')
    
    print(f"\nBehavior distribution:")
    print(df['behavior_type'].value_counts().to_string())
    
    # Re-index items and categories to contiguous IDs
    print("\nRe-indexing items and categories...")
    item_ids = df['item_id'].unique()
    category_ids = df['category_id'].unique()
    
    item_map = {old: new + 1 for new, old in enumerate(item_ids)}  # 0 reserved for padding
    category_map = {old: new + 1 for new, old in enumerate(category_ids)}
    
    df['item_idx'] = df['item_id'].map(item_map).astype('int32')
    df['category_idx'] = df['category_id'].map(category_map).astype('int32')
    
    num_items = len(item_map) + 1  # +1 for padding
    num_categories = len(category_map) + 1
    
    feature_dims = {
        'num_items': num_items,
        'num_categories': num_categories,
        'num_behaviors': NUM_BEHAVIOR_TYPES,
    }
    
    print(f"  Items: {num_items:,}, Categories: {num_categories:,}")
    
    # Sort by user and timestamp
    print("\nSorting by user and timestamp...")
    df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
    
    # Group by user
    print("Grouping by user...")
    user_groups = df.groupby('user_id')
    
    all_item_indices = list(range(1, num_items))  # For negative sampling
    
    train_samples = []
    val_samples = []
    test_samples = []
    
    print("Building samples...")
    processed_users = 0
    total_users = len(user_groups)
    
    for user_id, group in user_groups:
        processed_users += 1
        if processed_users % 100000 == 0:
            print(f"\r  Processed {processed_users:,}/{total_users:,} users "
                  f"(train: {len(train_samples):,}, val: {len(val_samples):,}, test: {len(test_samples):,})",
                  end='', flush=True)
        
        items = group['item_idx'].values.tolist()
        behaviors = group['behavior_id'].values.tolist()
        categories = group['category_idx'].values.tolist()
        
        if len(items) < min_hist_len + 2:  # Need at least min_hist_len + val + test
            continue
        
        user_item_set = set(items)
        
        # Test sample: last item
        test_hist_items = items[:-1][-max_seq_len:]
        test_hist_behaviors = behaviors[:-1][-max_seq_len:]
        test_hist_categories = categories[:-1][-max_seq_len:]
        
        test_samples.append({
            'hist_items': test_hist_items,
            'hist_behaviors': test_hist_behaviors,
            'hist_categories': test_hist_categories,
            'target_item': items[-1],
            'target_category': categories[-1],
            'label': 1,
        })
        
        # Negative samples for test
        for _ in range(neg_ratio):
            neg_item = random.choice(all_item_indices)
            while neg_item in user_item_set:
                neg_item = random.choice(all_item_indices)
            test_samples.append({
                'hist_items': test_hist_items,
                'hist_behaviors': test_hist_behaviors,
                'hist_categories': test_hist_categories,
                'target_item': neg_item,
                'target_category': 0,  # Unknown category for neg
                'label': 0,
            })
        
        # Validation sample: second-to-last item
        val_hist_items = items[:-2][-max_seq_len:]
        val_hist_behaviors = behaviors[:-2][-max_seq_len:]
        val_hist_categories = categories[:-2][-max_seq_len:]
        
        if len(val_hist_items) >= min_hist_len:
            val_samples.append({
                'hist_items': val_hist_items,
                'hist_behaviors': val_hist_behaviors,
                'hist_categories': val_hist_categories,
                'target_item': items[-2],
                'target_category': categories[-2],
                'label': 1,
            })
            
            for _ in range(neg_ratio):
                neg_item = random.choice(all_item_indices)
                while neg_item in user_item_set:
                    neg_item = random.choice(all_item_indices)
                val_samples.append({
                    'hist_items': val_hist_items,
                    'hist_behaviors': val_hist_behaviors,
                    'hist_categories': val_hist_categories,
                    'target_item': neg_item,
                    'target_category': 0,
                    'label': 0,
                })
        
        # Training samples: all prior items
        for t in range(min_hist_len, len(items) - 2):
            hist_items = items[:t][-max_seq_len:]
            hist_behaviors = behaviors[:t][-max_seq_len:]
            hist_categories = categories[:t][-max_seq_len:]
            
            train_samples.append({
                'hist_items': hist_items,
                'hist_behaviors': hist_behaviors,
                'hist_categories': hist_categories,
                'target_item': items[t],
                'target_category': categories[t],
                'label': 1,
            })
            
            for _ in range(neg_ratio):
                neg_item = random.choice(all_item_indices)
                while neg_item in user_item_set:
                    neg_item = random.choice(all_item_indices)
                train_samples.append({
                    'hist_items': hist_items,
                    'hist_behaviors': hist_behaviors,
                    'hist_categories': hist_categories,
                    'target_item': neg_item,
                    'target_category': 0,
                    'label': 0,
                })
    
    print(f"\n\nDataset Statistics:")
    print(f"  Train samples: {len(train_samples):,}")
    print(f"  Val samples:   {len(val_samples):,}")
    print(f"  Test samples:  {len(test_samples):,}")
    print(f"  Feature dims:  {feature_dims}")
    
    # Shuffle training data
    random.shuffle(train_samples)
    
    # Save cached data
    print(f"\nSaving processed data to {cache_path}...")
    with open(cache_path, 'wb') as f:
        pickle.dump({
            'feature_dims': feature_dims,
            'train': train_samples,
            'val': val_samples,
            'test': test_samples,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    cache_size = os.path.getsize(cache_path)
    print(f"  Cache size: {cache_size / (1024**3):.2f} GB")
    
    return feature_dims, train_samples, val_samples, test_samples


def get_dataloaders(data_path: str,
                    output_dir: str,
                    batch_size: int = 1024,
                    max_seq_len: int = 50,
                    num_workers: int = 4,
                    **kwargs) -> Tuple[Dict, DataLoader, DataLoader, DataLoader]:
    """
    Get train/val/test DataLoaders.
    
    Returns:
        feature_dims, train_loader, val_loader, test_loader
    """
    feature_dims, train_samples, val_samples, test_samples = preprocess_data(
        data_path, output_dir, max_seq_len=max_seq_len, **kwargs
    )
    
    train_dataset = UserBehaviorDataset(train_samples, max_seq_len)
    val_dataset = UserBehaviorDataset(val_samples, max_seq_len)
    test_dataset = UserBehaviorDataset(test_samples, max_seq_len)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    print(f"\nDataLoaders created:")
    print(f"  Train: {len(train_loader):,} batches")
    print(f"  Val:   {len(val_loader):,} batches")
    print(f"  Test:  {len(test_loader):,} batches")
    
    return feature_dims, train_loader, val_loader, test_loader
