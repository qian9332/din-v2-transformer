#!/usr/bin/env python3
"""
DIN-V2 预处理 —— 修复版
修复3个导致AUC虚高的BUG (详见 docs/bugfix_report.md)
"""
import os, sys, time, pickle, random
import numpy as np
import pandas as pd
from collections import defaultdict

def load_taobao(path):
    df = pd.read_csv(path, sep='\t', header=None, skiprows=1,
        names=['Id','user_id','age','gender','item_id','behavior_type','item_category','time','Province'],
        dtype=str, encoding='latin1')
    for c in ['Id','user_id','age','gender','item_id','behavior_type']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['Id','user_id','item_id','behavior_type'])
    df['behavior_type'] = df['behavior_type'].astype(int)
    df['item_id'] = df['item_id'].astype(int)
    df['item_category'] = df['item_category'].astype(str).str.strip()
    print(f"  Loaded {len(df):,} records")
    return df

def preprocess_fixed(data_path='data/taobao_raw.txt', output_dir='data/',
                     max_seq_len=20, min_seq_len=3, neg_ratio=1, seed=42):
    random.seed(seed); np.random.seed(seed)
    
    cache = os.path.join(output_dir, 'processed_fixed.pkl')
    if os.path.exists(cache):
        print(f"Loading cached: {cache}")
        with open(cache, 'rb') as f: data = pickle.load(f)
        for k in ['train','val','test']: print(f"  {k}: {len(data[k]):,}")
        return data
    
    t0 = time.time()
    df = load_taobao(data_path)
    
    # Re-index
    item_map = {old: i+1 for i, old in enumerate(df['item_id'].unique())}
    cat_map = {old: i+1 for i, old in enumerate(df['item_category'].unique())}
    df['iidx'] = df['item_id'].map(item_map).astype(int)
    df['cidx'] = df['item_category'].map(cat_map).astype(int)
    df['bidx'] = (df['behavior_type'] - 1).astype(int)
    
    num_items = len(item_map) + 1
    num_cats = len(cat_map) + 1
    
    # FIX-1: item→category 映射
    item_to_cat = dict(zip(df['iidx'], df['cidx']))
    print(f"  Items: {num_items:,}, Cats: {num_cats:,}, item→cat: {len(item_to_cat):,}")
    
    # FIX-2: 频率加权负采样池 (预生成大批量)
    item_freq = df.groupby('iidx').size()
    all_items_arr = item_freq.index.values
    freq_weights = np.power(item_freq.values.astype(np.float64), 0.75)
    freq_probs = freq_weights / freq_weights.sum()
    
    # 预生成100万个负样本候选
    NEG_POOL_SIZE = 1_000_000
    neg_pool = np.random.choice(all_items_arr, size=NEG_POOL_SIZE, p=freq_probs)
    neg_ptr = [0]  # mutable pointer
    
    def get_neg(exclude_set):
        """从预生成池取负样本"""
        for _ in range(50):
            item = int(neg_pool[neg_ptr[0] % NEG_POOL_SIZE])
            neg_ptr[0] += 1
            if item not in exclude_set:
                return item
        return int(neg_pool[neg_ptr[0] % NEG_POOL_SIZE])
    
    print(f"  Neg sampling pool ready ({NEG_POOL_SIZE:,} candidates, freq^0.75)")
    
    # Session构建
    print(f"  Building sessions...")
    df = df.sort_values(['item_category','time','Id']).reset_index(drop=True)
    
    train_s, val_s, test_s = [], [], []
    sess_count = 0
    
    for (cat, date), grp in df.groupby(['item_category','time']):
        n = len(grp)
        if n < min_seq_len + 2: continue
        sess_count += 1
        
        items = grp['iidx'].values.tolist()
        bids = grp['bidx'].values.tolist()
        cats = grp['cidx'].values.tolist()
        sess_set = set(items)
        
        # Test: last
        h = items[:-1][-max_seq_len:]
        hb = bids[:-1][-max_seq_len:]
        hc = cats[:-1][-max_seq_len:]
        
        test_s.append({'hist_items':h, 'hist_behaviors':hb, 'hist_categories':hc,
                       'target_item':items[-1], 'target_category':item_to_cat.get(items[-1],0), 'label':1})
        neg = get_neg(sess_set)
        test_s.append({'hist_items':h, 'hist_behaviors':hb, 'hist_categories':hc,
                       'target_item':neg, 'target_category':item_to_cat.get(neg,0), 'label':0})
        
        # Val: second-to-last
        if n >= min_seq_len + 3:
            h2 = items[:-2][-max_seq_len:]
            hb2 = bids[:-2][-max_seq_len:]
            hc2 = cats[:-2][-max_seq_len:]
            val_s.append({'hist_items':h2, 'hist_behaviors':hb2, 'hist_categories':hc2,
                          'target_item':items[-2], 'target_category':item_to_cat.get(items[-2],0), 'label':1})
            neg = get_neg(sess_set)
            val_s.append({'hist_items':h2, 'hist_behaviors':hb2, 'hist_categories':hc2,
                          'target_item':neg, 'target_category':item_to_cat.get(neg,0), 'label':0})
        
        # Train: sliding window (max 8 per session)
        positions = list(range(min_seq_len, n - 2))
        if len(positions) > 8: positions = random.sample(positions, 8)
        for t in positions:
            ht = items[:t][-max_seq_len:]
            hbt = bids[:t][-max_seq_len:]
            hct = cats[:t][-max_seq_len:]
            train_s.append({'hist_items':ht, 'hist_behaviors':hbt, 'hist_categories':hct,
                            'target_item':items[t], 'target_category':item_to_cat.get(items[t],0), 'label':1})
            neg = get_neg(sess_set)
            train_s.append({'hist_items':ht, 'hist_behaviors':hbt, 'hist_categories':hct,
                            'target_item':neg, 'target_category':item_to_cat.get(neg,0), 'label':0})
    
    random.shuffle(train_s)
    
    fdims = {'num_items': num_items, 'num_categories': num_cats, 'num_behaviors': 4}
    print(f"\n  Sessions: {sess_count:,}")
    print(f"  Train: {len(train_s):,}  Val: {len(val_s):,}  Test: {len(test_s):,}")
    print(f"  Feature dims: {fdims}")
    
    # 修复验证
    print(f"\n  === 修复验证 ===")
    pos_tc = [s['target_category'] for s in test_s if s['label']==1]
    neg_tc = [s['target_category'] for s in test_s if s['label']==0]
    pos0 = sum(1 for c in pos_tc if c==0)
    neg0 = sum(1 for c in neg_tc if c==0)
    print(f"  正样本 cat=0: {pos0}/{len(pos_tc)} ({pos0/max(len(pos_tc),1)*100:.1f}%)")
    print(f"  负样本 cat=0: {neg0}/{len(neg_tc)} ({neg0/max(len(neg_tc),1)*100:.1f}%)")
    print(f"  {'✅ BUG-1 FIX verified' if neg0/max(len(neg_tc),1) < 0.01 else '⚠️ check!'}")
    
    # 验证频率采样效果
    pos_items = [s['target_item'] for s in train_s[:2000] if s['label']==1]
    neg_items = [s['target_item'] for s in train_s[:2000] if s['label']==0]
    pos_mean_freq = np.mean([item_freq.get(i,0) for i in pos_items])
    neg_mean_freq = np.mean([item_freq.get(i,0) for i in neg_items])
    print(f"  正样本平均频率: {pos_mean_freq:.1f}")
    print(f"  负样本平均频率: {neg_mean_freq:.1f}")
    print(f"  频率比(pos/neg): {pos_mean_freq/max(neg_mean_freq,1):.2f}x")
    print(f"  {'✅ BUG-2 FIX effective' if pos_mean_freq/max(neg_mean_freq,1) < 5 else '⚠️ still skewed'}")
    
    print(f"  Time: {time.time()-t0:.1f}s")
    
    data = {'feature_dims': fdims, 'train': train_s, 'val': val_s, 'test': test_s}
    with open(cache, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved: {cache} ({os.path.getsize(cache)/(1024**2):.1f}MB)")
    return data

if __name__ == '__main__':
    preprocess_fixed()
