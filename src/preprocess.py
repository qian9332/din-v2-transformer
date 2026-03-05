#!/usr/bin/env python3
"""
Full training pipeline for DIN-V2 on CPU.
Step 1: Preprocess data (generates train/val/test pkl)
"""
import os, sys, time, pickle, random, numpy as np, pandas as pd
from collections import defaultdict

def preprocess(csv_path='data/UserBehavior.csv', output='data/processed_full.pkl', 
               max_seq_len=30, min_hist_len=5, neg_ratio=1, seed=42):
    random.seed(seed); np.random.seed(seed)
    t0 = time.time()
    
    if os.path.exists(output):
        print(f"Cache exists: {output}")
        with open(output, 'rb') as f:
            data = pickle.load(f)
        print(f"  Train: {len(data['train']):,}, Val: {len(data['val']):,}, Test: {len(data['test']):,}")
        return data
    
    print("Loading CSV...")
    df = pd.read_csv(csv_path, names=['user_id','item_id','category_id','behavior_type','timestamp'],
                     dtype={'user_id':'int32','item_id':'int32','category_id':'int32',
                            'behavior_type':'str','timestamp':'int32'})
    print(f"  Records: {len(df):,}")
    
    bmap = {'pv':0, 'fav':1, 'cart':2, 'buy':3}
    df['bid'] = df['behavior_type'].map(bmap).astype('int32')
    
    # Re-index
    item_map = {old: new+1 for new, old in enumerate(df['item_id'].unique())}
    cat_map = {old: new+1 for new, old in enumerate(df['category_id'].unique())}
    df['iidx'] = df['item_id'].map(item_map).astype('int32')
    df['cidx'] = df['category_id'].map(cat_map).astype('int32')
    
    num_items = len(item_map) + 1
    num_cats = len(cat_map) + 1
    feature_dims = {'num_items': num_items, 'num_categories': num_cats, 'num_behaviors': 4}
    
    print(f"  Items: {num_items:,}, Categories: {num_cats:,}")
    
    df = df.sort_values(['user_id','timestamp']).reset_index(drop=True)
    
    all_items = list(range(1, num_items))
    train_s, val_s, test_s = [], [], []
    
    print("Building samples...")
    for uid, grp in df.groupby('user_id'):
        items = grp['iidx'].values.tolist()
        bids = grp['bid'].values.tolist()
        cats = grp['cidx'].values.tolist()
        
        if len(items) < min_hist_len + 2:
            continue
        
        uset = set(items)
        
        # Test: last item
        h_items = items[:-1][-max_seq_len:]
        h_bids = bids[:-1][-max_seq_len:]
        h_cats = cats[:-1][-max_seq_len:]
        
        test_s.append({'hist_items':h_items, 'hist_behaviors':h_bids, 'hist_categories':h_cats,
                       'target_item':items[-1], 'target_category':cats[-1], 'label':1})
        neg = random.choice(all_items)
        while neg in uset: neg = random.choice(all_items)
        test_s.append({'hist_items':h_items, 'hist_behaviors':h_bids, 'hist_categories':h_cats,
                       'target_item':neg, 'target_category':0, 'label':0})
        
        # Val: second-to-last
        h2_items = items[:-2][-max_seq_len:]
        h2_bids = bids[:-2][-max_seq_len:]
        h2_cats = cats[:-2][-max_seq_len:]
        
        if len(h2_items) >= min_hist_len:
            val_s.append({'hist_items':h2_items, 'hist_behaviors':h2_bids, 'hist_categories':h2_cats,
                          'target_item':items[-2], 'target_category':cats[-2], 'label':1})
            neg = random.choice(all_items)
            while neg in uset: neg = random.choice(all_items)
            val_s.append({'hist_items':h2_items, 'hist_behaviors':h2_bids, 'hist_categories':h2_cats,
                          'target_item':neg, 'target_category':0, 'label':0})
        
        # Train: sliding window with sampling to control size
        positions = list(range(min_hist_len, len(items)-2))
        # Sample to keep training manageable (max 10 positions per user)
        if len(positions) > 10:
            positions = random.sample(positions, 10)
        
        for t in positions:
            h = items[:t][-max_seq_len:]
            hb = bids[:t][-max_seq_len:]
            hc = cats[:t][-max_seq_len:]
            
            train_s.append({'hist_items':h, 'hist_behaviors':hb, 'hist_categories':hc,
                            'target_item':items[t], 'target_category':cats[t], 'label':1})
            neg = random.choice(all_items)
            while neg in uset: neg = random.choice(all_items)
            train_s.append({'hist_items':h, 'hist_behaviors':hb, 'hist_categories':hc,
                            'target_item':neg, 'target_category':0, 'label':0})
    
    random.shuffle(train_s)
    
    print(f"\n  Train: {len(train_s):,}, Val: {len(val_s):,}, Test: {len(test_s):,}")
    print(f"  Time: {time.time()-t0:.1f}s")
    
    data = {'feature_dims': feature_dims, 'train': train_s, 'val': val_s, 'test': test_s}
    with open(output, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved to {output} ({os.path.getsize(output)/(1024**2):.1f} MB)")
    
    return data

if __name__ == '__main__':
    preprocess()
