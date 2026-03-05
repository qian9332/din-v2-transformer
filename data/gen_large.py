#!/usr/bin/env python3
"""
Generate data with proper Markov behavioral dependencies AFTER timestamp sorting.
"""
import os, time, numpy as np

def generate(num_users=50000, num_items=100000, num_cats=5000, output='data/UserBehavior.csv', seed=42):
    np.random.seed(seed)
    t0 = time.time()
    
    item_cats = np.random.randint(1, num_cats+1, size=num_items+1); item_cats[0] = 0
    pop = np.random.zipf(1.8, num_items).astype(float); pop /= pop.sum()
    items_arr = np.arange(1, num_items+1)
    
    activity = np.random.lognormal(3.2, 0.9, num_users).astype(int)
    activity = np.clip(activity, 10, 300)
    total = int(activity.sum())
    print(f"Generating: {num_users:,} users, {num_items:,} items | Total: {total:,}")
    
    # Realistic Markov transition matrix
    transition = np.array([
        [0.78, 0.09, 0.08, 0.05],  # pv  -> mostly browse, some escalation
        [0.50, 0.15, 0.22, 0.13],  # fav -> interest shown, higher cart/buy
        [0.35, 0.10, 0.25, 0.30],  # cart -> high conversion to buy
        [0.70, 0.12, 0.12, 0.06],  # buy -> restart cycle
    ])
    transition = transition / transition.sum(axis=1, keepdims=True)
    init_dist = np.array([0.85, 0.07, 0.05, 0.03])
    
    ts_start, ts_end = 1511539200, 1512316800
    btype_names = ['pv','fav','cart','buy']
    
    # Pre-generate items
    all_items_pool = np.random.choice(items_arr, size=total, p=pop)
    
    print("Generating per-user sequences with Markov behaviors...")
    # Build data per user: first assign timestamps (sorted), then apply Markov behavior chain
    records_uid = np.zeros(total, dtype=np.int32)
    records_item = np.zeros(total, dtype=np.int32)
    records_btype = np.zeros(total, dtype=np.int8)
    records_ts = np.zeros(total, dtype=np.int32)
    
    idx = 0
    for uid in range(num_users):
        n = int(activity[uid])
        # Sorted timestamps for this user
        ts = np.sort(np.random.randint(ts_start, ts_end, n))
        # Items from popularity
        items = all_items_pool[idx:idx+n]
        # Markov behavior chain
        btypes = np.zeros(n, dtype=np.int8)
        btypes[0] = np.random.choice(4, p=init_dist)
        for j in range(1, n):
            btypes[j] = np.random.choice(4, p=transition[btypes[j-1]])
        
        records_uid[idx:idx+n] = uid + 1
        records_item[idx:idx+n] = items
        records_btype[idx:idx+n] = btypes
        records_ts[idx:idx+n] = ts
        idx += n
        
        if (uid+1) % 10000 == 0:
            print(f"\r  Users: {uid+1:,}/{num_users:,} ({time.time()-t0:.0f}s)", end='', flush=True)
    
    records_cat = item_cats[records_item]
    print(f"\n  Generation done ({time.time()-t0:.1f}s)")
    
    # Write CSV
    print("Writing CSV...")
    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else '.', exist_ok=True)
    btype_map = np.array(btype_names)
    btype_strs = btype_map[records_btype]
    
    with open(output, 'w') as f:
        chunk = 100000
        for start in range(0, total, chunk):
            end = min(start + chunk, total)
            lines = [f"{records_uid[i]},{records_item[i]},{records_cat[i]},{btype_strs[i]},{records_ts[i]}\n" for i in range(start, end)]
            f.writelines(lines)
    
    fsize = os.path.getsize(output)
    bcount = np.bincount(records_btype, minlength=4)
    
    # Compute actual transition matrix
    trans_count = np.zeros((4,4), dtype=np.int64)
    idx = 0
    for uid in range(num_users):
        n = int(activity[uid])
        bt = records_btype[idx:idx+n]
        for j in range(n-1):
            trans_count[bt[j], bt[j+1]] += 1
        idx += n
    trans_pct = trans_count / np.maximum(trans_count.sum(axis=1, keepdims=True), 1) * 100
    
    print(f"\nDone in {time.time()-t0:.1f}s | {output} ({fsize/(1024**2):.1f} MB)")
    print(f"Records: {total:,} | Users: {num_users:,}")
    for i, name in enumerate(btype_names):
        print(f"  {name}: {bcount[i]:,} ({bcount[i]/total*100:.1f}%)")
    print("\nBehavior Transition Matrix (row=from, col=to):")
    print(f"{'':>6}  {'pv':>6} {'fav':>6} {'cart':>6} {'buy':>6}")
    for i, name in enumerate(btype_names):
        print(f"{name:>6}  {trans_pct[i,0]:>5.1f}% {trans_pct[i,1]:>5.1f}% {trans_pct[i,2]:>5.1f}% {trans_pct[i,3]:>5.1f}%")

if __name__ == '__main__':
    generate()
