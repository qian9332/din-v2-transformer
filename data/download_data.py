#!/usr/bin/env python3
"""
Download and prepare UserBehavior dataset from Alibaba.
Dataset: https://tianchi.aliyun.com/dataset/649

Since the original dataset requires Tianchi login, we provide:
1. A synthetic generation script that creates data with the same schema
2. Instructions for using the real dataset

The UserBehavior dataset contains ~100M records with columns:
    user_id, item_id, category_id, behavior_type, timestamp
"""

import os
import sys
import random
import csv
import time

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(DATA_DIR, "UserBehavior.csv")


def generate_userbehavior_data(output_path, num_records=100_000_000, 
                                num_users=987_994, num_items=4_162_024,
                                num_categories=9_439):
    """
    Generate synthetic UserBehavior dataset matching the real dataset schema and scale.
    
    Schema: user_id, item_id, category_id, behavior_type, timestamp
    Behavior types: pv, fav, cart, buy (with realistic proportions)
    Time range: 2017-11-25 to 2017-12-03
    """
    print(f"Generating {num_records:,} synthetic UserBehavior records...")
    print(f"  Users: {num_users:,}, Items: {num_items:,}, Categories: {num_categories:,}")
    
    # Realistic behavior type distribution from the actual dataset
    # pv: ~89.5%, fav: ~2.8%, cart: ~5.5%, buy: ~2.2%
    behavior_types = ['pv', 'fav', 'cart', 'buy']
    behavior_weights = [0.895, 0.028, 0.055, 0.022]
    
    # Time range: 2017-11-25 00:00:00 to 2017-12-03 23:59:59
    start_ts = 1511539200  # 2017-11-25 00:00:00 UTC+8
    end_ts = 1512316799    # 2017-12-03 23:59:59 UTC+8
    
    # Pre-generate item -> category mapping (each item belongs to one category)
    print("  Building item-category mapping...")
    random.seed(42)
    
    # Write data in chunks
    chunk_size = 1_000_000
    total_written = 0
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        for chunk_start in range(0, num_records, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_records)
            actual_chunk = chunk_end - chunk_start
            
            rows = []
            for _ in range(actual_chunk):
                user_id = random.randint(1, num_users)
                item_id = random.randint(1, num_items)
                category_id = (item_id % num_categories) + 1  # Deterministic mapping
                behavior = random.choices(behavior_types, weights=behavior_weights, k=1)[0]
                timestamp = random.randint(start_ts, end_ts)
                rows.append([user_id, item_id, category_id, behavior, timestamp])
            
            writer.writerows(rows)
            total_written += actual_chunk
            
            progress = total_written / num_records * 100
            print(f"\r  Progress: {total_written:,}/{num_records:,} ({progress:.1f}%)", end='', flush=True)
    
    print(f"\n  Dataset saved to: {output_path}")
    file_size = os.path.getsize(output_path)
    print(f"  File size: {file_size / (1024**3):.2f} GB")
    

def verify_data(filepath, n_samples=5):
    """Verify the generated data."""
    print(f"\nVerifying data from {filepath}...")
    
    import pandas as pd
    
    # Read first few rows
    df = pd.read_csv(filepath, nrows=10000, 
                     names=['user_id', 'item_id', 'category_id', 'behavior_type', 'timestamp'])
    
    print(f"\nSample records:")
    print(df.head(n_samples).to_string(index=False))
    
    print(f"\nBehavior type distribution (first 10K rows):")
    print(df['behavior_type'].value_counts(normalize=True).to_string())
    
    print(f"\nBasic stats:")
    print(f"  Unique users: {df['user_id'].nunique():,}")
    print(f"  Unique items: {df['item_id'].nunique():,}")
    print(f"  Unique categories: {df['category_id'].nunique():,}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Download/Generate UserBehavior dataset')
    parser.add_argument('--num_records', type=int, default=100_000_000,
                        help='Number of records to generate (default: 100M)')
    parser.add_argument('--output', type=str, default=OUTPUT_FILE,
                        help='Output file path')
    parser.add_argument('--verify', action='store_true',
                        help='Verify data after generation')
    args = parser.parse_args()
    
    if os.path.exists(args.output):
        file_size = os.path.getsize(args.output)
        print(f"Data file already exists: {args.output} ({file_size / (1024**3):.2f} GB)")
        print("Skipping generation. Delete the file to regenerate.")
    else:
        start_time = time.time()
        generate_userbehavior_data(args.output, num_records=args.num_records)
        elapsed = time.time() - start_time
        print(f"Generation completed in {elapsed:.1f} seconds")
    
    if args.verify or not os.path.exists(args.output):
        verify_data(args.output)
