#!/usr/bin/env python3
"""
Generate concentrated UserBehavior data for training.
Uses fewer users with more behaviors each to create meaningful sequences.
"""
import os
import random
import csv
import time

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(DATA_DIR, "UserBehavior.csv")

def generate_concentrated_data(output_path, num_users=50000, 
                                num_items=100000, num_categories=5000):
    """
    Generate UserBehavior data with concentrated user behaviors.
    Each user has 10-100 behaviors, creating meaningful sequences.
    """
    print(f"Generating concentrated UserBehavior data...")
    print(f"  Users: {num_users:,}, Items: {num_items:,}, Categories: {num_categories:,}")
    
    behavior_types = ['pv', 'fav', 'cart', 'buy']
    behavior_weights = [0.895, 0.028, 0.055, 0.022]
    
    start_ts = 1511539200
    end_ts = 1512316799
    
    random.seed(42)
    total_records = 0
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        for user_id in range(1, num_users + 1):
            # Each user has 10-100 behaviors (power law distribution)
            num_behaviors = min(int(random.paretovariate(1.5) * 10), 200)
            num_behaviors = max(num_behaviors, 10)
            
            # User has a preference for certain items
            user_item_pool = random.sample(range(1, num_items + 1), 
                                           min(num_behaviors * 3, num_items))
            
            timestamps = sorted([random.randint(start_ts, end_ts) for _ in range(num_behaviors)])
            
            for t_idx in range(num_behaviors):
                item_id = random.choice(user_item_pool)
                category_id = (item_id % num_categories) + 1
                behavior = random.choices(behavior_types, weights=behavior_weights, k=1)[0]
                writer.writerow([user_id, item_id, category_id, behavior, timestamps[t_idx]])
                total_records += 1
            
            if user_id % 10000 == 0:
                print(f"\r  Progress: {user_id:,}/{num_users:,} users, {total_records:,} records", 
                      end='', flush=True)
    
    print(f"\n  Total records: {total_records:,}")
    file_size = os.path.getsize(output_path)
    print(f"  File size: {file_size / (1024**2):.1f} MB")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_users', type=int, default=50000)
    parser.add_argument('--num_items', type=int, default=100000)
    args = parser.parse_args()
    
    start = time.time()
    generate_concentrated_data(OUTPUT_FILE, num_users=args.num_users, num_items=args.num_items)
    print(f"Done in {time.time() - start:.1f}s")
