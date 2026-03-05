#!/usr/bin/env python3
"""
Quick training script that runs in constrained environments.
Trains on a subset and generates complete training logs.
"""
import os, sys, time, json, random, csv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

from src.model import DINV2
from src.model_v1 import DINV1
from src.dataset import UserBehaviorDataset, BEHAVIOR_MAP, NUM_BEHAVIOR_TYPES
from src.utils import calculate_metrics, AverageMeter, count_parameters, format_time
from torch.utils.data import DataLoader

def generate_training_samples(data_path, max_samples=200000, max_seq_len=50, min_hist_len=3):
    """Fast data processing with sample limit."""
    import pandas as pd
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, 
                     names=['user_id', 'item_id', 'category_id', 'behavior_type', 'timestamp'],
                     dtype={'user_id':'int32','item_id':'int32','category_id':'int32',
                            'behavior_type':'str','timestamp':'int32'})
    
    print(f"Total records: {len(df):,}")
    df['behavior_id'] = df['behavior_type'].map(BEHAVIOR_MAP).astype('int32')
    
    # Re-index
    item_ids = df['item_id'].unique()
    cat_ids = df['category_id'].unique()
    item_map = {old: new+1 for new, old in enumerate(item_ids)}
    cat_map = {old: new+1 for new, old in enumerate(cat_ids)}
    df['item_idx'] = df['item_id'].map(item_map).astype('int32')
    df['cat_idx'] = df['category_id'].map(cat_map).astype('int32')
    
    num_items = len(item_map) + 1
    num_cats = len(cat_map) + 1
    
    df = df.sort_values(['user_id','timestamp']).reset_index(drop=True)
    
    all_items = list(range(1, num_items))
    random.seed(42)
    
    train_samples, val_samples, test_samples = [], [], []
    
    for user_id, group in df.groupby('user_id'):
        items = group['item_idx'].values.tolist()
        behaviors = group['behavior_id'].values.tolist()
        cats = group['cat_idx'].values.tolist()
        
        if len(items) < min_hist_len + 2:
            continue
        
        user_set = set(items)
        
        # Test
        test_samples.append({
            'hist_items': items[:-1][-max_seq_len:],
            'hist_behaviors': behaviors[:-1][-max_seq_len:],
            'hist_categories': cats[:-1][-max_seq_len:],
            'target_item': items[-1], 'target_category': cats[-1], 'label': 1
        })
        neg = random.choice(all_items)
        while neg in user_set: neg = random.choice(all_items)
        test_samples.append({
            'hist_items': items[:-1][-max_seq_len:],
            'hist_behaviors': behaviors[:-1][-max_seq_len:],
            'hist_categories': cats[:-1][-max_seq_len:],
            'target_item': neg, 'target_category': 0, 'label': 0
        })
        
        # Val
        if len(items[:-2]) >= min_hist_len:
            val_samples.append({
                'hist_items': items[:-2][-max_seq_len:],
                'hist_behaviors': behaviors[:-2][-max_seq_len:],
                'hist_categories': cats[:-2][-max_seq_len:],
                'target_item': items[-2], 'target_category': cats[-2], 'label': 1
            })
            neg = random.choice(all_items)
            while neg in user_set: neg = random.choice(all_items)
            val_samples.append({
                'hist_items': items[:-2][-max_seq_len:],
                'hist_behaviors': behaviors[:-2][-max_seq_len:],
                'hist_categories': cats[:-2][-max_seq_len:],
                'target_item': neg, 'target_category': 0, 'label': 0
            })
        
        # Train - sample subset of positions
        positions = list(range(min_hist_len, len(items)-2))
        if len(positions) > 3:
            positions = random.sample(positions, 3)
        
        for t in positions:
            train_samples.append({
                'hist_items': items[:t][-max_seq_len:],
                'hist_behaviors': behaviors[:t][-max_seq_len:],
                'hist_categories': cats[:t][-max_seq_len:],
                'target_item': items[t], 'target_category': cats[t], 'label': 1
            })
            neg = random.choice(all_items)
            while neg in user_set: neg = random.choice(all_items)
            train_samples.append({
                'hist_items': items[:t][-max_seq_len:],
                'hist_behaviors': behaviors[:t][-max_seq_len:],
                'hist_categories': cats[:t][-max_seq_len:],
                'target_item': neg, 'target_category': 0, 'label': 0
            })
        
        if len(train_samples) >= max_samples:
            break
    
    random.shuffle(train_samples)
    train_samples = train_samples[:max_samples]
    
    print(f"Samples: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")
    
    feature_dims = {'num_items': num_items, 'num_categories': num_cats, 'num_behaviors': NUM_BEHAVIOR_TYPES}
    return feature_dims, train_samples, val_samples, test_samples


def train_and_log():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f'logs/din_v2_{timestamp}'
    ckpt_dir = f'checkpoints/din_v2_{timestamp}'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    class Logger:
        def __init__(self, filepath):
            self.f = open(filepath, 'w')
        def info(self, msg):
            line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - {msg}"
            print(line)
            self.f.write(line + '\n')
            self.f.flush()
        def close(self):
            self.f.close()
    
    logger = Logger(log_file)
    
    logger.info("=" * 80)
    logger.info("DIN-V2: Behavior-Type-Aware Deep Interest Network with Transformer Encoder")
    logger.info("=" * 80)
    
    # Config
    config = {
        'model': 'v2', 'embed_dim': 64, 'num_heads': 4, 'num_transformer_layers': 2,
        'max_seq_len': 50, 'dropout': 0.1, 'epochs': 3, 'batch_size': 512,
        'lr': 1e-3, 'weight_decay': 1e-5, 'device': 'cpu'
    }
    logger.info(f"Config: {json.dumps(config, indent=2)}")
    
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("Using CPU")
    
    # Data
    logger.info("\n--- Loading & Processing Data ---")
    t0 = time.time()
    feature_dims, train_samples, val_samples, test_samples = generate_training_samples(
        'data/UserBehavior.csv', max_samples=200000
    )
    logger.info(f"Data processing: {time.time()-t0:.1f}s")
    logger.info(f"Feature dims: {feature_dims}")
    
    bs = config['batch_size']
    train_ds = UserBehaviorDataset(train_samples, config['max_seq_len'])
    val_ds = UserBehaviorDataset(val_samples, config['max_seq_len'])
    test_ds = UserBehaviorDataset(test_samples, config['max_seq_len'])
    
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=0)
    
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    
    # Model
    logger.info("\n--- Building DIN-V2 Model ---")
    model = DINV2(
        num_items=feature_dims['num_items'],
        num_categories=feature_dims['num_categories'],
        num_behaviors=NUM_BEHAVIOR_TYPES,
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_transformer_layers=config['num_transformer_layers'],
        max_seq_len=config['max_seq_len'],
        dropout=config['dropout'],
    ).to(device)
    
    total_p, train_p = count_parameters(model)
    logger.info(f"Model: DIN-V2")
    logger.info(f"Total params: {total_p:,}")
    logger.info(f"Trainable params: {train_p:,}")
    logger.info(f"Architecture:\n{model}")
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    
    # Training
    logger.info("\n--- Training Started ---")
    best_val_auc = 0.0
    history = []
    
    for epoch in range(config['epochs']):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch+1}/{config['epochs']}")
        logger.info(f"{'='*60}")
        
        model.train()
        loss_meter = AverageMeter()
        all_labels, all_preds = [], []
        epoch_start = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            hist_items = batch['hist_items'].to(device)
            hist_behaviors = batch['hist_behaviors'].to(device)
            hist_categories = batch['hist_categories'].to(device)
            mask = batch['mask'].to(device)
            target_item = batch['target_item'].to(device)
            target_category = batch['target_category'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(hist_items, hist_behaviors, hist_categories, mask, target_item, target_category)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            loss_meter.update(loss.item(), hist_items.size(0))
            preds = torch.sigmoid(logits).detach().cpu().numpy().flatten()
            all_labels.extend(labels.cpu().numpy().flatten().tolist())
            all_preds.extend(preds.tolist())
            
            if (batch_idx + 1) % 50 == 0:
                elapsed = time.time() - epoch_start
                speed = (batch_idx + 1) * bs / elapsed
                logger.info(f"  Batch [{batch_idx+1}/{len(train_loader)}] "
                           f"Loss: {loss_meter.avg:.4f} Speed: {speed:.0f} samples/s")
        
        train_time = time.time() - epoch_start
        train_metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
        train_metrics['loss'] = loss_meter.avg
        
        logger.info(f"\n[Train] Epoch {epoch+1} - {format_time(train_time)}")
        logger.info(f"  Loss: {train_metrics['loss']:.4f} AUC: {train_metrics['auc']:.4f} Acc: {train_metrics['accuracy']:.4f}")
        
        # Validation
        model.eval()
        val_labels, val_preds, val_loss_sum, val_n = [], [], 0, 0
        with torch.no_grad():
            for batch in val_loader:
                h = {k: v.to(device) for k, v in batch.items()}
                logits = model(h['hist_items'], h['hist_behaviors'], h['hist_categories'],
                              h['mask'], h['target_item'], h['target_category'])
                loss = criterion(logits, h['label'])
                preds = torch.sigmoid(logits).cpu().numpy().flatten()
                val_labels.extend(h['label'].cpu().numpy().flatten().tolist())
                val_preds.extend(preds.tolist())
                val_loss_sum += loss.item() * h['hist_items'].size(0)
                val_n += h['hist_items'].size(0)
        
        val_metrics = calculate_metrics(np.array(val_labels), np.array(val_preds))
        val_metrics['loss'] = val_loss_sum / max(val_n, 1)
        
        logger.info(f"[Val] Loss: {val_metrics['loss']:.4f} AUC: {val_metrics['auc']:.4f} "
                    f"LogLoss: {val_metrics['logloss']:.4f} Acc: {val_metrics['accuracy']:.4f}")
        
        history.append({
            'epoch': epoch+1, 'train_loss': train_metrics['loss'], 'train_auc': train_metrics['auc'],
            'val_loss': val_metrics['loss'], 'val_auc': val_metrics['auc'],
            'val_logloss': val_metrics['logloss'], 'val_accuracy': val_metrics['accuracy'],
            'train_time': train_time, 'lr': config['lr']
        })
        
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            torch.save({'epoch': epoch+1, 'model_state_dict': model.state_dict(),
                        'val_auc': val_metrics['auc'], 'config': config},
                       os.path.join(ckpt_dir, 'best_model.pt'))
            logger.info(f"  * Best model saved! Val AUC: {val_metrics['auc']:.4f}")
    
    # Test
    logger.info("\n--- Testing ---")
    ckpt = torch.load(os.path.join(ckpt_dir, 'best_model.pt'), map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    
    model.eval()
    test_labels, test_preds = [], []
    with torch.no_grad():
        for batch in test_loader:
            h = {k: v.to(device) for k, v in batch.items()}
            logits = model(h['hist_items'], h['hist_behaviors'], h['hist_categories'],
                          h['mask'], h['target_item'], h['target_category'])
            preds = torch.sigmoid(logits).cpu().numpy().flatten()
            test_labels.extend(h['label'].cpu().numpy().flatten().tolist())
            test_preds.extend(preds.tolist())
    
    test_metrics = calculate_metrics(np.array(test_labels), np.array(test_preds))
    logger.info(f"[Test] AUC: {test_metrics['auc']:.4f} LogLoss: {test_metrics['logloss']:.4f} Acc: {test_metrics['accuracy']:.4f}")
    
    # Save results
    results = {
        'model': 'DIN-V2', 'params': total_p, 'best_val_auc': best_val_auc,
        'test_metrics': test_metrics, 'history': history, 'config': config,
        'feature_dims': feature_dims
    }
    with open(os.path.join(log_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Training Complete! Best Val AUC: {best_val_auc:.4f}, Test AUC: {test_metrics['auc']:.4f}")
    logger.info(f"Logs: {log_dir}")
    logger.info(f"{'='*80}")
    logger.close()
    
    return log_dir, results

if __name__ == '__main__':
    train_and_log()
