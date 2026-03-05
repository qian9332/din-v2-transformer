#!/usr/bin/env python3
"""Incremental training that saves state between invocations."""
import os, sys, time, json, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import UserBehaviorDataset, NUM_BEHAVIOR_TYPES
from src.model import DINV2
from src.utils import calculate_metrics, AverageMeter, count_parameters, format_time
from torch.utils.data import DataLoader

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_FILE = os.path.join(PROJECT, 'checkpoints', 'train_state.pt')
LOG_FILE = os.path.join(PROJECT, 'logs', 'training_full.log')

def log(msg):
    line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {msg}"
    print(line)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')

def load_data():
    with open(os.path.join(PROJECT, 'data', 'processed_data.pkl'), 'rb') as f:
        data = pickle.load(f)
    return data['feature_dims'], data['train'], data['val'], data['test']

def run_epoch(epoch_num, total_epochs=3):
    feature_dims, train_s, val_s, test_s = load_data()
    
    bs = 512
    train_ds = UserBehaviorDataset(train_s, 50)
    val_ds = UserBehaviorDataset(val_s, 50)
    test_ds = UserBehaviorDataset(test_s, 50)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=0)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = DINV2(
        num_items=feature_dims['num_items'],
        num_categories=feature_dims['num_categories'],
        num_behaviors=NUM_BEHAVIOR_TYPES,
        embed_dim=64, num_heads=4, num_transformer_layers=2,
        max_seq_len=50, dropout=0.1
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    
    # Load state if resuming
    if epoch_num > 0 and os.path.exists(STATE_FILE):
        state = torch.load(STATE_FILE, map_location=device, weights_only=False)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        log(f"Resumed from epoch {state['epoch']}")
    
    if epoch_num == 0:
        tp, trp = count_parameters(model)
        log("=" * 80)
        log("DIN-V2: Behavior-Type-Aware Deep Interest Network + Transformer Encoder")
        log("=" * 80)
        log(f"Device: {device}")
        log(f"Feature dims: {feature_dims}")
        log(f"Total params: {tp:,}, Trainable: {trp:,}")
        log(f"Train: {len(train_s)}, Val: {len(val_s)}, Test: {len(test_s)}")
        log(f"Batch size: {bs}, Epochs: {total_epochs}")
        log(f"Model architecture:\n{model}")
    
    # Train
    log(f"\n{'='*60}")
    log(f"Epoch {epoch_num+1}/{total_epochs}")
    log(f"{'='*60}")
    
    model.train()
    loss_meter = AverageMeter()
    all_labels, all_preds = [], []
    t0 = time.time()
    
    for i, batch in enumerate(train_loader):
        h = {k: v.to(device) for k, v in batch.items()}
        logits = model(h['hist_items'], h['hist_behaviors'], h['hist_categories'],
                      h['mask'], h['target_item'], h['target_category'])
        loss = criterion(logits, h['label'])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        loss_meter.update(loss.item(), h['hist_items'].size(0))
        preds = torch.sigmoid(logits).detach().cpu().numpy().flatten()
        all_labels.extend(h['label'].cpu().numpy().flatten().tolist())
        all_preds.extend(preds.tolist())
        
        if (i+1) % 20 == 0:
            speed = (i+1)*bs/(time.time()-t0)
            log(f"  Batch [{i+1}/{len(train_loader)}] Loss: {loss_meter.avg:.4f} Speed: {speed:.0f} s/s")
    
    train_time = time.time() - t0
    train_m = calculate_metrics(np.array(all_labels), np.array(all_preds))
    log(f"[Train] {format_time(train_time)} Loss: {loss_meter.avg:.4f} AUC: {train_m['auc']:.4f} Acc: {train_m['accuracy']:.4f}")
    
    # Validate
    model.eval()
    vl, vp = [], []
    with torch.no_grad():
        for batch in val_loader:
            h = {k: v.to(device) for k, v in batch.items()}
            logits = model(h['hist_items'], h['hist_behaviors'], h['hist_categories'],
                          h['mask'], h['target_item'], h['target_category'])
            preds = torch.sigmoid(logits).cpu().numpy().flatten()
            vl.extend(h['label'].cpu().numpy().flatten().tolist())
            vp.extend(preds.tolist())
    
    val_m = calculate_metrics(np.array(vl), np.array(vp))
    log(f"[Val] AUC: {val_m['auc']:.4f} LogLoss: {val_m['logloss']:.4f} Acc: {val_m['accuracy']:.4f}")
    
    # Save state
    torch.save({
        'epoch': epoch_num + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_auc': val_m['auc'],
        'train_loss': loss_meter.avg,
    }, STATE_FILE)
    log(f"State saved to {STATE_FILE}")
    
    # Test on final epoch
    if epoch_num == total_epochs - 1:
        tl, tp_list = [], []
        with torch.no_grad():
            for batch in test_loader:
                h = {k: v.to(device) for k, v in batch.items()}
                logits = model(h['hist_items'], h['hist_behaviors'], h['hist_categories'],
                              h['mask'], h['target_item'], h['target_category'])
                preds = torch.sigmoid(logits).cpu().numpy().flatten()
                tl.extend(h['label'].cpu().numpy().flatten().tolist())
                tp_list.extend(preds.tolist())
        
        test_m = calculate_metrics(np.array(tl), np.array(tp_list))
        log(f"\n--- Final Test Results ---")
        log(f"[Test] AUC: {test_m['auc']:.4f} LogLoss: {test_m['logloss']:.4f} Acc: {test_m['accuracy']:.4f}")
        
        # Save results
        results = {
            'model': 'DIN-V2', 'test_auc': test_m['auc'], 'test_logloss': test_m['logloss'],
            'test_accuracy': test_m['accuracy'], 'val_auc': val_m['auc'],
            'feature_dims': feature_dims, 'total_epochs': total_epochs
        }
        with open(os.path.join(PROJECT, 'logs', 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        log("Results saved!")
        log("=" * 80)
    
    return val_m['auc']

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--epoch', type=int, required=True)
    p.add_argument('--total_epochs', type=int, default=3)
    args = p.parse_args()
    run_epoch(args.epoch, args.total_epochs)
