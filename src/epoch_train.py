#!/usr/bin/env python3
"""
Epoch-by-epoch training with state persistence.
Each invocation runs one epoch then saves state.
"""
import os, sys, time, json, pickle
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.dataset import UserBehaviorDataset, NUM_BEHAVIOR_TYPES
from src.model import DINV2
from src.utils import calculate_metrics, AverageMeter, count_parameters
from torch.utils.data import DataLoader

P = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
os.chdir(P)

STATE = os.path.join(P, 'checkpoints', 'train_state.pt')
LOG = os.path.join(P, 'logs', 'training_full.log')
RESULTS = os.path.join(P, 'logs', 'results.json')
os.makedirs(os.path.join(P,'checkpoints'), exist_ok=True)
os.makedirs(os.path.join(P,'logs'), exist_ok=True)

def log(msg):
    t = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    s = f"{t} - INFO - {msg}"
    print(s)
    with open(LOG, 'a') as f: f.write(s + '\n')

def run_one_epoch(epoch_idx, total_epochs=3):
    with open('data/processed_small.pkl','rb') as f: data=pickle.load(f)
    fd = data['feature_dims']
    
    train_s, val_s, test_s = data['train'], data['val'], data['test']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bs = 512
    
    model = DINV2(num_items=fd['num_items'], num_categories=fd['num_categories'],
                  num_behaviors=NUM_BEHAVIOR_TYPES, embed_dim=32, num_heads=4,
                  num_transformer_layers=2, max_seq_len=50, dropout=0.1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    
    best_auc = 0.0
    if os.path.exists(STATE):
        st = torch.load(STATE, map_location=device, weights_only=False)
        model.load_state_dict(st['model_state_dict'])
        optimizer.load_state_dict(st['optimizer_state_dict'])
        best_auc = st.get('best_auc', 0)
        log(f"Resumed from epoch {st['epoch']}, best_auc={best_auc:.4f}")
    
    if epoch_idx == 0:
        tp, _ = count_parameters(model)
        log("=" * 70)
        log("DIN-V2: Behavior-Type-Aware Deep Interest Network + Transformer Encoder")
        log("=" * 70)
        log(f"Device: {device} | Params: {tp:,} | embed=32, heads=4, layers=2")
        log(f"Data: train={len(train_s)}, val={len(val_s)}, test={len(test_s)}")
        log(f"Features: {fd}")
        log(f"Config: bs={bs}, lr=1e-3, epochs={total_epochs}, max_seq=50")
    
    # Train
    train_dl = DataLoader(UserBehaviorDataset(train_s, 50), batch_size=bs, shuffle=True, num_workers=0, drop_last=True)
    
    log(f"\n--- Epoch {epoch_idx+1}/{total_epochs} ---")
    model.train()
    lm = AverageMeter()
    al, ap = [], []
    t0 = time.time()
    
    for i, batch in enumerate(train_dl):
        h = {k: v.to(device) for k,v in batch.items()}
        logits = model(h['hist_items'], h['hist_behaviors'], h['hist_categories'],
                      h['mask'], h['target_item'], h['target_category'])
        loss = criterion(logits, h['label'])
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        lm.update(loss.item(), h['hist_items'].size(0))
        pr = torch.sigmoid(logits).detach().cpu().numpy().flatten()
        al.extend(h['label'].cpu().numpy().flatten().tolist())
        ap.extend(pr.tolist())
        if (i+1) % 5 == 0 or i == len(train_dl)-1:
            log(f"  Batch [{i+1}/{len(train_dl)}] Loss: {lm.avg:.4f} ({time.time()-t0:.1f}s)")
    
    tt = time.time()-t0
    tm = calculate_metrics(np.array(al), np.array(ap))
    log(f"[Train] {tt:.1f}s Loss:{lm.avg:.4f} AUC:{tm['auc']:.4f} Acc:{tm['accuracy']:.4f}")
    
    # Val
    val_dl = DataLoader(UserBehaviorDataset(val_s, 50), batch_size=bs, shuffle=False, num_workers=0)
    model.eval()
    vl, vp = [], []
    with torch.no_grad():
        for batch in val_dl:
            h = {k: v.to(device) for k,v in batch.items()}
            logits = model(h['hist_items'], h['hist_behaviors'], h['hist_categories'],
                          h['mask'], h['target_item'], h['target_category'])
            pr = torch.sigmoid(logits).cpu().numpy().flatten()
            vl.extend(h['label'].cpu().numpy().flatten().tolist())
            vp.extend(pr.tolist())
    vm = calculate_metrics(np.array(vl), np.array(vp))
    log(f"[Val] AUC:{vm['auc']:.4f} LogLoss:{vm['logloss']:.4f} Acc:{vm['accuracy']:.4f}")
    
    if vm['auc'] > best_auc:
        best_auc = vm['auc']
        log(f"  * New best! AUC={best_auc:.4f}")
    
    # Save state
    torch.save({
        'epoch': epoch_idx+1, 'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_auc': best_auc, 'val_auc': vm['auc'], 'train_loss': lm.avg
    }, STATE)
    log(f"State saved.")
    
    # Test on last epoch
    if epoch_idx == total_epochs - 1:
        test_dl = DataLoader(UserBehaviorDataset(test_s, 50), batch_size=bs, shuffle=False, num_workers=0)
        tl, tp_l = [], []
        with torch.no_grad():
            for batch in test_dl:
                h = {k: v.to(device) for k,v in batch.items()}
                logits = model(h['hist_items'], h['hist_behaviors'], h['hist_categories'],
                              h['mask'], h['target_item'], h['target_category'])
                pr = torch.sigmoid(logits).cpu().numpy().flatten()
                tl.extend(h['label'].cpu().numpy().flatten().tolist())
                tp_l.extend(pr.tolist())
        test_m = calculate_metrics(np.array(tl), np.array(tp_l))
        log(f"\n[Test] AUC:{test_m['auc']:.4f} LogLoss:{test_m['logloss']:.4f} Acc:{test_m['accuracy']:.4f}")
        
        with open(RESULTS, 'w') as f:
            json.dump({'model':'DIN-V2','test':test_m,'best_val_auc':best_auc,
                       'feature_dims':fd,'params':sum(p.numel() for p in model.parameters())}, f, indent=2, default=str)
        log("=" * 70)
        log(f"DONE! Best Val AUC: {best_auc:.4f} | Test AUC: {test_m['auc']:.4f}")
        log("=" * 70)

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--epoch', type=int, required=True)
    p.add_argument('--total', type=int, default=3)
    args = p.parse_args()
    run_one_epoch(args.epoch, args.total)
