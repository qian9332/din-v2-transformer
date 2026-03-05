#!/usr/bin/env python3
"""Fast single-shot training script that completes within 80 seconds."""
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

LOG = os.path.join(P, 'logs', 'training_full.log')
os.makedirs(os.path.dirname(LOG), exist_ok=True)
os.makedirs(os.path.join(P, 'checkpoints'), exist_ok=True)

lines = []
def log(msg):
    t = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    s = f"{t} - INFO - {msg}"
    print(s); lines.append(s)

log("=" * 70)
log("DIN-V2 Training: Behavior-Type-Aware + Transformer Encoder")
log("=" * 70)

with open('data/processed_small.pkl','rb') as f:
    data = pickle.load(f)
fd = data['feature_dims']
log(f"Feature dims: {fd}")
log(f"Train: {len(data['train'])}, Val: {len(data['val'])}, Test: {len(data['test'])}")

bs = 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log(f"Device: {device}")

train_dl = DataLoader(UserBehaviorDataset(data['train'], 50), batch_size=bs, shuffle=True, num_workers=0, drop_last=True)
val_dl = DataLoader(UserBehaviorDataset(data['val'], 50), batch_size=bs, shuffle=False, num_workers=0)
test_dl = DataLoader(UserBehaviorDataset(data['test'], 50), batch_size=bs, shuffle=False, num_workers=0)

model = DINV2(num_items=fd['num_items'], num_categories=fd['num_categories'],
              num_behaviors=NUM_BEHAVIOR_TYPES, embed_dim=64, num_heads=4,
              num_transformer_layers=2, max_seq_len=50, dropout=0.1).to(device)

tp, trp = count_parameters(model)
log(f"Model: DIN-V2, Params: {tp:,} (trainable: {trp:,})")
log(f"Architecture: 2-layer 4-head Transformer + Causal Mask + Learnable PosEmb + Target Attention")

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

history = []
best_auc = 0

for epoch in range(3):
    log(f"\n--- Epoch {epoch+1}/3 ---")
    model.train()
    lm = AverageMeter()
    al, ap = [], []
    t0 = time.time()
    
    for i, batch in enumerate(train_dl):
        h = {k: v.to(device) for k, v in batch.items()}
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
        if (i+1) % 10 == 0:
            log(f"  Batch [{i+1}/{len(train_dl)}] Loss: {lm.avg:.4f}")
    
    tt = time.time() - t0
    tm = calculate_metrics(np.array(al), np.array(ap))
    log(f"[Train] {tt:.1f}s Loss:{lm.avg:.4f} AUC:{tm['auc']:.4f} Acc:{tm['accuracy']:.4f}")
    
    model.eval()
    vl, vp = [], []
    with torch.no_grad():
        for batch in val_dl:
            h = {k: v.to(device) for k, v in batch.items()}
            logits = model(h['hist_items'], h['hist_behaviors'], h['hist_categories'],
                          h['mask'], h['target_item'], h['target_category'])
            pr = torch.sigmoid(logits).cpu().numpy().flatten()
            vl.extend(h['label'].cpu().numpy().flatten().tolist())
            vp.extend(pr.tolist())
    vm = calculate_metrics(np.array(vl), np.array(vp))
    log(f"[Val] AUC:{vm['auc']:.4f} LogLoss:{vm['logloss']:.4f} Acc:{vm['accuracy']:.4f}")
    
    history.append({'epoch':epoch+1,'train_loss':lm.avg,'train_auc':tm['auc'],'val_auc':vm['auc'],'val_logloss':vm['logloss'],'time':tt})
    
    if vm['auc'] > best_auc:
        best_auc = vm['auc']
        torch.save({'epoch':epoch+1,'model_state_dict':model.state_dict(),'val_auc':vm['auc']},
                   os.path.join(P,'checkpoints','best_model.pt'))
        log(f"  * Best model saved! AUC: {vm['auc']:.4f}")

# Test
log("\n--- Final Test ---")
ckpt = torch.load(os.path.join(P,'checkpoints','best_model.pt'), map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
tl, tp_l = [], []
with torch.no_grad():
    for batch in test_dl:
        h = {k: v.to(device) for k, v in batch.items()}
        logits = model(h['hist_items'], h['hist_behaviors'], h['hist_categories'],
                      h['mask'], h['target_item'], h['target_category'])
        pr = torch.sigmoid(logits).cpu().numpy().flatten()
        tl.extend(h['label'].cpu().numpy().flatten().tolist())
        tp_l.extend(pr.tolist())
test_m = calculate_metrics(np.array(tl), np.array(tp_l))
log(f"[Test] AUC:{test_m['auc']:.4f} LogLoss:{test_m['logloss']:.4f} Acc:{test_m['accuracy']:.4f}")

results = {'model':'DIN-V2','test':test_m,'val_best_auc':best_auc,'history':history,'feature_dims':fd,'params':tp}
with open(os.path.join(P,'logs','results.json'),'w') as f:
    json.dump(results, f, indent=2, default=str)

log(f"\n{'='*70}")
log(f"COMPLETE! Best Val AUC: {best_auc:.4f} | Test AUC: {test_m['auc']:.4f}")
log(f"{'='*70}")

with open(LOG, 'w') as f:
    f.write('\n'.join(lines) + '\n')
print(f"\nLog saved to {LOG}")
