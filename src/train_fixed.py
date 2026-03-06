#!/usr/bin/env python3
"""
DIN-V2 修复版训练脚本
支持分步训练（状态恢复）、V1/V2对比、详细日志
"""
import os, sys, time, json, pickle, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ========== Dataset ==========
class DINDataset(Dataset):
    def __init__(self, samples, max_seq_len=20):
        self.samples = samples
        self.max_seq_len = max_seq_len
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        hi = s['hist_items'][-self.max_seq_len:]
        hb = s['hist_behaviors'][-self.max_seq_len:]
        hc = s['hist_categories'][-self.max_seq_len:]
        sl = len(hi)
        pad = self.max_seq_len - sl
        return {
            'hist_items': torch.LongTensor(hi + [0]*pad),
            'hist_behaviors': torch.LongTensor(hb + [0]*pad),
            'hist_categories': torch.LongTensor(hc + [0]*pad),
            'mask': torch.FloatTensor([1]*sl + [0]*pad),
            'target_item': torch.LongTensor([s['target_item']]),
            'target_category': torch.LongTensor([s['target_category']]),
            'label': torch.FloatTensor([s['label']]),
        }

# ========== Models ==========
class TargetAttention(nn.Module):
    def __init__(self, dim, hdim=64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim*4,hdim),nn.PReLU(),nn.Linear(hdim,1))
    def forward(self, q, k, mask=None):
        q = q.unsqueeze(1).expand(-1,k.size(1),-1)
        a = self.mlp(torch.cat([q,k,q-k,q*k],dim=-1)).squeeze(-1)
        if mask is not None: a = a.masked_fill(mask==0, -1e9)
        w = torch.softmax(a, dim=-1)
        return torch.bmm(w.unsqueeze(1), k).squeeze(1)

class DINV1(nn.Module):
    """V1基线: 无行为类型区分, 无Transformer"""
    def __init__(self, ni, nc, nb=4, dim=16, hash_buckets=10000, **kw):
        super().__init__()
        self.model_name = "DIN-V1"
        self.hb = hash_buckets
        self.ie = nn.Embedding(hash_buckets+1, dim, padding_idx=0)
        self.ce = nn.Embedding(nc, dim, padding_idx=0)
        self.att = TargetAttention(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim*3,128), nn.BatchNorm1d(128), nn.PReLU(), nn.Dropout(0.1),
            nn.Linear(128,64), nn.BatchNorm1d(64), nn.PReLU(), nn.Dropout(0.1),
            nn.Linear(64,1))
        self._init()
    def _init(self):
        for m in self.modules():
            if isinstance(m,nn.Linear): nn.init.xavier_uniform_(m.weight)
            elif isinstance(m,nn.Embedding): nn.init.normal_(m.weight, std=0.01)
    def _hash(self, x): return (x % self.hb).clamp(min=0)
    def forward(self, hi, hb, hc, mask, ti, tc, **kw):
        he = self.ie(self._hash(hi))
        te = self.ie(self._hash(ti.squeeze(1)))
        tce = self.ce(tc.squeeze(1))
        ui = self.att(te, he, mask)
        return self.mlp(torch.cat([ui,te,tce],dim=-1))

class DINV2(nn.Module):
    """V2: 行为类型Embedding + Transformer Encoder + Target Attention"""
    def __init__(self, ni, nc, nb=4, dim=16, nheads=2, nlayers=1,
                 max_seq_len=20, hash_buckets=10000, **kw):
        super().__init__()
        self.model_name = "DIN-V2"
        self.hb = hash_buckets
        self.ie = nn.Embedding(hash_buckets+1, dim, padding_idx=0)
        self.ce = nn.Embedding(nc, dim, padding_idx=0)
        self.be = nn.Embedding(nb, dim)  # 行为类型Embedding
        self.pe = nn.Embedding(max_seq_len, dim)  # 可学习位置编码
        
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=nheads, dim_feedforward=dim*4,
            dropout=0.1, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=nlayers,
                                                  enable_nested_tensor=False)
        self.ln = nn.LayerNorm(dim)
        self.att = TargetAttention(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim*3,128), nn.BatchNorm1d(128), nn.PReLU(), nn.Dropout(0.1),
            nn.Linear(128,64), nn.BatchNorm1d(64), nn.PReLU(), nn.Dropout(0.1),
            nn.Linear(64,1))
        self._init()
    def _init(self):
        for m in self.modules():
            if isinstance(m,nn.Linear): nn.init.xavier_uniform_(m.weight)
            elif isinstance(m,nn.Embedding): nn.init.normal_(m.weight, std=0.01)
    def _hash(self, x): return (x % self.hb).clamp(min=0)
    def forward(self, hi, hb, hc, mask, ti, tc, **kw):
        B, S = hi.shape
        # Step1: Item + Behavior Type composite embedding
        he = self.ie(self._hash(hi)) + self.be(hb)
        he = he * mask.unsqueeze(-1)
        # Step2: Add position encoding
        pos = torch.arange(S, device=hi.device).unsqueeze(0).expand(B,-1)
        he = he + self.pe(pos)
        # Step3: Transformer Encoder with causal mask
        causal = nn.Transformer.generate_square_subsequent_mask(S, device=hi.device)
        he = self.transformer(he, mask=causal, is_causal=True)
        he = self.ln(he) * mask.unsqueeze(-1)
        # Step4: Target Attention
        te = self.ie(self._hash(ti.squeeze(1)))
        tce = self.ce(tc.squeeze(1))
        ui = self.att(te, he, mask)
        # Step5: MLP
        return self.mlp(torch.cat([ui,te,tce],dim=-1))

# ========== Training ==========
def calc_metrics(labels, preds):
    try: auc = roc_auc_score(labels, preds)
    except: auc = 0.5
    try: ll = log_loss(labels, np.clip(preds,1e-7,1-1e-7))
    except: ll = float('inf')
    acc = accuracy_score(labels, (preds>=0.5).astype(int))
    return {'auc':auc, 'logloss':ll, 'accuracy':acc}

def train_one_epoch(model, loader, optimizer, criterion, device, max_batches=None):
    model.train()
    all_loss, all_labels, all_preds = [], [], []
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches: break
        hi = batch['hist_items'].to(device)
        hb = batch['hist_behaviors'].to(device)
        hc = batch['hist_categories'].to(device)
        mk = batch['mask'].to(device)
        ti = batch['target_item'].to(device)
        tc = batch['target_category'].to(device)
        lb = batch['label'].to(device)
        
        logits = model(hi, hb, hc, mk, ti, tc)
        loss = criterion(logits, lb)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        all_loss.append(loss.item())
        p = torch.sigmoid(logits).detach().cpu().numpy().flatten()
        all_labels.extend(lb.cpu().numpy().flatten())
        all_preds.extend(p)
    
    m = calc_metrics(np.array(all_labels), np.array(all_preds))
    m['loss'] = np.mean(all_loss)
    return m

def evaluate(model, loader, criterion, device, max_batches=None):
    model.eval()
    all_labels, all_preds, total_loss, total_n = [], [], 0, 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches and i >= max_batches: break
            hi = batch['hist_items'].to(device)
            hb = batch['hist_behaviors'].to(device)
            hc = batch['hist_categories'].to(device)
            mk = batch['mask'].to(device)
            ti = batch['target_item'].to(device)
            tc = batch['target_category'].to(device)
            lb = batch['label'].to(device)
            logits = model(hi, hb, hc, mk, ti, tc)
            loss = criterion(logits, lb)
            total_loss += loss.item()*hi.size(0); total_n += hi.size(0)
            p = torch.sigmoid(logits).cpu().numpy().flatten()
            all_labels.extend(lb.cpu().numpy().flatten())
            all_preds.extend(p)
    m = calc_metrics(np.array(all_labels), np.array(all_preds))
    m['loss'] = total_loss / max(total_n,1)
    return m

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='v2', choices=['v1','v2'])
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--embed_dim', type=int, default=16)
    parser.add_argument('--max_seq_len', type=int, default=20)
    parser.add_argument('--hash_buckets', type=int, default=10000)
    parser.add_argument('--max_train_samples', type=int, default=0, help='0=all')
    parser.add_argument('--max_batches', type=int, default=0, help='0=all')
    parser.add_argument('--log_file', type=str, default='')
    parser.add_argument('--state_file', type=str, default='')
    parser.add_argument('--start_epoch', type=int, default=0)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)
    
    # Load data
    pkl = 'data/processed_fixed.pkl'
    print(f"Loading {pkl}...", flush=True)
    with open(pkl,'rb') as f: data = pickle.load(f)
    fdims = data['feature_dims']
    
    train_data = data['train']
    if args.max_train_samples > 0:
        train_data = train_data[:args.max_train_samples]
    
    print(f"Train: {len(train_data):,}, Val: {len(data['val']):,}, Test: {len(data['test']):,}", flush=True)
    
    train_loader = DataLoader(DINDataset(train_data, args.max_seq_len),
                              batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(DINDataset(data['val'], args.max_seq_len),
                            batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(DINDataset(data['test'], args.max_seq_len),
                             batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Build model
    if args.model == 'v2':
        model = DINV2(fdims['num_items'], fdims['num_categories'], fdims['num_behaviors'],
                      dim=args.embed_dim, max_seq_len=args.max_seq_len,
                      hash_buckets=args.hash_buckets)
    else:
        model = DINV1(fdims['num_items'], fdims['num_categories'], fdims['num_behaviors'],
                      dim=args.embed_dim, hash_buckets=args.hash_buckets)
    
    model = model.to(device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"Model: {model.model_name}, Params: {nparams:,}", flush=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    
    # Load state
    start_epoch = args.start_epoch
    if args.state_file and os.path.exists(args.state_file):
        state = torch.load(args.state_file, map_location=device, weights_only=False)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        start_epoch = state['epoch']
        print(f"Resumed from epoch {start_epoch}", flush=True)
    
    log_lines = []
    max_b = args.max_batches if args.max_batches > 0 else None
    
    header = f"{'Epoch':>5} | {'Train Loss':>10} | {'Train AUC':>9} | {'Val Loss':>8} | {'Val AUC':>7} | {'Time':>6}"
    sep = "-"*len(header)
    print(sep, flush=True); print(header, flush=True); print(sep, flush=True)
    log_lines.extend([sep, header, sep])
    
    best_val_auc = 0
    history = []
    
    for ep in range(start_epoch, args.epochs):
        t0 = time.time()
        tm = train_one_epoch(model, train_loader, optimizer, criterion, device, max_b)
        vm = evaluate(model, val_loader, criterion, device, max_b)
        dt = time.time() - t0
        
        line = f"{ep+1:>5} | {tm['loss']:>10.4f} | {tm['auc']:>9.4f} | {vm['loss']:>8.4f} | {vm['auc']:>7.4f} | {dt:>5.1f}s"
        print(line, flush=True)
        log_lines.append(line)
        
        history.append({
            'epoch':ep+1, 'train_loss':round(tm['loss'],4), 'train_auc':round(tm['auc'],4),
            'val_loss':round(vm['loss'],4), 'val_auc':round(vm['auc'],4),
            'val_logloss':round(vm['logloss'],4), 'val_acc':round(vm['accuracy'],4), 'time':round(dt,1)
        })
        
        if vm['auc'] > best_val_auc:
            best_val_auc = vm['auc']
        
        # Save state
        if args.state_file:
            torch.save({'model':model.state_dict(), 'optimizer':optimizer.state_dict(),
                        'epoch':ep+1, 'best_val_auc':best_val_auc}, args.state_file)
    
    # Test
    print(sep, flush=True); log_lines.append(sep)
    testm = evaluate(model, test_loader, criterion, device, max_b)
    tline = f"TEST  | {'':>10} | {'':>9} | {testm['loss']:>8.4f} | {testm['auc']:>7.4f} |"
    print(tline, flush=True); log_lines.append(tline)
    print(f"Test AUC={testm['auc']:.4f}, LogLoss={testm['logloss']:.4f}, Acc={testm['accuracy']:.4f}", flush=True)
    log_lines.append(f"Test AUC={testm['auc']:.4f}, LogLoss={testm['logloss']:.4f}, Acc={testm['accuracy']:.4f}")
    
    # Save log
    if args.log_file:
        os.makedirs(os.path.dirname(args.log_file) or '.', exist_ok=True)
        with open(args.log_file, 'w') as f:
            f.write('\n'.join(log_lines))
    
    # Save results
    results = {
        'model': args.model, 'model_name': model.model_name,
        'params': nparams, 'embed_dim': args.embed_dim, 'hash_buckets': args.hash_buckets,
        'train_samples': len(train_data), 'val_samples': len(data['val']), 'test_samples': len(data['test']),
        'best_val_auc': round(best_val_auc,4),
        'test_auc': round(testm['auc'],4), 'test_logloss': round(testm['logloss'],4),
        'test_accuracy': round(testm['accuracy'],4),
        'history': history,
        'feature_dims': fdims,
        'args': vars(args),
    }
    res_file = args.log_file.replace('.log','.json') if args.log_file else f'logs/{args.model}_results.json'
    os.makedirs(os.path.dirname(res_file) or '.', exist_ok=True)
    with open(res_file,'w') as f: json.dump(results, f, indent=2)
    print(f"Results saved: {res_file}", flush=True)

if __name__ == '__main__':
    main()
