#!/usr/bin/env python3
"""
Two-phase epoch training: 
Phase 1 (warmup): Small warmup forward/backward to trigger JIT, then save state
Phase 2+: Normal training
"""
import os, sys, time, json, pickle, logging
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import DINV2

DATA='data/processed_30k.pkl'; STATE='checkpoints/es.pt'; BEST='checkpoints/best.pt'
LOG='logs/full_training.log'; RES='logs/full_results.json'
C=dict(em=32,hd=4,ly=2,sq=30,bs=256,lr=0.001,dr=0.1,ff=128,ep=5)

class DS(Dataset):
    def __init__(s,d,m=30): s.d=d; s.m=m
    def __len__(s): return len(s.d)
    def __getitem__(s,i):
        r=s.d[i]; h=r['hist_items'][-s.m:]; b=r['hist_behaviors'][-s.m:]
        c=r['hist_categories'][-s.m:]; l=len(h); p=s.m-l
        return dict(hi=torch.LongTensor(h+[0]*p),hb=torch.LongTensor(b+[0]*p),
                    hc=torch.LongTensor(c+[0]*p),mk=torch.FloatTensor([1]*l+[0]*p),
                    ti=torch.LongTensor([r['target_item']]),tc=torch.LongTensor([r['target_category']]),
                    y=torch.FloatTensor([r['label']]))

def mets(l,p):
    from sklearn.metrics import roc_auc_score,accuracy_score
    pc=np.clip(p,1e-7,1-1e-7)
    try: a=roc_auc_score(l,p)
    except: a=0.5
    return {'auc':round(a,4),'logloss':round(float(-np.mean(l*np.log(pc)+(1-l)*np.log(1-pc))),4),'acc':round(float(accuracy_score(l,(p>=0.5).astype(int))),4)}

def fwd(m,b): return m(b['hi'],b['hb'],b['hc'],b['mk'],b['ti'],b['tc'])

def ev(m,dl,cr):
    m.eval(); al,ap,tl,tn=[],[],0.,0
    with torch.no_grad():
        for b in dl:
            o=fwd(m,b); lo=cr(o,b['y']); n=b['y'].size(0); tl+=lo.item()*n; tn+=n
            al.extend(b['y'].numpy().flatten().tolist()); ap.extend(torch.sigmoid(o).numpy().flatten().tolist())
    r=mets(np.array(al),np.array(ap)); r['loss']=round(tl/max(tn,1),4); return r

def main():
    os.makedirs('logs',exist_ok=True); os.makedirs('checkpoints',exist_ok=True)
    lg=logging.getLogger('ep2'); lg.setLevel(logging.INFO); lg.handlers=[]
    fh=logging.FileHandler(LOG,mode='a'); ch=logging.StreamHandler()
    f_=logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(f_); ch.setFormatter(f_); lg.addHandler(fh); lg.addHandler(ch)
    
    with open(DATA,'rb') as f: data=pickle.load(f)
    fd=data['feature_dims']
    model=DINV2(num_items=fd['num_items'],num_categories=fd['num_categories'],num_behaviors=4,
                embed_dim=C['em'],num_heads=C['hd'],num_transformer_layers=C['ly'],
                max_seq_len=C['sq'],dropout=C['dr'],ff_dim=C['ff'])
    cr=nn.BCEWithLogitsLoss(); op=optim.Adam(model.parameters(),lr=C['lr'],weight_decay=1e-5)
    tp=sum(p.numel() for p in model.parameters())
    
    ce,ba,hist,tt,warmed=1,0.,[],0.,False
    if os.path.exists(STATE):
        st=torch.load(STATE,weights_only=False)
        model.load_state_dict(st['ms']); op.load_state_dict(st['os'])
        ce=st['ce']; ba=st['ba']; hist=st['h']; tt=st['tt']; warmed=st.get('w',False)
        for pg in op.param_groups: pg['lr']=C['lr']*(0.5**((ce-1)//2))
    
    if not warmed:
        # Warmup phase - trigger JIT compilation with tiny batch
        lg.info("="*70)
        lg.info("DIN-V2 Full Training")
        lg.info("="*70)
        lg.info(f"Params:{tp:,}|Embed:{C['em']}|Trans:{C['ly']}L{C['hd']}H|BS:{C['bs']}|Seq:{C['sq']}")
        lg.info(f"Data:Train={len(data['train']):,} Val={len(data['val']):,} Test={len(data['test']):,}")
        lg.info(f"Features:{fd} | Epochs:{C['ep']} | CPU({os.cpu_count()} cores)")
        lg.info("="*70)
        
        lg.info("Warming up JIT compilation...")
        t0=time.time()
        model.train()
        hi=torch.randint(0,100,(4,C['sq'])); hb=torch.randint(0,4,(4,C['sq']))
        hc=torch.randint(0,100,(4,C['sq'])); mk=torch.ones(4,C['sq'])
        ti=torch.randint(1,100,(4,1)); tc=torch.randint(1,100,(4,1)); y=torch.zeros(4,1)
        o=model(hi,hb,hc,mk,ti,tc); lo=cr(o,y); op.zero_grad(); lo.backward(); op.step()
        lg.info(f"  Warmup done in {time.time()-t0:.1f}s")
        
        warmed=True
        torch.save({'ms':model.state_dict(),'os':op.state_dict(),'ce':ce,'ba':ba,'h':hist,'tt':tt,'w':True},STATE)
        lg.info("  State saved. Run again to start training.")
        print("WARMUP_DONE"); return
    
    if ce>C['ep']:
        lg.info("\n--- Final Test ---")
        if os.path.exists(BEST):
            ck=torch.load(BEST,weights_only=False); model.load_state_dict(ck['ms']); lg.info(f"Best model epoch {ck['ep']}")
        tl=DataLoader(DS(data['test'],C['sq']),batch_size=C['bs'],shuffle=False,num_workers=0)
        tm=ev(model,tl,cr)
        lg.info(f"[Test] AUC={tm['auc']:.4f} LogLoss={tm['logloss']:.4f} Acc={tm['acc']:.4f}")
        r={'model':'DIN-V2','params':tp,'arch':{'behavior_emb':'4x32','transformer':'2L4H causal learnable-pos',
           'target_attn':'MLP-based','mlp':'96->256->128->64->1'},'history':hist,'test':tm,'best_val':ba,
           'time_s':round(tt,1),'raw_data':'1.83M records (50K users, 100K items)',
           'train_data':{'train':len(data['train']),'val':len(data['val']),'test':len(data['test'])},
           'features':fd}
        with open(RES,'w') as f: json.dump(r,f,indent=2,default=str)
        lg.info(f"COMPLETE! Best Val={ba:.4f} Test={tm['auc']:.4f} Time={tt:.0f}s")
        print("DONE"); return
    
    # Train epoch
    torch.manual_seed(42+ce)
    trl=DataLoader(DS(data['train'],C['sq']),batch_size=C['bs'],shuffle=True,num_workers=0,drop_last=True)
    lg.info(f"\n--- Epoch {ce}/{C['ep']} ({len(trl)} batches) ---")
    
    model.train(); t0=time.time(); tl_,tn_=0.,0; al,ap=[],[]
    for i,b in enumerate(trl):
        o=fwd(model,b); lo=cr(o,b['y'])
        op.zero_grad(); lo.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); op.step()
        n=b['y'].size(0); tl_+=lo.item()*n; tn_+=n
        al.extend(b['y'].numpy().flatten().tolist()); ap.extend(torch.sigmoid(o).detach().numpy().flatten().tolist())
        if (i+1)%25==0: lg.info(f"  B[{i+1}/{len(trl)}] Loss:{tl_/tn_:.4f}")
    
    et=time.time()-t0; tt+=et
    tm_=mets(np.array(al),np.array(ap)); tm_['loss']=round(tl_/max(tn_,1),4)
    lg.info(f"[Train] E{ce}: Loss={tm_['loss']:.4f} AUC={tm_['auc']:.4f} Acc={tm_['acc']:.4f} ({et:.0f}s)")
    
    vl=DataLoader(DS(data['val'],C['sq']),batch_size=C['bs'],shuffle=False,num_workers=0)
    vm=ev(model,vl,cr)
    lg.info(f"[Val]   E{ce}: Loss={vm['loss']:.4f} AUC={vm['auc']:.4f} Acc={vm['acc']:.4f}")
    
    ib=vm['auc']>ba
    if ib: ba=vm['auc']; torch.save({'ms':model.state_dict(),'ep':ce,'va':vm['auc']},BEST); lg.info(f"  ★ Best AUC={ba:.4f}")
    
    hist.append({'epoch':ce,'tr_loss':tm_['loss'],'tr_auc':tm_['auc'],'tr_acc':tm_['acc'],
                 'v_loss':vm['loss'],'v_auc':vm['auc'],'v_acc':vm['acc'],'v_ll':vm['logloss'],
                 'lr':op.param_groups[0]['lr'],'best':ib,'time':round(et,1)})
    ce+=1
    for pg in op.param_groups: pg['lr']=C['lr']*(0.5**((ce-1)//2))
    torch.save({'ms':model.state_dict(),'os':op.state_dict(),'ce':ce,'ba':ba,'h':hist,'tt':tt,'w':True},STATE)
    print(f"E{ce-1} DONE (next:{ce})")

if __name__=='__main__': main()
