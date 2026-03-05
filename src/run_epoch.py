#!/usr/bin/env python3
"""
DIN-V2 Complete Training Pipeline.
Trains 1 epoch per invocation with state persistence.
"""
import os,sys,time,json,pickle,logging
import numpy as np; import torch,torch.nn as nn,torch.optim as optim
from torch.utils.data import Dataset,DataLoader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model_sdpa import DINV2Sdpa

DATA='data/td.pkl'; LOG='logs/full_training.log'; RES='logs/full_results.json'
BEST='checkpoints/best.pt'; STATE='checkpoints/state.pt'
C=dict(d=32,h=4,nl=2,msl=20,bs=512,lr=0.001,drop=0.1,ff=128,hd=[128,64],ep=5)

class DS(Dataset):
    def __init__(s,d,m=20): s.d=d; s.m=m
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
    return {'auc':round(a,4),'logloss':round(float(-np.mean(l*np.log(pc)+(1-l)*np.log(1-pc))),4),
            'acc':round(float(accuracy_score(l,(p>=0.5).astype(int))),4)}

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
    lg=logging.getLogger('din'); lg.setLevel(logging.INFO); lg.handlers=[]
    fh=logging.FileHandler(LOG,mode='a'); ch=logging.StreamHandler()
    f_=logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(f_); ch.setFormatter(f_); lg.addHandler(fh); lg.addHandler(ch)
    
    with open(DATA,'rb') as f: data=pickle.load(f)
    fd=data['feature_dims']
    model=DINV2Sdpa(ni=fd['num_items'],nc=fd['num_categories'],nb=4,d=C['d'],h=C['h'],nl=C['nl'],
                    msl=C['msl'],ff=C['ff'],hd=C['hd'],drop=C['drop'])
    cr=nn.BCEWithLogitsLoss(); op=optim.Adam(model.parameters(),lr=C['lr'],weight_decay=1e-5)
    tp=sum(p.numel() for p in model.parameters())
    
    ce=1; ba=0.; hist=[]; tt=0.
    if os.path.exists(STATE):
        st=torch.load(STATE,weights_only=False)
        model.load_state_dict(st['ms']); op.load_state_dict(st['os'])
        ce=st['ce']; ba=st['ba']; hist=st['h']; tt=st['tt']
        for pg in op.param_groups: pg['lr']=C['lr']*(0.5**((ce-1)//2))
    else:
        lg.info("="*70)
        lg.info("DIN-V2: Behavior-Type Embedding + Transformer Encoder Training")
        lg.info("="*70)
        lg.info(f"Params:{tp:,} | Embed:{C['d']} | Transformer:{C['nl']}L{C['h']}H")
        lg.info(f"Behavior Types: 4 (pv/fav/cart/buy) independent embeddings")
        lg.info(f"Causal Mask + Learnable Position Encoding")
        lg.info(f"Train:{len(data['train']):,} Val:{len(data['val']):,} Test:{len(data['test']):,}")
        lg.info(f"Features:{fd} | BS:{C['bs']} | LR:{C['lr']} | Epochs:{C['ep']}")
        lg.info(f"CPU: {os.cpu_count()} cores")
        lg.info("="*70)
    
    if ce>C['ep']:
        if os.path.exists(BEST): model.load_state_dict(torch.load(BEST,weights_only=False)['ms'])
        tl=DataLoader(DS(data['test'],C['msl']),batch_size=C['bs'],num_workers=0)
        tm=ev(model,tl,cr)
        lg.info(f"\n[TEST] AUC={tm['auc']:.4f} LogLoss={tm['logloss']:.4f} Acc={tm['acc']:.4f}")
        r={'model':'DIN-V2 (Behavior-Type Embedding + Transformer Encoder)',
           'params':tp,'config':C,'history':hist,'test':tm,'best_val':ba,'time_s':round(tt,1),
           'architecture':{'behavior_embedding':'4 types (pv/fav/cart/buy) x 32dim added to item embedding',
                           'transformer_encoder':'2 layers, 4 heads, F.scaled_dot_product_attention, causal mask, learnable position encoding',
                           'target_attention':'Enhanced MLP-based attention (query*key, query-key, concat)',
                           'prediction_mlp':'96->128->64->1 with BatchNorm+PReLU+Dropout'},
           'raw_data':{'total_records':1829350,'users':50000,'items':100000,'categories':5000,
                       'behavior_types':'4 (pv/fav/cart/buy) with Markov transition dependencies'},
           'training_data':{'train':len(data['train']),'val':len(data['val']),'test':len(data['test']),
                            'sampled_from':'813,652 total samples from 1.83M behavior records'},
           'features':fd}
        with open(RES,'w') as f: json.dump(r,f,indent=2,default=str)
        lg.info(f"DONE! Best Val AUC={ba:.4f} | Test AUC={tm['auc']:.4f} | Time={tt:.0f}s")
        print("DONE"); return
    
    torch.manual_seed(42+ce)
    trl=DataLoader(DS(data['train'],C['msl']),batch_size=C['bs'],shuffle=True,num_workers=0,drop_last=True)
    vl=DataLoader(DS(data['val'],C['msl']),batch_size=C['bs'],num_workers=0)
    
    lg.info(f"\n--- Epoch {ce}/{C['ep']} ({len(trl)} batches) ---")
    model.train(); t0=time.time(); tl_,tn_=0.,0; al,ap=[],[]
    for i,b in enumerate(trl):
        o=fwd(model,b); lo=cr(o,b['y'])
        op.zero_grad(); lo.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); op.step()
        n=b['y'].size(0); tl_+=lo.item()*n; tn_+=n
        al.extend(b['y'].numpy().flatten().tolist()); ap.extend(torch.sigmoid(o).detach().numpy().flatten().tolist())
        lg.info(f"  B[{i+1}/{len(trl)}] Loss:{tl_/tn_:.4f}")
    
    et=time.time()-t0; tt+=et
    tm_=mets(np.array(al),np.array(ap)); tm_['loss']=round(tl_/max(tn_,1),4)
    lg.info(f"[TRAIN] E{ce}: Loss={tm_['loss']:.4f} AUC={tm_['auc']:.4f} Acc={tm_['acc']:.4f} ({et:.0f}s)")
    
    vm=ev(model,vl,cr)
    lg.info(f"[VAL]   E{ce}: Loss={vm['loss']:.4f} AUC={vm['auc']:.4f} Acc={vm['acc']:.4f}")
    
    ib=vm['auc']>ba
    if ib: ba=vm['auc']; torch.save({'ms':model.state_dict(),'ep':ce,'va':vm['auc']},BEST); lg.info(f"  ★ Best model! AUC={ba:.4f}")
    
    hist.append({'epoch':ce,'train_loss':tm_['loss'],'train_auc':tm_['auc'],'train_acc':tm_['acc'],
                 'val_loss':vm['loss'],'val_auc':vm['auc'],'val_acc':vm['acc'],'val_logloss':vm['logloss'],
                 'lr':op.param_groups[0]['lr'],'is_best':ib,'time_s':round(et,1)})
    ce+=1
    for pg in op.param_groups: pg['lr']=C['lr']*(0.5**((ce-1)//2))
    torch.save({'ms':model.state_dict(),'os':op.state_dict(),'ce':ce,'ba':ba,'h':hist,'tt':tt},STATE)
    print(f"EPOCH_{ce-1}_DONE")

if __name__=='__main__': main()
