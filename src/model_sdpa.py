#!/usr/bin/env python3
"""DIN-V2 using F.scaled_dot_product_attention for speed."""
import torch, torch.nn as nn, torch.nn.functional as F

class SdpaBlock(nn.Module):
    def __init__(self, d, h, ff, drop=0.1):
        super().__init__()
        self.h=h; self.hd=d//h
        self.qkv=nn.Linear(d,3*d); self.out=nn.Linear(d,d)
        self.ff=nn.Sequential(nn.Linear(d,ff),nn.GELU(),nn.Dropout(drop),nn.Linear(ff,d))
        self.ln1=nn.LayerNorm(d); self.ln2=nn.LayerNorm(d); self.drop=nn.Dropout(drop)
    def forward(self,x):
        B,S,D=x.shape; nx=self.ln1(x)
        qkv=self.qkv(nx).reshape(B,S,3,self.h,self.hd).permute(2,0,3,1,4)
        q,k,v=qkv[0],qkv[1],qkv[2]
        a=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        a=a.transpose(1,2).reshape(B,S,D)
        x=x+self.drop(self.out(a))
        x=x+self.drop(self.ff(self.ln2(x)))
        return x

class SdpaEncoder(nn.Module):
    def __init__(self,d,h=4,nl=2,msl=20,drop=0.1,ff=128):
        super().__init__()
        self.pos=nn.Embedding(msl,d)
        self.blocks=nn.ModuleList([SdpaBlock(d,h,ff,drop) for _ in range(nl)])
        self.norm=nn.LayerNorm(d); self.drop=nn.Dropout(drop)
    def forward(self,x,mask=None):
        B,S,D=x.shape
        if mask is not None: x=x*mask.unsqueeze(-1)
        p=torch.arange(S,device=x.device).unsqueeze(0).expand(B,-1)
        x=self.drop(x+self.pos(p))
        for b in self.blocks: x=b(x)
        x=self.norm(x)
        if mask is not None: x=x*mask.unsqueeze(-1)
        return x

class TargetAttn(nn.Module):
    def __init__(self,d,hd=64):
        super().__init__()
        self.m=nn.Sequential(nn.Linear(d*4,hd),nn.PReLU(),nn.Linear(hd,hd),nn.PReLU(),nn.Linear(hd,1))
    def forward(self,q,k,mask=None):
        S=k.size(1); q=q.unsqueeze(1).expand(-1,S,-1)
        s=self.m(torch.cat([q,k,q-k,q*k],dim=-1)).squeeze(-1)
        if mask is not None: s=s.masked_fill(mask==0,-1e9)
        return torch.bmm(F.softmax(s,dim=-1).unsqueeze(1),k).squeeze(1)

class DINV2Sdpa(nn.Module):
    def __init__(self,ni,nc,nb=4,d=32,h=4,nl=2,msl=20,hd=[128,64],drop=0.1,ff=128,**kw):
        super().__init__()
        self.model_name="DIN-V2"
        self.ie=nn.Embedding(ni,d,padding_idx=0); self.ce=nn.Embedding(nc,d,padding_idx=0)
        self.be=nn.Embedding(nb,d)
        self.enc=SdpaEncoder(d,h,nl,msl,drop,ff)
        self.ta=TargetAttn(d,64)
        layers=[]
        prev=d*3
        for dim in hd: layers.extend([nn.Linear(prev,dim),nn.BatchNorm1d(dim),nn.PReLU(),nn.Dropout(drop)]); prev=dim
        layers.append(nn.Linear(prev,1))
        self.mlp=nn.Sequential(*layers)
        self._init()
    def _init(self):
        for m in self.modules():
            if isinstance(m,nn.Linear): nn.init.xavier_uniform_(m.weight); m.bias is not None and nn.init.zeros_(m.bias)
            elif isinstance(m,nn.Embedding): nn.init.normal_(m.weight,std=0.01); m.padding_idx is not None and nn.init.zeros_(m.weight[m.padding_idx])
    def forward(self,hi,hb,hc,mk,ti,tc,**kw):
        h=self.ie(hi)+self.be(hb); e=self.enc(h,mk)
        te=self.ie(ti.squeeze(1)); tce=self.ce(tc.squeeze(1))
        u=self.ta(te,e,mk)
        return self.mlp(torch.cat([u,te,tce],dim=-1))
