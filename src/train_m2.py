"""Fine-tune DistilRoBERTa for binary need detection without touching test data."""
import argparse, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

MODEL = "distilroberta-base"

class Posts(Dataset):
    def __init__(self, enc, y): self.enc, self.y = enc, torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return {**{k: torch.tensor(v[i]) for k,v in self.enc.items()}, "labels": self.y[i]}

def threshold(y,p):
    choices=np.unique(np.r_[0,p,1]); ok=[t for t in choices if (p[y==1]>=t).mean()>=.9]
    return float(max(ok)) if ok else 0.0

def main():
    root=Path(__file__).resolve().parents[2]
    ap=argparse.ArgumentParser(); ap.add_argument('--data',type=Path,default=root/'data/processed/train_labels.parquet'); ap.add_argument('--out',type=Path,default=root/'models/m2_distilroberta'); ap.add_argument('--epochs',type=int,default=5); args=ap.parse_args()
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    df=pd.read_parquet(args.data); y=df.is_need.astype(int).to_numpy(); groups=df.dup_key.fillna(df.post_id).to_numpy()
    tr,va=next(StratifiedGroupKFold(n_splits=7,shuffle=True,random_state=42).split(df,y,groups))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print(f'device={device}; train={len(tr)} validation={len(va)}')
    tok=AutoTokenizer.from_pretrained(MODEL); enc=tok(df.text.fillna('').tolist(),truncation=True,padding=True,max_length=256)
    train=Posts({k:np.asarray(v)[tr] for k,v in enc.items()},y[tr]); valid=Posts({k:np.asarray(v)[va] for k,v in enc.items()},y[va])
    model=AutoModelForSequenceClassification.from_pretrained(MODEL,num_labels=2).to(device)
    weights=torch.tensor([1, (len(tr)-y[tr].sum())/y[tr].sum()],dtype=torch.float,device=device); opt=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01)
    best=(-1,None)
    for epoch in range(args.epochs):
        model.train()
        for batch in DataLoader(train,batch_size=16,shuffle=True):
            labels=batch.pop('labels').to(device); batch={k:v.to(device) for k,v in batch.items()}; out=model(**batch)
            loss=torch.nn.functional.cross_entropy(out.logits,labels,weight=weights); loss.backward(); opt.step(); opt.zero_grad()
        model.eval(); probs=[]
        with torch.no_grad():
            for batch in DataLoader(valid,batch_size=16):
                batch.pop('labels'); batch={k:v.to(device) for k,v in batch.items()}; probs.extend(torch.softmax(model(**batch).logits,1)[:,1].cpu().numpy())
        pr=average_precision_score(y[va],probs); t=threshold(y[va],np.array(probs)); pred=np.array(probs)>=t; _,rec,f1,_=precision_recall_fscore_support(y[va],pred,average='binary',zero_division=0); print(f'epoch={epoch+1} val_pr_auc={pr:.3f} recall={rec:.3f} f1={f1:.3f}')
        if f1>best[0]: best=(f1,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
    model.load_state_dict(best[1]); args.out.mkdir(parents=True,exist_ok=True); model.save_pretrained(args.out); tok.save_pretrained(args.out)
    model.eval(); probs=[]
    with torch.no_grad():
      for batch in DataLoader(valid,batch_size=16):
        batch.pop('labels'); batch={k:v.to(device) for k,v in batch.items()}; probs.extend(torch.softmax(model(**batch).logits,1)[:,1].cpu().numpy())
    p=np.array(probs); t=threshold(y[va],p); prec,rec,f1,_=precision_recall_fscore_support(y[va],p>=t,average='binary',zero_division=0)
    (root/'models/m2_threshold.json').write_text(json.dumps({'threshold':t},indent=2)); (root/'reports').mkdir(exist_ok=True); (root/'reports/m2_val.md').write_text(f'# M2 validation\n\nPrecision {prec:.3f}; recall {rec:.3f}; F1 {f1:.3f}; PR-AUC {average_precision_score(y[va],p):.3f}; threshold {t:.4f}.\n')
if __name__=='__main__': main()
