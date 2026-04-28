"""
dnabert.py — DNABERT-2 fine-tuning
label=0=BENIGN(minority), label=1=PATHOGENIC(majority)
Saves dnabert2_splice.pt + dnabert2_threshold.json
"""
import os, json, math, sys
from pathlib import Path
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

def patch_cache():
    import re
    base = Path.home() / ".cache" / "huggingface"
    NEW_REBUILD = '''\
    def rebuild_alibi_tensor(self, size, device=None):
        # WIN_ALIBI_REWRITE
        import math, torch
        n = self.num_attention_heads
        def get_slopes(n):
            def _p2(n):
                s = 2**(-(2**(-(math.log2(n)-3))))
                return [s * s**i for i in range(n)]
            if math.log2(n).is_integer(): return _p2(n)
            p = 2**math.floor(math.log2(n))
            return _p2(p) + get_slopes(2*p)[::2][:n-p]
        sl  = torch.tensor(get_slopes(n), dtype=torch.float32)
        pos = torch.arange(size, dtype=torch.float32)
        rel = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs().unsqueeze(0)
        alibi = sl.view(-1,1,1) * -rel
        self.register_buffer("alibi", alibi.unsqueeze(1), persistent=False)
'''
    for p in base.rglob("bert_layers.py"):
        text = p.read_text(encoding="utf-8")
        if "WIN_ALIBI_REWRITE" not in text:
            m = re.search(r'    def rebuild_alibi_tensor\(self.*?(?=\n    def |\nclass |\Z)', text, flags=re.DOTALL)
            if m:
                text = text[:m.start()] + NEW_REBUILD + text[m.end():]
        lines = []
        for line in text.splitlines():
            s = line.strip()
            if any(s.startswith(x) for x in ["import triton","from triton","import flash_attn","from flash_attn"]) and not s.startswith("#"):
                lines.append("# WIN_REMOVED: " + line)
            else:
                lines.append(line)
        p.write_text("\n".join(lines), encoding="utf-8")
    for p in base.rglob("flash_attn_triton.py"):
        p.write_text("# WIN_STUB\n", encoding="utf-8")
    for p in base.rglob("bert_padding.py"):
        text = p.read_text(encoding="utf-8")
        if "triton" in text and "WIN_REMOVED" not in text:
            lines = ["# WIN_REMOVED: " + l if "triton" in l and not l.strip().startswith("#") else l for l in text.splitlines()]
            p.write_text("\n".join(lines), encoding="utf-8")

patch_cache()

import importlib.util as _ilu
_dmu_spec = _ilu.find_spec("transformers.dynamic_module_utils")
_dmu_mod  = _ilu.module_from_spec(_dmu_spec)
_dmu_spec.loader.exec_module(_dmu_mod)
_orig_check = _dmu_mod.check_imports
def _patched_check(filename):
    try: return _orig_check(filename)
    except ImportError as e:
        if "triton" in str(e) or "flash_attn" in str(e): return []
        raise
_dmu_mod.check_imports = _patched_check
sys.modules["transformers.dynamic_module_utils"] = _dmu_mod

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoConfig, get_cosine_schedule_with_warmup
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    classification_report, average_precision_score
)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _safe_alibi(self, size, device=None):
    n = self.num_attention_heads
    def gs(n):
        def p2(n):
            s = 2**(-(2**(-(math.log2(n)-3))))
            return [s*s**i for i in range(n)]
        if math.log2(n).is_integer(): return p2(n)
        p = 2**math.floor(math.log2(n))
        return p2(p) + gs(2*p)[::2][:n-p]
    sl  = torch.tensor(gs(n), dtype=torch.float32)
    pos = torch.arange(size, dtype=torch.float32)
    rel = (pos.unsqueeze(0)-pos.unsqueeze(1)).abs().unsqueeze(0)
    alibi = sl.view(-1,1,1)*-rel
    self.register_buffer("alibi",alibi.unsqueeze(1),persistent=False)

def _patch_all():
    import importlib.util as ilu
    from transformers.models.bert.configuration_bert import BertConfig as StdCfg
    cache = Path.home()/".cache"/"huggingface"/"modules"
    for cfg_file in cache.rglob("configuration_bert.py"):
        mname = ".".join(cfg_file.with_suffix("").relative_to(cache).parts)
        if mname not in sys.modules:
            sp = ilu.spec_from_file_location(mname, cfg_file)
            m  = ilu.module_from_spec(sp); sp.loader.exec_module(m)
            m.BertConfig = StdCfg; sys.modules[mname] = m
        else:
            sys.modules[mname].BertConfig = StdCfg
    for mn, mod in list(sys.modules.items()):
        if "transformers_modules" in mn and hasattr(mod, "BertConfig"):
            mod.BertConfig = StdCfg
        if "bert_layers" in mn and hasattr(mod, "BertEncoder"):
            mod.BertEncoder.rebuild_alibi_tensor = _safe_alibi

PARQUET_PATH  = "splice_windows.parquet"
MODEL_NAME    = "zhihan1996/DNABERT-2-117M"
WINDOW        = 200
MAX_LEN       = 128
BATCH_SIZE    = 32
EPOCHS        = 15
LR            = 2e-5
WARMUP_RATIO  = 0.1
WEIGHT_DECAY  = 0.01
GRAD_CLIP     = 1.0
SAVE_PATH     = "dnabert2_splice.pt"
RESULTS_PATH  = "dnabert2_results.json"
THRESH_PATH   = "dnabert2_threshold.json"
SEED          = 42

torch.manual_seed(SEED); np.random.seed(SEED)
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()
print(f"Device: {DEVICE} | AMP: {USE_AMP}")


class SpliceDataset(Dataset):
    def __init__(self, df, tokenizer, window, max_len):
        self.labels = df["label"].values.astype(np.int64)
        rc, ac = f"ref_seq_{window}", f"alt_seq_{window}"
        seqs = []
        for _, row in df.iterrows():
            ref = str(row[rc]).upper() if pd.notna(row.get(rc)) else "N"*window
            alt = str(row[ac]).upper() if pd.notna(row.get(ac)) else ref
            # Training format: ref + [SEP] + alt — must match inference exactly
            seqs.append(ref + " [SEP] " + alt)
        print(f"  Tokenising {len(seqs):,}...")
        self.enc = tokenizer(seqs, max_length=max_len, padding="max_length",
                             truncation=True, return_tensors="pt")
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return {"input_ids":self.enc["input_ids"][i],
                "attention_mask":self.enc["attention_mask"][i],
                "labels":torch.tensor(self.labels[i],dtype=torch.long)}


def load_model(device):
    _patch_all()
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    _patch_all()
    base = None
    for cand in ["bert_layers.BertModel","modeling_bert.BertModel"]:
        try:
            cls = get_class_from_dynamic_module(cand, MODEL_NAME)
            from transformers.models.bert.configuration_bert import BertConfig as StdCfg
            cls.config_class = StdCfg
            base = cls.from_pretrained(MODEL_NAME, config=config,
                                       trust_remote_code=True, low_cpu_mem_usage=False)
            print(f"Loaded base via {cand}")
            break
        except Exception as e:
            print(f"  {cand} failed: {e}")
    if base is None:
        raise RuntimeError("Cannot load BertModel")
    _patch_all()

    class DNABERTClassifier(nn.Module):
        def __init__(self, b):
            super().__init__()
            self.base = b
            self.drop = nn.Dropout(0.1)
            self.classifier = nn.Linear(b.config.hidden_size, 2)
        def forward(self, input_ids, attention_mask):
            out = self.base(input_ids=input_ids, attention_mask=attention_mask)
            # out is a tuple: (last_hidden_state, pooler_output, ...)
            pooled = out[1] if (len(out)>=2 and out[1] is not None and out[1].dim()==2) else out[0][:,0]
            return self.classifier(self.drop(pooled))

    return DNABERTClassifier(base).float().to(device)


def tune_threshold(probs, labels):
    best_t, best = 0.5, 0.0
    for t in np.arange(0.02, 0.98, 0.02):
        p = (probs>=t).astype(int)
        f = (f1_score(labels,p,pos_label=0,zero_division=0)+
             f1_score(labels,p,pos_label=1,zero_division=0))/2
        if f>best: best,best_t=f,float(t)
    return best_t, best


def train_epoch(model, loader, optimizer, scheduler, scaler, criterion):
    model.train(); total=0.0
    for batch in loader:
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        lbl  = batch["labels"].to(DEVICE)
        optimizer.zero_grad()
        if USE_AMP:
            with torch.amp.autocast(device_type="cuda"):
                loss = criterion(model(input_ids=ids,attention_mask=mask), lbl)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer); scaler.update()
        else:
            loss = criterion(model(input_ids=ids,attention_mask=mask), lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
        scheduler.step(); total += loss.item()
    return total/len(loader)


@torch.no_grad()
def evaluate(model, loader, threshold=0.5):
    model.eval(); all_p,all_l=[],[]
    for batch in loader:
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        if USE_AMP:
            with torch.amp.autocast(device_type="cuda"):
                logits = model(input_ids=ids,attention_mask=mask)
        else:
            logits = model(input_ids=ids,attention_mask=mask)
        all_p.extend(torch.softmax(logits,-1)[:,1].cpu().numpy())
        all_l.extend(batch["labels"].numpy())
    p,l = np.array(all_p),np.array(all_l)
    pr  = (p>=threshold).astype(int)
    return {"accuracy":accuracy_score(l,pr),"auc":roc_auc_score(l,p),
            "pr_auc":average_precision_score(l,p),
            "f1_path":f1_score(l,pr,pos_label=1,zero_division=0),
            "f1_ben":f1_score(l,pr,pos_label=0,zero_division=0),
            "probs":p,"labels":l}


def run():
    df = pd.read_parquet(PARQUET_PATH)
    window = WINDOW
    if f"ref_seq_{window}" not in df.columns:
        window = int([c for c in df.columns if c.startswith("ref_seq_")][0].split("_")[-1])

    train_df = df[df["split"]=="train"].reset_index(drop=True)
    val_df   = df[df["split"]=="val"].reset_index(drop=True)
    test_df  = df[df["split"]=="test"].reset_index(drop=True)

    n_ben = (train_df["label"]==0).sum()
    n_pat = (train_df["label"]==1).sum()
    ratio = n_pat / max(n_ben, 1)
    print(f"Benign={n_ben:,}  Pathogenic={n_pat:,}  ratio={ratio:.1f}x")

    # Up-weight benign (minority) in loss
    w_ben = min(ratio, 15.0)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([w_ben, 1.0], dtype=torch.float32, device=DEVICE)
    )
    print(f"Loss weights: benign={w_ben:.1f}  pathogenic=1.0")

    # Oversample benign in each epoch
    sample_weights = np.where(train_df["label"].values==0, float(ratio), 1.0)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=int(2*n_pat), replacement=True
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    _patch_all()

    print("Loading model...")
    model = load_model(DEVICE)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    print("Building datasets...")
    train_ds = SpliceDataset(train_df, tokenizer, window, MAX_LEN)
    val_ds   = SpliceDataset(val_df,   tokenizer, window, MAX_LEN)
    test_ds  = SpliceDataset(test_df,  tokenizer, window, MAX_LEN)

    kw = dict(num_workers=0, pin_memory=USE_AMP)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE*2, shuffle=False,   **kw)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE*2, shuffle=False,   **kw)

    no_decay = ["bias","LayerNorm.weight"]
    optimizer = torch.optim.AdamW([
        {"params":[p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)],"weight_decay":WEIGHT_DECAY},
        {"params":[p for n,p in model.named_parameters() if     any(nd in n for nd in no_decay)],"weight_decay":0.0},
    ], lr=LR)
    total_steps = EPOCHS * len(train_loader)
    scheduler   = get_cosine_schedule_with_warmup(optimizer, int(WARMUP_RATIO*total_steps), total_steps)
    scaler      = torch.amp.GradScaler(device="cuda") if USE_AMP else None

    print(f"\nTraining {EPOCHS} epochs...\n")
    best_auc=0.0; losses=[]; history=[]

    for epoch in range(1, EPOCHS+1):
        tl = train_epoch(model, train_loader, optimizer, scheduler, scaler, criterion)
        vm = evaluate(model, val_loader)
        losses.append(tl); history.append(vm)
        print(f"Ep {epoch:02d}/{EPOCHS}  loss={tl:.4f}  auc={vm['auc']:.4f}  "
              f"pr_auc={vm['pr_auc']:.4f}  f1_path={vm['f1_path']:.4f}  f1_ben={vm['f1_ben']:.4f}")
        if vm["auc"]>best_auc:
            best_auc=vm["auc"]
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  ✓ saved (AUC={best_auc:.4f})")

    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE, weights_only=False))
    vm_best = evaluate(model, val_loader)
    best_t, best_f1 = tune_threshold(vm_best["probs"], vm_best["labels"])
    print(f"\nThreshold: {best_t:.2f}  macro-F1={best_f1:.4f}")
    json.dump({"threshold":best_t,"val_macro_f1":best_f1}, open(THRESH_PATH,"w"), indent=2)

    tm = evaluate(model, test_loader, threshold=best_t)
    print(f"\nTest: AUC={tm['auc']:.4f}  PR-AUC={tm['pr_auc']:.4f}")
    print(classification_report(tm["labels"],(tm["probs"]>=best_t).astype(int),
                                target_names=["Benign","Pathogenic"]))

    json.dump({"test_auc":tm["auc"],"test_pr_auc":tm["pr_auc"],
               "test_f1_benign":tm["f1_ben"],"test_f1_path":tm["f1_path"],
               "best_val_auc":best_auc,"threshold":best_t,
               "model":MODEL_NAME,"window":window,"epochs":EPOCHS},
              open(RESULTS_PATH,"w"), indent=2)

    fig,axes=plt.subplots(1,4,figsize=(18,4))
    ep=range(1,len(losses)+1)
    for ax,vals,title in zip(axes,
        [losses,[m["auc"] for m in history],[m["pr_auc"] for m in history],[m["f1_ben"] for m in history]],
        ["Train Loss","Val AUC","Val PR-AUC","Val F1 Benign"]):
        ax.plot(ep,vals); ax.set_title(title); ax.set_xlabel("Epoch"); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("dnabert2_curves.png",dpi=150); plt.close()
    print("Done.")

if __name__ == "__main__":
    run()