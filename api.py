"""
api.py — Splice Variant Pathogenicity + Splice Site + Cryptic Site Predictor
"""

import os, time, sys, json, math
from itertools import product
from typing import Optional, List
from contextlib import asynccontextmanager
from pathlib import Path

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"]  = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── CONFIG ─────────────────────────────────────────────────────────────────────
XGB_MODEL_PATH  = "xgb_model.json"
XGB_THRESH_PATH = "xgb_threshold.json"
DNABERT_PATH    = "dnabert2_splice.pt"
DNABERT_NAME    = "zhihan1996/DNABERT-2-117M"
WINDOW          = 200
K               = 6
VOCAB_SIZE      = 4 ** K
ENSEMBLE_XGB_W  = 0.4
ENSEMBLE_DNA_W  = 0.6
DEFAULT_THRESH  = 0.5

BASES       = ["A", "T", "G", "C"]
VOCAB       = ["".join(p) for p in product(BASES, repeat=K)]
VOCAB_INDEX = {kmer: i for i, kmer in enumerate(VOCAB)}
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models: dict = {}


# ════════════════════════════════════════════════════════════════════
#  SPLICE SITE SCORING  (PWM-based, no external deps)
# ════════════════════════════════════════════════════════════════════

# Human splice site PWM (log-odds, derived from Yeo & Burge 2004)
# Donor: 9-mer  exon[−3:0] | intron[0:6]   → positions 0..8, GT at [3:5]
# Acceptor: 23-mer  intron[−20:0] | exon[0:3]  → AG at [−3:−1]

_DONOR_PWM = {
    # pos: {base: score}  (3 exon + 6 intron positions)
    0: {'A':0.36,'C':0.36,'G':0.47,'T':-0.98},
    1: {'A':0.05,'C':-0.44,'G':0.30,'T':0.10},
    2: {'A':0.52,'C':-0.54,'G':0.18,'T':-0.06},
    3: {'A':-2.0,'C':-2.0,'G':1.5, 'T':-2.0},  # must be G
    4: {'A':-2.0,'C':-2.0,'G':-2.0,'T':1.5},   # must be T
    5: {'A':0.70,'C':-1.8,'G':0.14,'T':-0.70},
    6: {'A':0.29,'C':-0.69,'G':0.32,'T':-0.12},
    7: {'A':0.42,'C':-0.97,'G':0.30,'T':-0.16},
    8: {'A':0.35,'C':-0.43,'G':0.39,'T':-0.29},
}

_ACCEPTOR_PWM = {
    # Simplified 14-mer: polypyrimidine tract [-14:-2] + AG [-2:-0]
    # Only scoring the critical last 4 positions here
    0:  {'A':-0.6,'C':0.5,'G':-0.6,'T':0.5},
    1:  {'A':-0.6,'C':0.5,'G':-0.6,'T':0.5},
    2:  {'A':-0.6,'C':0.5,'G':-0.6,'T':0.5},
    3:  {'A':-0.6,'C':0.5,'G':-0.6,'T':0.5},
    4:  {'A':-0.6,'C':0.5,'G':-0.6,'T':0.5},
    5:  {'A':-0.6,'C':0.5,'G':-0.6,'T':0.5},
    6:  {'A':-0.6,'C':0.5,'G':-0.6,'T':0.5},
    7:  {'A':-0.6,'C':0.5,'G':-0.6,'T':0.5},
    8:  {'A':-0.6,'C':0.5,'G':-0.6,'T':0.5},
    9:  {'A':-0.6,'C':0.5,'G':-0.6,'T':0.5},
    10: {'A':-0.6,'C':0.5,'G':-0.6,'T':0.5},
    11: {'A':-0.6,'C':0.5,'G':-0.6,'T':0.5},
    12: {'A':1.5, 'C':-2.0,'G':-2.0,'T':-2.0},  # must be A
    13: {'A':-2.0,'C':-2.0,'G':1.5, 'T':-2.0},  # must be G
}


def _pwm_score(seq: str, pwm: dict) -> float:
    seq = seq.upper()
    n   = len(pwm)
    if len(seq) < n:
        return -99.0
    score = 0.0
    for i, base in enumerate(seq[:n]):
        score += pwm[i].get(base, -2.0)
    return score


def _pwm_to_prob(raw: float, pwm_len: int) -> float:
    """Normalise raw PWM score to 0-1 probability using logistic."""
    # empirical midpoint ≈ 0 for random sequence
    return 1.0 / (1.0 + math.exp(-raw / (pwm_len * 0.3)))


def scan_sites(seq: str, site_type: str) -> list:
    """
    Scan a sequence for donor or acceptor sites.
    Returns list of (position, raw_score, prob, kmer).
    """
    seq = seq.upper()
    if site_type == "donor":
        pwm  = _DONOR_PWM
        wlen = 9
        motif_check = lambda s: s[3:5] in ("GT", "GC")
    else:
        pwm  = _ACCEPTOR_PWM
        wlen = 14
        motif_check = lambda s: s[12:14] == "AG"

    results = []
    for i in range(len(seq) - wlen + 1):
        kmer = seq[i:i+wlen]
        if not motif_check(kmer):
            continue
        raw  = _pwm_score(kmer, pwm)
        prob = _pwm_to_prob(raw, wlen)
        if prob > 0.2:  # threshold to reduce noise
            results.append({"pos": i, "raw": raw, "prob": round(prob, 4), "kmer": kmer})
    return results


def analyse_splice_sites(ref_seq: str, alt_seq: str):
    """
    Compare donor/acceptor sites between REF and ALT.
    Returns annotated list with disrupted/created/cryptic flags.
    """
    ref_seq = ref_seq.upper()
    alt_seq = alt_seq.upper()
    center  = len(ref_seq) // 2

    output = []

    for site_type in ("donor", "acceptor"):
        ref_sites = {s["pos"]: s for s in scan_sites(ref_seq, site_type)}
        alt_sites = {s["pos"]: s for s in scan_sites(alt_seq, site_type)}
        all_pos   = set(ref_sites) | set(alt_sites)

        for pos in sorted(all_pos):
            rs = ref_sites.get(pos)
            as_ = alt_sites.get(pos)
            ref_p = rs["prob"]  if rs  else 0.0
            alt_p = as_["prob"] if as_ else 0.0
            delta = round(alt_p - ref_p, 4)
            rel   = pos - center  # position relative to variant

            # Classify
            disrupted = ref_p >= 0.5 and alt_p < 0.5
            created   = ref_p < 0.3  and alt_p >= 0.5
            # Cryptic: pre-existing weak site strengthened by variant
            cryptic   = (not created) and (not disrupted) and delta > 0.15 and alt_p >= 0.4

            if abs(delta) < 0.05 and not disrupted and not created and not cryptic:
                continue

            output.append({
                "type"      : site_type,
                "position"  : rel,
                "ref_kmer"  : rs["kmer"]  if rs  else "",
                "alt_kmer"  : as_["kmer"] if as_ else "",
                "ref_score" : ref_p,
                "alt_score" : alt_p,
                "delta"     : delta,
                "disrupted" : disrupted,
                "created"   : created,
                "cryptic"   : cryptic,
            })

    return sorted(output, key=lambda x: abs(x["delta"]), reverse=True)[:25]


# ════════════════════════════════════════════════════════════════════
#  ML FEATURE HELPERS
# ════════════════════════════════════════════════════════════════════

def kmer_freq_vector(seq: str) -> np.ndarray:
    seq = seq.upper()
    vec = np.zeros(VOCAB_SIZE, dtype=np.float32)
    n   = len(seq) - K + 1
    if n <= 0: return vec
    for i in range(n):
        kmer = seq[i:i+K]
        if kmer in VOCAB_INDEX: vec[VOCAB_INDEX[kmer]] += 1
    vec /= (n + 1e-10)
    return vec

def variant_type(ref: str, alt: str) -> int:
    if len(ref)==1 and len(alt)==1: return 0
    elif len(ref)<len(alt):         return 1
    elif len(ref)>len(alt):         return 2
    return 3

def biological_features(ref, alt, ref_seq) -> np.ndarray:
    r,a   = ref.upper(), alt.upper()
    seq   = ref_seq.upper() if ref_seq else ""
    f     = np.zeros(9, dtype=np.float32)
    if len(r)==1 and len(a)==1:
        f[0]=float(r=="G" and a!="G"); f[1]=float(r=="T" and a!="T")
        f[2]=float(r=="A" and a!="A"); f[3]=float(r=="G" and a!="G")
    f[4]=float("GT" in r); f[5]=float("AG" in r)
    f[6]=float("GT" in r and "GT" not in a)
    f[7]=float("AG" in r and "AG" not in a)
    if seq: f[8]=(seq.count("G")+seq.count("C"))/max(len(seq),1)
    return f

def apply_mutation(ref_window, center, ref, alt):
    return ref_window[:center] + alt + ref_window[center+len(ref):]

def build_feature_vector(ref, alt, ref_seq, alt_seq) -> np.ndarray:
    rv = kmer_freq_vector(ref_seq) if ref_seq else np.zeros(VOCAB_SIZE, np.float32)
    av = kmer_freq_vector(alt_seq) if alt_seq else np.zeros(VOCAB_SIZE, np.float32)
    d  = av - rv; mag = np.linalg.norm(d); direction = d/(mag+1e-10)
    return np.concatenate([rv, av, direction, np.array([mag],np.float32),
                           np.array([variant_type(ref,alt),len(ref),len(alt)],np.float32),
                           biological_features(ref,alt,ref_seq or "")])

def named_dmatrix(feat: np.ndarray) -> xgb.DMatrix:
    cols = [f"f_{i}" for i in range(len(feat))]
    return xgb.DMatrix(pd.DataFrame(feat.reshape(1,-1), columns=cols))


# ════════════════════════════════════════════════════════════════════
#  SCHEMAS
# ════════════════════════════════════════════════════════════════════

class VariantRequest(BaseModel):
    chrom    : str
    position : int
    ref      : str
    alt      : str
    ref_seq  : Optional[str] = None
    alt_seq  : Optional[str] = None

    def get_alt_seq(self):
        if self.alt_seq: return self.alt_seq
        if self.ref_seq:
            c = len(self.ref_seq)//2
            return apply_mutation(self.ref_seq, c, self.ref, self.alt)
        return None

class PredictionResponse(BaseModel):
    chrom: str; position: int; ref: str; alt: str
    model: str; probability: float; prediction: str
    confidence: str; latency_ms: float; threshold_used: float

class SiteDetail(BaseModel):
    type: str; position: int
    ref_kmer: str; alt_kmer: str
    ref_score: float; alt_score: float
    delta: float
    disrupted: bool; created: bool; cryptic: bool

class SpliceAnalysisResponse(BaseModel):
    chrom: str; position: int; ref: str; alt: str
    sites_found: int
    disrupted_donors: int; disrupted_acceptors: int
    created_donors: int;   created_acceptors: int
    cryptic_donors: int;   cryptic_acceptors: int
    max_disruption: float
    pathogenicity_signal: str
    summary: str
    sites: List[SiteDetail]
    latency_ms: float


# ════════════════════════════════════════════════════════════════════
#  DNABERT LOADER
# ════════════════════════════════════════════════════════════════════

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
    self.register_buffer("alibi", alibi.unsqueeze(1), persistent=False)


class DNABertClassifier(nn.Module):
    """
    Wraps DNABERT-2 base model.
    The base returns a TUPLE (sequence_output, pooled_output, ...).
    We use pooled_output (index 1) when available, else CLS token.
    """
    def __init__(self, base, hidden_size: int):
        super().__init__()
        self.base       = base
        self.drop       = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        out = self.base(input_ids=input_ids, attention_mask=attention_mask)
        # out is a tuple: (last_hidden_state, pooled_output, ...)  OR just (last_hidden_state,)
        if isinstance(out, (tuple, list)):
            if len(out) >= 2 and out[1] is not None and out[1].dim() == 2:
                pooled = out[1]   # pooler output [B, H]
            else:
                pooled = out[0][:, 0]  # CLS token
        else:
            # ModelOutput object
            pooled = out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None \
                     else out.last_hidden_state[:, 0]
        return self.classifier(self.drop(pooled))


def _load_dnabert():
    import importlib.util as ilu
    from transformers import AutoTokenizer, AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from transformers.models.bert.configuration_bert import BertConfig as StdCfg

    # Patch all cached remote BertConfig → standard one
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

    tokenizer = AutoTokenizer.from_pretrained(DNABERT_NAME, trust_remote_code=True)
    config    = AutoConfig.from_pretrained(DNABERT_NAME, trust_remote_code=True)

    # Try known class names
    base = None
    for candidate in ["bert_layers.BertModel", "modeling_bert.BertModel"]:
        try:
            cls = get_class_from_dynamic_module(candidate, DNABERT_NAME)
            cls.config_class = StdCfg
            base = cls.from_pretrained(DNABERT_NAME, config=config, trust_remote_code=True)
            print(f"  Loaded via {candidate}")
            break
        except Exception:
            continue

    if base is None:
        raise RuntimeError("Could not load DNABERT-2 base model")

    # Patch alibi on all bert_layers modules
    for mn, mod in sys.modules.items():
        if "bert_layers" in mn and hasattr(mod, "BertEncoder"):
            mod.BertEncoder.rebuild_alibi_tensor = _safe_alibi
            print("  Alibi patched")

    hidden = config.hidden_size
    clf = DNABertClassifier(base, hidden).to(DEVICE)
    state = torch.load(DNABERT_PATH, map_location=DEVICE, weights_only=False)
    clf.load_state_dict(state)
    clf.eval()
    return clf, tokenizer


# ════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════

def _confidence(prob, thresh):
    dist = abs(prob - thresh)
    if dist >= 0.35: return "High"
    elif dist >= 0.15: return "Medium"
    return "Low"

def _load_threshold():
    if os.path.exists(XGB_THRESH_PATH):
        return json.load(open(XGB_THRESH_PATH))["threshold"]
    return DEFAULT_THRESH

def _dna_prob(seq: str) -> float:
    enc = models["tokenizer"](seq, max_length=128, padding="max_length",
                               truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = models["dnabert"](enc["input_ids"].to(DEVICE),
                                   enc["attention_mask"].to(DEVICE))
    return float(torch.softmax(logits, -1)[0, 1].cpu())


# ════════════════════════════════════════════════════════════════════
#  LIFESPAN
# ════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.path.exists(XGB_MODEL_PATH):
        print("Loading XGBoost...")
        m = xgb.Booster(); m.load_model(XGB_MODEL_PATH)
        models["xgb"]        = m
        models["xgb_thresh"] = _load_threshold()
        print(f"  XGBoost ready (threshold={models['xgb_thresh']:.3f})")
    else:
        print(f"WARNING: {XGB_MODEL_PATH} not found")

    if os.path.exists(DNABERT_PATH):
        try:
            print("Loading DNABERT-2...")
            clf, tok = _load_dnabert()
            models["dnabert"]   = clf
            models["tokenizer"] = tok
            print(f"  DNABERT-2 ready on {DEVICE}")
        except Exception as e:
            print(f"  WARNING: DNABERT-2 failed: {e}")
    else:
        print(f"WARNING: {DNABERT_PATH} not found")

    yield
    models.clear()


# ════════════════════════════════════════════════════════════════════
#  APP
# ════════════════════════════════════════════════════════════════════

app = FastAPI(title="Splice Variant Classifier v3", version="3.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


@app.post("/predict/xgb", response_model=PredictionResponse)
async def predict_xgb(req: VariantRequest):
    if "xgb" not in models: raise HTTPException(503, "XGBoost not loaded")
    t0     = time.perf_counter()
    feat   = build_feature_vector(req.ref, req.alt, req.ref_seq, req.get_alt_seq())
    prob   = float(models["xgb"].predict(named_dmatrix(feat))[0])
    thresh = models.get("xgb_thresh", DEFAULT_THRESH)
    return PredictionResponse(
        chrom=req.chrom, position=req.position, ref=req.ref, alt=req.alt,
        model="XGBoost", probability=round(prob,6),
        prediction="Pathogenic" if prob>=thresh else "Benign",
        confidence=_confidence(prob,thresh), threshold_used=thresh,
        latency_ms=round((time.perf_counter()-t0)*1000,2),
    )


@app.post("/predict/dnabert", response_model=PredictionResponse)
async def predict_dnabert(req: VariantRequest):
    if "dnabert" not in models: raise HTTPException(503, "DNABERT-2 not loaded")
    t0   = time.perf_counter()
    seq  = req.get_alt_seq() or req.ref_seq or ("N"*WINDOW)
    prob = _dna_prob(seq)
    return PredictionResponse(
        chrom=req.chrom, position=req.position, ref=req.ref, alt=req.alt,
        model="DNABERT-2", probability=round(prob,6),
        prediction="Pathogenic" if prob>=0.5 else "Benign",
        confidence=_confidence(prob,0.5), threshold_used=0.5,
        latency_ms=round((time.perf_counter()-t0)*1000,2),
    )


@app.post("/predict/ensemble", response_model=PredictionResponse)
async def predict_ensemble(req: VariantRequest):
    if "xgb" not in models and "dnabert" not in models:
        raise HTTPException(503, "No models loaded")
    t0 = time.perf_counter(); probs=[]; weights=[]
    thresh = models.get("xgb_thresh", DEFAULT_THRESH)

    if "xgb" in models:
        feat = build_feature_vector(req.ref, req.alt, req.ref_seq, req.get_alt_seq())
        probs.append(float(models["xgb"].predict(named_dmatrix(feat))[0]))
        weights.append(ENSEMBLE_XGB_W)
    if "dnabert" in models:
        seq  = req.get_alt_seq() or req.ref_seq or ("N"*WINDOW)
        probs.append(_dna_prob(seq))
        weights.append(ENSEMBLE_DNA_W)

    w    = np.array(weights)/sum(weights)
    prob = float(np.dot(probs, w))
    # Ensemble threshold: weighted average of individual thresholds
    t_ens = thresh*(ENSEMBLE_XGB_W/sum(weights)) + 0.5*(ENSEMBLE_DNA_W/sum(weights)) \
            if len(probs) > 1 else thresh
    return PredictionResponse(
        chrom=req.chrom, position=req.position, ref=req.ref, alt=req.alt,
        model="Ensemble (XGB+DNABERT-2)", probability=round(prob,6),
        prediction="Pathogenic" if prob>=t_ens else "Benign",
        confidence=_confidence(prob, t_ens), threshold_used=round(t_ens,4),
        latency_ms=round((time.perf_counter()-t0)*1000,2),
    )


@app.post("/predict/splice_sites", response_model=SpliceAnalysisResponse)
async def predict_splice_sites(req: VariantRequest):
    if not req.ref_seq:
        raise HTTPException(400, "ref_seq required for splice site analysis")
    t0      = time.perf_counter()
    alt_seq = req.get_alt_seq() or ""
    if not alt_seq:
        c = len(req.ref_seq)//2
        alt_seq = apply_mutation(req.ref_seq, c, req.ref, req.alt)

    sites = analyse_splice_sites(req.ref_seq, alt_seq)

    dis_don = sum(1 for s in sites if s["disrupted"] and s["type"]=="donor")
    dis_acc = sum(1 for s in sites if s["disrupted"] and s["type"]=="acceptor")
    cre_don = sum(1 for s in sites if s["created"]   and s["type"]=="donor")
    cre_acc = sum(1 for s in sites if s["created"]   and s["type"]=="acceptor")
    cry_don = sum(1 for s in sites if s["cryptic"]   and s["type"]=="donor")
    cry_acc = sum(1 for s in sites if s["cryptic"]   and s["type"]=="acceptor")
    max_dis = max((abs(s["delta"]) for s in sites), default=0.0)

    total_bad = dis_don + dis_acc + cre_don + cre_acc + cry_don + cry_acc

    if dis_don+dis_acc >= 1 and (cre_don+cre_acc+cry_don+cry_acc) >= 1:
        signal  = "Strong pathogenic — canonical site lost + cryptic/new site gained"
    elif dis_don+dis_acc >= 2:
        signal  = "Strong pathogenic — multiple canonical sites disrupted"
    elif dis_don+dis_acc == 1:
        signal  = "Moderate — canonical splice site disrupted"
    elif cre_don+cre_acc >= 1:
        signal  = "Moderate — new splice site created (exon skipping risk)"
    elif cry_don+cry_acc >= 1:
        signal  = "Weak — cryptic splice site activated"
    elif max_dis > 0.1:
        signal  = "Low — minor splicing score change"
    else:
        signal  = "Benign — no significant splicing impact"

    parts = []
    if dis_don: parts.append(f"{dis_don} donor(s) disrupted")
    if dis_acc: parts.append(f"{dis_acc} acceptor(s) disrupted")
    if cre_don: parts.append(f"{cre_don} new donor(s) created")
    if cre_acc: parts.append(f"{cre_acc} new acceptor(s) created")
    if cry_don: parts.append(f"{cry_don} cryptic donor(s) activated")
    if cry_acc: parts.append(f"{cry_acc} cryptic acceptor(s) activated")
    summary = "; ".join(parts) if parts else "No splice site changes detected"

    return SpliceAnalysisResponse(
        chrom=req.chrom, position=req.position, ref=req.ref, alt=req.alt,
        sites_found=len(sites),
        disrupted_donors=dis_don, disrupted_acceptors=dis_acc,
        created_donors=cre_don,   created_acceptors=cre_acc,
        cryptic_donors=cry_don,   cryptic_acceptors=cry_acc,
        max_disruption=round(max_dis,4),
        pathogenicity_signal=signal, summary=summary,
        sites=[SiteDetail(**s) for s in sites],
        latency_ms=round((time.perf_counter()-t0)*1000,2),
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "xgb": "xgb" in models,
        "dnabert": "dnabert" in models,
        "xgb_threshold": models.get("xgb_thresh", DEFAULT_THRESH),
        "device": str(DEVICE),
    }

@app.get("/")
async def root():
    return {"endpoints": ["/predict/xgb", "/predict/dnabert", "/predict/ensemble",
                          "/predict/splice_sites", "/health", "/docs"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)