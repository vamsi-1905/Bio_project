"""
api.py — Splice Variant Pathogenicity + Splice Site + Cryptic Site Predictor
FIXES:
  - DNABERT inference now uses ref_seq + [SEP] + alt_seq to match training format
  - XGB sample weights via DMatrix weight param to handle benign underrepresentation
  - Splice analysis returns full cryptic site details
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
XGB_MODEL_PATH    = "xgb_model.json"
XGB_THRESH_PATH   = "xgb_threshold.json"
DNA_THRESH_PATH   = "dnabert2_threshold.json"   # FIX: load tuned threshold, not hardcoded 0.5
DNABERT_PATH      = "dnabert2_splice.pt"
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
#  SPLICE SITE SCORING (PWM-based)
# ════════════════════════════════════════════════════════════════════

_DONOR_PWM = {
    0: {'A':0.36,'C':0.36,'G':0.47,'T':-0.98},
    1: {'A':0.05,'C':-0.44,'G':0.30,'T':0.10},
    2: {'A':0.52,'C':-0.54,'G':0.18,'T':-0.06},
    3: {'A':-2.0,'C':-2.0,'G':1.5, 'T':-2.0},
    4: {'A':-2.0,'C':-2.0,'G':-2.0,'T':1.5},
    5: {'A':0.70,'C':-1.8,'G':0.14,'T':-0.70},
    6: {'A':0.29,'C':-0.69,'G':0.32,'T':-0.12},
    7: {'A':0.42,'C':-0.97,'G':0.30,'T':-0.16},
    8: {'A':0.35,'C':-0.43,'G':0.39,'T':-0.29},
}

_ACCEPTOR_PWM = {
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
    12: {'A':1.5, 'C':-2.0,'G':-2.0,'T':-2.0},
    13: {'A':-2.0,'C':-2.0,'G':1.5, 'T':-2.0},
}


def _pwm_score(seq: str, pwm: dict) -> float:
    seq = seq.upper()
    n   = len(pwm)
    if len(seq) < n:
        return -99.0
    return sum(pwm[i].get(seq[i], -2.0) for i in range(n))


def _pwm_to_prob(raw: float, pwm_len: int) -> float:
    return 1.0 / (1.0 + math.exp(-raw / (pwm_len * 0.3)))


def scan_sites(seq: str, site_type: str) -> list:
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
        if prob > 0.2:
            results.append({"pos": i, "raw": raw, "prob": round(prob, 4), "kmer": kmer})
    return results


def _classify_mutation(ref: str, alt: str) -> dict:
    ref, alt = ref.upper(), alt.upper()
    r_len, a_len = len(ref), len(alt)

    if r_len == 1 and a_len == 1:
        mut_type = "SNV"
        if ref == "G":
            canon = "+1G (invariant donor)"
            mechanism = "Disrupts the invariant +1G of GT donor dinucleotide. Nearly always pathogenic — this position is >99% conserved."
        elif ref == "T":
            canon = "+2T (donor)"
            mechanism = "Disrupts the +2T of GT donor dinucleotide. Highly conserved; most substitutions abolish splicing."
        elif ref == "A":
            canon = "-2A (acceptor)"
            mechanism = "Disrupts the -2A of AG acceptor dinucleotide. High conservation; loss typically causes exon skipping."
        elif ref == "C" and alt in ("T","G","A"):
            canon = "non-canonical position"
            mechanism = f"C→{alt} substitution at non-invariant position. Moderate splice region effect — may weaken ESE binding."
        else:
            canon = "splice region"
            mechanism = f"{ref}→{alt} substitution in splice region. Effect depends on local sequence context."
        detail = f"Single nucleotide variant {ref}→{alt} at {canon}"
    elif r_len > a_len:
        mut_type = "Deletion"
        del_len  = r_len - a_len
        mechanism = f"{del_len}bp deletion. Frame{'shift' if del_len%3!=0 else '-preserving'} deletion — likely disrupts splice site geometry."
        detail = f"{del_len}bp deletion ({ref}→{alt})"
    elif r_len < a_len:
        mut_type = "Insertion"
        ins_len  = a_len - r_len
        mechanism = f"{ins_len}bp insertion. Frame{'shift' if ins_len%3!=0 else '-preserving'} insertion — may create or destroy splice signals."
        detail = f"{ins_len}bp insertion ({ref}→{alt})"
    else:
        mut_type = "MNV"
        mechanism = f"Multi-nucleotide variant ({r_len}bp). Complex substitution — multiple splice positions potentially affected."
        detail = f"{r_len}bp complex substitution ({ref}→{alt})"

    return {"mutation_type": mut_type, "detail": detail, "mechanism": mechanism}


def analyse_splice_sites(ref_seq: str, alt_seq: str, ref: str, alt: str):
    ref_seq = ref_seq.upper()
    alt_seq = alt_seq.upper()
    center  = len(ref_seq) // 2

    mut_info = _classify_mutation(ref, alt)
    output   = []

    for site_type in ("donor", "acceptor"):
        ref_sites = {s["pos"]: s for s in scan_sites(ref_seq, site_type)}
        alt_sites = {s["pos"]: s for s in scan_sites(alt_seq, site_type)}
        all_pos   = set(ref_sites) | set(alt_sites)

        for pos in sorted(all_pos):
            rs  = ref_sites.get(pos)
            as_ = alt_sites.get(pos)
            ref_p = rs["prob"]  if rs  else 0.0
            alt_p = as_["prob"] if as_ else 0.0
            delta = round(alt_p - ref_p, 4)
            rel   = pos - center

            disrupted = ref_p >= 0.5 and alt_p < 0.5
            created   = ref_p < 0.3  and alt_p >= 0.5
            # FIX: lower cryptic threshold slightly so we catch more
            cryptic   = (not created) and (not disrupted) and delta > 0.10 and alt_p >= 0.35

            if abs(delta) < 0.05 and not disrupted and not created and not cryptic:
                continue

            pos_label = f"+{rel}" if rel >= 0 else str(rel)
            if disrupted:
                reasoning = (
                    f"Canonical {site_type} site at position {pos_label} "
                    f"loses score from {ref_p:.2f} → {alt_p:.2f}. "
                    f"{mut_info['mechanism']} "
                    f"Predicted consequence: exon skipping or intron retention."
                )
            elif created:
                reasoning = (
                    f"New {site_type} site emerges at position {pos_label} "
                    f"(score {ref_p:.2f} → {alt_p:.2f}). "
                    f"Competing splice site may redirect splicing machinery, "
                    f"causing partial exon inclusion or novel isoform."
                )
            elif cryptic:
                reasoning = (
                    f"Pre-existing weak {site_type} site at {pos_label} "
                    f"is strengthened ({ref_p:.2f} → {alt_p:.2f}). "
                    f"Cryptic site activation — {mut_info['mechanism']} "
                    f"May cause aberrant splicing in a fraction of transcripts."
                )
            else:
                chg = "weakened" if delta < 0 else "strengthened"
                reasoning = (
                    f"{site_type.capitalize()} site at {pos_label} {chg} "
                    f"({ref_p:.2f} → {alt_p:.2f}). "
                    f"Subthreshold change — monitor in combination with other variants."
                )

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
                "reasoning" : reasoning,
            })

    return (
        sorted(output, key=lambda x: abs(x["delta"]), reverse=True)[:30],
        mut_info
    )


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
    mutation_type: str; mechanism: str

class SiteDetail(BaseModel):
    type: str; position: int
    ref_kmer: str; alt_kmer: str
    ref_score: float; alt_score: float
    delta: float
    disrupted: bool; created: bool; cryptic: bool
    reasoning: str

class SpliceAnalysisResponse(BaseModel):
    chrom: str; position: int; ref: str; alt: str
    mutation_type: str; mutation_detail: str; mutation_mechanism: str
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


def _load_dnabert():
    import importlib.util as ilu
    from transformers import AutoTokenizer, AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    _patch_all()
    tokenizer = AutoTokenizer.from_pretrained(DNABERT_NAME, trust_remote_code=True)
    _patch_all()
    config = AutoConfig.from_pretrained(DNABERT_NAME, trust_remote_code=True)
    _patch_all()

    BertModelCls = get_class_from_dynamic_module("bert_layers.BertModel", DNABERT_NAME)
    base = BertModelCls.from_pretrained(
        DNABERT_NAME, config=config,
        trust_remote_code=True, low_cpu_mem_usage=False
    )
    _patch_all()

    class DNABERTClassifier(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base       = base
            self.drop       = nn.Dropout(0.1)
            self.classifier = nn.Linear(base.config.hidden_size, 2)

        def forward(self, input_ids, attention_mask):
            out = self.base(input_ids=input_ids, attention_mask=attention_mask)
            pooled = out[1] if (len(out) >= 2 and out[1] is not None
                                and out[1].dim() == 2) else out[0][:, 0]
            return self.classifier(self.drop(pooled))

    clf = DNABERTClassifier(base).float().to(DEVICE)
    state = torch.load(DNABERT_PATH, map_location=DEVICE, weights_only=False)
    missing, unexpected = clf.load_state_dict(state, strict=False)
    print(f"  DNABERT weights: missing={len(missing)} unexpected={len(unexpected)}")
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
        t = json.load(open(XGB_THRESH_PATH))["threshold"]
        print(f"  Loaded XGB threshold: {t:.4f}")
        return t
    print(f"  WARNING: {XGB_THRESH_PATH} not found, using {DEFAULT_THRESH}")
    return DEFAULT_THRESH

def _load_dna_threshold():
    if os.path.exists(DNA_THRESH_PATH):
        t = json.load(open(DNA_THRESH_PATH))["threshold"]
        print(f"  Loaded DNABERT threshold: {t:.4f}")
        return t
    print(f"  WARNING: {DNA_THRESH_PATH} not found, using {DEFAULT_THRESH}")
    return DEFAULT_THRESH

def _dna_prob(ref_seq: str, alt_seq: str) -> float:
    """
    FIX: Match training format — model was trained on ref + [SEP] + alt concatenated.
    Passing only alt_seq causes distribution mismatch with training.
    """
    combined = ref_seq + " [SEP] " + alt_seq
    enc = models["tokenizer"](combined, max_length=128, padding="max_length",
                               truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = models["dnabert"](
            enc["input_ids"].to(DEVICE),
            enc["attention_mask"].to(DEVICE)
        )
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
        print(f"  XGBoost ready (threshold={models['xgb_thresh']:.4f})")
    else:
        print(f"WARNING: {XGB_MODEL_PATH} not found")

    if os.path.exists(DNABERT_PATH):
        try:
            print("Loading DNABERT-2...")
            clf, tok = _load_dnabert()
            models["dnabert"]      = clf
            models["tokenizer"]    = tok
            models["dna_thresh"]   = _load_dna_threshold()
            print(f"  DNABERT-2 ready on {DEVICE} (threshold={models['dna_thresh']:.4f})")
        except Exception as e:
            import traceback
            print(f"  WARNING: DNABERT-2 failed: {e}")
            traceback.print_exc()
    else:
        print(f"WARNING: {DNABERT_PATH} not found")

    yield
    models.clear()


# ════════════════════════════════════════════════════════════════════
#  APP
# ════════════════════════════════════════════════════════════════════

app = FastAPI(title="Splice Variant Classifier", version="4.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


def _make_pred_response(req, model_name, prob, thresh):
    mut = _classify_mutation(req.ref, req.alt)
    return PredictionResponse(
        chrom=req.chrom, position=req.position, ref=req.ref, alt=req.alt,
        model=model_name, probability=round(prob, 6),
        prediction="Pathogenic" if prob >= thresh else "Benign",
        confidence=_confidence(prob, thresh), threshold_used=thresh,
        mutation_type=mut["mutation_type"],
        mechanism=mut["mechanism"],
        latency_ms=0.0,
    )


@app.post("/predict/xgb", response_model=PredictionResponse)
async def predict_xgb(req: VariantRequest):
    if "xgb" not in models: raise HTTPException(503, "XGBoost not loaded")
    t0     = time.perf_counter()
    feat   = build_feature_vector(req.ref, req.alt, req.ref_seq, req.get_alt_seq())
    prob   = float(models["xgb"].predict(named_dmatrix(feat))[0])
    thresh = models.get("xgb_thresh", DEFAULT_THRESH)
    r = _make_pred_response(req, "XGBoost", prob, thresh)
    r.latency_ms = round((time.perf_counter()-t0)*1000, 2)
    return r


@app.post("/predict/dnabert", response_model=PredictionResponse)
async def predict_dnabert(req: VariantRequest):
    if "dnabert" not in models: raise HTTPException(503, "DNABERT-2 not loaded")
    t0      = time.perf_counter()
    ref_seq = req.ref_seq or ("N" * WINDOW)
    alt_seq = req.get_alt_seq() or ref_seq
    # FIX: use tuned threshold from val set, not hardcoded 0.5
    thresh = models.get("dna_thresh", DEFAULT_THRESH)
    prob = _dna_prob(ref_seq, alt_seq)
    r = _make_pred_response(req, "DNABERT-2", prob, thresh)
    r.latency_ms = round((time.perf_counter()-t0)*1000, 2)
    return r


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
        ref_seq = req.ref_seq or ("N" * WINDOW)
        alt_seq = req.get_alt_seq() or ref_seq
        # FIX: same here — ref+sep+alt
        probs.append(_dna_prob(ref_seq, alt_seq))
        weights.append(ENSEMBLE_DNA_W)

    w    = np.array(weights)/sum(weights)
    prob = float(np.dot(probs, w))
    # ensemble threshold: weighted average of each model's tuned threshold
    dna_thresh = models.get("dna_thresh", DEFAULT_THRESH)
    t_ens = thresh if len(probs)==1 else (
        thresh*(ENSEMBLE_XGB_W/sum(weights)) + dna_thresh*(ENSEMBLE_DNA_W/sum(weights))
    )
    r = _make_pred_response(req, "Ensemble (XGB+DNABERT-2)", prob, round(t_ens,4))
    r.latency_ms = round((time.perf_counter()-t0)*1000, 2)
    return r


@app.post("/predict/splice_sites", response_model=SpliceAnalysisResponse)
async def predict_splice_sites(req: VariantRequest):
    if not req.ref_seq:
        raise HTTPException(400, "ref_seq required for splice site analysis")
    t0      = time.perf_counter()
    alt_seq = req.get_alt_seq() or ""
    if not alt_seq:
        c = len(req.ref_seq)//2
        alt_seq = apply_mutation(req.ref_seq, c, req.ref, req.alt)

    sites, mut_info = analyse_splice_sites(req.ref_seq, alt_seq, req.ref, req.alt)

    dis_don = sum(1 for s in sites if s["disrupted"] and s["type"]=="donor")
    dis_acc = sum(1 for s in sites if s["disrupted"] and s["type"]=="acceptor")
    cre_don = sum(1 for s in sites if s["created"]   and s["type"]=="donor")
    cre_acc = sum(1 for s in sites if s["created"]   and s["type"]=="acceptor")
    cry_don = sum(1 for s in sites if s["cryptic"]   and s["type"]=="donor")
    cry_acc = sum(1 for s in sites if s["cryptic"]   and s["type"]=="acceptor")
    max_dis = max((abs(s["delta"]) for s in sites), default=0.0)

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
        mutation_type=mut_info["mutation_type"],
        mutation_detail=mut_info["detail"],
        mutation_mechanism=mut_info["mechanism"],
        sites_found=len(sites),
        disrupted_donors=dis_don, disrupted_acceptors=dis_acc,
        created_donors=cre_don,   created_acceptors=cre_acc,
        cryptic_donors=cry_don,   cryptic_acceptors=cry_acc,
        max_disruption=round(max_dis,4),
        pathogenicity_signal=signal, summary=summary,
        sites=[SiteDetail(**s) for s in sites],
        latency_ms=round((time.perf_counter()-t0)*1000, 2),
    )


@app.get("/health")
async def health():
    return {
        "status"        : "ok",
        "xgb"           : "xgb" in models,
        "dnabert"       : "dnabert" in models,
        "xgb_threshold" : models.get("xgb_thresh", DEFAULT_THRESH),
        "dna_threshold" : models.get("dna_thresh", DEFAULT_THRESH),
        "device"        : str(DEVICE),
    }

@app.get("/")
async def root():
    return {"endpoints": ["/predict/xgb", "/predict/dnabert", "/predict/ensemble",
                          "/predict/splice_sites", "/health", "/docs"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)