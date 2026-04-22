"""
api.py — Step 5
FastAPI serving XGBoost + DNABERT-2 for splice variant prediction.

Endpoints:
  POST /predict/xgb      → XGBoost prediction (fast, CPU)
  POST /predict/dnabert  → DNABERT-2 prediction (GPU if available)
  POST /predict/ensemble → Weighted ensemble of both
  GET  /health           → Health check

Requirements:
    pip install fastapi uvicorn xgboost transformers torch pandas numpy pyarrow

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import json
import time
import numpy as np
from itertools import product
from typing import Optional
from contextlib import asynccontextmanager

import torch
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── CONFIG ────────────────────────────────────────────────────────────────────
XGB_MODEL_PATH    = "xgb_model.json"
DNABERT_PATH      = "dnabert2_splice.pt"
DNABERT_NAME      = "zhihan1996/DNABERT-2-117M"
WINDOW            = 200
K                 = 6
VOCAB_SIZE        = 4 ** K
ENSEMBLE_XGB_W    = 0.4    # weight for XGB in ensemble
ENSEMBLE_DNA_W    = 0.6    # weight for DNABERT in ensemble
# ─────────────────────────────────────────────────────────────────────────────

# Build k-mer vocab once
BASES       = ["A", "T", "G", "C"]
VOCAB       = ["".join(p) for p in product(BASES, repeat=K)]
VOCAB_INDEX = {kmer: i for i, kmer in enumerate(VOCAB)}
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global model holders
models = {}


# ── Feature extraction (mirrors features.py) ─────────────────────────────────
def kmer_freq_vector(seq: str) -> np.ndarray:
    seq = seq.upper()
    vec = np.zeros(VOCAB_SIZE, dtype=np.float32)
    n   = len(seq) - K + 1
    if n <= 0:
        return vec
    for i in range(n):
        kmer = seq[i:i+K]
        if kmer in VOCAB_INDEX:
            vec[VOCAB_INDEX[kmer]] += 1
    vec /= (n + 1e-10)
    return vec


def variant_type(ref: str, alt: str) -> int:
    if len(ref) == 1 and len(alt) == 1:
        return 0
    elif len(ref) < len(alt):
        return 1
    elif len(ref) > len(alt):
        return 2
    return 3


def biological_features(ref: str, alt: str, ref_seq: str) -> np.ndarray:
    r, a   = ref.upper(), alt.upper()
    seq    = ref_seq.upper() if ref_seq else ""
    feats  = np.zeros(9, dtype=np.float32)

    if len(r) == 1 and len(a) == 1:
        feats[0] = float(r == "G" and a != "G")
        feats[1] = float(r == "T" and a != "T")
        feats[2] = float(r == "A" and a != "A")
        feats[3] = float(r == "G" and a != "G")

    feats[4] = float("GT" in r)
    feats[5] = float("AG" in r)
    feats[6] = float("GT" in r and "GT" not in a)
    feats[7] = float("AG" in r and "AG" not in a)

    if seq:
        feats[8] = (seq.count("G") + seq.count("C")) / max(len(seq), 1)

    return feats


def apply_mutation(ref_window: str, center: int, ref: str, alt: str) -> str:
    return ref_window[:center] + alt + ref_window[center + len(ref):]


def build_feature_vector(ref: str, alt: str,
                          ref_seq: Optional[str],
                          alt_seq: Optional[str]) -> np.ndarray:
    ref_vec = kmer_freq_vector(ref_seq) if ref_seq else np.zeros(VOCAB_SIZE, np.float32)
    alt_vec = kmer_freq_vector(alt_seq) if alt_seq else np.zeros(VOCAB_SIZE, np.float32)

    delta     = alt_vec - ref_vec
    magnitude = np.linalg.norm(delta)
    direction = delta / (magnitude + 1e-10)

    structural = np.array([variant_type(ref, alt), len(ref), len(alt)], np.float32)
    bio        = biological_features(ref, alt, ref_seq or "")
    mag_arr    = np.array([magnitude], np.float32)

    return np.concatenate([ref_vec, alt_vec, direction, mag_arr, structural, bio])


# ── Pydantic request models ───────────────────────────────────────────────────
class VariantRequest(BaseModel):
    chrom    : str         = Field(..., example="1")
    position : int         = Field(..., example=925952)
    ref      : str         = Field(..., example="G")
    alt      : str         = Field(..., example="A")
    ref_seq  : Optional[str] = Field(None, description="Reference window sequence (recommended)")
    alt_seq  : Optional[str] = Field(None, description="Alt window sequence (auto-built if omitted)")

    def resolved_alt_seq(self) -> Optional[str]:
        if self.alt_seq:
            return self.alt_seq
        if self.ref_seq:
            center = len(self.ref_seq) // 2
            return apply_mutation(self.ref_seq, center, self.ref, self.alt)
        return None


class PredictionResponse(BaseModel):
    chrom          : str
    position       : int
    ref            : str
    alt            : str
    model          : str
    probability    : float
    prediction     : str
    confidence     : str
    latency_ms     : float


# ── Lifespan (load models once) ───────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # XGBoost
    if os.path.exists(XGB_MODEL_PATH):
        print(f"Loading XGBoost model from {XGB_MODEL_PATH}...")
        xgb_model = xgb.Booster()
        xgb_model.load_model(XGB_MODEL_PATH)
        models["xgb"] = xgb_model
        print("  XGBoost ready.")
    else:
        print(f"  WARNING: {XGB_MODEL_PATH} not found. /predict/xgb will be unavailable.")

    # DNABERT-2
    if os.path.exists(DNABERT_PATH):
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            print(f"Loading DNABERT-2 from {DNABERT_NAME}...")
            tokenizer = AutoTokenizer.from_pretrained(DNABERT_NAME, trust_remote_code=True)
            dna_model = AutoModelForSequenceClassification.from_pretrained(
                DNABERT_NAME, num_labels=2, trust_remote_code=True
            ).to(DEVICE)
            dna_model.load_state_dict(
                torch.load(DNABERT_PATH, map_location=DEVICE)
            )
            dna_model.eval()
            models["dnabert"] = dna_model
            models["tokenizer"] = tokenizer
            print(f"  DNABERT-2 ready on {DEVICE}.")
        except Exception as e:
            print(f"  WARNING: DNABERT-2 load failed: {e}")
    else:
        print(f"  WARNING: {DNABERT_PATH} not found. /predict/dnabert will be unavailable.")

    yield
    models.clear()


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Splice Variant Classifier",
    description = "XGBoost + DNABERT-2 splice variant pathogenicity prediction",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


def _confidence(prob: float) -> str:
    if prob >= 0.9 or prob <= 0.1:
        return "High"
    elif prob >= 0.75 or prob <= 0.25:
        return "Medium"
    return "Low"


# ── XGBoost endpoint ─────────────────────────────────────────────────────────
@app.post("/predict/xgb", response_model=PredictionResponse)
async def predict_xgb(req: VariantRequest):
    if "xgb" not in models:
        raise HTTPException(503, "XGBoost model not loaded")

    t0 = time.perf_counter()

    feat = build_feature_vector(req.ref, req.alt, req.ref_seq, req.resolved_alt_seq())
    dmat = xgb.DMatrix(feat.reshape(1, -1))
    prob = float(models["xgb"].predict(dmat)[0])

    latency = (time.perf_counter() - t0) * 1000

    return PredictionResponse(
        chrom       = req.chrom,
        position    = req.position,
        ref         = req.ref,
        alt         = req.alt,
        model       = "XGBoost",
        probability = round(prob, 6),
        prediction  = "Pathogenic" if prob >= 0.5 else "Benign",
        confidence  = _confidence(prob),
        latency_ms  = round(latency, 2),
    )


# ── DNABERT-2 endpoint ────────────────────────────────────────────────────────
@app.post("/predict/dnabert", response_model=PredictionResponse)
async def predict_dnabert(req: VariantRequest):
    if "dnabert" not in models:
        raise HTTPException(503, "DNABERT-2 model not loaded")

    t0 = time.perf_counter()

    seq = req.resolved_alt_seq() or req.ref_seq or ("N" * WINDOW)
    tokenizer = models["tokenizer"]
    enc = tokenizer(
        seq, max_length=128, padding="max_length",
        truncation=True, return_tensors="pt"
    )

    with torch.no_grad():
        out   = models["dnabert"](
            input_ids      = enc["input_ids"].to(DEVICE),
            attention_mask = enc["attention_mask"].to(DEVICE),
        )
        prob = float(torch.softmax(out.logits, dim=-1)[0, 1].cpu())

    latency = (time.perf_counter() - t0) * 1000

    return PredictionResponse(
        chrom       = req.chrom,
        position    = req.position,
        ref         = req.ref,
        alt         = req.alt,
        model       = "DNABERT-2",
        probability = round(prob, 6),
        prediction  = "Pathogenic" if prob >= 0.5 else "Benign",
        confidence  = _confidence(prob),
        latency_ms  = round(latency, 2),
    )


# ── Ensemble endpoint ─────────────────────────────────────────────────────────
@app.post("/predict/ensemble", response_model=PredictionResponse)
async def predict_ensemble(req: VariantRequest):
    xgb_available = "xgb"     in models
    dna_available = "dnabert" in models

    if not xgb_available and not dna_available:
        raise HTTPException(503, "No models loaded")

    t0 = time.perf_counter()
    probs, weights = [], []

    if xgb_available:
        feat = build_feature_vector(req.ref, req.alt, req.ref_seq, req.resolved_alt_seq())
        dmat = xgb.DMatrix(feat.reshape(1, -1))
        probs.append(float(models["xgb"].predict(dmat)[0]))
        weights.append(ENSEMBLE_XGB_W)

    if dna_available:
        seq = req.resolved_alt_seq() or req.ref_seq or ("N" * WINDOW)
        tokenizer = models["tokenizer"]
        enc = models["tokenizer"](
            seq, max_length=128, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            out  = models["dnabert"](
                input_ids      = enc["input_ids"].to(DEVICE),
                attention_mask = enc["attention_mask"].to(DEVICE),
            )
            probs.append(float(torch.softmax(out.logits, dim=-1)[0, 1].cpu()))
            weights.append(ENSEMBLE_DNA_W)

    w_arr  = np.array(weights) / sum(weights)
    prob   = float(np.dot(probs, w_arr))
    latency = (time.perf_counter() - t0) * 1000

    return PredictionResponse(
        chrom       = req.chrom,
        position    = req.position,
        ref         = req.ref,
        alt         = req.alt,
        model       = "Ensemble (XGB+DNABERT-2)",
        probability = round(prob, 6),
        prediction  = "Pathogenic" if prob >= 0.5 else "Benign",
        confidence  = _confidence(prob),
        latency_ms  = round(latency, 2),
    )


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status"          : "ok",
        "models_loaded"   : list(k for k in models if k != "tokenizer"),
        "device"          : str(DEVICE),
    }


@app.get("/")
async def root():
    return {
        "message"   : "Splice Variant Classifier API",
        "endpoints" : ["/predict/xgb", "/predict/dnabert",
                       "/predict/ensemble", "/health", "/docs"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)