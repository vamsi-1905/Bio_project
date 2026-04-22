"""
features.py — Step 2
Extracts features from splice_windows.parquet.

Features per variant:
  - 6-mer frequencies from REF and ALT (4096 each)
  - Structural: variant type, ref_len, alt_len
  - Biological: disrupts_donor_GT, disrupts_acceptor_AG, distance proxy
  - Direction + magnitude of delta (from 6-mer space)

Output: features.parquet
    columns: [chrom, position, ref, alt, label, split, f_0 ... f_N]
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product

# ── CONFIG ─────────────────────────────────────────────────────────────────────
IN_PATH  = "splice_windows.parquet"
OUT_PATH = "features.parquet"
K        = 6
WINDOW   = 200
# ──────────────────────────────────────────────────────────────────────────────

BASES       = ["A", "T", "G", "C"]
VOCAB       = ["".join(p) for p in product(BASES, repeat=K)]
VOCAB_INDEX = {kmer: i for i, kmer in enumerate(VOCAB)}
VOCAB_SIZE  = len(VOCAB)   # 4096


def kmer_freq_vector(seq, k=K):
    seq = str(seq).upper()
    vec = np.zeros(VOCAB_SIZE, dtype=np.float32)
    n   = len(seq) - k + 1
    if n <= 0:
        return vec
    for i in range(n):
        kmer = seq[i:i+k]
        if kmer in VOCAB_INDEX:
            vec[VOCAB_INDEX[kmer]] += 1
    vec /= (n + 1e-10)
    return vec


def variant_type(ref, alt):
    r, a = str(ref), str(alt)
    if len(r) == 1 and len(a) == 1:
        return 0
    elif len(r) < len(a):
        return 1
    elif len(r) > len(a):
        return 2
    else:
        return 3


def biological_features(ref, alt, ref_seq):
    r   = str(ref).upper()
    a   = str(alt).upper()
    seq = str(ref_seq).upper() if pd.notna(ref_seq) and ref_seq else ""

    feats = np.zeros(9, dtype=np.float32)

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
        gc = (seq.count("G") + seq.count("C")) / max(len(seq), 1)
        feats[8] = gc

    return feats


def delta_features(ref_vec, alt_vec):
    delta     = alt_vec - ref_vec
    magnitude = np.linalg.norm(delta)
    direction = delta / (magnitude + 1e-10)
    return magnitude, direction


def extract_features(row, window):
    ref     = row["ref"]
    alt     = row["alt"]
    ref_col = f"ref_seq_{window}"
    alt_col = f"alt_seq_{window}"

    # FIX: use pandas Series indexing, not .get()
    ref_seq = row[ref_col] if ref_col in row.index else None
    alt_seq = row[alt_col] if alt_col in row.index else None

    # Treat NaN as None
    if pd.isna(ref_seq):
        ref_seq = None
    if pd.isna(alt_seq):
        alt_seq = None

    ref_vec = kmer_freq_vector(ref_seq) if ref_seq else np.zeros(VOCAB_SIZE, np.float32)
    alt_vec = kmer_freq_vector(alt_seq) if alt_seq else np.zeros(VOCAB_SIZE, np.float32)

    magnitude, direction = delta_features(ref_vec, alt_vec)

    vtype      = variant_type(ref, alt)
    ref_len    = len(str(ref))
    alt_len    = len(str(alt))
    bio        = biological_features(ref, alt, ref_seq)
    structural = np.array([vtype, ref_len, alt_len], dtype=np.float32)
    magnitude  = np.array([magnitude], dtype=np.float32)

    full = np.concatenate([
        ref_vec,    # 4096
        alt_vec,    # 4096
        direction,  # 4096
        magnitude,  # 1
        structural, # 3
        bio         # 9
    ])

    return full


def run():
    print(f"Loading {IN_PATH}...")
    df = pd.read_parquet(IN_PATH)
    print(f"Loaded {len(df):,} variants")
    print(f"Columns: {df.columns.tolist()}")

    # FIX: resolve window at module level, not inside a function with global
    window = WINDOW
    ref_col = f"ref_seq_{window}"
    if ref_col not in df.columns:
        available = [c for c in df.columns if c.startswith("ref_seq_")]
        print(f"Warning: {ref_col} not found. Available: {available}")
        if not available:
            raise ValueError("No ref_seq_* columns found in parquet.")
        window = int(available[0].split("_")[-1])
        print(f"Using window size {window} instead.")

    n_features = VOCAB_SIZE * 3 + 1 + 3 + 9   # 12301
    print(f"Feature vector size: {n_features}")
    print("Extracting features...")

    feature_matrix = np.zeros((len(df), n_features), dtype=np.float32)

    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Extracting features")):
        feature_matrix[i] = extract_features(row, window)

    feat_cols = [f"f_{i}" for i in range(n_features)]
    feat_df   = pd.DataFrame(feature_matrix, columns=feat_cols)

    meta = df[["chrom", "position", "ref", "alt", "label", "split"]].reset_index(drop=True)
    out  = pd.concat([meta, feat_df], axis=1)

    out.to_parquet(OUT_PATH, index=False)

    print(f"\nDone.")
    print(f"  Feature matrix : {feature_matrix.shape}")
    print(f"  Train          : {(out['split']=='train').sum():,}")
    print(f"  Val            : {(out['split']=='val').sum():,}")
    print(f"  Test           : {(out['split']=='test').sum():,}")
    print(f"  Saved to       : {OUT_PATH}")


if __name__ == "__main__":
    run()