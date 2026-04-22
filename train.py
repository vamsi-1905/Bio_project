"""
train.py — Step 3
XGBoost baseline on features.parquet.

Output:
  - xgb_model.json          (trained model)
  - xgb_results.json        (metrics)
  - xgb_confusion.png
  - xgb_roc.png
  - xgb_feature_importance.png

Requirements:
    pip install xgboost scikit-learn pandas numpy matplotlib seaborn pyarrow
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    classification_report, confusion_matrix, roc_curve
)

# ── CONFIG ────────────────────────────────────────────────────────────────────
IN_PATH        = "features.parquet"
MODEL_OUT      = "xgb_model.json"
RESULTS_OUT    = "xgb_results.json"
EARLY_STOP     = 50
N_ESTIMATORS   = 1000
MAX_DEPTH      = 6
LEARNING_RATE  = 0.05
SUBSAMPLE      = 0.8
COLSAMPLE      = 0.8
SCALE_POS      = None   # set to None → auto-computed from class ratio
# ─────────────────────────────────────────────────────────────────────────────


def load_splits(path):
    print(f"Loading {path}...")
    df = pd.read_parquet(path)
    print(f"  Total rows : {len(df):,}")

    feat_cols = [c for c in df.columns if c.startswith("f_")]
    print(f"  Features   : {len(feat_cols)}")

    train = df[df["split"] == "train"]
    val   = df[df["split"] == "val"]
    test  = df[df["split"] == "test"]

    X_tr, y_tr = train[feat_cols].values, train["label"].values
    X_val, y_val = val[feat_cols].values,   val["label"].values
    X_te, y_te  = test[feat_cols].values,   test["label"].values

    print(f"  Train : {len(X_tr):,}  |  Val : {len(X_val):,}  |  Test : {len(X_te):,}")
    return X_tr, y_tr, X_val, y_val, X_te, y_te, feat_cols


def plot_confusion(cm, title, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Benign", "Pathogenic"],
                yticklabels=["Benign", "Pathogenic"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_roc(fpr, tpr, auc, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (Test Set)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_importance(model, feat_cols, path, top_n=30):
    scores = model.get_score(importance_type="gain")
    # map f_N → index
    items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names = [feat_cols[int(k[2:])] if k.startswith("f_") else k for k, _ in items]
    vals  = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(names[::-1], vals[::-1])
    ax.set_xlabel("Gain")
    ax.set_title(f"Top {top_n} Feature Importances (Gain)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def run():
    X_tr, y_tr, X_val, y_val, X_te, y_te, feat_cols = load_splits(IN_PATH)

    # Auto scale_pos_weight for class imbalance
    neg = (y_tr == 0).sum()
    pos = (y_tr == 1).sum()
    spw = neg / max(pos, 1) if SCALE_POS is None else SCALE_POS
    print(f"\nClass balance  →  neg={neg:,}  pos={pos:,}  scale_pos_weight={spw:.2f}")

    dtrain = xgb.DMatrix(X_tr,  label=y_tr,  feature_names=feat_cols)
    dval   = xgb.DMatrix(X_val, label=y_val, feature_names=feat_cols)
    dtest  = xgb.DMatrix(X_te,  label=y_te,  feature_names=feat_cols)

    params = {
        "objective"        : "binary:logistic",
        "eval_metric"      : ["logloss", "auc"],
        "max_depth"        : MAX_DEPTH,
        "learning_rate"    : LEARNING_RATE,
        "subsample"        : SUBSAMPLE,
        "colsample_bytree" : COLSAMPLE,
        "scale_pos_weight" : spw,
        "seed"             : 42,
        "tree_method"      : "hist",   # fast on CPU; GPU picks up automatically if available
        "device"           : "cuda" if _has_cuda() else "cpu",
    }

    print(f"\nTraining XGBoost  (device={params['device']})...")
    evals_result = {}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round = N_ESTIMATORS,
        evals           = [(dtrain, "train"), (dval, "val")],
        early_stopping_rounds = EARLY_STOP,
        evals_result    = evals_result,
        verbose_eval    = 50,
    )

    model.save_model(MODEL_OUT)
    print(f"\nModel saved → {MODEL_OUT}")

    # ── Evaluation ──────────────────────────────────────────────────────────
    print("\n── Evaluation ──")
    for split_name, dmat, y_true in [("Val", dval, y_val), ("Test", dtest, y_te)]:
        probs  = model.predict(dmat)
        preds  = (probs >= 0.5).astype(int)
        acc    = accuracy_score(y_true, preds)
        auc    = roc_auc_score(y_true, probs)
        f1     = f1_score(y_true, preds)
        cm     = confusion_matrix(y_true, preds)

        print(f"\n{split_name} set:")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  ROC-AUC  : {auc:.4f}")
        print(f"  F1       : {f1:.4f}")
        print(classification_report(y_true, preds,
                                    target_names=["Benign", "Pathogenic"]))

        plot_confusion(cm, f"Confusion Matrix ({split_name})",
                       f"xgb_confusion_{split_name.lower()}.png")

        if split_name == "Test":
            fpr, tpr, _ = roc_curve(y_true, probs)
            plot_roc(fpr, tpr, auc, "xgb_roc.png")

    # Feature importance
    plot_importance(model, feat_cols, "xgb_feature_importance.png")

    # Save metrics
    test_probs = model.predict(dtest)
    test_preds = (test_probs >= 0.5).astype(int)
    results = {
        "test_accuracy" : float(accuracy_score(y_te, test_preds)),
        "test_auc"      : float(roc_auc_score(y_te, test_probs)),
        "test_f1"       : float(f1_score(y_te, test_preds)),
        "best_iteration": int(model.best_iteration),
        "n_features"    : len(feat_cols),
    }
    with open(RESULTS_OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {RESULTS_OUT}")
    print(json.dumps(results, indent=2))


def _has_cuda():
    try:
        import subprocess
        r = subprocess.run(["nvidia-smi"], capture_output=True)
        return r.returncode == 0
    except Exception:
        return False


if __name__ == "__main__":
    run()