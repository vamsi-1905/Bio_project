

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    classification_report, confusion_matrix, roc_curve, average_precision_score,
    recall_score
)

IN_PATH      = "features.parquet"
MODEL_OUT    = "xgb_model.json"
THRESH_OUT   = "xgb_threshold.json"
RESULTS_OUT  = "xgb_results.json"
EARLY_STOP   = 50
N_ESTIMATORS = 1000
MAX_DEPTH    = 6
LR           = 0.05
SUBSAMPLE    = 0.8
COLSAMPLE    = 0.8


def load_splits(path):
    print(f"Loading {path}...")
    df = pd.read_parquet(path)
    print(f"  Total rows : {len(df):,}")
    feat_cols = [c for c in df.columns if c.startswith("f_")]
    print(f"  Features   : {len(feat_cols)}")
    train = df[df["split"] == "train"]
    val   = df[df["split"] == "val"]
    test  = df[df["split"] == "test"]
    X_tr,  y_tr  = train[feat_cols].values, train["label"].values
    X_val, y_val = val[feat_cols].values,   val["label"].values
    X_te,  y_te  = test[feat_cols].values,  test["label"].values
    print(f"  Train={len(X_tr):,}  Val={len(X_val):,}  Test={len(X_te):,}")
    return X_tr, y_tr, X_val, y_val, X_te, y_te, feat_cols


def make_sample_weights(y, ratio):
    """
    Benign (label=0) is minority — upweight by ratio so XGB treats each
    benign sample as `ratio` pathogenic samples.
    """
    w = np.where(y == 0, ratio, 1.0).astype(np.float32)
    return w


def tune_threshold(probs, labels):
    """
    Tune threshold to maximize balanced sensitivity:
    harmonic mean of recall for benign and recall for pathogenic.
    This prevents the majority class from dominating threshold selection.
    """
    best_t, best_score = 0.5, 0.0
    for t in np.arange(0.05, 0.95, 0.01):
        preds = (probs >= t).astype(int)
        rec_ben = recall_score(labels, preds, pos_label=0, zero_division=0)
        rec_pat = recall_score(labels, preds, pos_label=1, zero_division=0)
        # harmonic mean of recalls — punishes if either class recall collapses
        if rec_ben + rec_pat == 0:
            continue
        score = 2 * rec_ben * rec_pat / (rec_ben + rec_pat)
        if score > best_score:
            best_score, best_t = score, float(t)
    return best_t, best_score


def plot_confusion(cm, title, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Benign","Pathogenic"],
                yticklabels=["Benign","Pathogenic"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved: {path}")


def plot_roc(fpr, tpr, auc, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label=f"AUC={auc:.4f}")
    ax.plot([0,1],[0,1],"k--",lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC Curve (Test)")
    ax.legend(); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved: {path}")


def plot_importance(model, feat_cols, path, top_n=30):
    scores = model.get_score(importance_type="gain")
    items  = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names  = [feat_cols[int(k[2:])] if k.startswith("f_") else k for k,_ in items]
    vals   = [v for _,v in items]
    fig, ax = plt.subplots(figsize=(8,6))
    ax.barh(names[::-1], vals[::-1]); ax.set_xlabel("Gain")
    ax.set_title(f"Top {top_n} Feature Importances")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved: {path}")


def run():
    X_tr, y_tr, X_val, y_val, X_te, y_te, feat_cols = load_splits(IN_PATH)

    n_ben = (y_tr == 0).sum()   # benign = minority
    n_pat = (y_tr == 1).sum()   # pathogenic = majority
    ratio = n_ben / max(n_pat, 1)
    print(f"\nClass balance → benign={n_ben:,}  pathogenic={n_pat:,}  ratio={ratio:.1f}x")

    # FIX: sample_weight upweights benign (minority, label=0)
    # scale_pos_weight only upweights label=1 (pathogenic = majority here) → wrong
    sample_w_tr  = make_sample_weights(y_tr,  ratio)
    sample_w_val = make_sample_weights(y_val, ratio)
    print(f"  Benign sample weight={ratio:.1f}  Pathogenic sample weight=1.0")

    dtrain = xgb.DMatrix(X_tr,  label=y_tr,  feature_names=feat_cols, weight=sample_w_tr)
    dval   = xgb.DMatrix(X_val, label=y_val, feature_names=feat_cols, weight=sample_w_val)
    dtest  = xgb.DMatrix(X_te,  label=y_te,  feature_names=feat_cols)

    params = {
        "objective"        : "binary:logistic",
        "eval_metric"      : ["logloss", "auc"],
        "max_depth"        : MAX_DEPTH,
        "learning_rate"    : LR,
        "subsample"        : SUBSAMPLE,
        "colsample_bytree" : COLSAMPLE,
        "seed"             : 42,
        "tree_method"      : "hist",
        "device"           : "cuda" if _has_cuda() else "cpu",
        "min_child_weight" : 1,   # lower = more splits on rare benign samples
        "gamma"            : 0.1,
    }

    print(f"\nTraining XGBoost (device={params['device']})...")
    evals_result = {}
    model = xgb.train(
        params, dtrain,
        num_boost_round       = N_ESTIMATORS,
        evals                 = [(dtrain,"train"),(dval,"val")],
        early_stopping_rounds = EARLY_STOP,
        evals_result          = evals_result,
        verbose_eval          = 50,
    )
    model.save_model(MODEL_OUT)
    print(f"\nModel saved → {MODEL_OUT}")

    # FIX: tune threshold on val using balanced recall harmonic mean
    val_probs = model.predict(dval)
    best_t, best_score = tune_threshold(val_probs, y_val)
    print(f"Optimal threshold: {best_t:.2f}  (val balanced-recall-hmean={best_score:.4f})")

    # Print per-threshold breakdown so you can see what's happening
    print("\nThreshold sweep (val):")
    print(f"{'thresh':>8} {'rec_ben':>9} {'rec_pat':>9} {'f1_ben':>8} {'f1_pat':>8}")
    for t in np.arange(0.1, 0.9, 0.1):
        preds = (val_probs >= t).astype(int)
        rb = recall_score(y_val, preds, pos_label=0, zero_division=0)
        rp = recall_score(y_val, preds, pos_label=1, zero_division=0)
        fb = f1_score(y_val, preds, pos_label=0, zero_division=0)
        fp = f1_score(y_val, preds, pos_label=1, zero_division=0)
        marker = " ← selected" if abs(t - best_t) < 0.05 else ""
        print(f"{t:>8.1f} {rb:>9.3f} {rp:>9.3f} {fb:>8.3f} {fp:>8.3f}{marker}")

    with open(THRESH_OUT, "w") as f:
        json.dump({"threshold": best_t}, f)
    print(f"\nThreshold saved → {THRESH_OUT}")

    # Evaluate
    print("\n── Evaluation ──")
    for split_name, dmat, y_true in [("Val",dval,y_val),("Test",dtest,y_te)]:
        probs  = model.predict(dmat)
        preds  = (probs >= best_t).astype(int)
        acc    = accuracy_score(y_true, preds)
        auc    = roc_auc_score(y_true, probs)
        pr_auc = average_precision_score(y_true, probs)
        f1p    = f1_score(y_true, preds, pos_label=1, zero_division=0)
        f1b    = f1_score(y_true, preds, pos_label=0, zero_division=0)
        cm     = confusion_matrix(y_true, preds)

        print(f"\n{split_name}:  Acc={acc:.4f}  AUC={auc:.4f}  PR-AUC={pr_auc:.4f}  "
              f"F1-path={f1p:.4f}  F1-benign={f1b:.4f}")
        print(classification_report(y_true, preds, target_names=["Benign","Pathogenic"]))
        plot_confusion(cm, f"Confusion ({split_name})", f"xgb_confusion_{split_name.lower()}.png")
        if split_name == "Test":
            fpr, tpr, _ = roc_curve(y_true, probs)
            plot_roc(fpr, tpr, auc, "xgb_roc.png")

    plot_importance(model, feat_cols, "xgb_feature_importance.png")

    test_probs = model.predict(dtest)
    test_preds = (test_probs >= best_t).astype(int)
    results = {
        "test_accuracy" : float(accuracy_score(y_te, test_preds)),
        "test_auc"      : float(roc_auc_score(y_te, test_probs)),
        "test_pr_auc"   : float(average_precision_score(y_te, test_probs)),
        "test_f1_path"  : float(f1_score(y_te, test_preds, pos_label=1, zero_division=0)),
        "test_f1_benign": float(f1_score(y_te, test_preds, pos_label=0, zero_division=0)),
        "threshold"     : best_t,
        "best_iteration": int(model.best_iteration),
        "n_features"    : len(feat_cols),
        "class_balance" : {"benign": int(n_ben), "pathogenic": int(n_pat), "ratio": float(ratio)},
    }
    with open(RESULTS_OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults → {RESULTS_OUT}")
    print(json.dumps(results, indent=2))


def _has_cuda():
    try:
        import subprocess
        return subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0
    except Exception:
        return False


if __name__ == "__main__":
    run()