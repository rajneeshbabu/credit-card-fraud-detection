"""
train.py — Smart Training Pipeline  (creditcard.csv → models/)
══════════════════════════════════════════════════════════════
STEP 1  Load data from creditcard.csv
STEP 2  Feature engineering + split + scale
STEP 3  Quick model comparison (3-fold CV, ~2 min)
            LR  |  Random Forest  |  XGBoost  |  LightGBM
STEP 4  Full RandomizedSearchCV on BEST model only (n_iter=40, cv=5)
STEP 5  PyTorch Autoencoder on Mac GPU (MPS) / CUDA / CPU
STEP 6  Ensemble  (best supervised × 0.6  +  autoencoder × 0.4)
STEP 7  Save all artefacts → models/

Usage:
    python train.py                   # full pipeline
    python train.py --skip-ae        # skip autoencoder (faster)
    python train.py --n-iter 20      # fewer search iterations
"""

import argparse, json, os, time, warnings
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, classification_report,
    precision_recall_curve, roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold, RandomizedSearchCV,
    cross_val_score, train_test_split,
)
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# ─────────────────────────────────────────────
def banner(msg: str):
    print(f"\n{'═'*62}\n  {msg}\n{'═'*62}")

def best_threshold(y_true, probs) -> float:
    p, r, t = precision_recall_curve(y_true, probs)
    f1 = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-10)
    return float(t[np.argmax(f1)])


# ══════════════════════════════════════════════
#  STEP 1+2 — Data
# ══════════════════════════════════════════════
def load_and_prepare(csv_path: str):
    banner("STEP 1+2 — Load & Prepare  creditcard.csv")
    df = pd.read_csv(csv_path)
    print(f"  Shape     : {df.shape}")
    vc = df["Class"].value_counts()
    print(f"  Legit     : {vc[0]:,}   Fraud : {vc[1]:,}   ({vc[1]/len(df)*100:.4f}%)")

    df["Hour"] = (df["Time"] / 3600) % 24
    X = df.drop(columns=["Class", "Time", "Amount"])
    y = df["Class"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    sc = StandardScaler()
    X_tr_s = pd.DataFrame(sc.fit_transform(X_tr), columns=X.columns)
    X_te_s = pd.DataFrame(sc.transform(X_te),     columns=X.columns)

    print(f"  Train     : {len(X_tr):,}  |  Test : {len(X_te):,}")

    sm = SMOTE(random_state=42, k_neighbors=5)
    X_res, y_res = sm.fit_resample(X_tr_s, y_tr)
    print(f"  After SMOTE → {len(X_res):,} training samples (balanced)")

    return X_tr_s, X_te_s, y_tr, y_te, X_res, y_res, sc, list(X.columns)


# ══════════════════════════════════════════════
#  STEP 3 — Quick model comparison
# ══════════════════════════════════════════════
def quick_compare(X_res, y_res, y_tr) -> str:
    banner("STEP 3 — Quick Model Comparison  (3-fold CV, small estimators)")

    scale_pos = int(y_tr.value_counts()[0] / max(y_tr.value_counts()[1], 1))
    cv3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=500, C=0.1, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=5,
            scale_pos_weight=scale_pos, eval_metric="aucpr",
            random_state=42, n_jobs=-1, verbosity=0),
        "LightGBM": LGBMClassifier(
            n_estimators=150, learning_rate=0.05,
            class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1),
    }

    print(f"\n  {'Model':<22} {'CV PR-AUC':>12}  {'Time':>6}")
    print(f"  {'─'*22} {'─'*12}  {'─'*6}")

    scores = {}
    for name, clf in candidates.items():
        t0 = time.time()
        s  = cross_val_score(clf, X_res, y_res,
                             cv=cv3, scoring="average_precision", n_jobs=-1)
        scores[name] = s.mean()
        print(f"  {name:<22} {s.mean():.4f} ±{s.std():.4f}  {time.time()-t0:>5.1f}s")

    best = max(scores, key=scores.get)
    print(f"\n  🏆  Best model → {best}  (PR-AUC = {scores[best]:.4f})")
    return best


# ══════════════════════════════════════════════
#  STEP 4 — Full tune of winner
# ══════════════════════════════════════════════
PARAM_GRIDS = {
    "LightGBM": {
        "n_estimators":      [300, 500, 700, 1000],
        "learning_rate":     [0.01, 0.03, 0.05, 0.1],
        "max_depth":         [5, 7, 9, -1],
        "num_leaves":        [31, 63, 127],
        "subsample":         [0.6, 0.7, 0.8, 0.9],
        "colsample_bytree":  [0.6, 0.7, 0.8, 1.0],
        "min_child_samples": [10, 20, 30],
        "reg_alpha":         [0, 0.1, 0.5, 1.0],
        "reg_lambda":        [0, 0.1, 0.5, 1.0],
    },
    "XGBoost": {
        "n_estimators":      [200, 300, 500, 700],
        "learning_rate":     [0.01, 0.03, 0.05, 0.1],
        "max_depth":         [4, 5, 6, 8],
        "subsample":         [0.6, 0.7, 0.8, 1.0],
        "colsample_bytree":  [0.6, 0.7, 0.8, 1.0],
        "gamma":             [0, 0.1, 0.3, 0.5],
        "reg_alpha":         [0, 0.1, 0.5],
        "reg_lambda":        [0.5, 1.0, 1.5, 2.0],
        "min_child_weight":  [1, 3, 5],
    },
    "Random Forest": {
        "n_estimators":      [200, 300, 500],
        "max_depth":         [10, 15, 20, None],
        "min_samples_leaf":  [1, 2, 4],
        "min_samples_split": [2, 5, 10],
        "max_features":      ["sqrt", "log2", 0.3],
    },
    "Logistic Regression": {
        "C":       [0.001, 0.01, 0.1, 1, 10, 100],
        "penalty": ["l1", "l2"],
        "solver":  ["liblinear", "saga"],
    },
}

BASE_MODELS = lambda y_tr: {
    "LightGBM": LGBMClassifier(
        class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1),
    "XGBoost": XGBClassifier(
        scale_pos_weight=int(y_tr.value_counts()[0]/max(y_tr.value_counts()[1],1)),
        eval_metric="aucpr", random_state=42, n_jobs=-1, verbosity=0),
    "Random Forest": RandomForestClassifier(
        class_weight="balanced", random_state=42, n_jobs=-1),
    "Logistic Regression": LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=42),
}


def full_tune(best_name: str, X_res, y_res, X_te_s, y_te, y_tr, n_iter: int):
    banner(f"STEP 4 — Full Tune → {best_name}  (n_iter={n_iter}, cv=5)")

    base   = BASE_MODELS(y_tr)[best_name]
    params = PARAM_GRIDS[best_name]
    cv5    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        base, params, n_iter=n_iter, scoring="average_precision",
        cv=cv5, refit=True, n_jobs=-1, random_state=42, verbose=1,
    )
    t0 = time.time()
    search.fit(X_res, y_res)
    print(f"\n  Done in {time.time()-t0:.1f}s  |  CV PR-AUC: {search.best_score_:.4f}")
    print("  Best params:")
    for k, v in search.best_params_.items():
        print(f"    {k:<25}: {v}")

    model  = search.best_estimator_
    probs  = model.predict_proba(X_te_s)[:, 1]
    thr    = best_threshold(y_te, probs)
    roc    = roc_auc_score(y_te, probs)
    pr_auc = average_precision_score(y_te, probs)

    print(f"\n  Test ROC-AUC : {roc:.4f}")
    print(f"  Test PR-AUC  : {pr_auc:.4f}")
    print(f"  Threshold    : {thr:.4f}")
    print(classification_report(y_te, (probs >= thr).astype(int),
                                 target_names=["Legit", "Fraud"]))
    return model, probs, thr, roc, pr_auc, search.best_score_, search.best_params_


# ══════════════════════════════════════════════
#  STEP 5 — PyTorch Autoencoder  (Mac MPS ⚡)
# ══════════════════════════════════════════════
def train_autoencoder(X_tr_s, X_te_s, y_tr, y_te):
    banner("STEP 5 — PyTorch Autoencoder  (Mac GPU via MPS)")
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
        from model_arch import FraudAutoencoder, get_device
    except ImportError as e:
        print(f"  ⚠️  {e}\n  Install: pip install torch\n  Skipping autoencoder.")
        return None, None, None

    device, dev_label = get_device()
    print(f"  Device : {dev_label}")

    # ── Legit-only training tensors
    X_legit = X_tr_s[y_tr == 0].values.astype("float32")
    X_val   = X_te_s[y_te  == 0].values.astype("float32")
    X_test  = X_te_s.values.astype("float32")

    t_legit = torch.tensor(X_legit).to(device)
    t_val   = torch.tensor(X_val).to(device)
    t_test  = torch.tensor(X_test).to(device)

    ds      = TensorDataset(t_legit, t_legit)
    loader  = DataLoader(ds, batch_size=256, shuffle=True)

    print(f"  Training on {len(X_legit):,} legit transactions  ({X_legit.shape[1]} features)")

    # ── Model
    input_dim = X_legit.shape[1]
    ae        = FraudAutoencoder(input_dim).to(device)
    opt       = torch.optim.Adam(ae.parameters(), lr=1e-3, weight_decay=1e-5)
    sched     = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt, patience=5, factor=0.5, min_lr=1e-6, verbose=False)
    criterion = nn.MSELoss()

    # ── Training loop with early stopping
    best_val, patience_cnt, best_state = float("inf"), 0, None
    PATIENCE, EPOCHS = 10, 120

    print(f"\n  {'Epoch':>5}  {'Train Loss':>11}  {'Val Loss':>10}  {'LR':>10}")
    print(f"  {'─'*5}  {'─'*11}  {'─'*10}  {'─'*10}")

    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        # train
        ae.train()
        train_loss = 0.0
        for xb, _ in loader:
            opt.zero_grad()
            loss = criterion(ae(xb), xb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(X_legit)

        # validate
        ae.eval()
        with torch.no_grad():
            val_loss = criterion(ae(t_val), t_val).item()

        sched.step(val_loss)
        lr_now = opt.param_groups[0]["lr"]

        if epoch % 10 == 0 or epoch == 1:
            print(f"  {epoch:>5}  {train_loss:>11.6f}  {val_loss:>10.6f}  {lr_now:>10.2e}")

        # early stopping
        if val_loss < best_val - 1e-6:
            best_val, patience_cnt = val_loss, 0
            best_state = {k: v.clone() for k, v in ae.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"\n  ⏹  Early stop at epoch {epoch}  (best val loss: {best_val:.6f})")
                break

    ae.load_state_dict(best_state)
    ae.eval()
    print(f"  Training time: {time.time()-t0:.1f}s")

    # ── Reconstruction errors & normalisation
    with torch.no_grad():
        raw_err = torch.mean((t_test - ae(t_test)) ** 2, dim=1).cpu().numpy()

    err_min, err_max = float(raw_err.min()), float(raw_err.max())
    norm_err = (raw_err - err_min) / (err_max - err_min + 1e-10)
    norm_err = np.clip(norm_err, 0.0, 1.0)

    # ── Optimal threshold
    thr    = best_threshold(y_te, norm_err)
    roc    = roc_auc_score(y_te, norm_err)
    pr_auc = average_precision_score(y_te, norm_err)

    print(f"\n  AE ROC-AUC : {roc:.4f}")
    print(f"  AE PR-AUC  : {pr_auc:.4f}")
    print(f"  AE Threshold: {thr:.4f}")
    print(classification_report((norm_err >= thr).astype(int), y_te,
                                 target_names=["Legit", "Fraud"]))

    ae_meta = {
        "recon_err_min": err_min, "recon_err_max": err_max,
        "ae_threshold":  thr,    "ae_pr_auc":     pr_auc,
        "ae_roc_auc":    roc,    "ae_input_dim":  input_dim,
    }
    return ae, ae_meta, norm_err


# ══════════════════════════════════════════════
#  STEP 6 — Ensemble
# ══════════════════════════════════════════════
def build_ensemble(sup_probs, ae_scores, y_te):
    banner("STEP 6 — Ensemble  (supervised × 0.6 + autoencoder × 0.4)")
    if ae_scores is None:
        print("  Skipping — autoencoder not available")
        return None

    W_S, W_A = 0.6, 0.4
    ens    = W_S * sup_probs + W_A * ae_scores
    thr    = best_threshold(y_te, ens)
    roc    = roc_auc_score(y_te, ens)
    pr_auc = average_precision_score(y_te, ens)

    print(f"  Ensemble PR-AUC   : {pr_auc:.4f}")
    print(f"  Ensemble ROC-AUC  : {roc:.4f}")
    print(f"  Ensemble threshold: {thr:.4f}")
    return {"w_sup": W_S, "w_ae": W_A,
            "ens_threshold": thr, "ens_pr_auc": pr_auc, "ens_roc_auc": roc}


# ══════════════════════════════════════════════
#  STEP 7 — Save
# ══════════════════════════════════════════════
def save_all(out_dir, sc, features, best_name, sup_model, sup_thr,
             sup_cv, sup_pr, sup_roc, sup_params,
             ae_model, ae_meta, ens_meta):
    banner("STEP 7 — Save All Artefacts")
    os.makedirs(out_dir, exist_ok=True)

    joblib.dump(sup_model, f"{out_dir}/best_model.pkl")
    joblib.dump(sc,        f"{out_dir}/scaler.pkl")
    print(f"  ✅  {out_dir}/best_model.pkl  ({best_name})")
    print(f"  ✅  {out_dir}/scaler.pkl")

    meta = {
        "best_model":      best_name,
        "trained_on":      "creditcard.csv",
        "best_threshold":  sup_thr,
        "features":        features,
        "best_cv_pr_auc":  sup_cv,
        "test_pr_auc":     sup_pr,
        "test_roc_auc":    sup_roc,
        "best_params":     sup_params,
    }
    if ae_meta:   meta.update(ae_meta)
    if ens_meta:  meta.update(ens_meta)

    with open(f"{out_dir}/model_info.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  ✅  {out_dir}/model_info.json")

    if ae_model is not None:
        import torch
        torch.save({
            "state_dict": ae_model.state_dict(),
            "input_dim":  ae_meta["ae_input_dim"],
        }, f"{out_dir}/autoencoder.pt")
        print(f"  ✅  {out_dir}/autoencoder.pt  (PyTorch)")

    print(f"\n  All artefacts saved → {out_dir}/")


# ══════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",      default="creditcard.csv")
    ap.add_argument("--out",      default="models")
    ap.add_argument("--n-iter",   type=int, default=40,
                    help="RandomizedSearchCV iterations (default 40)")
    ap.add_argument("--skip-ae",  action="store_true",
                    help="Skip autoencoder training")
    args = ap.parse_args()

    print("\n" + "█"*62)
    print("  CREDIT CARD FRAUD DETECTION — TRAINING PIPELINE")
    print("  Smart model selection  +  PyTorch Autoencoder (Mac GPU)")
    print("█"*62)

    # Steps 1+2
    X_tr_s, X_te_s, y_tr, y_te, X_res, y_res, sc, features = \
        load_and_prepare(args.csv)

    # Step 3 — quick compare → pick winner
    best_name = quick_compare(X_res, y_res, y_tr)

    # Step 4 — full tune of winner only
    sup_model, sup_probs, sup_thr, sup_roc, sup_pr, sup_cv, sup_params = \
        full_tune(best_name, X_res, y_res, X_te_s, y_te, y_tr, args.n_iter)

    # Step 5 — PyTorch autoencoder
    ae_model, ae_meta, ae_scores = (None, None, None)
    if not args.skip_ae:
        ae_model, ae_meta, ae_scores = \
            train_autoencoder(X_tr_s, X_te_s, y_tr, y_te)

    # Step 6 — ensemble
    ens_meta = build_ensemble(sup_probs, ae_scores, y_te)

    # Step 7 — save
    save_all(args.out, sc, features, best_name, sup_model, sup_thr,
             sup_cv, sup_pr, sup_roc, sup_params,
             ae_model, ae_meta, ens_meta)

    print("\n" + "█"*62)
    print(f"  ✅  DONE   Best model : {best_name}")
    print(f"             PR-AUC    : {sup_pr:.4f}  (supervised)")
    if ae_meta:
        print(f"             PR-AUC    : {ae_meta['ae_pr_auc']:.4f}  (autoencoder)")
    if ens_meta:
        print(f"             PR-AUC    : {ens_meta['ens_pr_auc']:.4f}  (ensemble)")
    print("█"*62 + "\n")


if __name__ == "__main__":
    main()
