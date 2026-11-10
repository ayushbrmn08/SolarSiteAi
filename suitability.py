"""Task 1: suitability. Presence-background learning, spatial-block CV, model comparison, AHP baseline."""
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from data import LAYER_NAMES

def get_models(seed):
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, eval_metric="logloss", random_state=seed)
        xgb_name = "XGBoost"
    except ImportError:   # fallback keeps the pipeline runnable; install xgboost for the real comparison
        xgb = HistGradientBoostingClassifier(max_depth=4, random_state=seed)
        xgb_name = "XGBoost (fallback: sklearn HistGB)"
    return {
        "LR": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
        "RF": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                                     n_jobs=-1, random_state=seed),
        xgb_name: xgb,
        "SVM": make_pipeline(StandardScaler(), SVC(probability=True, class_weight="balanced", random_state=seed)),
    }

def block_ids(shape, block):
    r, c = np.indices(shape)
    return (r // block) * (shape[1] // block + 1) + (c // block)

def sample_presence_background(positives, allowed, cfg):
    """Positives = existing farms. Background = random allowed non-farm cells (unlabeled, NOT negatives)."""
    rng = np.random.default_rng(cfg.seed + 2)
    pos = np.flatnonzero(positives.ravel() == 1)
    pool = np.flatnonzero((positives.ravel() == 0) & allowed.ravel())
    bg = rng.choice(pool, size=min(len(pool), cfg.background_ratio * len(pos)), replace=False)
    idx = np.concatenate([pos, bg])
    y = np.concatenate([np.ones(len(pos), int), np.zeros(len(bg), int)])
    return idx, y

def _metrics(y, p):
    yhat = (p >= 0.5).astype(int)
    return dict(Accuracy=accuracy_score(y, yhat), Precision=precision_score(y, yhat, zero_division=0),
                Recall=recall_score(y, yhat), F1=f1_score(y, yhat),
                ROC_AUC=roc_auc_score(y, p), PR_AUC=average_precision_score(y, p))

def spatial_cv(models, X, y, groups, cfg):
    """GroupKFold over spatial blocks: neighbouring pixels never sit in both train and test."""
    import pandas as pd
    rows = []
    for name, m in models.items():
        folds = []
        for tr, te in GroupKFold(n_splits=cfg.n_splits).split(X, y, groups):
            if y[te].sum() == 0 or y[tr].sum() == 0:
                continue
            mm = clone(m).fit(X[tr], y[tr])
            folds.append(_metrics(y[te], mm.predict_proba(X[te])[:, 1]))
        df = pd.DataFrame(folds)
        rows.append({"Model": name, **df.mean().to_dict(), **{f"{k}_std": v for k, v in df.std().items() if k in ("ROC_AUC", "PR_AUC")}})
    return pd.DataFrame(rows).set_index("Model")

def ahp_weighted_overlay(layers, allowed):
    """MCDM baseline with FIXED expert weights (illustrative; replace with your AHP pairwise-comparison result)."""
    def mm(a): return (a - a.min()) / (a.max() - a.min() + 1e-12)
    w = dict(ghi=0.35, slope=0.20, dist_grid=0.15, dist_road=0.10, ndvi=0.05, temp=0.05, income=0.10)
    s = (w["ghi"] * mm(layers["ghi"]) + w["slope"] * (1 - mm(layers["slope"])) + w["dist_grid"] * (1 - mm(layers["dist_grid"]))
         + w["dist_road"] * (1 - mm(layers["dist_road"])) + w["ndvi"] * (1 - mm(layers["ndvi"]))
         + w["temp"] * (1 - mm(layers["temp"])) + w["income"] * mm(layers["income"]))
    return s * allowed

def fit_and_map(model, X_all, idx, y, shape, allowed):
    m = clone(model).fit(X_all.reshape(-1, X_all.shape[-1])[idx], y)
    p = m.predict_proba(X_all.reshape(-1, X_all.shape[-1]))[:, 1].reshape(shape)
    return m, p * allowed     # hard exclusions applied after scoring

def importance(model, X, y):
    """Permutation importance if the estimator has no native feature_importances_."""
    if hasattr(model, "feature_importances_"):
        v = model.feature_importances_
    else:
        from sklearn.inspection import permutation_importance
        v = permutation_importance(model, X, y, n_repeats=5, random_state=0, scoring="roc_auc").importances_mean
    return dict(sorted(zip(LAYER_NAMES, v), key=lambda t: -t[1]))
