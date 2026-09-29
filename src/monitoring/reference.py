"""Builds models/reference.json: the training-time distributions that live
traffic is compared against.

    python -m src.monitoring.reference

Monitored features: the top N by the model's gain importance (drift in a
feature the model barely uses matters less), numeric and categorical.
Feature reference = the training split (what the model learned from).
Also builds a baseline: PSI of consecutive validation windows against the
reference, so alert levels reflect how much each feature normally moves.
Score reference = the model's scores on the validation split, not on
training rows, because the model has seen the training rows and scores them
more confidently than it scores new data; comparing live scores to those
would show phantom drift on day one.
"""
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from src.monitoring import drift
from src.serving.model import FraudModel
from src.training.common import DROP_COLS, MODELS, PROCESSED, TARGET, load_parquet_lean, prep_features, time_split

REFERENCE_PATH = MODELS / "reference.json"
MODEL_PATH = MODELS / "phase2_lgbm.pkl"
N_FEATURES = 30
N_CATEGORIES = 20


WINDOW_ROWS = 3000  # window size the alert levels are calibrated for; matches the default monitor runs


def _baseline(features: dict, val_df: pd.DataFrame, val_scores: np.ndarray, score_spec: dict, window_rows: int) -> dict:
    """How much each monitored feature and the score normally move: PSI against
    the reference for consecutive equal-sized windows of held-out data
    (already sorted by time). p95 across windows sets the alert levels."""
    n_windows = len(val_df) // window_rows
    per_feature = {name: [] for name in features}
    score_psis = []
    for i in range(n_windows):
        sl = slice(i * window_rows, (i + 1) * window_rows)
        w = val_df.iloc[sl]
        for name, spec in features.items():
            per_feature[name].append(drift.column_psi(spec, w[name].tolist()))
        s = val_scores[sl]
        score_psis.append(drift.psi(score_spec["props"], drift.numeric_props(s, score_spec["edges"])))

    def summ(v):
        v = np.asarray(v)
        return {"median": float(np.median(v)), "p95": float(np.percentile(v, 95)), "max": float(v.max())}

    return {
        "window_rows": window_rows,
        "n_windows": n_windows,
        "features": {k: summ(v) for k, v in per_feature.items()},
        "score": summ(score_psis),
    }


def build_reference(
    train_df: pd.DataFrame,
    val_scores: np.ndarray,
    fm: FraudModel,
    n_features: int = N_FEATURES,
    val_df: pd.DataFrame = None,
    window_rows: int = WINDOW_ROWS,
) -> dict:
    """val_df (optional, same row order as val_scores, sorted by time) enables
    the calibration baseline."""
    gain = fm.model.booster_.feature_importance(importance_type="gain")
    ranked = [c for _, c in sorted(zip(gain, fm.feature_names), reverse=True)][:n_features]

    features = {}
    for name in ranked:
        if name not in train_df.columns:
            continue
        col = train_df[name]
        if name in fm.categorical_cols:
            cats = col.dropna().astype(str).value_counts().head(N_CATEGORIES).index.tolist()
            props = drift.categorical_props(col.where(col.notna(), None).astype(object).tolist(), cats)
            features[name] = {"type": "categorical", "categories": cats, "props": props.tolist()}
        else:
            arr = pd.to_numeric(col, errors="coerce").to_numpy(dtype="float64")
            edges = drift.numeric_edges(arr)
            features[name] = {"type": "numeric", "edges": edges, "props": drift.numeric_props(arr, edges).tolist()}

    s_edges = drift.numeric_edges(val_scores)
    score_spec = {"edges": s_edges, "props": drift.numeric_props(val_scores, s_edges).tolist()}
    baseline = None
    if val_df is not None and len(val_df) >= 2 * window_rows:
        baseline = _baseline(features, val_df, val_scores, score_spec, window_rows)
    return {
        "model_version": fm.version,
        "created": datetime.now(timezone.utc).isoformat(),
        "n_train_rows": int(len(train_df)),
        "n_score_rows": int(len(val_scores)),
        "features": features,
        "score": score_spec,
        "baseline": baseline,
    }


def main():
    fm = FraudModel.load(MODEL_PATH)
    print("loading data/processed/train_features.parquet...")
    df = load_parquet_lean(PROCESSED / "train_features.parquet")
    train_df, val_df = time_split(df)
    val_df = val_df.sort_values("TransactionDT")  # windows must be consecutive in time

    drop_cols = DROP_COLS + ["entity_id"]
    X_val = prep_features(val_df, drop_cols)
    val_scores = joblib.load(MODEL_PATH).predict_proba(X_val)[:, 1]

    ref = build_reference(train_df, val_scores, fm, val_df=val_df)
    REFERENCE_PATH.write_text(json.dumps(ref))
    print(f"monitoring {len(ref['features'])} features + score; model_version {fm.version}")
    b = ref["baseline"]
    if b:
        print(f"calibrated alert levels on {b['n_windows']} windows of {b['window_rows']} rows; score PSI normally p95 {b['score']['p95']:.3f}")
    print(f"saved {REFERENCE_PATH}")


if __name__ == "__main__":
    main()
