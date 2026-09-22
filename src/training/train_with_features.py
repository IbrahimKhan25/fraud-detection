"""Phase 2: retrains the LightGBM model with the Phase 2 engineered
features added, using the exact same split/imbalance handling/evaluation as
Phase 1 (src/training/train_baseline.py), so the PR-AUC numbers are directly
comparable. The only difference from train_baseline.py is the input columns:
this reads data/processed/train_features.parquet (raw columns + the 8
FEATURE_COLUMNS from src/features/offline.py) instead of the raw join, and
drops entity_id from X (it's a join key, not a feature -- its information is
already present via the raw card/addr columns it's built from).

Run from the repo root, after build_training_set.py:
    python -m src.training.train_with_features
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score

from src.features.definitions import FEATURE_COLUMNS
from src.training.common import DROP_COLS, MODELS, PROCESSED, TARGET, load_parquet_lean, prep_features, time_split


def main():
    print("loading data/processed/train_features.parquet...")
    df = load_parquet_lean(PROCESSED / "train_features.parquet")
    train_df, val_df = time_split(df)
    print(f"train: {len(train_df):,} rows | val: {len(val_df):,} rows")

    drop_cols = DROP_COLS + ["entity_id"]
    X_train, y_train = prep_features(train_df, drop_cols), train_df[TARGET]
    X_val, y_val = prep_features(val_df, drop_cols), val_df[TARGET]
    print(f"feature count: {X_train.shape[1]} (raw columns + {len(FEATURE_COLUMNS)} engineered)")

    model = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=50)],
    )

    print(f"\nbest_iteration: {model.best_iteration_} (out of n_estimators={model.n_estimators})")

    val_scores = model.predict_proba(X_val)[:, 1]
    pr_auc = average_precision_score(y_val, val_scores)
    random_baseline = y_val.mean()
    print(f"\nPR-AUC: {pr_auc:.4f}  (random baseline: {random_baseline:.4f})")

    for alert_rate in (0.01, 0.02, 0.05):
        threshold = np.quantile(val_scores, 1 - alert_rate)
        flagged = val_scores >= threshold
        precision = y_val[flagged].mean()
        recall = y_val[flagged].sum() / y_val.sum()
        print(f"alert rate {alert_rate:>4.0%}: precision {precision:.4f}, recall {recall:.4f}")

    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    print("\ntop 20 features by importance:")
    print(importances.sort_values(ascending=False).head(20))

    print("\nengineered feature ranks (out of {}):".format(len(importances)))
    ranked = importances.sort_values(ascending=False)
    rank_of = {c: (ranked.index.get_loc(c) + 1) for c in FEATURE_COLUMNS if c in ranked.index}
    for c, r in sorted(rank_of.items(), key=lambda kv: kv[1]):
        print(f"  {c}: rank {r}, importance {importances[c]}")

    MODELS.mkdir(exist_ok=True)
    import joblib

    joblib.dump(model, MODELS / "phase2_lgbm.pkl")
    print(f"\nsaved model to {MODELS / 'phase2_lgbm.pkl'}")


if __name__ == "__main__":
    main()
