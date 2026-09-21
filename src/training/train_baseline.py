"""Phase 1 baseline: LightGBM on raw IEEE-CIS columns, time-based split,
class weighting for imbalance. Proves the modeling problem is understood
before any serving/feature-store infrastructure gets built.

Run from the repo root:
    python -m src.training.train_baseline
"""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
MODELS = ROOT / "models"

TARGET = "isFraud"
DROP_COLS = ["TransactionID", "isFraud", "TransactionDT"]  # ID, target, raw time index


def load_data() -> pd.DataFrame:
    cached = RAW / "train_joined.parquet"
    if cached.exists():
        return pd.read_parquet(cached)
    tx = pd.read_csv(RAW / "train_transaction.csv")
    ident = pd.read_csv(RAW / "train_identity.csv")
    return tx.merge(ident, on="TransactionID", how="left")


def time_split(df: pd.DataFrame, cutoff_quantile: float = 0.80):
    cutoff_dt = df["TransactionDT"].quantile(cutoff_quantile)
    train_df = df[df["TransactionDT"] < cutoff_dt]
    val_df = df[df["TransactionDT"] >= cutoff_dt]
    return train_df, val_df


def prep_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.drop(columns=DROP_COLS)
    # LightGBM handles categoricals natively if the dtype is "category" -- no
    # one-hot encoding needed, which matters here since P_emaildomain etc.
    # would otherwise explode into hundreds of columns.
    obj_cols = X.select_dtypes(include="object").columns
    X[obj_cols] = X[obj_cols].astype("category")
    return X


def main():
    print("loading data...")
    df = load_data()
    train_df, val_df = time_split(df)
    print(f"train: {len(train_df):,} rows | val: {len(val_df):,} rows")

    X_train, y_train = prep_features(train_df), train_df[TARGET]
    X_val, y_val = prep_features(val_df), val_df[TARGET]

    # Weight the minority class using ONLY the train split's class counts --
    # using the full dataset (including val) here would leak val information
    # into a training decision, the same mistake as a random split would make.
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    print(f"train class balance: {neg:,} negative / {pos:,} positive ({pos / (neg + pos):.4%} fraud)")

    # No scale_pos_weight. Tested it at the full class ratio (~28x) and a milder
    # 5x -- both destabilized boosting (best_iteration_ stuck at 1 and 12, vs 388
    # unweighted) and produced worse PR-AUC. LightGBM's ranking-based objective
    # handles this level of imbalance on its own; see README for the comparison.
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
        eval_metric="auc",  # early-stopping signal; PR-AUC is reported separately below
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=50)],
    )

    print(f"\nbest_iteration: {model.best_iteration_} (out of n_estimators={model.n_estimators})")

    val_scores = model.predict_proba(X_val)[:, 1]
    pr_auc = average_precision_score(y_val, val_scores)
    random_baseline = y_val.mean()  # a random classifier's PR-AUC equals class prevalence
    print(f"\nPR-AUC: {pr_auc:.4f}  (random baseline: {random_baseline:.4f})")

    # Recall at a fixed alert budget: if a fraud team can only review the top 1%
    # of transactions by score, how much actual fraud do they catch?
    for alert_rate in (0.01, 0.02, 0.05):
        threshold = np.quantile(val_scores, 1 - alert_rate)
        flagged = val_scores >= threshold
        precision = y_val[flagged].mean()
        recall = y_val[flagged].sum() / y_val.sum()
        print(f"alert rate {alert_rate:>4.0%}: precision {precision:.4f}, recall {recall:.4f}")

    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    print("\ntop 20 features by importance:")
    print(importances.sort_values(ascending=False).head(20))

    MODELS.mkdir(exist_ok=True)
    import joblib

    joblib.dump(model, MODELS / "baseline_lgbm.pkl")
    print(f"\nsaved model to {MODELS / 'baseline_lgbm.pkl'}")


if __name__ == "__main__":
    main()
