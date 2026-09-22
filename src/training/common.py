"""Shared data loading / splitting / prep, used by both train_baseline.py
(Phase 1) and train_with_features.py (Phase 2), so the two runs are
comparable: same rows, same split, same categorical handling, the only
difference is which columns are in X.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"

TARGET = "isFraud"
DROP_COLS = ["TransactionID", "isFraud", "TransactionDT"]  # ID, target, raw time index


def load_raw() -> pd.DataFrame:
    cached = RAW / "train_joined.parquet"
    if cached.exists():
        return pd.read_parquet(cached)
    tx = pd.read_csv(RAW / "train_transaction.csv")
    ident = pd.read_csv(RAW / "train_identity.csv")
    return tx.merge(ident, on="TransactionID", how="left")


def load_parquet_lean(path) -> pd.DataFrame:
    """Reads a parquet file and downcasts float64 -> float32, keeping peak
    memory well below a naive pd.read_parquet + df.astype. Two things make
    the difference: pyarrow's to_pandas(split_blocks=True, self_destruct=True)
    frees each column's Arrow buffer as it's converted instead of holding
    the whole Arrow table and the whole pandas frame in memory at once, and
    downcasting one column at a time (rather than
    df[float_cols].astype('float32') on the whole block) avoids briefly
    holding a second full-width float64 copy alongside the float32 one.
    Doesn't matter on a workstation with plenty of RAM; matters in a
    memory-constrained container, which is why this exists as a separate
    function rather than being load_raw()'s default. (Built because this
    session's own sandboxed dev environment only had 4-8GB RAM and OOM-killed
    on a naive load of the 439-column feature set -- your machine may not
    need this at all, but it's here if you hit the same wall.)"""
    import gc

    import pyarrow.parquet as pq

    table = pq.read_table(path, memory_map=True)
    df = table.to_pandas(split_blocks=True, self_destruct=True)
    del table
    gc.collect()
    for c in df.select_dtypes(include="float64").columns:
        df[c] = df[c].astype("float32")
    gc.collect()
    return df


def time_split(df: pd.DataFrame, cutoff_quantile: float = 0.80):
    """Same 80/20 time-based split Phase 1 used: validation is strictly
    later in time than training, so it mirrors how the model is actually
    used (scoring transactions it has never seen, all of them later than
    everything it trained on) rather than a random split, which would let
    the model see the "future" during training."""
    cutoff_dt = df["TransactionDT"].quantile(cutoff_quantile)
    train_df = df[df["TransactionDT"] < cutoff_dt]
    val_df = df[df["TransactionDT"] >= cutoff_dt]
    return train_df, val_df


def prep_features(df: pd.DataFrame, drop_cols=None) -> pd.DataFrame:
    X = df.drop(columns=drop_cols if drop_cols is not None else DROP_COLS)
    # LightGBM handles categoricals natively if the dtype is "category" -- no
    # one-hot encoding needed, which matters here since P_emaildomain etc.
    # would otherwise explode into hundreds of columns.
    obj_cols = X.select_dtypes(include="object").columns
    X[obj_cols] = X[obj_cols].astype("category")
    return X
