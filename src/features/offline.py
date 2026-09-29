"""Batch, point-in-time-correct feature computation for training data.

Processes the full historical dataset as a chronological replay: for every
transaction, its engineered features are computed from that entity's state
strictly BEFORE this transaction. This is what keeps training features
honest about what would actually have been known at scoring time -- the
standard leakage mistake here is computing a rolling stat over a full group
(including future rows, or the row itself) rather than only strictly-prior
rows.

Implementation note: an earlier version of this used
`groupby(...).rolling(window, on="dt", closed="left")`, pandas' built-in
time-based rolling. That breaks on this data: IEEE-CIS has second-resolution
timestamps and some entities have multiple transactions in the same second,
so `on="dt"` produces a non-unique index and pandas refuses to reindex it
back onto the frame. Windows are computed manually instead, with
`np.searchsorted` per entity on the (already time-sorted) TransactionDT
array: `right = searchsorted(t, t[i], side="left")` finds the first row at
this exact timestamp (so any row sharing this transaction's own timestamp is
excluded from its own window, not just future rows), and
`left = searchsorted(t, t[i] - window, side="left")` finds the window's
start. `count = right - left`; the sum comes from a prefix-sum array the
same way. This also sidesteps the tie-breaking ambiguity of "closed='left'"
under duplicate timestamps by treating same-instant transactions as mutually
invisible to each other, which is the same practical assumption pandas'
version was making, just made explicit.

See src/features/store.py for the online counterpart used at serving time,
and src/features/definitions.py for the shared feature contract both must
satisfy.
"""
import numpy as np
import pandas as pd

from src.features.definitions import (
    ENTITY_KEY_COLS,
    FEATURE_COLUMNS,
    WINDOW_LONG_SECONDS,
    WINDOW_SHORT_SECONDS,
)
from src.features.entity import build_entity_id


def _changed_flag(current: pd.Series, previous: pd.Series, is_first: pd.Series) -> pd.Series:
    """1.0 if current != previous, 0.0 if equal, NaN if either side is
    missing/unknown (including the entity's first-ever transaction). "Both
    missing" is NaN, not "unchanged": if addr1 is missing on both this
    transaction and the last one, we don't actually know whether the address
    changed, so calling it unchanged would be overclaiming."""
    both_present = current.notna() & previous.notna()
    result = pd.Series(np.nan, index=current.index, dtype="float64")
    result[both_present] = (current[both_present] != previous[both_present]).astype(float)
    result[is_first] = np.nan
    return result


def _window_counts_and_sums(group: pd.DataFrame) -> pd.DataFrame:
    """Per-entity trailing-window count/sum, strictly excluding the current
    row and anything at or after its own timestamp. `group` must already be
    sorted by TransactionDT ascending (compute_features_batch guarantees
    this before grouping)."""
    t = group["TransactionDT"].to_numpy()
    amt = group["TransactionAmt"].to_numpy(dtype="float64")
    prefix_sum = np.concatenate(([0.0], np.cumsum(amt)))

    right = np.searchsorted(t, t, side="left")  # excludes this txn + any same-instant ties

    def window(seconds: int):
        left = np.searchsorted(t, t - seconds, side="left")
        count = (right - left).astype("float64")
        total = prefix_sum[right] - prefix_sum[left]
        return count, total

    count_1h, _ = window(WINDOW_SHORT_SECONDS)
    count_24h, sum_24h = window(WINDOW_LONG_SECONDS)
    mean_24h = np.where(count_24h > 0, sum_24h / np.where(count_24h == 0, 1, count_24h), np.nan)

    return pd.DataFrame(
        {
            "entity_txn_count_1h": count_1h,
            "entity_txn_count_24h": count_24h,
            "entity_amt_sum_24h": sum_24h,
            "entity_amt_mean_24h": mean_24h,
        },
        index=group.index,
    )


def compute_features_batch(df: pd.DataFrame, entity_col: str = "entity_id") -> pd.DataFrame:
    """Return a DataFrame aligned to df.index with columns = FEATURE_COLUMNS.

    df must contain: TransactionID, TransactionDT, TransactionAmt,
    DeviceInfo, and either an existing `entity_col` column or the raw
    ENTITY_KEY_COLS (entity_id is built automatically if missing).
    """
    required = {"TransactionID", "TransactionDT", "TransactionAmt", "DeviceInfo"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"compute_features_batch missing required columns: {missing_cols}")

    work = df[["TransactionID", "TransactionDT", "TransactionAmt", "DeviceInfo"]].copy()
    if entity_col in df.columns:
        work[entity_col] = df[entity_col]
    else:
        work[entity_col] = build_entity_id(df[ENTITY_KEY_COLS + ["TransactionID"]])

    # Sort by (entity, time) so every per-entity op below only looks
    # backward within that sorted order. _orig_idx carries df's original
    # index so results can be reindexed back to it at the end.
    work = work.reset_index(names="_orig_idx").sort_values([entity_col, "TransactionDT"]).reset_index(drop=True)

    grouped = work.groupby(entity_col, sort=False)

    work["entity_txn_seq_num"] = grouped.cumcount()
    is_first = work["entity_txn_seq_num"] == 0

    work["entity_time_since_last_sec"] = grouped["TransactionDT"].diff()

    window_feats = grouped.apply(_window_counts_and_sums, include_groups=False)
    window_feats = window_feats.reset_index(level=0, drop=True)
    work = work.join(window_feats)

    prev_device = grouped["DeviceInfo"].shift(1)
    work["entity_device_changed"] = _changed_flag(work["DeviceInfo"], prev_device, is_first)

    result = work.set_index("_orig_idx")[FEATURE_COLUMNS]
    return result.reindex(df.index)
