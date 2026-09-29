"""Proves the offline (training) and online (serving) feature pipelines
agree, transaction by transaction, on real data and on synthetic data built
to hit the edge cases (the synthetic test is the one CI runs, since the
IEEE-CIS data is not in the repo). This is the actual
guarantee behind "training and serving share the same feature logic" --
not a comment claiming it, a test checking it.

Uses fakeredis (an in-memory redis-py-compatible server) instead of a real
Redis instance so this test has no external dependency and runs anywhere.
"""
import math

import fakeredis
import numpy as np
import pandas as pd
import pytest

from src.features.definitions import ENTITY_KEY_COLS, FEATURE_COLUMNS
from src.features.entity import build_entity_id
from src.features.offline import compute_features_batch
from src.features.store import RedisFeatureStore

DATA_PATH = "data/raw/train_joined.parquet"


def _load_sample(n_entities: int = 60, seed: int = 0) -> pd.DataFrame:
    """A chronologically-sorted sample covering entities with enough
    transactions to exercise the rolling windows, plus a few singleton
    (missing-key-field) rows to exercise that edge case too."""
    cols = list(
        dict.fromkeys(
            ["TransactionID", "TransactionDT", "TransactionAmt", "addr1", "addr2", "DeviceInfo"]
            + ENTITY_KEY_COLS
        )
    )
    df = pd.read_parquet(DATA_PATH, columns=cols)
    df["entity_id"] = build_entity_id(df)

    counts = df["entity_id"].value_counts()
    busy_entities = counts[counts >= 5].sample(n_entities, random_state=seed).index
    sample = df[df["entity_id"].isin(busy_entities)]

    singleton_sample = df[df["entity_id"].str.startswith("UNK_")].sample(20, random_state=seed)

    combined = pd.concat([sample, singleton_sample]).sort_values("TransactionDT").reset_index(drop=True)
    return combined


@pytest.fixture(scope="module")
def sample_df():
    try:
        return _load_sample()
    except FileNotFoundError:
        pytest.skip("data/raw/train_joined.parquet not present (run src/training/train_baseline.py first)")


def _synthetic_sample(n_entities: int = 40, seed: int = 0) -> pd.DataFrame:
    """Transactions shaped like the real data's key columns. Gaps are drawn
    from seconds to days so rows fall on both sides of the 1h and 24h window
    edges, some entities repeat a timestamp (same-second ties), devices are
    sometimes missing or change, and some rows miss a key field so they get
    singleton IDs."""
    rng = np.random.default_rng(seed)
    gap_choices = np.array([0, 1, 30, 600, 3_599, 3_600, 3_601, 7_200, 86_399, 86_400, 86_401, 200_000])
    rows = []
    for e in range(n_entities):
        n = int(rng.integers(2, 25))
        t = float(rng.integers(0, 1_000_000))
        devices = rng.choice(["Windows", "iOS Device", "MacOS", None], size=2)
        for _ in range(n):
            t += float(rng.choice(gap_choices))
            rows.append({
                "TransactionDT": t,
                "TransactionAmt": round(float(rng.gamma(2, 40)), 2),
                "card1": 10_000 + e, "card2": 100.0 + e % 3, "card3": 150.0, "card5": 226.0,
                "addr1": 299.0, "addr2": 87.0,
                "DeviceInfo": devices[int(rng.random() < 0.3)],
            })
    df = pd.DataFrame(rows)
    df["card1"] = df["card1"].astype("int64")
    missing = rng.random(len(df)) < 0.05
    df.loc[missing, "card2"] = np.nan
    df = df.sort_values("TransactionDT", kind="stable").reset_index(drop=True)
    df["TransactionID"] = np.arange(len(df)) + 1
    df["entity_id"] = build_entity_id(df)
    return df


def _assert_offline_online_agree(sample_df: pd.DataFrame) -> None:
    offline_feats = compute_features_batch(sample_df)

    fake_client = fakeredis.FakeRedis(decode_responses=True)
    store = RedisFeatureStore(fake_client)

    # Online path processes transactions strictly in time order, exactly as
    # a serving system would receive them one at a time. Same-second ties are
    # broken by TransactionID, the order the offline pipeline sees them in:
    # seq_num, time_since_last and device_changed depend on arrival order
    # within a tie (the window features do not, ties are mutually invisible).
    ordered = sample_df.sort_values(["TransactionDT", "TransactionID"])
    online_rows = {}
    for idx, row in ordered.iterrows():
        online_rows[idx] = store.get_features_and_update(
            entity_id=row["entity_id"],
            txn_id=row["TransactionID"],
            dt=float(row["TransactionDT"]),
            amt=float(row["TransactionAmt"]),
            device=None if pd.isna(row["DeviceInfo"]) else row["DeviceInfo"],
        )

    online_feats = pd.DataFrame.from_dict(online_rows, orient="index")[FEATURE_COLUMNS]
    online_feats = online_feats.reindex(offline_feats.index)

    mismatches = []
    for col in FEATURE_COLUMNS:
        off = offline_feats[col].to_numpy(dtype="float64")
        on = online_feats[col].to_numpy(dtype="float64")
        both_nan = pd.isna(off) & pd.isna(on)
        close = both_nan | (pd.notna(off) & pd.notna(on) & (abs(off - on) < 1e-6))
        if not close.all():
            bad_idx = offline_feats.index[~close]
            mismatches.append((col, len(bad_idx), bad_idx[:5].tolist()))

    assert not mismatches, f"offline/online feature mismatch: {mismatches}"


def test_offline_and_online_agree_on_real_data(sample_df):
    _assert_offline_online_agree(sample_df)


def test_offline_and_online_agree_on_synthetic_data():
    df = _synthetic_sample()
    # Guard the generator itself: the edge cases must actually be present.
    feats = compute_features_batch(df)
    assert df["entity_id"].str.startswith("UNK_").any()
    assert df.duplicated(["entity_id", "TransactionDT"]).any()
    assert (feats["entity_txn_count_1h"] > 0).any() and (feats["entity_txn_count_24h"] > feats["entity_txn_count_1h"]).any()
    assert set(feats["entity_device_changed"].dropna().unique()) == {0.0, 1.0}
    _assert_offline_online_agree(df)


def test_entity_id_missing_key_fields_get_singleton_ids():
    df = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "card1": [100, 100, 200],
            "card2": [1.0, None, 1.0],
            "card3": [1.0, 1.0, 1.0],
            "card5": [1.0, 1.0, 1.0],
            "addr1": [1.0, 1.0, 1.0],
            "addr2": [1.0, 1.0, 1.0],
        }
    )
    eid = build_entity_id(df)
    assert eid.iloc[0] != eid.iloc[1]  # row 2's missing card2 must not collapse it onto row 1
    assert eid.iloc[1] == "UNK_2"
    assert eid.iloc[0] == eid.iloc[0]  # sanity
