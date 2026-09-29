"""Proves the offline (training) and online (serving) feature pipelines
agree, transaction by transaction, on real data. This is the actual
guarantee behind "training and serving share the same feature logic" --
not a comment claiming it, a test checking it.

Uses fakeredis (an in-memory redis-py-compatible server) instead of a real
Redis instance so this test has no external dependency and runs anywhere.
"""
import math

import fakeredis
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


def test_offline_and_online_agree_on_real_data(sample_df):
    offline_feats = compute_features_batch(sample_df)

    fake_client = fakeredis.FakeRedis(decode_responses=True)
    store = RedisFeatureStore(fake_client)

    # Online path processes transactions strictly in time order, exactly as
    # a serving system would receive them one at a time.
    ordered = sample_df.sort_values("TransactionDT")
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
