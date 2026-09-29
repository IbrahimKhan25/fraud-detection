"""Backfilled Redis state must be indistinguishable from state built by
replaying history online: features for later transactions match the offline
pipeline."""
import fakeredis
import numpy as np
import pandas as pd
import pytest

from src.features.backfill import backfill_state
from src.features.definitions import FEATURE_COLUMNS
from src.features.entity import build_entity_id
from src.features.offline import compute_features_batch
from src.features.store import RedisFeatureStore


def _synthetic(n=400, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "TransactionID": np.arange(n) + 1,
            "TransactionDT": np.sort(rng.integers(0, 400_000, n)).astype(float),
            "TransactionAmt": rng.gamma(2, 40, n).round(2),
            "card1": pd.Series(rng.choice([1000, 1001, 1002], n), dtype="int64"),
            "card2": 100.0, "card3": 150.0, "card5": 226.0, "addr1": 299.0, "addr2": 87.0,
            "DeviceInfo": rng.choice(["Windows", "iOS Device", None], n),
        }
    )
    df["entity_id"] = build_entity_id(df)
    return df


def _compare(df, cutoff):
    offline = compute_features_batch(df)
    client = fakeredis.FakeRedis(decode_responses=True)
    backfill_state(client, df, cutoff)
    store = RedisFeatureStore(client)
    later = df[df["TransactionDT"] >= cutoff].sort_values("TransactionDT")
    assert len(later) > 20
    for idx, row in later.iterrows():
        online = store.get_features_and_update(
            entity_id=row["entity_id"], txn_id=int(row["TransactionID"]), dt=float(row["TransactionDT"]),
            amt=float(row["TransactionAmt"]), device=None if pd.isna(row["DeviceInfo"]) else row["DeviceInfo"],
        )
        for col in FEATURE_COLUMNS:
            off, on = offline.loc[idx, col], online[col]
            assert (pd.isna(off) and pd.isna(on)) or abs(off - on) < 1e-6, (idx, col, off, on)


def test_backfilled_state_matches_offline_features_synthetic():
    df = _synthetic()
    _compare(df, cutoff=float(df["TransactionDT"].quantile(0.8)))


def test_cold_store_disagrees_with_offline_without_backfill():
    """The reason backfill exists: with an empty store the online features differ."""
    df = _synthetic()
    cutoff = float(df["TransactionDT"].quantile(0.8))
    offline = compute_features_batch(df)
    store = RedisFeatureStore(fakeredis.FakeRedis(decode_responses=True))
    later = df[df["TransactionDT"] >= cutoff].sort_values("TransactionDT")
    diffs = 0
    for idx, row in later.iterrows():
        on = store.get_features_and_update(row["entity_id"], int(row["TransactionID"]), float(row["TransactionDT"]),
                                           float(row["TransactionAmt"]), None if pd.isna(row["DeviceInfo"]) else row["DeviceInfo"])
        diffs += on["entity_txn_seq_num"] != offline.loc[idx, "entity_txn_seq_num"]
    assert diffs > 0.5 * len(later)


def test_singletons_are_skipped():
    df = _synthetic(50)
    df.loc[0, "entity_id"] = "UNK_1"
    client = fakeredis.FakeRedis(decode_responses=True)
    backfill_state(client, df, cutoff_dt=1e12)
    assert client.exists("entity:UNK_1:state") == 0


def test_backfilled_state_matches_offline_features_real_data():
    from tests.test_feature_parity import DATA_PATH, _load_sample

    try:
        df = _load_sample(n_entities=60)
    except FileNotFoundError:
        pytest.skip(f"{DATA_PATH} not present")
    _compare(df, cutoff=float(df["TransactionDT"].quantile(0.7)))
