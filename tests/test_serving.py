"""Serving tests. Use a tiny LightGBM model trained on synthetic data with
the same shape of problem as the real one (numeric + string categorical
columns + the 8 engineered columns), and fakeredis, so nothing here needs the
real dataset, a trained pickle, or a Redis server.
"""
import fakeredis
import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from lightgbm import LGBMClassifier

from src.features.definitions import ENTITY_KEY_COLS, FEATURE_COLUMNS
from src.features.entity import build_entity_id
from src.features.offline import compute_features_batch
from src.features.store import RedisFeatureStore
from src.serving.app import app, entity_id_for
from src.serving.model import FraudModel
from src.serving.schemas import Transaction

RAW_COLS = ["TransactionAmt", "card1", "card2", "addr1", "C1", "P_emaildomain", "DeviceInfo"]


def _toy_model() -> FraudModel:
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame(
        {
            "TransactionAmt": rng.gamma(2, 50, n),
            "card1": rng.integers(1000, 1100, n).astype(float),
            "card2": rng.integers(100, 200, n).astype(float),
            "addr1": rng.integers(200, 300, n).astype(float),
            "C1": rng.integers(0, 5, n).astype(float),
            "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", None], n),
            "DeviceInfo": rng.choice(["Windows", "iOS Device", None], n),
        }
    )
    for c in FEATURE_COLUMNS:
        df[c] = rng.normal(size=n)
    y = (df["TransactionAmt"] + rng.normal(0, 30, n) > 130).astype(int)
    X = df.copy()
    for c in ["P_emaildomain", "DeviceInfo"]:
        X[c] = X[c].astype("category")  # same as prep_features
    model = LGBMClassifier(n_estimators=20, num_leaves=7, verbose=-1, random_state=0).fit(X, y)
    return FraudModel(model)


@pytest.fixture()
def client():
    if hasattr(app.state, "model"):
        del app.state.model
    app.state.model = _toy_model()
    app.state.store = RedisFeatureStore(fakeredis.FakeRedis(decode_responses=True))
    with TestClient(app) as c:
        yield c
    del app.state.model
    del app.state.store


def _txn(i, dt, **over):
    body = {
        "TransactionID": i, "TransactionDT": dt, "TransactionAmt": 50.0,
        "card1": 1050, "card2": 150.0, "card3": 150.0, "card5": 226.0,
        "addr1": 299.0, "addr2": 87.0, "DeviceInfo": "Windows",
        "features": {"C1": 1, "P_emaildomain": "gmail.com"},
    }
    body.update(over)
    return body


def test_model_detects_categoricals_from_the_pickle():
    m = _toy_model()
    assert m.categorical_cols == {"P_emaildomain", "DeviceInfo"}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_score_returns_probability_and_updates_entity_state(client):
    r1 = client.post("/score", json=_txn(1, 1000.0)).json()
    assert 0.0 <= r1["fraud_score"] <= 1.0
    assert r1["engineered_features"]["entity_txn_seq_num"] == 0
    assert r1["engineered_features"]["entity_time_since_last_sec"] is None

    r2 = client.post("/score", json=_txn(2, 1600.0)).json()
    assert r2["entity_id"] == r1["entity_id"]
    f = r2["engineered_features"]
    assert f["entity_txn_seq_num"] == 1
    assert f["entity_time_since_last_sec"] == 600.0
    assert f["entity_txn_count_1h"] == 1.0
    assert f["entity_amt_sum_24h"] == 50.0
    assert f["entity_device_changed"] == 0.0


def test_unseen_category_and_missing_fields_do_not_crash(client):
    r = client.post("/score", json=_txn(3, 10.0, DeviceInfo=None, features={"P_emaildomain": "never-seen.io"}))
    assert r.status_code == 200


def test_missing_key_field_gets_singleton_entity(client):
    a = client.post("/score", json=_txn(10, 10.0, card2=None)).json()
    b = client.post("/score", json=_txn(11, 20.0, card2=None)).json()
    assert a["entity_id"] == "UNK_10" and b["entity_id"] == "UNK_11"


def test_bad_request_is_422(client):
    assert client.post("/score", json={"TransactionID": 1}).status_code == 422


def test_entity_id_matches_training_path_formatting():
    # Training frame dtypes as they come out of the raw CSV: card1 int64, rest float64.
    train = pd.DataFrame(
        {
            "TransactionID": [2987000],
            "card1": pd.Series([13926], dtype="int64"),
            "card2": [np.nan if False else 100.0],
            "card3": [150.0], "card5": [142.0], "addr1": [315.0], "addr2": [87.0],
        }
    )
    expected = build_entity_id(train).iloc[0]
    txn = Transaction(TransactionID=2987000, TransactionDT=0, TransactionAmt=1, card1=13926,
                      card2=100.0, card3=150.0, card5=142.0, addr1=315.0, addr2=87.0)
    assert entity_id_for(txn) == expected == "13926_100.0_150.0_142.0_315.0_87.0"
    # a JSON client sending 13926.0 for card1 must land on the same entity
    txn2 = Transaction.model_validate({**txn.model_dump(), "card1": 13926.0})
    assert entity_id_for(txn2) == expected


def test_endpoint_features_match_offline_pipeline(client):
    rng = np.random.default_rng(1)
    n = 40
    df = pd.DataFrame(
        {
            "TransactionID": np.arange(n) + 1,
            "TransactionDT": np.sort(rng.integers(0, 200_000, n)).astype(float),
            "TransactionAmt": rng.gamma(2, 40, n).round(2),
            "card1": pd.Series(rng.choice([1000, 1001], n), dtype="int64"),
            "card2": 100.0, "card3": 150.0, "card5": 226.0,
            "addr1": rng.choice([299.0, 300.0], n), "addr2": 87.0,
            "DeviceInfo": rng.choice(["Windows", "iOS Device", None], n),
        }
    )
    df["entity_id"] = build_entity_id(df)
    offline = compute_features_batch(df)

    for i, row in df.iterrows():
        body = {
            "TransactionID": int(row.TransactionID), "TransactionDT": float(row.TransactionDT),
            "TransactionAmt": float(row.TransactionAmt), "card1": int(row.card1), "card2": row.card2,
            "card3": row.card3, "card5": row.card5, "addr1": row.addr1, "addr2": row.addr2,
            "DeviceInfo": None if pd.isna(row.DeviceInfo) else row.DeviceInfo,
        }
        online = client.post("/score", json=body).json()["engineered_features"]
        for col in FEATURE_COLUMNS:
            off = offline.loc[i, col]
            on = online[col]
            assert (pd.isna(off) and on is None) or abs(off - on) < 1e-6, (i, col, off, on)


def test_fast_predict_matches_pandas_path_after_pickle_roundtrip(tmp_path):
    """Trains with early stopping (so best_iteration_ < n_estimators), pickles
    and reloads like production, then checks the NumPy path equals the
    pandas path, including unseen and missing categories."""
    rng = np.random.default_rng(3)
    n = 800
    X = pd.DataFrame(
        {
            "TransactionAmt": rng.gamma(2, 50, n),
            "card1": rng.integers(1000, 1100, n).astype(float),
            "C1": rng.integers(0, 5, n).astype(float),
            "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", "hotmail.com", None], n),
            "DeviceInfo": rng.choice(["Windows", "iOS Device", None], n),
        }
    )
    y = ((X["TransactionAmt"] > 110) ^ (X["P_emaildomain"] == "yahoo.com")).astype(int)
    for c in ["P_emaildomain", "DeviceInfo"]:
        X[c] = X[c].astype("category")
    from lightgbm import early_stopping
    clf = LGBMClassifier(n_estimators=300, num_leaves=7, learning_rate=0.3, verbose=-1, random_state=0)
    clf.fit(X[:600], y[:600], eval_set=[(X[600:], y[600:])], callbacks=[early_stopping(10, verbose=False)])
    assert clf.best_iteration_ < 300
    path = tmp_path / "m.pkl"
    joblib.dump(clf, path)
    m = FraudModel.load(path)

    cases = [
        {"TransactionAmt": 40.0, "card1": 1010, "C1": 2, "P_emaildomain": "yahoo.com", "DeviceInfo": "Windows"},
        {"TransactionAmt": 300.0, "card1": 1090, "C1": 0, "P_emaildomain": "gmail.com", "DeviceInfo": None},
        {"TransactionAmt": 120.0, "P_emaildomain": "never-seen.io", "DeviceInfo": "iOS Device"},
        {},
        {"TransactionAmt": "not-a-number", "P_emaildomain": float("nan")},
    ]
    for c in cases:
        assert abs(m.predict(c) - m.predict_reference(c)) < 1e-9, c


def test_entity_id_fast_path_matches_build_entity_id_on_random_rows():
    rng = np.random.default_rng(5)
    n = 300
    df = pd.DataFrame(
        {
            "TransactionID": np.arange(n) + 2987000,
            "card1": pd.Series(rng.integers(1000, 18000, n), dtype="int64"),
            "card2": rng.choice([100.0, 321.0, 555.5, np.nan], n),
            "card3": rng.choice([150.0, 185.0, np.nan], n),
            "card5": rng.choice([142.0, 226.0, np.nan], n),
            "addr1": rng.choice([315.0, 299.0, np.nan], n),
            "addr2": rng.choice([87.0, 60.0, np.nan], n),
        }
    )
    expected = build_entity_id(df)
    for i, row in df.iterrows():
        kw = {c: (None if pd.isna(row[c]) else row[c]) for c in ENTITY_KEY_COLS}
        kw["card1"] = int(row["card1"])
        txn = Transaction(TransactionID=int(row.TransactionID), TransactionDT=0, TransactionAmt=1, **kw)
        assert entity_id_for(txn) == expected.iloc[i]


def test_out_of_order_arrival_gives_nan_gap_and_keeps_latest_time(client):
    client.post("/score", json=_txn(20, 1600.0))
    late = client.post("/score", json=_txn(21, 1000.0, DeviceInfo="iOS Device")).json()["engineered_features"]
    assert late["entity_time_since_last_sec"] is None  # not -600
    assert late["entity_txn_seq_num"] == 1
    # "latest seen" stayed at 1600, so the next in-order txn measures from it
    nxt = client.post("/score", json=_txn(22, 2200.0)).json()["engineered_features"]
    assert nxt["entity_time_since_last_sec"] == 600.0
    # the late txn's device did not overwrite the latest one (Windows)
    assert nxt["entity_device_changed"] == 0.0


def test_same_timestamp_gap_is_zero_like_offline(client):
    client.post("/score", json=_txn(30, 500.0))
    f = client.post("/score", json=_txn(31, 500.0)).json()["engineered_features"]
    assert f["entity_time_since_last_sec"] == 0.0
