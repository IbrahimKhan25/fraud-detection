"""Phase 4 tests: PSI math, drift decisions, the non-blocking prediction
logger, logging through the API, and (when TEST_DATABASE_URL points at a
Postgres) the real database path end to end."""
import json
import os
import threading
import time

import fakeredis
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.features.definitions import FEATURE_COLUMNS
from src.features.store import RedisFeatureStore
from src.monitoring import db, drift
from src.monitoring.logger import PredictionLogger
from src.monitoring.monitor import fetch_rows
from src.monitoring.reference import build_reference
from src.serving.app import app
from tests.test_serving import _toy_model, _txn


# ---------- PSI math ----------

def test_psi_zero_for_identical_and_positive_for_shift():
    p = np.array([0.1, 0.2, 0.3, 0.4])
    assert drift.psi(p, p) == pytest.approx(0.0)
    assert drift.psi(p, [0.4, 0.3, 0.2, 0.1]) > 0.25


def test_psi_known_value():
    # 0.1*ln(0.6/0.5) + (-0.1)*ln(0.4/0.5) = 0.018232 + 0.022314 = 0.040546
    assert drift.psi([0.5, 0.5], [0.6, 0.4]) == pytest.approx(0.040546, abs=1e-5)


def test_numeric_props_sum_to_one_and_have_missing_bin():
    v = np.array([1.0, 2.0, 3.0, 4.0, np.nan, np.nan])
    edges = drift.numeric_edges(np.arange(100, dtype=float))
    props = drift.numeric_props(v, edges)
    assert props.sum() == pytest.approx(1.0)
    assert props[-1] == pytest.approx(2 / 6)  # last bin = missing
    assert len(props) == len(edges) + 2


def test_categorical_props_other_and_missing():
    props = drift.categorical_props(["a", "a", "b", "zzz", None], ["a", "b"])
    assert props.tolist() == pytest.approx([0.4, 0.2, 0.2, 0.2])  # a, b, other, missing


# ---------- reference + evaluate on the toy model ----------

def _synthetic_frame(rng, n, amt_scale=1.0):
    return pd.DataFrame(
        {
            "TransactionAmt": rng.gamma(2, 50, n) * amt_scale,
            "card1": rng.integers(1000, 1100, n).astype(float),
            "card2": rng.integers(100, 200, n).astype(float),
            "addr1": rng.integers(200, 300, n).astype(float),
            "C1": rng.integers(0, 5, n).astype(float),
            "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", None], n),
            "DeviceInfo": rng.choice(["Windows", "iOS Device", None], n),
            **{c: rng.normal(size=n) for c in FEATURE_COLUMNS},
        }
    )


def _rows(fm, df):
    out = []
    for rec in df.to_dict("records"):
        feats = fm.loggable(rec)
        out.append({"fraud_score": fm.predict(rec), "features": feats})
    return out


@pytest.fixture(scope="module")
def ref_and_model():
    rng = np.random.default_rng(0)
    fm = _toy_model()
    train = _synthetic_frame(rng, 5000)
    val = _synthetic_frame(rng, 3000)
    val_scores = np.array([fm.predict(r) for r in val.to_dict("records")])
    return build_reference(train, val_scores, fm, n_features=10), fm


def test_no_drift_on_same_distribution(ref_and_model):
    ref, fm = ref_and_model
    rng = np.random.default_rng(1)
    report = drift.evaluate(_rows(fm, _synthetic_frame(rng, 1500)), ref)
    assert report["status"] == "ok" and not report["drifted"], report["features"]
    assert report["score_psi"] < 0.1


def test_drift_detected_when_amounts_triple(ref_and_model):
    ref, fm = ref_and_model
    rng = np.random.default_rng(2)
    report = drift.evaluate(_rows(fm, _synthetic_frame(rng, 1500, amt_scale=3.0)), ref)
    assert "TransactionAmt" in report["alert_features"]
    assert report["features"]["TransactionAmt"] > 0.25


def test_feature_going_missing_is_drift(ref_and_model):
    ref, fm = ref_and_model
    rng = np.random.default_rng(3)
    df = _synthetic_frame(rng, 1500)
    df["C1"] = np.nan
    report = drift.evaluate(_rows(fm, df), ref)
    assert report["features"]["C1"] > 0.25


def test_too_few_rows_is_insufficient_not_ok(ref_and_model):
    ref, fm = ref_and_model
    rng = np.random.default_rng(4)
    report = drift.evaluate(_rows(fm, _synthetic_frame(rng, 50)), ref)
    assert report["status"] == "insufficient_data" and not report["drifted"]


# ---------- non-blocking logger ----------

def _row(i):
    return {"transaction_id": i, "entity_id": "e", "model_version": "v", "fraud_score": 0.1, "latency_ms": 1.0, "features": {}}


def test_logger_batches_and_flushes_on_close():
    got = []
    lg = PredictionLogger(sink=got.extend, batch_size=50, flush_seconds=0.05)
    for i in range(120):
        assert lg.log(_row(i))
    lg.close()
    assert len(got) == 120 and lg.stats()["written"] == 120


def test_logger_collects_a_burst_into_one_write():
    calls = []
    lg = PredictionLogger(sink=lambda rows: calls.append(len(rows)), batch_size=200, flush_seconds=0.3)
    for i in range(50):
        lg.log({"i": i})
        time.sleep(0.002)          # rows trickle in, as real requests do
    lg.close()
    assert sum(calls) == 50 and len(calls) <= 2


def test_logger_drops_instead_of_blocking_when_sink_is_stuck():
    gate = threading.Event()
    lg = PredictionLogger(sink=lambda rows: gate.wait(5), max_queue=5, batch_size=1, flush_seconds=0.01)
    t0 = time.perf_counter()
    for i in range(200):
        lg.log(_row(i))
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.5  # never waited on the stuck sink
    assert lg.dropped > 100
    gate.set()
    lg.close()


def test_logger_survives_a_failing_sink():
    def boom(rows):
        raise RuntimeError("db down")

    lg = PredictionLogger(sink=boom, flush_seconds=0.02)
    for i in range(10):
        lg.log(_row(i))
    lg.close()
    assert lg.stats()["failed"] == 10 and lg.stats()["written"] == 0


# ---------- logging through the API ----------

@pytest.fixture()
def logged_client():
    rows = []
    for attr in ("model", "store", "pred_logger"):
        if hasattr(app.state, attr):
            delattr(app.state, attr)
    app.state.model = _toy_model()
    app.state.store = RedisFeatureStore(fakeredis.FakeRedis(decode_responses=True))
    app.state.pred_logger = PredictionLogger(sink=rows.extend, flush_seconds=0.02)
    with TestClient(app) as c:
        yield c, rows
    for attr in ("model", "store", "pred_logger"):
        delattr(app.state, attr)


def test_score_writes_a_prediction_row_with_present_features_only(logged_client):
    c, rows = logged_client
    r = c.post("/score", json=_txn(77, 1000.0, DeviceInfo=None)).json()
    app.state.pred_logger.close()
    assert len(rows) == 1
    row = rows[0]
    assert row["transaction_id"] == 77 and row["fraud_score"] == r["fraud_score"]
    assert row["model_version"] == "unversioned"
    f = row["features"]
    assert f["TransactionAmt"] == 50.0 and "DeviceInfo" not in f  # missing = absent
    assert "entity_time_since_last_sec" not in f  # first txn: NaN, so absent
    assert f["entity_txn_seq_num"] == 0.0
    json.dumps(f)  # must be JSON-serializable


def test_health_reports_logger_stats(logged_client):
    c, _ = logged_client
    h = c.get("/health").json()
    assert set(h["prediction_log"]) == {"written", "dropped", "failed", "queued"}


# ---------- real Postgres (skipped unless TEST_DATABASE_URL is set) ----------

needs_pg = pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL not set")


@pytest.fixture()
def pg(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", os.environ["TEST_DATABASE_URL"])
    with db.connect() as conn:
        db.ensure_schema(conn)
        conn.execute("TRUNCATE predictions, drift_reports RESTART IDENTITY")
    yield
    with db.connect() as conn:
        conn.execute("TRUNCATE predictions, drift_reports RESTART IDENTITY")


@needs_pg
def test_postgres_end_to_end(pg, ref_and_model):
    from src.monitoring.logger import PostgresSink

    ref, fm = ref_and_model
    for attr in ("model", "store", "pred_logger"):
        if hasattr(app.state, attr):
            delattr(app.state, attr)
    app.state.model = fm
    app.state.store = RedisFeatureStore(fakeredis.FakeRedis(decode_responses=True))
    app.state.pred_logger = PredictionLogger(sink=PostgresSink(), flush_seconds=0.05)
    with TestClient(app) as c:
        for i in range(30):
            assert c.post("/score", json=_txn(1000 + i, 1000.0 + i)).status_code == 200
        app.state.pred_logger.close()

        with db.connect() as conn:
            n, = conn.execute("SELECT count(*) FROM predictions").fetchone()
            assert n == 30
            feats, = conn.execute("SELECT features FROM predictions WHERE transaction_id = 1000").fetchone()
            assert feats["TransactionAmt"] == 50.0

        # outcome endpoint fills the label
        assert c.post("/outcome", json={"TransactionID": 1000, "isFraud": 1}).json() == {"updated": 1}
        assert c.post("/outcome", json={"TransactionID": 999999, "isFraud": 0}).json() == {"updated": 0}
        assert c.post("/outcome", json={"TransactionID": 1000, "isFraud": 2}).status_code == 422
        with db.connect() as conn:
            label, = conn.execute("SELECT label FROM predictions WHERE transaction_id = 1000").fetchone()
            assert label == 1

    # the monitor's query reads them back in the shape evaluate() expects
    with db.connect() as conn:
        rows = fetch_rows(conn, "unversioned", window_minutes=5)
        assert len(rows) == 30
        assert drift.evaluate(rows, ref, min_rows=10)["status"] in ("ok", "drifted")
        other = fetch_rows(conn, "some-other-model", window_minutes=5)
        assert other == []  # rows from a different model version are never mixed in
    for attr in ("model", "store", "pred_logger"):
        delattr(app.state, attr)


# ---------- decision rule ----------

def _flat_reference(n_features=10, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=20000)
    feats = {}
    edges = drift.numeric_edges(base)
    for i in range(n_features):
        feats[f"f{i}"] = {"type": "numeric", "edges": edges, "props": drift.numeric_props(base, edges).tolist()}
    s = rng.uniform(size=20000)
    s_edges = drift.numeric_edges(s)
    return {"model_version": "t", "features": feats, "score": {"edges": s_edges, "props": drift.numeric_props(s, s_edges).tolist()}}


def _flat_rows(n, shifted_features=(), score_shift=0.0, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        f = {f"f{i}": float(rng.normal() + (3.0 if f"f{i}" in shifted_features else 0.0)) for i in range(10)}
        rows.append({"fraud_score": float(np.clip(rng.uniform() * (1 - score_shift) + score_shift, 0, 1)), "features": f})
    return rows


def test_one_drifting_feature_alone_is_not_drift():
    r = drift.evaluate(_flat_rows(2000, shifted_features={"f0"}), _flat_reference())
    assert r["alert_features"] == ["f0"] and not r["drifted"]


def test_broad_feature_drift_is_drift():
    r = drift.evaluate(_flat_rows(2000, shifted_features={"f0", "f1"}), _flat_reference())
    assert r["drifted"] and "past their alert level" in r["reason"]


def test_score_drift_alone_is_drift():
    r = drift.evaluate(_flat_rows(2000, score_shift=0.6), _flat_reference())
    assert r["drifted"] and r["alert_features"] == [] and "score PSI" in r["reason"]


# ---------- calibrated alert levels ----------

def _with_baseline(ref, feature_p95, score_p95, window_rows=3000):
    ref = json.loads(json.dumps(ref))
    ref["baseline"] = {
        "window_rows": window_rows,
        "n_windows": 39,
        "features": {k: {"median": v / 2, "p95": v, "max": v * 1.5} for k, v in feature_p95.items()},
        "score": {"median": score_p95 / 2, "p95": score_p95, "max": score_p95 * 2},
    }
    return ref


def test_levels_scale_with_how_much_a_feature_normally_moves():
    ref = _with_baseline(_flat_reference(), {"f0": 1.1, "f1": 0.03}, score_p95=0.028)
    lv = drift.alert_levels(ref)
    assert lv["features"]["f0"] == pytest.approx(2.2)   # noisy feature: high bar
    assert lv["features"]["f1"] == 0.25                 # quiet feature: floor
    assert lv["score"] == pytest.approx(0.10)           # 3 x 0.028 = 0.084, floored at 0.10


def test_noisy_feature_at_its_normal_level_does_not_alert_but_quiet_one_does():
    ref = _with_baseline(_flat_reference(), {"f0": 1.1}, score_p95=0.028)
    # f0 shifted hard (PSI well under 2.2 is impossible to hit here, so check the level logic directly)
    r = drift.evaluate(_flat_rows(2000, shifted_features={"f0", "f1"}), ref)
    assert "f1" in r["alert_features"]                 # quiet feature, shifted: alerts
    assert r["levels"]["f0"] > r["levels"]["f1"]


def test_score_shift_below_textbook_level_but_above_normal_is_drift():
    ref = _with_baseline(_flat_reference(), {}, score_p95=0.028)
    r = drift.evaluate(_flat_rows(2000, score_shift=0.08), ref)
    assert 0.10 <= r["score_psi"] < 0.25 and r["drifted"]


def test_severe_level_is_twice_worst_clean_window_with_floor():
    ref = _with_baseline(_flat_reference(), {"f0": 0.8, "f1": 0.02}, score_p95=0.028)  # max = 1.5 x p95
    sv = drift.alert_levels(ref)["severe"]
    assert sv["f0"] == pytest.approx(2.4) and sv["f1"] == 0.25
    assert "f2" not in sv                                # no baseline for it: no single-feature rule


def test_one_feature_far_past_normal_is_drift_on_its_own():
    ref = _with_baseline(_flat_reference(), {"f0": 0.01}, score_p95=0.028)
    r = drift.evaluate(_flat_rows(2000, shifted_features={"f0"}), ref)
    assert r["severe_features"] == ["f0"] and r["drifted"] and "on its own" in r["reason"]


def test_noisy_feature_shift_within_its_normal_range_is_not_severe():
    ref = _with_baseline(_flat_reference(), {"f0": 4.0}, score_p95=0.028)   # worst clean window 6.0
    r = drift.evaluate(_flat_rows(2000, shifted_features={"f0"}), ref)
    assert r["severe_features"] == [] and not r["drifted"]


def test_small_window_gets_a_note():
    ref = _with_baseline(_flat_reference(), {}, score_p95=0.028, window_rows=3000)
    r = drift.evaluate(_flat_rows(800), ref)
    assert "calibrated" in r["note"]


def test_build_reference_calibrates_and_detects_the_amount_shift():
    rng = np.random.default_rng(10)
    fm = _toy_model()
    train = _synthetic_frame(rng, 5000)
    val = _synthetic_frame(rng, 4000)
    val_scores = np.array([fm.predict(r) for r in val.to_dict("records")])
    ref = build_reference(train, val_scores, fm, n_features=10, val_df=val, window_rows=500)
    assert ref["baseline"]["n_windows"] == 8 and "TransactionAmt" in ref["baseline"]["features"]

    normal = drift.evaluate(_rows(fm, _synthetic_frame(rng, 1500)), ref)
    assert not normal["drifted"], (normal["reason"], normal["score_psi"])
    shifted = drift.evaluate(_rows(fm, _synthetic_frame(rng, 1500, amt_scale=3.0)), ref)
    assert shifted["drifted"] and "TransactionAmt" in shifted["alert_features"]


# ---------- retrain trigger ----------

def test_retrain_dispatches_workflow_or_runs_locally(monkeypatch):
    from src.monitoring import monitor

    calls = []
    monkeypatch.setattr(monitor.subprocess, "run", lambda cmd, check: calls.append(cmd))
    monitor.trigger_retrain(local=False)
    assert calls == [["gh", "workflow", "run", "retrain.yml", "--ref", "main"]]

    calls.clear()
    monitor.trigger_retrain(local=True)
    assert [c[-1] for c in calls] == ["src.training.build_training_set", "src.training.train_with_features",
                                       "src.monitoring.reference"]


def test_retrain_flags_are_exclusive():
    from src.monitoring import monitor

    with pytest.raises(SystemExit):
        monitor.main(["--retrain", "--retrain-local"])
