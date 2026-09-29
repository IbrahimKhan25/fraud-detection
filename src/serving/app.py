"""Phase 3: FastAPI scoring service.

POST /score  takes one transaction, looks up the entity's state in Redis,
computes the 8 engineered features (same RedisFeatureStore the parity test
covers), scores with the Phase 2 LightGBM model, then records the transaction
into the entity's state. Order matters: features are computed from state
strictly before this transaction, then state is updated.

Config via environment variables:
    MODEL_PATH  default models/phase2_lgbm.pkl
    REDIS_URL   default redis://localhost:6379/0

Run locally (repo root, Redis up):
    uvicorn src.serving.app:app --port 8000
"""
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import redis
from fastapi import FastAPI, HTTPException

from src.features.definitions import ENTITY_KEY_COLS, FEATURE_COLUMNS
from src.features.store import RedisFeatureStore
from src.serving.model import FraudModel
from src.monitoring import db
from src.monitoring.logger import PostgresSink, PredictionLogger
from src.serving.schemas import Outcome, ScoreResponse, Transaction

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = ROOT / "models" / "phase2_lgbm.pkl"


def entity_id_for(txn: Transaction) -> str:
    """Same proxy entity as build_entity_id() in src/features/entity.py, done
    with plain string formatting because a one-row pandas frame cost ~1.5 ms
    per request. Must stay identical to it: any missing key field gives a
    singleton "UNK_<TransactionID>", card1 formats as an int ("13926") and
    the rest as floats ("100.0"), as in the raw training frame.
    tests/test_serving.py compares the two on randomized rows."""
    parts = []
    for c in ENTITY_KEY_COLS:
        v = getattr(txn, c)
        if v is None or v != v:
            return f"UNK_{txn.TransactionID}"
        parts.append(str(int(v)) if c == "card1" else str(float(v)))
    return "_".join(parts)


def _nan_to_none(x: float):
    return None if x != x else float(x)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Tests can preload app.state.model / app.state.store before startup.
    if not hasattr(app.state, "model"):
        app.state.model = FraudModel.load(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
    if not hasattr(app.state, "store"):
        client = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True)
        app.state.store = RedisFeatureStore(client)
    # Prediction logging is on only when DATABASE_URL is set.
    if not hasattr(app.state, "pred_logger"):
        app.state.pred_logger = PredictionLogger(PostgresSink()) if os.getenv("DATABASE_URL") else None
    yield
    if app.state.pred_logger is not None:
        app.state.pred_logger.close()


app = FastAPI(title="Fraud scoring", lifespan=lifespan)


@app.get("/health")
def health():
    try:
        app.state.store.client.ping()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"redis unavailable: {e}")
    out = {
        "status": "ok",
        "n_model_features": len(app.state.model.feature_names),
        "model_version": app.state.model.version,
    }
    if app.state.pred_logger is not None:
        out["prediction_log"] = app.state.pred_logger.stats()
    return out


@app.post("/score", response_model=ScoreResponse)
def score(txn: Transaction):
    t0 = time.perf_counter()
    entity_id = entity_id_for(txn)

    try:
        engineered = app.state.store.get_features_and_update(
            entity_id=entity_id,
            txn_id=txn.TransactionID,
            dt=txn.TransactionDT,
            amt=txn.TransactionAmt,
            device=txn.DeviceInfo,
        )
    except redis.RedisError as e:
        raise HTTPException(status_code=503, detail=f"feature store unavailable: {e}")

    values = dict(txn.features)
    values.update(
        TransactionAmt=txn.TransactionAmt,
        card1=txn.card1, card2=txn.card2, card3=txn.card3, card5=txn.card5,
        addr1=txn.addr1, addr2=txn.addr2, DeviceInfo=txn.DeviceInfo,
    )
    values.update(engineered)
    fraud_score = app.state.model.predict(values)

    latency_ms = (time.perf_counter() - t0) * 1000
    if app.state.pred_logger is not None:
        app.state.pred_logger.log(
            {
                "transaction_id": txn.TransactionID,
                "entity_id": entity_id,
                "model_version": app.state.model.version,
                "fraud_score": fraud_score,
                "latency_ms": latency_ms,
                "features": app.state.model.loggable(values),
            }
        )

    return ScoreResponse(
        transaction_id=txn.TransactionID,
        fraud_score=fraud_score,
        entity_id=entity_id,
        engineered_features={k: _nan_to_none(engineered[k]) for k in FEATURE_COLUMNS},
        latency_ms=latency_ms,
    )


@app.post("/outcome")
def outcome(o: Outcome):
    """Record the true label for a transaction scored earlier. Prediction
    rows are written asynchronously, so an outcome sent within about a second
    of the score can match 0 rows; `updated` says how many matched."""
    if app.state.pred_logger is None:
        raise HTTPException(status_code=503, detail="prediction logging is not configured (DATABASE_URL unset)")
    try:
        with db.connect() as conn:
            cur = conn.execute(
                "UPDATE predictions SET label = %s, label_ts = now() WHERE transaction_id = %s",
                (o.isFraud, o.TransactionID),
            )
            return {"updated": cur.rowcount}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"database unavailable: {e}")
