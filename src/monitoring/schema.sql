-- Loaded by the Postgres container on first start (docker-entrypoint-initdb.d)
-- and by monitor.py / tests via db.ensure_schema(). Idempotent.

-- One row per scored transaction. `features` holds the model's non-missing
-- inputs (raw + engineered) exactly as the model saw them, so drift can be
-- measured on what the model actually received. A key absent from the JSON
-- means the value was missing. `label` is filled in later, when the outcome
-- (fraud or not) becomes known.
CREATE TABLE IF NOT EXISTS predictions (
    id             BIGSERIAL PRIMARY KEY,
    ts             TIMESTAMPTZ NOT NULL DEFAULT now(),
    transaction_id BIGINT NOT NULL,
    entity_id      TEXT NOT NULL,
    model_version  TEXT NOT NULL,
    fraud_score    DOUBLE PRECISION NOT NULL,
    latency_ms     DOUBLE PRECISION,
    features       JSONB NOT NULL,
    label          SMALLINT,
    label_ts       TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS predictions_ts_idx ON predictions (ts);
CREATE INDEX IF NOT EXISTS predictions_txn_idx ON predictions (transaction_id);

-- One row per drift check, so alerts have a history.
CREATE TABLE IF NOT EXISTS drift_reports (
    id               BIGSERIAL PRIMARY KEY,
    ts               TIMESTAMPTZ NOT NULL DEFAULT now(),
    model_version    TEXT,
    n_rows           INTEGER NOT NULL,
    score_psi        DOUBLE PRECISION,
    n_alert_features INTEGER NOT NULL,
    drifted          BOOLEAN NOT NULL,
    details          JSONB NOT NULL
);
