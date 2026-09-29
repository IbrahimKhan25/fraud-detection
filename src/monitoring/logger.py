"""Prediction logger: gets scored transactions into Postgres without ever
slowing down or breaking scoring.

log() only puts a dict on an in-memory queue (microseconds). A background
thread drains the queue and writes batches with one INSERT round trip per
batch. If Postgres is slow or down, the queue fills and new rows are dropped
and counted. Scoring never waits on the database and never fails because of
it: for a fraud API, an unlogged prediction is a smaller problem than a
rejected or delayed transaction. `dropped` and `failed` are exposed on
/health so loss is visible, not silent.
"""
import json
import logging
import queue
import threading
import time
from typing import Callable, Dict, List, Optional

from src.monitoring import db

log = logging.getLogger("prediction_logger")

_INSERT = """
INSERT INTO predictions (transaction_id, entity_id, model_version, fraud_score, latency_ms, features)
VALUES (%s, %s, %s, %s, %s, %s::jsonb)
"""


class PostgresSink:
    """Writes a batch of rows. Keeps one connection open and reconnects
    after a failure."""

    def __init__(self):
        self._conn = None

    def __call__(self, rows: List[Dict]) -> None:
        try:
            if self._conn is None or self._conn.closed:
                self._conn = db.connect()
            params = [
                (r["transaction_id"], r["entity_id"], r["model_version"], r["fraud_score"],
                 r["latency_ms"], json.dumps(r["features"]))
                for r in rows
            ]
            with self._conn.cursor() as cur:
                cur.executemany(_INSERT, params)
        except Exception:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:  # noqa: BLE001
                    pass
            self._conn = None
            raise


class PredictionLogger:
    def __init__(
        self,
        sink: Optional[Callable[[List[Dict]], None]] = None,
        max_queue: int = 10_000,
        batch_size: int = 200,
        flush_seconds: float = 1.0,
    ):
        self._sink = sink or PostgresSink()
        self._q: "queue.Queue[Dict]" = queue.Queue(maxsize=max_queue)
        self._batch_size = batch_size
        self._flush_seconds = flush_seconds
        self._stop = threading.Event()
        self._last_warn = 0.0
        self.written = 0
        self.dropped = 0
        self.failed = 0
        self._thread = threading.Thread(target=self._run, name="prediction-logger", daemon=True)
        self._thread.start()

    def log(self, row: Dict) -> bool:
        try:
            self._q.put_nowait(row)
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def stats(self) -> Dict[str, int]:
        return {"written": self.written, "dropped": self.dropped, "failed": self.failed, "queued": self._q.qsize()}

    def _run(self) -> None:
        while not (self._stop.is_set() and self._q.empty()):
            batch: List[Dict] = []
            try:
                batch.append(self._q.get(timeout=self._flush_seconds))
            except queue.Empty:
                continue
            # Keep collecting until the batch is full or flush_seconds has passed
            # since the first row, so one commit covers many rows instead of
            # waking for every request.
            deadline = time.monotonic() + self._flush_seconds
            while len(batch) < self._batch_size and not self._stop.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    batch.append(self._q.get(timeout=remaining))
                except queue.Empty:
                    break
            while len(batch) < self._batch_size:  # shutdown: take whatever is left
                try:
                    batch.append(self._q.get_nowait())
                except queue.Empty:
                    break
            self._flush(batch)

    def _flush(self, batch: List[Dict]) -> None:
        try:
            self._sink(batch)
            self.written += len(batch)
        except Exception as e:  # noqa: BLE001
            self.failed += len(batch)
            now = time.monotonic()
            if now - self._last_warn > 30:  # don't spam the log during an outage
                self._last_warn = now
                log.warning("prediction log write failed (%d rows lost so far): %s", self.failed, e)

    def close(self, timeout: float = 5.0) -> None:
        """Stop accepting work and flush what is queued (up to `timeout`)."""
        self._stop.set()
        self._thread.join(timeout)
