"""Online, Redis-backed feature store for serving-time feature computation.

Computes the exact same FEATURE_COLUMNS as src/features/offline.py, but
incrementally: one transaction at a time, using the entity's current stored
state, instead of a batch replay over the full history. This is the "online"
half of the training/serving parity story -- see
tests/test_feature_parity.py, which replays real transactions through both
this class and compute_features_batch() and asserts they agree.

Per entity, Redis holds:
  - `entity:{id}:events`  a ZSET, score = TransactionDT, member =
    "{TransactionID}:{TransactionAmt}". Used to answer "how many
    transactions, and what did they sum to, in the trailing N seconds"
    without re-scanning full history -- ZRANGEBYSCORE/ZCOUNT do that in
    O(log n + k). Members older than the longest window in use are trimmed
    on every write so this never grows unbounded per entity.
  - `entity:{id}:state`  a HASH: last_dt, last_addr1, last_device,
    txn_seq_num. Used for time-since-last and the addr/device-changed
    flags without querying the ZSET.

get_features_and_update() reads state BEFORE writing the new transaction
into it, mirroring offline.py's closed='left' / strictly-prior semantics:
a transaction's own amount, address, and device never leak into its own
features.

Known simplification: read-then-write here is not wrapped in a Redis
transaction (MULTI/WATCH or a Lua script), so two concurrent requests for
the same entity arriving within the same instant could both read the same
"before" state. Fine for this project's scale and for the parity test; a
production version handling real concurrent traffic per entity would want
that atomicity.
"""
import math
from typing import Optional

from src.features.definitions import FEATURE_COLUMNS, WINDOW_LONG_SECONDS, WINDOW_SHORT_SECONDS

_MISSING = ""  # sentinel for "no value" in a Redis hash field (Redis has no null)

# How long to keep an idle entity's keys around before Redis expires them.
_KEY_TTL_SECONDS = 7 * 86400


class RedisFeatureStore:
    def __init__(self, client, ttl_seconds: int = _KEY_TTL_SECONDS):
        """`client` is any redis-py-compatible client constructed with
        decode_responses=True (a real redis.Redis, or a fakeredis.FakeRedis
        for tests -- both implement the same commands used here)."""
        self.client = client
        self.ttl_seconds = ttl_seconds

    def _events_key(self, entity_id: str) -> str:
        return f"entity:{entity_id}:events"

    def _state_key(self, entity_id: str) -> str:
        return f"entity:{entity_id}:state"

    def get_features_and_update(
        self,
        entity_id: str,
        txn_id,
        dt: float,
        amt: float,
        addr1: Optional[str],
        device: Optional[str],
    ) -> dict:
        """Compute this transaction's features from entity_id's state as of
        strictly before `dt`, then record this transaction into that state.
        Returns a dict with exactly the keys in FEATURE_COLUMNS, so it can
        be compared row-for-row against compute_features_batch()."""
        events_key = self._events_key(entity_id)
        state_key = self._state_key(entity_id)

        # Redis hash values are always strings; stringify here so a value
        # read back from a prior call (already a string) compares equal to
        # the same value passed in fresh (e.g. a pandas float64) instead of
        # failing every comparison on type alone.
        addr1 = str(addr1) if addr1 is not None else None
        device = str(device) if device is not None else None

        state = self.client.hgetall(state_key)  # client must be constructed with decode_responses=True
        has_prior = bool(state)
        last_dt = float(state["last_dt"]) if has_prior else None
        last_addr1 = state.get("last_addr1") or None if has_prior else None
        last_device = state.get("last_device") or None if has_prior else None
        txn_seq_num = int(state.get("txn_seq_num", 0)) if has_prior else 0

        count_1h = self.client.zcount(events_key, dt - WINDOW_SHORT_SECONDS, f"({dt}")
        count_24h = self.client.zcount(events_key, dt - WINDOW_LONG_SECONDS, f"({dt}")
        members_24h = self.client.zrangebyscore(events_key, dt - WINDOW_LONG_SECONDS, f"({dt}")
        sum_24h = sum(float(m.split(":")[1]) for m in members_24h)
        mean_24h = sum_24h / count_24h if count_24h > 0 else math.nan

        is_first = txn_seq_num == 0
        time_since_last = (dt - last_dt) if (not is_first and last_dt is not None) else math.nan
        addr_changed = self._changed(addr1, last_addr1, is_first)
        device_changed = self._changed(device, last_device, is_first)

        features = {
            "entity_txn_seq_num": float(txn_seq_num),
            "entity_time_since_last_sec": time_since_last,
            "entity_txn_count_1h": float(count_1h),
            "entity_txn_count_24h": float(count_24h),
            "entity_amt_sum_24h": sum_24h,
            "entity_amt_mean_24h": mean_24h,
            "entity_addr_changed": addr_changed,
            "entity_device_changed": device_changed,
        }
        assert list(features.keys()) == FEATURE_COLUMNS

        # Update state AFTER computing features above, not before.
        pipe = self.client.pipeline()
        pipe.zadd(events_key, {f"{txn_id}:{amt}": dt})
        pipe.zremrangebyscore(events_key, "-inf", f"({dt - WINDOW_LONG_SECONDS}")
        pipe.expire(events_key, self.ttl_seconds)
        pipe.hset(
            state_key,
            mapping={
                "last_dt": dt,
                "last_addr1": addr1 if addr1 is not None else _MISSING,
                "last_device": device if device is not None else _MISSING,
                "txn_seq_num": txn_seq_num + 1,
            },
        )
        pipe.expire(state_key, self.ttl_seconds)
        pipe.execute()

        return features

    @staticmethod
    def _changed(current, previous, is_first: bool) -> float:
        if is_first or current is None or previous is None:
            return math.nan
        return 1.0 if current != previous else 0.0
