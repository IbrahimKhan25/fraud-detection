"""Loads entity history into Redis so the online feature store starts warm.

Without this, a freshly started Redis knows nothing: every entity looks
brand new (txn_seq_num 0, no 24h history), while the model was trained on
features computed from the full history. That is training/serving skew from
a cold start, and it showed up in the first Phase 4 drift run as an
entity_txn_seq_num PSI of 2.7 (0.17 when measured offline on the same rows).

    python -m src.features.backfill              # history up to the training cutoff
    python -m src.features.backfill --flush      # wipe Redis first

For every entity seen before the cutoff it writes exactly what
RedisFeatureStore would hold had it processed those transactions live:
txn_seq_num, latest TransactionDT, last device, and the last 24h of
(TransactionID:amount) events. tests/test_backfill.py proves that: backfill,
then replay later transactions online, and the features equal the offline
pipeline's. Singleton entities (UNK_...) are skipped: by definition they are
never seen again.
"""
import argparse
import math
import os

import pandas as pd

from src.features.definitions import WINDOW_LONG_SECONDS
from src.features.store import _KEY_TTL_SECONDS, _MISSING

REQUIRED = ["entity_id", "TransactionID", "TransactionDT", "TransactionAmt", "DeviceInfo"]


def backfill_state(client, history: pd.DataFrame, cutoff_dt: float, ttl_seconds: int = _KEY_TTL_SECONDS, batch: int = 500) -> int:
    """Write entity state for all transactions in `history` with
    TransactionDT < cutoff_dt. Returns the number of entities written."""
    h = history.loc[history["TransactionDT"] < cutoff_dt, REQUIRED]
    h = h[~h["entity_id"].astype(str).str.startswith("UNK_")]
    h = h.sort_values(["entity_id", "TransactionDT"], kind="stable")

    pipe = client.pipeline()
    n_entities = 0
    for entity_id, g in h.groupby("entity_id", sort=False):
        last = g.iloc[-1]
        device = last["DeviceInfo"]
        device = _MISSING if (device is None or (isinstance(device, float) and math.isnan(device))) else str(device)
        recent = g[g["TransactionDT"] >= cutoff_dt - WINDOW_LONG_SECONDS]

        events_key, state_key = f"entity:{entity_id}:events", f"entity:{entity_id}:state"
        pipe.delete(events_key, state_key)
        if len(recent):
            pipe.zadd(
                events_key,
                {f"{int(r.TransactionID)}:{float(r.TransactionAmt)}": float(r.TransactionDT) for r in recent.itertuples()},
            )
            pipe.expire(events_key, ttl_seconds)
        pipe.hset(
            state_key,
            mapping={"last_dt": float(last["TransactionDT"]), "last_device": device, "txn_seq_num": len(g)},
        )
        pipe.expire(state_key, ttl_seconds)
        n_entities += 1
        if n_entities % batch == 0:
            pipe.execute()
            pipe = client.pipeline()
    pipe.execute()
    return n_entities


def main():
    import redis

    from src.training.common import PROCESSED

    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff-quantile", type=float, default=0.80, help="same split point as the training/validation split")
    ap.add_argument("--cutoff-dt", type=float, default=None, help="explicit TransactionDT cutoff (overrides --cutoff-quantile)")
    ap.add_argument("--flush", action="store_true", help="FLUSHDB first")
    args = ap.parse_args()

    df = pd.read_parquet(PROCESSED / "train_features.parquet", columns=REQUIRED)
    cutoff = args.cutoff_dt if args.cutoff_dt is not None else float(df["TransactionDT"].quantile(args.cutoff_quantile))
    client = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True)
    if args.flush:
        client.flushdb()
    n = backfill_state(client, df, cutoff)
    print(f"backfilled {n:,} entities from transactions before TransactionDT {cutoff:,.0f}")


if __name__ == "__main__":
    main()
