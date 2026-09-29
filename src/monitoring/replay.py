"""Sends real transactions to a running API so the prediction log fills up,
optionally with an injected shift so you can watch the drift check fire.

    python -m src.monitoring.replay --n 3000                    # validation-period traffic
    python -m src.monitoring.replay --n 3000 --amt-multiplier 3 # amounts tripled (synthetic drift)
    python -m src.monitoring.replay --n 3000 --offset 30000     # a later slice (backfill with --cutoff-dt = the printed first TransactionDT)

Traffic comes from the validation slice (the last 20% of the data in time),
sent in time order, because that is data the model was not trained on and
it arrives in the order a live system would see it.
"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

from loadtest.make_payloads import extra_columns, row_to_payload
from src.training.common import PROCESSED, load_parquet_lean, time_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--amt-multiplier", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--offset", type=int, default=0, help="skip this many validation rows first (pick a different slice)")
    args = ap.parse_args()

    df = load_parquet_lean(PROCESSED / "train_features.parquet")
    _, val_df = time_split(df)
    val_df = val_df.sort_values("TransactionDT").iloc[args.offset : args.offset + args.n]
    print(f"slice: rows {args.offset:,}-{args.offset + len(val_df):,} of validation, "
          f"first TransactionDT {val_df['TransactionDT'].iloc[0]:.0f}")
    extra = extra_columns(val_df.columns)
    payloads = [row_to_payload(r, extra) for _, r in val_df.iterrows()]
    for p in payloads:
        p["TransactionAmt"] *= args.amt_multiplier

    client = httpx.Client(base_url=args.url, timeout=10)
    failures = 0

    def send(p):
        nonlocal failures
        r = client.post("/score", json=p)
        if r.status_code != 200:
            failures += 1

    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(send, payloads))
    print(f"sent {len(payloads)} transactions in {time.time() - t0:.1f}s, {failures} failures")


if __name__ == "__main__":
    main()
