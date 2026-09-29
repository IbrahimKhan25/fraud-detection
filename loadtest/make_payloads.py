"""Samples real transactions from data/processed/train_features.parquet into
loadtest/payloads.jsonl, in the /score request shape, so the load test sends
full-width (~430 field) requests like production would, not toy payloads.

Run from the repo root:  python -m loadtest.make_payloads [N]
"""
import json
import sys
from pathlib import Path

import pandas as pd

from src.features.definitions import ENTITY_KEY_COLS, FEATURE_COLUMNS
from src.training.common import DROP_COLS, PROCESSED

CORE = ["TransactionID", "TransactionDT", "TransactionAmt", "DeviceInfo"] + ENTITY_KEY_COLS
OUT = Path(__file__).parent / "payloads.jsonl"


def _clean(v):
    return None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)


def row_to_payload(row, extra_cols) -> dict:
    """One /score request body from a feature-table row."""
    body = {c: _clean(row[c]) for c in CORE}
    body["TransactionID"] = int(body["TransactionID"])
    if body["card1"] is not None:
        body["card1"] = int(body["card1"])
    body["features"] = {c: _clean(row[c]) for c in extra_cols}
    return body


def extra_columns(columns) -> list:
    skip = set(CORE) | set(DROP_COLS) | set(FEATURE_COLUMNS) | {"entity_id"}
    return [c for c in columns if c not in skip]


def main(n: int = 2000):
    df = pd.read_parquet(PROCESSED / "train_features.parquet").sample(n, random_state=0)
    extra_cols = extra_columns(df.columns)
    with open(OUT, "w") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row_to_payload(row, extra_cols)) + "\n")
    print(f"wrote {n} payloads to {OUT}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2000)
