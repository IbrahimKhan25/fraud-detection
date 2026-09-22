"""Proxy entity resolution.

IEEE-CIS has no user ID. The standard workaround for this dataset (used by
most competition writeups) is to build a proxy "entity" out of the card and
address fields: card1, card2, card3, card5, addr1, addr2. This is NOT a real
user ID: card1-6 are Vesta's own categorical bins, not literal account
numbers, so two different people can land in the same bucket, and a genuine
person whose address changes mid-history can get split into two buckets.
It's a documented approximation, not ground truth.

Bug this module exists to avoid: card2/card3/card5/addr1/addr2 are missing
on 1.5-11% of rows each (addr1/addr2 alone are missing on ~11%, and 12.85%
of rows are missing at least one of the six key fields). A naive
str(card1)+"_"+str(card2)+... concatenation turns each missing field into
the literal substring "nan", which is identical across rows -- so every
transaction missing addr1 and addr2, regardless of which card it used, gets
merged into one shared "entity". On the actual data this produced a fake
"entity" with 9,900 transactions (1.7% of the entire dataset) that was
nothing but the union of every row with no address on file. Rolling
features built on top of that bucket would be pure noise dressed up as
signal.

Fix: any row missing one or more key fields gets its own singleton ID
(prefixed UNK_ + its TransactionID) instead of joining a shared bucket. That
row is correctly treated as a first-time/no-history transaction for feature
purposes -- which is also the practically correct behavior: if we can't
confidently resolve identity, we shouldn't be borrowing another
transaction's history for it.
"""
import pandas as pd

from src.features.definitions import ENTITY_KEY_COLS


def build_entity_id(df: pd.DataFrame) -> pd.Series:
    """Return a proxy entity ID per row. See module docstring for the
    missing-field singleton fix -- do not replace this with a plain
    concatenation of ENTITY_KEY_COLS.astype(str)."""
    missing = df[ENTITY_KEY_COLS].isna().any(axis=1)

    parts = [df[c].astype(str) for c in ENTITY_KEY_COLS]
    concatenated = parts[0]
    for p in parts[1:]:
        concatenated = concatenated + "_" + p

    singleton = "UNK_" + df["TransactionID"].astype(str)
    return concatenated.where(~missing, singleton)
