# %% [markdown]
# # 01 EDA: IEEE-CIS Fraud Detection
# Run one cell at a time (Shift+Enter, or the "Run Cell" link above each `# %%`).
# Goal of this file: understand the data before modeling anything.

# %% Load and join
from pathlib import Path

import pandas as pd

# Find the project root from the working directory, so this runs the same whether
# the kernel starts in the repo root or in notebooks/.
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "data" / "raw").is_dir())
RAW = ROOT / "data" / "raw"
tx = pd.read_csv(RAW / "train_transaction.csv")
ident = pd.read_csv(RAW / "train_identity.csv")

# Left join: keep every transaction, attach identity info where it exists.
df = tx.merge(ident, on="TransactionID", how="left").copy()  # copy defragments the wide frame
print(tx.shape, ident.shape, df.shape)

# %% Cache as parquet so later loads take seconds instead of a minute
df.to_parquet(RAW / "train_joined.parquet")

# %% Class balance: the whole reason this project is hard
print(df["isFraud"].value_counts())
print(f"fraud rate: {df['isFraud'].mean():.4%}")

# %% Time: TransactionDT is seconds from an unknown start point, not a date
days = df["TransactionDT"] / 86400
print(f"span: {days.min():.1f} to {days.max():.1f} days")
print("is the file sorted by time?", df["TransactionDT"].is_monotonic_increasing)

# %% Missing-ness: which columns are mostly empty?
missing = df.isna().mean().sort_values(ascending=False)
print(missing.head(15))
print("columns >50% missing:", (missing > 0.5).sum(), "of", df.shape[1])

# %% Does having identity info relate to fraud?
df["has_identity"] = df["id_01"].notna()
print(df.groupby("has_identity")["isFraud"].agg(["mean", "count"]))

# %% Build the entity proxy: no user ID exists, so group by card1 + addr1
# card1 is an encoded card identifier, addr1 is a billing region code.
# Together they're the closest thing to "this is probably the same person."
df["entity_id"] = df["card1"].astype(str) + "_" + df["addr1"].astype(str)

entity_stats = df.groupby("entity_id").agg(
    n_transactions=("TransactionID", "count"),
    n_days_active=("TransactionDT", lambda s: (s.max() - s.min()) / 86400),
    fraud_rate=("isFraud", "mean"),
)
print("total entities:", entity_stats.shape[0], "vs", df.shape[0], "transactions")
print(entity_stats["n_transactions"].describe())

# %% How concentrated is activity? A few huge entities can be a red flag (shared card
# infra, e.g. a corporate card) rather than one real person.
print("entities with only 1 transaction:", (entity_stats["n_transactions"] == 1).mean())
print(entity_stats.sort_values("n_transactions", ascending=False).head(10))

# %% Widen the key: card1 alone collides too much (top entity spans 182 days,
# 5885 transactions -- clearly many different people sharing one bucket).
# Stack more card/address fields to narrow it, the way top IEEE-CIS solutions do.
key_cols = ["card1", "card2", "card3", "card5", "addr1", "addr2"]
# Vectorized concat with "+" (not .agg(join, axis=1), which is slow and can pass
# a stray NaN through as a float instead of the string "nan").
df["entity_id_v2"] = df[key_cols[0]].astype(str)
for col in key_cols[1:]:
    df["entity_id_v2"] += "_" + df[col].astype(str)

entity_stats_v2 = df.groupby("entity_id_v2").agg(
    n_transactions=("TransactionID", "count"),
    n_days_active=("TransactionDT", lambda s: (s.max() - s.min()) / 86400),
    fraud_rate=("isFraud", "mean"),
)
print("total entities (v2):", entity_stats_v2.shape[0], "vs", df.shape[0], "transactions")
print(entity_stats_v2["n_transactions"].describe())
print("singleton share (v2):", (entity_stats_v2["n_transactions"] == 1).mean())
print(entity_stats_v2.sort_values("n_transactions", ascending=False).head(10))

# %% [markdown]
# ## Time-based split
# TODO (Phase 2): entity_id_v2 barely beat card1+addr1 -- card3/card5/addr2 are
# near-constant. Revisit with a D1 (account tenure) refinement once the rolling
# features are built, and A/B it against validation PR-AUC instead of guessing.

# %% Time-based train/validation split
# A random split lets the model train on rows from AFTER some of its validation
# rows -- info leaks backward in time that production will never have. Split on
# TransactionDT instead: everything before the cutoff is train, everything after
# is validation, same as how the model will actually be used.
cutoff_dt = df["TransactionDT"].quantile(0.80)

train_df = df[df["TransactionDT"] < cutoff_dt]
val_df = df[df["TransactionDT"] >= cutoff_dt]

print(f"cutoff: day {cutoff_dt / 86400:.1f} of {df['TransactionDT'].max() / 86400:.1f}")
print(f"train: {train_df.shape[0]:>7,} rows, {train_df['isFraud'].mean():.4%} fraud")
print(f"val:   {val_df.shape[0]:>7,} rows, {val_df['isFraud'].mean():.4%} fraud")

# Sanity check: no time overlap between the two sets.
print("train max day:", train_df["TransactionDT"].max() / 86400)
print("val min day:  ", val_df["TransactionDT"].min() / 86400)
