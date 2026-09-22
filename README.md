# Real-Time Fraud Detection System

Train a model to flag fraudulent transactions, then build the infrastructure
to serve it in real time with monitoring and retraining, not just a notebook
that outputs an AUC score.

## Status

Phase 1 (data and baseline model) done. Phase 2 (feature pipeline) done. See
results below.

## Dataset

[IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection)
(Kaggle), not the Kaggle Credit Card Fraud dataset. The Credit Card dataset
has only anonymized PCA columns and no card, address, or user field, so
there's no way to build the entity-level rolling features Phase 2 needs.
IEEE-CIS has no explicit user ID either, but its card and address fields let
us construct a proxy entity (see Phase 2 notes once that's built).

Not committed to this repo (`data/raw/` is gitignored, ~700MB). To reproduce:
join the competition on Kaggle, accept its rules, download
`train_transaction.csv` and `train_identity.csv` into `data/raw/`.

## Phase 1: baseline model

`src/training/train_baseline.py`. LightGBM, time-based 80/20 train/validation
split (not random) so validation mirrors how the model is actually used in
production: scoring transactions it has never seen, all of them later in
time than everything it trained on. Categorical columns use pandas `category`
dtype, which LightGBM splits on natively, instead of one-hot encoding
(avoids exploding `P_emaildomain` etc. into hundreds of columns).

**Evaluated with PR-AUC, not accuracy.** At 3.5% fraud prevalence, a model
that always predicts "not fraud" scores 96.5% accuracy. A random classifier's
PR-AUC equals class prevalence (~0.034), so that's the real baseline to beat.

**Class imbalance: no `scale_pos_weight`.** Tested it at the full class ratio
(~28x) and a milder 5x. Both destabilized LightGBM's boosting: validation AUC
stopped improving after 1 and 12 trees respectively, versus 388 unweighted.
Heavier reweighting means a handful of minority-class examples dominate each
round's gradient, so every tree overcorrects for those specific examples
instead of finding patterns that generalize, classic overfitting, confirmed
by inspecting `best_iteration_` directly rather than assuming.

| scale_pos_weight | best_iteration | PR-AUC |
|---|---|---|
| 28.0 (full class ratio) | 1 | 0.2629 |
| 5.0 (mild) | 12 | 0.4450 |
| none | 388 | **0.5707** |

Went with no reweighting. LightGBM's tree-based objective handles this level
of imbalance on its own; the imbalance is instead handled at the alert-rate
threshold, which is also how a real fraud team would operate the model, by
picking a review capacity, not by reweighting a loss function.

### Results (final baseline, no reweighting)

| Alert rate | Precision | Recall |
|---|---|---|
| 1% | 0.8976 | 0.2611 |
| 2% | 0.7245 | 0.4213 |
| 5% | 0.4299 | 0.6248 |

At a 2% alert budget, the model catches 42% of all fraud at 72% precision.

Top features by split count: `card1`, `card2`, `DeviceInfo`, `addr1`,
`TransactionAmt`, `id_31` (browser string), `P_emaildomain`. Card and address
fields dominating tracks with the earlier EDA finding that specific card/addr
combinations carry different fraud rates, legitimate signal, not leakage,
since it's known at scoring time.

## Phase 2: feature pipeline

No user ID exists in IEEE-CIS. The proxy entity used throughout
(`src/features/entity.py`) is `card1+card2+card3+card5+addr1+addr2`, the
standard workaround for this dataset. It's an approximation, not ground
truth: card1-6 are Vesta's own categorical bins, not real account numbers, so
two different people can collide into the same bucket, and a genuine
person's address change mid-history can split them into two.

**Bug caught before it shipped:** a naive `str(card1)+"_"+str(card2)+...`
concatenation turns every missing field into the literal substring `"nan"`,
which is identical across rows. On this data that merged every transaction
missing an address into one fake "entity" of 9,900 transactions (1.7% of the
whole dataset) with rolling features computed over pure noise. Fix: any row
missing a key field gets a per-row singleton ID instead of joining a shared
bucket, which is also the practically correct call: if identity can't be
resolved, don't borrow another transaction's history for it. 12.85% of rows
get singleton IDs; the real entity population is 38,197 proxy entities
across 514,655 rows, median 2 transactions each, max 5,862.

**No distance feature.** The brief asks for "distance from last transaction
location." IEEE-CIS has no raw geo coordinates to compute one from — `dist1`
and `dist2` are Vesta's own pre-engineered distances, already sitting in the
data, not something to derive ourselves. Rather than fabricate coordinates
to check a box, this is documented as a dataset limitation and substituted
with `entity_addr_changed` / `entity_device_changed`: whether this
transaction's address or device differs from the entity's last one. Same
class of signal (identity/location mismatch), actually computable here.

**Point-in-time correctness.** Every feature for a transaction is computed
from that entity's state strictly *before* this transaction — its own
amount, address, or device never leaks into its own features. The batch
pipeline (`src/features/offline.py`) does this with `np.searchsorted` per
entity on the time-sorted transaction array rather than pandas' built-in
`groupby().rolling(on=...)`: IEEE-CIS has second-resolution timestamps and
some entities transact more than once in the same second, which produces a
non-unique index that built-in time-rolling can't reindex back onto the
frame. Same-instant transactions are treated as mutually invisible to each
other, which is the explicit version of the assumption pandas' rolling was
making implicitly.

**Feature store: Redis.** Sub-millisecond reads/writes and native support
for the access pattern here (per-entity sorted set for windowed
count/sum, hash for last-seen state) fit Phase 3's <100ms serving target
directly. `src/features/store.py`'s `RedisFeatureStore` reads an entity's
state, computes features, *then* writes the new transaction into that state
— same before/after ordering as the offline pipeline, and the reason the two
don't drift apart into different definitions of "before."

**Proof they don't drift apart:** `tests/test_feature_parity.py` replays
real transactions through both the offline batch pipeline and the online
Redis path (via `fakeredis`, no server needed to run it) and asserts they
produce identical features row for row. This actually caught a bug during
development: Redis stores hash values as strings, and comparing a freshly
passed `addr1` (a float from pandas) against a value read back from Redis
(a string) meant `299.0 != "299.0"` was true on every single comparison —
`entity_addr_changed` was reporting "changed" almost every time online, not
because addresses were changing, but because of a type mismatch invisible
without a test that actually compares the two paths. That's the skew this
whole exercise exists to catch, caught before Phase 3 serving ever saw it.

### Does it improve the model?

Retrained the Phase 1 model with the 8 engineered columns added (439
features total vs. Phase 1's raw columns), same time-based split, same
`LGBMClassifier` config, same evaluation.

| | Phase 1 (baseline) | Phase 2 (+ entity features) |
|---|---|---|
| best_iteration | 388 | 392 |
| PR-AUC | **0.5707** | 0.5673 |
| alert 1%: precision / recall | 0.898 / 0.261 | 0.892 / 0.259 |
| alert 2%: precision / recall | 0.725 / 0.421 | 0.711 / 0.413 |
| alert 5%: precision / recall | 0.430 / 0.625 | 0.426 / 0.619 |

Slightly worse, not better. Being straight about this rather than cherry-
picking a favorable cut: adding the features did not raise PR-AUC on this
model.

That's not the same as the features being useless. `entity_txn_seq_num`
(how many times we've seen this entity before) ranks 7th of 439 by
importance, and `entity_time_since_last_sec` ranks 13th — both used
heavily by the model. The reason shows up in the data: transactions from an
entity never seen before make up 15.7% of the validation period but carry a
9.8% fraud rate, nearly 3x the validation average (3.4%) and over 4x the
rate for returning entities (2.3%). "This looks like a brand-new identity"
is real signal, and `entity_txn_seq_num == 0` encodes exactly that.

Two likely reasons it didn't net out positive anyway: first, IEEE-CIS
already ships `D1` ("days since account creation" style deltas) and the
`C1`-`C14` count features, which plausibly capture much of the same
"is this identity established" signal already, through Vesta's own
(better-resolved) entity linkage rather than this project's 6-field proxy —
so the new columns are partly redundant, not purely additive. Second, half
of `entity_amt_mean_24h` and most of `entity_device_changed` are NaN (a
transaction's entity had no transaction in the trailing 24h, or no device
info at all on one side of the comparison), adding width without much
signal for the majority of rows, while `num_leaves=63` was left unchanged
from Phase 1 — the model has the same split budget spread across more,
partly-redundant, partly-sparse columns.

The honest read: Phase 2's job was proving the feature pipeline is built
correctly for production — point-in-time correct, parity-tested against
online serving, with a real entity-resolution bug caught before it shipped
— not proving these particular 8 features raise this particular metric.
That's what it delivered. Whether the features themselves are worth
keeping past Phase 3 is a separate, smaller question (better entity
resolution, or letting the model use more capacity, might change the
answer) and isn't blocking Phase 3.

### Reproducing

```
python -m src.training.build_training_set   # writes data/processed/train_features.parquet
python -m src.training.train_with_features   # trains, saves models/phase2_lgbm.pkl
pytest tests/test_feature_parity.py -v
```

`build_training_set.py` loads all 434 raw columns into memory at once; on a
memory-constrained machine (this was developed and validated in an 8GB
sandbox, which OOM'd on a naive load) see `load_parquet_lean` in
`src/training/common.py`, used by `train_with_features.py`, for a leaner
loading path.

Local Redis for exercising `RedisFeatureStore` against something real
(the test suite itself doesn't need this, it uses `fakeredis`):
```
docker compose -f docker/docker-compose.yml up -d
```

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On macOS, LightGBM needs OpenMP, which isn't preinstalled: `brew install libomp`.

## Roadmap

- Phase 3: FastAPI serving endpoint, containerized, load tested (Locust/k6)
  for real latency numbers. `src/features/store.py`'s `RedisFeatureStore` is
  what the endpoint will call per request.
- Phase 4: prediction logging, drift detection (PSI or KS test), retraining
  trigger.
- Phase 5: CI/CD, scheduled or drift-triggered retraining and redeploy.
