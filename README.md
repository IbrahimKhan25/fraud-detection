# Real-Time Fraud Detection System

Train a model to flag fraudulent transactions, then build the infrastructure
to serve it in real time with monitoring and retraining, not just a notebook
that outputs an AUC score.

## Status

Phase 1 (data and baseline model) done. Phase 2 (feature pipeline) done.
Phase 3 (serving) done and load tested. Phase 4 (monitoring) done and validated
(see "How sensitive is it"). Phase 5 (CI/CD) in progress: CI runs the tests and
builds the image on every push and pull request. See results below.

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
location." IEEE-CIS has no raw geo coordinates to compute one from. `dist1`
and `dist2` are Vesta's own pre-engineered distances, already sitting in the
data, not something to derive ourselves. Rather than fabricate coordinates
to check a box, this is documented as a dataset limitation and substituted
with `entity_device_changed`: whether this transaction's device differs
from the entity's last one. Same class of signal (identity mismatch),
actually computable here. (An `entity_addr_changed` twin was built first and
later removed as a constant, see "Removing a dead feature" below.)

**Point-in-time correctness.** Every feature for a transaction is computed
from that entity's state strictly *before* this transaction, so its own
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
state, computes features, *then* writes the new transaction into that state,
the same before/after ordering as the offline pipeline, and the reason the two
don't drift apart into different definitions of "before."

**Proof they don't drift apart:** `tests/test_feature_parity.py` replays
real transactions through both the offline batch pipeline and the online
Redis path (via `fakeredis`, no server needed to run it) and asserts they
produce identical features row for row. This actually caught a bug during
development: Redis stores hash values as strings, and comparing a freshly
passed `addr1` (a float from pandas) against a value read back from Redis
(a string) meant `299.0 != "299.0"` was true on every single comparison, so
`entity_addr_changed` was reporting "changed" almost every time online, not
because addresses were changing, but because of a type mismatch invisible
without a test that actually compares the two paths. That's the skew this
whole exercise exists to catch, caught before Phase 3 serving ever saw it.
(The feature itself turned out to be dead for a different reason, below.)

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
importance, and `entity_time_since_last_sec` ranks 13th, and both are used
heavily by the model. The reason shows up in the data: transactions from an
entity never seen before make up 15.7% of the validation period but carry a
9.8% fraud rate, nearly 3x the validation average (3.4%) and over 4x the
rate for returning entities (2.3%). "This looks like a brand-new identity"
is real signal, and `entity_txn_seq_num == 0` encodes exactly that.

Two likely reasons it didn't net out positive anyway: first, IEEE-CIS
already ships `D1` ("days since account creation" style deltas) and the
`C1`-`C14` count features, which plausibly capture much of the same
"is this identity established" signal already, through Vesta's own
(better-resolved) entity linkage rather than this project's 6-field proxy,
so the new columns are partly redundant, not purely additive. Second, half
of `entity_amt_mean_24h` and most of `entity_device_changed` are NaN (a
transaction's entity had no transaction in the trailing 24h, or no device
info at all on one side of the comparison), adding width without much
signal for the majority of rows, while `num_leaves=63` was left unchanged
from Phase 1, so the model has the same split budget spread across more,
partly-redundant, partly-sparse columns.

The honest read: Phase 2's job was proving the feature pipeline is built
correctly for production: point-in-time correct, parity-tested against
online serving, with a real entity-resolution bug caught before it shipped,
not proving these particular 8 features raise this particular metric.
That's what it delivered. Whether the features themselves are worth
keeping past Phase 3 is a separate, smaller question (better entity
resolution, or letting the model use more capacity, might change the
answer) and isn't blocking Phase 3.

### Removing a dead feature

The Phase 2 feature summary showed `entity_addr_changed` with mean 0, std 0
and max 0: it was 0.0 on every row where it had a value. `addr1` is one of
the six fields that define the entity (`ENTITY_KEY_COLS`), so an entity can
never have a different `addr1` from its previous transaction. The flag could
not vary by construction. Nothing had caught it because the online/offline
parity test passed (both sides agreed on a constant) and the model simply
never split on it.

Removed it from the offline pipeline, the Redis store, the API and the tests,
rebuilt the training set and retrained on the Mac (438 features, 7
engineered). Result, same split and config:

| | with dead column | without |
|---|---|---|
| best_iteration | 392 | 392 |
| PR-AUC | 0.5673 | 0.5673 |
| alert 2%: precision / recall | 0.711 / 0.413 | 0.711 / 0.413 |

Identical, as expected for removing a column the trees never used. The point
of the change is honesty and a smaller serving path (one fewer Redis field per
entity), not accuracy. Ranks among the 438 features: `entity_txn_seq_num` 7,
`entity_time_since_last_sec` 13, `entity_amt_mean_24h` 24, `entity_amt_sum_24h`
44, `entity_txn_count_24h` 45, `entity_txn_count_1h` 196,
`entity_device_changed` 241. Whether a better entity definition (for example
without `addr1`/`addr2`, which would also make an address-change flag
meaningful) would help is untested and left as a possible experiment.

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

## Phase 3: serving

`src/serving/app.py`. `POST /score` takes one transaction, reads the entity's
state from Redis, computes the 7 engineered features with the same
`RedisFeatureStore` the parity test covers, scores with `models/phase2_lgbm.pkl`,
then records the transaction. `GET /health` pings Redis.

The request has a few core fields (`TransactionID`, `TransactionDT`,
`TransactionAmt`, the six entity key columns, `DeviceInfo`) plus a `features`
dict for every other raw column the model was trained on. Missing keys score as
NaN. Categorical columns are detected from the model itself and cast to
`category` so LightGBM re-maps them onto the training categories.

Entity ID formatting matters: `card1` is int64 in the raw data and the other
key columns are float64, so `13926` and `13926.0` would build different
entities. `entity_id_for()` reuses `build_entity_id` with those dtypes, and
`tests/test_serving.py` checks it.

```
docker compose -f docker/docker-compose.yml up -d redis
uvicorn src.serving.app:app --port 8000          # http://localhost:8000/docs
uvicorn src.serving.app:app --port 8000 --workers 4   # several processes, for throughput
pytest tests/test_serving.py -v                  # no Redis or real model needed

python -m loadtest.make_payloads                 # needs data/processed/train_features.parquet
locust -f loadtest/locustfile.py --host http://127.0.0.1:8000 --headless -u 20 -r 5 -t 60s --csv loadtest/results   # ~20 req/s (1 per user)

docker compose -f docker/docker-compose.yml up -d --build   # Redis + API container on :8000
```

### Load test results

Locust, `loadtest/locustfile.py`: 2,000 real transactions sampled from the
training set (full ~430-field payloads), replayed at a fixed rate of 1
request per second per simulated user, 60 s per run. MacBook (Apple silicon),
4 uvicorn workers, local Redis, Locust running on the same machine, not in
Docker. Latency in ms.

| Rate | Requests | p50 | p95 | p99 | max | Failures |
|---|---|---|---|---|---|---|
| ~20 req/s (20 users) | 1,170 | 14 | 25 | 46 | 63 | 0 |
| ~47 req/s (50 users) | 2,775 | 17 | 29 | 45 | 53 | 0 |
| ~85 req/s avg (100 users) | 5,050 | 17 | 42 | 120 | 198 | 0 |

Same test, 47 req/s (50 users), against the containers from
`docker compose up --build` (API + Redis, Docker Desktop on the same Mac, 8
CPUs allocated):

| Setup | Requests | p50 | p95 | p99 | max | Failures |
|---|---|---|---|---|---|---|
| API container, 1 worker | 2,775 | 64 | 100 | 150 | 170 | 0 |
| 4 workers (`WEB_CONCURRENCY=4`) | 2,775 | 37 | 66 | 130 | 180 | 0 |
| 4 workers + `OMP_NUM_THREADS=1` | 2,775 | 28 | 44 | 61 | 100 | 0 |
| same, retrained model (438 features), two runs | 2,775 each | 27 to 29 | 47 to 53 | 59 to 130 | 130 to 190 | 0 |

With prediction logging to Postgres switched on (Phase 4), same test, 4 workers,
`OMP_NUM_THREADS=1`, all in Docker on the same Mac:

| Setup | p50 | p95 | p99 | max | Failures |
|---|---|---|---|---|---|
| logging off, two runs | 35, 32 | 54, 49 | 69, 87 | 140, 130 | 0 |
| logging on, first version, warm, three runs | 35, 37, 40 | 71, 73, 79 | 140, 110, 99 | 190, 139, 121 | 0 |
| logging on, batched writes, first run after start | 26 | 65 | 160 | 214 | 0 |
| logging on, batched writes, warm | 28 | 50 | 120 | 161 | 0 |

The first version of the logger cost about 20 ms at p95. Postgres was not the
cause (`docker stats` during the run: Postgres 3 to 7% of one CPU, API 15 to
25%). The cause was the writer thread: it woke as soon as one row arrived and
wrote immediately, so 2,775 requests produced 1,233 commits (about 2.3 rows
each; counted with `pg_stat_database.xact_commit`). It now collects rows for up
to `flush_seconds` (1 s) or `batch_size` (200) before one write: 208 commits for
the same run, about 13 rows each, and p50 and p95 came back to the logging-off
level. p99 (120 ms here) is still above the 100 ms target on this run but inside
the 59 to 130 ms range seen without logging, and single runs on a shared laptop
vary by about 10 ms, so treat it as unresolved, not fixed. The trade-off of
batching is that a row can reach Postgres up to a second after the request.
The first run after a restart or rebuild has a worse tail than later ones.

The rows from the third down in the earlier container table use the current `docker/docker-compose.yml`. The
first run after a rebuild had a worse tail (p99 130 ms) than the second (59 ms),
so quote the range, not the best run. LightGBM starts one
OpenMP thread per core in every process by default; with 4 workers that was
~490 threads in the container (`docker stats` PIDs) competing for CPU on
single-row predictions that gain nothing from threading. Pinning to one
thread per worker dropped that to 58 PIDs and cut p99 from 130 to 61 ms. The
same setting may help a native multi-worker run; I have not tested that.
`docker stats` during the run showed the API at ~20% of the 8 allocated
CPUs and Redis under 2%, so this setup is nowhere near compute-bound at
47 req/s.

Natively, p99 stays under the 100 ms target up to about 47 req/s. At the 100-user run
p95 is still 42 ms but p99 exceeds 100 ms, so the practical ceiling for this
setup is between 47 and ~85 req/s. Single runs on a shared laptop CPU, so
treat the numbers as indicative, not a benchmark.

What moved the numbers: the first version (one process, pandas frame per
request, saturating load with no pause between requests) gave p50 380 ms and
p95 550 ms at 46 req/s, all of it queueing behind ~21 ms of per-request CPU.
Per-request profiling showed 5.5 ms building a 439-column DataFrame, 7.8 ms
in `predict_proba` and 1.6 ms building the entity ID with a one-row frame.
Replacing the frame with a NumPy row plus training-time category codes,
building the entity ID with string formatting, and running 4 workers removed
most of that. `tests/test_serving.py` checks the fast prediction equals the
pandas path and the entity ID equals `build_entity_id`.

Late-arriving transactions (older than the latest one already seen for that
entity) get `entity_time_since_last_sec = NaN`, not a negative gap, and do not
move the entity's stored latest time or device backwards.

Known limits: read-then-write on entity state is not atomic under concurrent
requests for the same entity (see `store.py`), and `/score` mutates state, so
retries of the same transaction double count. No alert threshold is applied
yet; the endpoint returns the raw score.

## Phase 4: monitoring and drift

Every `/score` call now writes a row to Postgres (`predictions`): transaction
ID, entity ID, model version (a hash of the model file), score, latency, and
the model's non-missing inputs as JSON, so drift is measured on exactly what the
model received. `POST /outcome` fills in the true label later.

**Logging never blocks scoring.** `log()` only drops a dict on an in-memory
queue; a background thread writes batches. If Postgres is slow or down the queue
fills and new rows are dropped and counted (`/health` shows written, dropped,
failed, queued). Scoring never waits on, or fails because of, the database.
The cost is that during an outage predictions go unlogged.

**Drift = PSI** (Population Stability Index) per feature and on the score,
computed by `src/monitoring/drift.py`. Bins come from the reference data's
deciles plus a missing bin; categoricals use top categories plus other and
missing. Rule of thumb: under 0.1 stable, 0.1 to 0.25 moderate, over 0.25
significant. Those are textbook cutoffs, and a single cutoff does not work here:
features differ hugely in how much they normally move (`id_31` sits around
PSI 0.8 between ordinary 3,000-row windows of validation data, `TransactionAmt`
near 0.02). So `reference.py` also measures each monitored feature's and the
score's PSI across consecutive validation windows and stores the p95. A
feature's alert level is max(0.25, 2 x its p95); the score's is max(0.10,
3 x its p95). Drift means the score passes its level, or at least 2 features
pass theirs, or any one feature passes its severe level: max(0.25, 2 x its
worst PSI across the clean calibration windows). The third rule was added
after testing (see "How sensitive is it"). The monitor refuses to decide on fewer than 500 rows and notes
when the window is much smaller than the 3,000 rows the levels were
calibrated on. These levels come from one sample of 39 windows of one
dataset: a calibration, not a guarantee, and with ~30 features an occasional
chance alert is expected, which is why two are required.
The reference (`models/reference.json`, built once per model by
`src.monitoring.reference`) covers the top 30 features by model gain;
feature references come from the training split, the score reference from the
model's validation-split scores (training rows would look more confident than
new data and show false drift on day one). Predictions are only compared with
the reference of the same model version.

```
docker compose -f docker/docker-compose.yml up -d --build      # Redis + Postgres (host port 5433) + API
python -m src.features.backfill --flush                        # warm Redis with entity history up to the training cutoff
python -m src.monitoring.reference                             # once per model: writes models/reference.json
python -m src.monitoring.replay --n 3000                       # send validation-period traffic
python -m src.monitoring.monitor --last-n 3000                 # drift report (exit code 0 ok, 2 drifted, 3 too little data)
python -m src.monitoring.replay --n 3000 --amt-multiplier 3    # inject a shift: amounts x3
python -m src.monitoring.monitor --last-n 3000                 # should report drift
# a different slice of validation traffic (cutoff = the "first TransactionDT" that replay prints for that slice):
python -m src.features.backfill --flush --cutoff-dt <first TransactionDT of the slice>
python -m src.monitoring.replay --n 3000 --offset 50000
docker exec -it $(docker ps -qf name=postgres) psql -U fraud -c "select ts, n_rows, score_psi, drifted from drift_reports"
```

### How sensitive is it

The first sweep (amounts multiplied on the first 3,000 validation rows) looked
like it worked, but it was misleading: `D1` alerts on that slice even with no
shift, so any single shifted feature was counted as the required second alert.
Rerunning on clean slices (validation rows 20,000 and 50,000, 3,000 rows each,
Redis backfilled to each slice's start) with only the score and
two-feature rules gave:

| Amount x | Score PSI (slice 20k / 50k) | `TransactionAmt` PSI | Result (20k / 50k) |
|---|---|---|---|
| 1.25 | 0.023 / 0.003 | 0.47 / 0.41 | OK / OK |
| 1.5 | 0.039 / 0.008 | 0.57 / 0.45 | OK / OK |
| 2 | 0.062 / 0.017 | 0.88 / 0.67 | OK / OK |
| 3 | 0.136 / 0.055 | 1.82 / 1.56 | DRIFT / **OK** |

Tripling every amount was missed on one slice: the model's score barely reacts
to amounts, and one alerting feature was not enough. The feature-level PSI was
clear, the decision rule threw it away. A simple fix (any feature at 2x its
alert level) was tested against the calibration windows first and rejected:
`card6`, `card5`, `R_emaildomain` and `card2` already reach 2x their level in
clean windows. The severe rule uses each feature's own worst clean window
instead (`TransactionAmt`: worst clean PSI 0.143, severe level 0.287). Result
on the same two slices:

| Amount x | Result (20k / 50k) | Fired |
|---|---|---|
| 1.0 (clean) | OK / OK | nothing |
| 1.25 | DRIFT / DRIFT | `TransactionAmt` severe |
| 1.5 | DRIFT / DRIFT | `TransactionAmt` severe |
| 2 | DRIFT / DRIFT | `TransactionAmt` severe |
| 3 | DRIFT / DRIFT | `TransactionAmt` severe (+ score on slice 20k) |

Limits, stated plainly. The thresholds and the test slices both come from the
validation period, so this is not an out-of-sample test. Only one kind of shift
(amount scaling) on two slices was tried, and each had one clean control run.
The severe rule watches stable features tightly and noisy ones (`id_31`,
`card6`) loosely: a moderate shift in those would be missed. The score rule
alone catches only large shifts, because the model is insensitive to amounts.

### What the first live run taught

The first replay of ordinary validation-period traffic was flagged as drift
by the original rule (3 features past 0.25). Checking against the parquet
offline, each alert had a different cause:

- `entity_txn_seq_num` PSI 2.68 live vs 0.17 offline on the same rows. The
  Redis store was cold, so every entity looked new. That is a real
  training/serving skew: a fresh deployment knows no history. Fixed by
  `src/features/backfill.py`, which loads each entity's count, latest time,
  last device and last 24h of events into Redis.
  `tests/test_backfill.py` checks that features computed after a backfill
  equal the offline pipeline's (synthetic data, and real data when present).
  Replaying twice also inflated it, since the first replay changed the state.
- `id_31` (browser) PSI 0.635 is real drift in the data (the browser mix
  shifted over time; the share of unseen browsers is unchanged, 2.2% vs
  2.3%) and does not hurt the model. The live number matched the offline
  computation exactly.
- `D1` PSI 0.443 came from a 3,000-row window covering a short slice of time;
  over the whole validation period it is 0.024. Short windows are noisy.

The score PSI on this traffic was 0.014. My first fix (score-led rule, with
feature drift counting only if 20% of features alerted) then missed a real
shift: with amounts tripled and a warm store, `TransactionAmt` PSI was 1.745
but the score PSI only 0.150 (I had judged the rule from a cold-store run
whose score PSI of 0.39 was inflated by the seq_num artifact). Offline across
39 consecutive validation windows the score PSI was median 0.009, p95 0.028,
max 0.107, so 0.150 was abnormal but under the fixed 0.25. That is why alert
levels are now calibrated from those windows instead of fixed.

`monitor --retrain` runs the training pipeline when drift is found. Honest
limit: today that retrains on the same static IEEE-CIS file, so it exercises
the trigger path but would not fix real drift. A useful retrain needs new
labelled data (logged predictions joined to outcomes via the `label` column),
and the API has to be redeployed to load a new model. Both belong to Phase 5.

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On macOS, LightGBM needs OpenMP, which isn't preinstalled: `brew install libomp`.

## Roadmap

- Phase 5, done so far: `.github/workflows/ci.yml` runs pytest (with a real
  Postgres service, so the end-to-end logging test runs) and builds the API
  image on every push and pull request. Dependencies are pinned, since the
  model is a pickle and must load with the scikit-learn and LightGBM versions
  it was trained with. The IEEE-CIS data is not in the repo, so CI checks
  training/serving parity on synthetic data built to hit the edge cases
  (same-second ties, both sides of the 1h and 24h window edges, missing
  devices, singleton entities); the real-data parity tests run locally.
- Phase 5, next: a retrain workflow (download data, retrain, PR-AUC gate,
  publish the model and an image tagged with its version to GHCR). There is no
  deploy target, so redeploy means publishing that image.
