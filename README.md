# Real-Time Fraud Detection System

Train a model to flag fraudulent transactions, then build the infrastructure
to serve it in real time with monitoring and retraining, not just a notebook
that outputs an AUC score.

## Status

Phase 1 (data and baseline model) done. See results below. Phase 2 (feature
pipeline) not started.

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

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On macOS, LightGBM needs OpenMP, which isn't preinstalled: `brew install libomp`.

## Roadmap

- Phase 2: feature pipeline computing rolling counts/velocity per entity as
  they'd actually be available at inference time, backed by a simple feature
  store (Redis or Postgres) so training and serving share the same logic.
- Phase 3: FastAPI serving endpoint, containerized, load tested (Locust/k6)
  for real latency numbers.
- Phase 4: prediction logging, drift detection (PSI or KS test), retraining
  trigger.
- Phase 5: CI/CD, scheduled or drift-triggered retraining and redeploy.
