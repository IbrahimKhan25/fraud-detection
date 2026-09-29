"""Drift measurement with the Population Stability Index (PSI).

PSI compares how a feature was distributed in the reference data (training)
with how it is distributed in recent live traffic. Both are turned into
proportions over the same bins; PSI = sum((live - ref) * ln(live / ref)).
0 means identical. Common rule of thumb: < 0.1 stable, 0.1 to 0.25 moderate
shift, > 0.25 significant shift.

Decision rule: drift is reported when (1) the model's score PSI passes its
alert level, (2) at least MIN_ALERT_FEATURES features pass theirs, or (3) any
single feature passes its severe level: twice its worst PSI across the clean
calibration windows (floor 0.25). Rule 3 exists because rules 1 and 2 alone
ignored a 3x amount shift on a quiet feature (TransactionAmt PSI 1.5) when the
model's score barely reacted. It only trusts a lone feature by a wide margin
over anything seen in clean data, so noisy features are effectively exempt.
Alert levels are
calibrated per feature (see alert_levels): a single fixed cutoff cannot work
because features differ enormously in how much they normally move. On
IEEE-CIS, id_31 (browser) sits around PSI 0.8 between ordinary time windows
while TransactionAmt sits near 0.02. A fixed 0.25 both false-alarms on the
first and, combined with a "share of features" rule, missed a 3x amount shift
in testing. The levels come from one sample of 39 validation windows, so they
are a calibration, not a guarantee; with ~30 features an occasional chance
alert is expected, which is why two are required.

Bins: numeric features use the reference data's decile edges plus a separate
"missing" bin (a feature that suddenly goes missing is drift too);
categorical features use the reference's top categories plus "other" and
"missing".

The reference (models/reference.json, built by reference.py) is computed
once per model. This module has no database or model dependency, so it is
plain functions over numbers and easy to test.
"""
import math
from typing import Dict, List, Optional, Sequence

import numpy as np

PSI_WARN = 0.10
PSI_ALERT = 0.25        # textbook "significant shift"; also the floor for any feature's alert level
SCORE_ALERT_FLOOR = 0.10
FEATURE_P95_MULT = 2.0  # feature alert level = max(PSI_ALERT, 2 x its normal p95)
SCORE_P95_MULT = 3.0    # score alert level   = max(SCORE_ALERT_FLOOR, 3 x its normal p95)
SEVERE_MAX_MULT = 2.0   # severe level = max(PSI_ALERT, 2 x the feature's worst clean-window PSI)
MIN_ALERT_FEATURES = 2  # this many features past their own level counts as drift
MIN_ROWS = 500          # below this, PSI is too noisy to act on
EPS = 1e-4              # floor for empty bins so ln() stays finite

OTHER = "__other__"
MISSING = "__missing__"


def psi(ref: Sequence[float], cur: Sequence[float]) -> float:
    ref = np.clip(np.asarray(ref, dtype="float64"), EPS, None)
    cur = np.clip(np.asarray(cur, dtype="float64"), EPS, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def numeric_edges(values: np.ndarray, n_bins: int = 10) -> List[float]:
    v = values[~np.isnan(values)]
    if len(v) == 0:
        return []
    return [float(e) for e in np.unique(np.quantile(v, np.linspace(0, 1, n_bins + 1)[1:-1]))]


def numeric_props(values: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    """Proportions over len(edges)+1 value bins followed by one missing bin."""
    values = np.asarray(values, dtype="float64")
    nan = np.isnan(values)
    idx = np.searchsorted(np.asarray(edges, dtype="float64"), values[~nan], side="left")
    counts = np.bincount(idx, minlength=len(edges) + 1).astype("float64")
    counts = np.append(counts, nan.sum())
    total = counts.sum()
    return counts / total if total else counts


def categorical_props(values: Sequence[Optional[str]], categories: Sequence[str]) -> np.ndarray:
    """Proportions over `categories`, then OTHER, then MISSING."""
    pos = {c: i for i, c in enumerate(categories)}
    counts = np.zeros(len(categories) + 2)
    for v in values:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            counts[-1] += 1
        else:
            counts[pos.get(str(v), len(categories))] += 1
    total = counts.sum()
    return counts / total if total else counts


def _to_float(v) -> float:
    try:
        return float("nan") if v is None else float(v)
    except (TypeError, ValueError):
        return float("nan")


def column_psi(spec: Dict, values: Sequence) -> float:
    """PSI of one monitored column against its reference spec. `values` are raw
    values (floats, strings, None or NaN for missing)."""
    if spec["type"] == "numeric":
        cur = numeric_props(np.array([_to_float(v) for v in values]), spec["edges"])
    else:
        cur = categorical_props(values, spec["categories"])
    return psi(spec["props"], cur)


def alert_levels(reference: Dict) -> Dict:
    """PSI level at which each feature (and the score) counts as alerting.

    Calibrated from how much each one normally moves between equal-sized
    windows of held-out data (reference["baseline"], built by reference.py):
    a feature like id_31 that always sits around PSI 0.8 needs a much higher
    level than one that normally sits near 0.02. Without a baseline (older
    reference files) it falls back to fixed levels."""
    base = reference.get("baseline") or {}
    feats = {}
    for name in reference["features"]:
        p95 = (base.get("features") or {}).get(name, {}).get("p95")
        feats[name] = PSI_ALERT if p95 is None else max(PSI_ALERT, FEATURE_P95_MULT * p95)
    sp95 = (base.get("score") or {}).get("p95")
    score = PSI_ALERT if sp95 is None else max(SCORE_ALERT_FLOOR, SCORE_P95_MULT * sp95)
    severe = {}
    for name in reference["features"]:
        fmax = (base.get("features") or {}).get(name, {}).get("max")
        if fmax is not None:  # no baseline, no single-feature rule
            severe[name] = max(PSI_ALERT, SEVERE_MAX_MULT * fmax)
    return {"features": feats, "score": score, "severe": severe, "window_rows": base.get("window_rows")}


def evaluate(rows: List[Dict], reference: Dict, min_rows: int = MIN_ROWS) -> Dict:
    """rows: [{"fraud_score": float, "features": {name: value}}, ...] from the
    prediction log. Returns a report dict; `drifted` is the go/no-go."""
    n = len(rows)
    levels = alert_levels(reference)
    report: Dict = {
        "n_rows": n,
        "model_version": reference.get("model_version"),
        "score_psi": None,
        "score_level": levels["score"],
        "features": {},
        "levels": levels["features"],
        "severe_levels": levels["severe"],
        "severe_features": [],
        "warn_features": [],
        "alert_features": [],
        "drifted": False,
        "status": "ok",
        "reason": "",
        "note": "",
    }
    if n < min_rows:
        report["status"] = "insufficient_data"
        report["reason"] = f"{n} rows in window, need at least {min_rows}"
        return report
    w = levels["window_rows"]
    if w and n < w / 2:
        report["note"] = f"alert levels were calibrated on {w}-row windows; a {n}-row window is noisier"

    scores = np.array([r["fraud_score"] for r in rows], dtype="float64")
    sref = reference["score"]
    report["score_psi"] = psi(sref["props"], numeric_props(scores, sref["edges"]))

    for name, spec in reference["features"].items():
        report["features"][name] = column_psi(spec, [r["features"].get(name) for r in rows])

    report["alert_features"] = sorted(k for k, v in report["features"].items() if v >= levels["features"][k])
    report["warn_features"] = sorted(
        k for k, v in report["features"].items() if PSI_WARN <= v and k not in report["alert_features"]
    )

    report["severe_features"] = sorted(
        k for k, v in report["features"].items() if k in levels["severe"] and v >= levels["severe"][k]
    )

    reasons = []
    if report["score_psi"] >= levels["score"]:
        reasons.append(f"score PSI {report['score_psi']:.3f} >= its alert level {levels['score']:.3f}")
    if len(report["alert_features"]) >= MIN_ALERT_FEATURES:
        reasons.append(
            f"{len(report['alert_features'])} features past their alert level ({', '.join(report['alert_features'])})"
        )
    if report["severe_features"]:
        reasons.append(f"far past normal on its own ({', '.join(report['severe_features'])})")
    report["drifted"] = bool(reasons)
    report["status"] = "drifted" if reasons else "ok"
    report["reason"] = "; ".join(reasons)
    return report
