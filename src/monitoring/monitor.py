"""Drift check over recent logged predictions. Run it by hand, from cron,
or from CI (Phase 5).

    python -m src.monitoring.monitor                       # last 60 minutes
    python -m src.monitoring.monitor --window-minutes 15
    python -m src.monitoring.monitor --last-n 5000
    python -m src.monitoring.monitor --retrain             # retrain if drifted

Exit code: 0 ok, 2 drifted, 3 not enough data. The report is also stored in
the drift_reports table so alerts have a history.

--retrain runs the existing training pipeline. Honest limit: with the
current setup that retrains on the same static IEEE-CIS file, so it proves the
trigger path works but would not fix real drift. A useful retrain needs new
labelled data: the logged predictions joined with their outcomes (the
`label` column, filled by POST /outcome). The serving container also has to
be redeployed to pick up a new model file. Both belong to Phase 5.
"""
import argparse
import json
import subprocess
import sys

from src.monitoring import db, drift
from src.monitoring.reference import REFERENCE_PATH

RETRAIN_STEPS = [
    [sys.executable, "-m", "src.training.build_training_set"],
    [sys.executable, "-m", "src.training.train_with_features"],
    [sys.executable, "-m", "src.monitoring.reference"],
]


def fetch_rows(conn, model_version, window_minutes=None, last_n=None):
    if last_n:
        cur = conn.execute(
            "SELECT fraud_score, features FROM predictions WHERE model_version = %s ORDER BY id DESC LIMIT %s",
            (model_version, last_n),
        )
    else:
        cur = conn.execute(
            "SELECT fraud_score, features FROM predictions WHERE model_version = %s "
            "AND ts >= now() - make_interval(mins => %s)",
            (model_version, window_minutes),
        )
    return [{"fraud_score": s, "features": f} for s, f in cur.fetchall()]


def print_report(report):
    print(f"model {report['model_version']}  rows {report['n_rows']}  status {report['status'].upper()}")
    if report["status"] == "insufficient_data":
        print(report["reason"])
        return
    print(f"score PSI: {report['score_psi']:.3f}  (alert level {report['score_level']:.3f})")
    print(f"{'feature':<28}{'PSI':>8}{'alert at':>10}  status")
    for name, v in sorted(report["features"].items(), key=lambda kv: -kv[1] / report["levels"][kv[0]])[:15]:
        lvl = report["levels"][name]
        status = "SEVERE" if name in report["severe_features"] else "ALERT" if v >= lvl else "warn" if v >= drift.PSI_WARN else "ok"
        print(f"{name:<28}{v:>8.3f}{lvl:>10.3f}  {status}")
    print(f"alert features: {len(report['alert_features'])}, severe: {len(report['severe_features'])}, "
          f"warn features: {len(report['warn_features'])}")
    if report["note"]:
        print(f"note: {report['note']}")
    if report["drifted"]:
        print(f"DRIFT: {report['reason']}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-minutes", type=int, default=60)
    ap.add_argument("--last-n", type=int, default=None, help="use the last N predictions instead of a time window")
    ap.add_argument("--min-rows", type=int, default=drift.MIN_ROWS)
    ap.add_argument("--retrain", action="store_true")
    args = ap.parse_args(argv)

    reference = json.loads(REFERENCE_PATH.read_text())
    with db.connect() as conn:
        db.ensure_schema(conn)
        rows = fetch_rows(conn, reference["model_version"], args.window_minutes, args.last_n)
        report = drift.evaluate(rows, reference, min_rows=args.min_rows)
        print_report(report)
        conn.execute(
            "INSERT INTO drift_reports (model_version, n_rows, score_psi, n_alert_features, drifted, details) "
            "VALUES (%s, %s, %s, %s, %s, %s::jsonb)",
            (report["model_version"], report["n_rows"], report["score_psi"], len(report["alert_features"]),
             report["drifted"], json.dumps(report)),
        )

    if report["status"] == "insufficient_data":
        return 3
    if report["drifted"]:
        if args.retrain:
            print("\nretraining...")
            for step in RETRAIN_STEPS:
                print("$", " ".join(step))
                subprocess.run(step, check=True)
            print("retrain finished; redeploy the API to load the new model (Phase 5)")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
