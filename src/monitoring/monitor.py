"""Drift check over recent logged predictions. Run it by hand or from cron.

    python -m src.monitoring.monitor                       # last 60 minutes
    python -m src.monitoring.monitor --window-minutes 15
    python -m src.monitoring.monitor --last-n 5000
    python -m src.monitoring.monitor --retrain             # if drifted, start the Retrain workflow on GitHub
    python -m src.monitoring.monitor --retrain-local       # if drifted, retrain on this machine instead
    python -m src.monitoring.monitor --reference path/to/reference.json

Exit code: 0 ok, 2 drifted, 3 not enough data. The report is also stored in
the drift_reports table so alerts have a history.

--reference: predictions are only compared with the reference of the same
model version. The published image's model (a `model-*` GitHub Release) is not
byte-identical to a locally trained one, so monitor it with the
reference.json from that Release, not models/reference.json.

--retrain runs `gh workflow run retrain.yml` (needs the gh CLI, logged in);
the workflow gates the new model and publishes it as a Release and a GHCR
image. --retrain-local runs the same training steps here. Honest limit for
both: they retrain on the same static IEEE-CIS file, so they prove the
trigger path works but would not fix real drift. A useful retrain needs new
labelled data: the logged predictions joined with their outcomes (the
`label` column, filled by POST /outcome).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from src.monitoring import db, drift
from src.monitoring.reference import REFERENCE_PATH

RETRAIN_STEPS = [
    [sys.executable, "-m", "src.training.build_training_set"],
    [sys.executable, "-m", "src.training.train_with_features"],
    [sys.executable, "-m", "src.monitoring.reference"],
]
RETRAIN_WORKFLOW = ["gh", "workflow", "run", "retrain.yml", "--ref", "main"]


def trigger_retrain(local: bool) -> None:
    steps = RETRAIN_STEPS if local else [RETRAIN_WORKFLOW]
    print("\nretraining" + (" locally" if local else " on GitHub") + "...")
    for step in steps:
        print("$", " ".join(step))
        subprocess.run(step, check=True)
    if local:
        print("retrain finished; rebuild or restart the API to load the new model")
    else:
        print("Retrain workflow started; if its gate passes it publishes a Release and a GHCR image")


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
    ap.add_argument("--reference", type=Path, default=REFERENCE_PATH,
                    help="reference.json of the model being monitored (default models/reference.json)")
    retrain = ap.add_mutually_exclusive_group()
    retrain.add_argument("--retrain", action="store_true", help="if drifted, start the Retrain workflow on GitHub")
    retrain.add_argument("--retrain-local", action="store_true", help="if drifted, retrain on this machine")
    args = ap.parse_args(argv)

    reference = json.loads(args.reference.read_text())
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
        if args.retrain or args.retrain_local:
            trigger_retrain(local=args.retrain_local)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
