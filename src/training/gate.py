"""Promotion gate for a retrained model: decides whether it may replace the
current champion (the model in the latest `model-*` GitHub Release).

    python -m src.training.gate --new models/metrics.json --champion champion/metrics.json

Passes if PR-AUC is at least FLOOR and no more than MAX_DROP below the
champion's. With no champion file (the first release) only the floor applies.
Exit code 0 pass, 1 fail.

The floor sits just under the measured 0.5673 of the current model, so it
catches a broken pipeline (wrong features, lost columns, leakage removed by
accident) rather than rewarding noise. MAX_DROP allows for the small
run-to-run differences LightGBM can show across machines. Both are judgment
calls from one dataset, not tuned values.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

FLOOR = 0.55
MAX_DROP = 0.005


def check(new: dict, champion: Optional[dict], floor: float = FLOOR, max_drop: float = MAX_DROP) -> Tuple[bool, List[str]]:
    """Returns (passed, one line per rule explaining the decision)."""
    pr_auc = new["pr_auc"]
    lines = []
    ok = pr_auc >= floor
    lines.append(f"{'pass' if ok else 'FAIL'}: PR-AUC {pr_auc:.4f} vs floor {floor:.4f}")
    if champion is None:
        lines.append("no champion yet: floor only")
    else:
        minimum = champion["pr_auc"] - max_drop
        beats = pr_auc >= minimum
        ok = ok and beats
        lines.append(
            f"{'pass' if beats else 'FAIL'}: PR-AUC {pr_auc:.4f} vs champion {champion['pr_auc']:.4f} "
            f"({champion.get('model_version', '?')}), minimum {minimum:.4f}"
        )
    return ok, lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", default="models/metrics.json")
    ap.add_argument("--champion", default=None, help="champion metrics.json; missing file means no champion")
    ap.add_argument("--floor", type=float, default=FLOOR)
    ap.add_argument("--max-drop", type=float, default=MAX_DROP)
    args = ap.parse_args(argv)

    new = json.loads(Path(args.new).read_text())
    champion = None
    if args.champion and Path(args.champion).exists():
        champion = json.loads(Path(args.champion).read_text())

    ok, lines = check(new, champion, args.floor, args.max_drop)
    print("\n".join(lines))
    print("gate passed" if ok else "gate failed")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
