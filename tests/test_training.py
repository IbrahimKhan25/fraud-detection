"""The retrain gate and the metrics it reads. No data or real model needed."""
import json

import numpy as np
import pytest

from src.serving.model import model_version
from src.training import gate
from src.training.train_with_features import evaluate


def test_gate_floor_only_without_champion():
    assert gate.check({"pr_auc": 0.56}, None)[0]
    assert not gate.check({"pr_auc": 0.54}, None)[0]


def test_gate_against_champion():
    champ = {"pr_auc": 0.5673, "model_version": "abc"}
    assert gate.check({"pr_auc": 0.5673}, champ)[0]
    assert gate.check({"pr_auc": 0.5630}, champ)[0]      # within MAX_DROP
    assert not gate.check({"pr_auc": 0.5620}, champ)[0]  # more than MAX_DROP below
    assert not gate.check({"pr_auc": 0.54}, {"pr_auc": 0.54})[0]  # the floor still applies


def test_gate_cli_exit_codes(tmp_path):
    new, champ = tmp_path / "new.json", tmp_path / "champ.json"
    new.write_text(json.dumps({"pr_auc": 0.56}))
    champ.write_text(json.dumps({"pr_auc": 0.60}))
    assert gate.main(["--new", str(new), "--champion", str(tmp_path / "missing.json")]) == 0
    assert gate.main(["--new", str(new), "--champion", str(champ)]) == 1


def test_evaluate_alert_rates():
    y = np.array([0] * 90 + [1] * 10)
    scores = np.linspace(0, 1, 100)  # the 10 positives get the 10 highest scores
    m = evaluate(y, scores)
    assert m["pr_auc"] == pytest.approx(1.0)
    assert m["random_baseline"] == pytest.approx(0.1)
    assert m["alert_rates"]["0.05"] == {"precision": 1.0, "recall": 0.5}


def test_model_version_is_file_hash(tmp_path):
    a, b = tmp_path / "a.pkl", tmp_path / "b.pkl"
    a.write_bytes(b"model")
    b.write_bytes(b"model")
    assert model_version(a) == model_version(b) and len(model_version(a)) == 12
    b.write_bytes(b"other")
    assert model_version(a) != model_version(b)
