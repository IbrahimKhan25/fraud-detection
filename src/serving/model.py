"""Wraps the trained LightGBM model so a single transaction dict can be
scored with exactly the column order and dtypes the model was trained on.

Two things make single-row scoring differ from the training path:

1. Column order/set. The model expects its training columns in training
   order. Anything the request doesn't supply is NaN.
2. Categoricals. Training used pandas `category` dtype on object columns.
   LightGBM remembers the category list per column and re-maps incoming
   values onto it, but only if the incoming column is also `category`
   dtype and the columns are in the same order. Values never seen in
   training become NaN, same as a missing value.

Speed: building a 439-column pandas frame per request cost ~5 ms, about a
third of the whole request. predict() therefore skips pandas: it fills a
NumPy row directly and maps each categorical string to the integer code
LightGBM assigned during training (the position in the booster's
`pandas_categorical` list; unseen values stay NaN, same as the pandas path),
then calls the booster directly. `predict_reference()` keeps the original
pandas path so a test can prove the two agree.

Which columns are categorical is read from the model itself
(`feature_infos` in the model dump: categorical features list their
category values, numeric ones list a [min:max] range), so this doesn't
depend on a side file that could drift from the pickle.
"""
import hashlib
import math
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd


def model_version(path) -> str:
    """First 12 hex chars of the model file's SHA-256, so every logged
    prediction says exactly which model file produced it. Training uses the
    same function to name the release, so the two always agree."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


class FraudModel:
    def __init__(self, model, version: str = "unversioned"):
        self.model = model
        self.version = version
        self.feature_names: List[str] = list(model.booster_.feature_name())
        info = model.booster_.dump_model(num_iteration=1)["feature_infos"]
        self.categorical_cols = {
            name for name, fi in info.items() if fi.get("values")
        }
        self.numeric_cols = [c for c in self.feature_names if c not in self.categorical_cols]

        # Fast-path lookup tables. pandas_categorical lists each categorical
        # column's training categories, in the order those columns appear.
        cat_cols_in_order = [c for c in self.feature_names if c in self.categorical_cols]
        cat_lists = model.booster_.pandas_categorical or []
        if len(cat_lists) != len(cat_cols_in_order):
            raise ValueError(
                f"model has {len(cat_lists)} pandas category lists but {len(cat_cols_in_order)} categorical columns"
            )
        self._cat_maps = {
            c: {str(v): i for i, v in enumerate(cats)} for c, cats in zip(cat_cols_in_order, cat_lists)
        }
        self._index = {c: i for i, c in enumerate(self.feature_names)}
        # Early stopping picked best_iteration_; sklearn's predict_proba uses
        # it, so the direct booster call must too.
        self._num_iteration = getattr(model, "best_iteration_", None) or None

    @classmethod
    def load(cls, path) -> "FraudModel":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"model file not found: {path}")
        return cls(joblib.load(path), version=model_version(path))

    def loggable(self, values: Dict[str, object]) -> Dict[str, object]:
        """The model's inputs that are actually present (missing ones are
        left out: an absent key means missing). This is what gets logged."""
        idx = self._index
        return {k: v for k, v in values.items() if k in idx and v is not None and v == v}

    def _row(self, values: Dict[str, object]) -> np.ndarray:
        row = np.full((1, len(self.feature_names)), np.nan)
        index, cat_maps = self._index, self._cat_maps
        for name, v in values.items():
            i = index.get(name)
            if i is None or v is None:
                continue
            cmap = cat_maps.get(name)
            if cmap is not None:
                code = cmap.get(str(v))
                if code is not None:
                    row[0, i] = code
            else:
                try:
                    row[0, i] = float(v)
                except (TypeError, ValueError):
                    pass
        return row

    def _frame(self, values: Dict[str, object]) -> pd.DataFrame:
        row = {}
        for c in self.feature_names:
            v = values.get(c)
            if c in self.categorical_cols:
                row[c] = None if v is None or (isinstance(v, float) and math.isnan(v)) else str(v)
            else:
                try:
                    row[c] = np.nan if v is None else float(v)
                except (TypeError, ValueError):
                    row[c] = np.nan
        df = pd.DataFrame([row], columns=self.feature_names)
        for c in self.categorical_cols:
            df[c] = df[c].astype("category")
        return df

    def predict(self, values: Dict[str, object]) -> float:
        return float(self.model.booster_.predict(self._row(values), num_iteration=self._num_iteration)[0])

    def predict_reference(self, values: Dict[str, object]) -> float:
        """Original, slower pandas path. Kept so tests can check predict()."""
        return float(self.model.predict_proba(self._frame(values))[0, 1])
