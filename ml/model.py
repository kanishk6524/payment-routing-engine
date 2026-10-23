"""Success-probability model (Stage 5): P(success | transaction, gateway, recent gateway health).

XGBoost is the model. A scikit-learn gradient-boosting stand-in (`--model hgb`) exists only for
environments where XGBoost cannot be installed. Both consume the same encoded features.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .features import CATEGORICAL_FEATURES, FEATURE_COLUMNS

ARTIFACT_PATH = "ml/artifacts/model.joblib"
MODEL_KINDS = ("xgboost", "hgb")


def learn_levels(train: pd.DataFrame) -> dict:
    """Category levels seen in training. Saved with the model so serving encodes identically."""
    return {c: sorted(train[c].unique()) for c in CATEGORICAL_FEATURES}


def encode(df: pd.DataFrame, levels: dict) -> pd.DataFrame:
    """Model input: FEATURE_COLUMNS with categorical dtypes on a fixed set of levels.

    A level never seen in training becomes missing instead of raising an error.
    """
    X = df[FEATURE_COLUMNS].copy()
    for col, cats in levels.items():
        X[col] = pd.Categorical(X[col], categories=cats)
    return X


def make_model(kind: str = "xgboost", seed: int = 42):
    if kind == "xgboost":
        try:
            from xgboost import XGBClassifier
        except Exception as exc:  # ImportError, or a missing OpenMP library on macOS
            raise RuntimeError(
                "XGBoost could not be imported. Try `pip install xgboost`; on macOS also "
                "`brew install libomp`. For a scikit-learn stand-in use --model hgb."
            ) from exc
        return XGBClassifier(
            n_estimators=1000, learning_rate=0.05, max_depth=6, min_child_weight=5,
            subsample=0.8, colsample_bytree=0.8, tree_method="hist", enable_categorical=True,
            eval_metric="logloss", early_stopping_rounds=50, n_jobs=-1, random_state=seed,
        )
    if kind == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=50,
            categorical_features="from_dtype", early_stopping=False, random_state=seed,
        )
    raise ValueError(f"unknown model kind {kind!r}; choose from {MODEL_KINDS}")


def fit_model(model, kind: str, X_train, y_train, X_valid, y_valid):
    if kind == "xgboost":
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    else:
        model.fit(X_train, y_train)
    return model


def predict_proba(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def expected_calibration_error(y, p, bins: int = 10) -> float:
    """Average gap between predicted and observed success, weighted by bin size."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    gap = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            gap += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(gap)


def score(y, p) -> dict:
    y = np.asarray(y)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "auc": float(roc_auc_score(y, p)),
        "logloss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
    }


def save_bundle(path: str, bundle: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path: str = ARTIFACT_PATH) -> dict:
    return joblib.load(path)
