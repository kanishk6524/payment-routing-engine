"""Diagnose whether XGBoost's explanation values add up to the model's raw output.

    python scripts/debug_explain.py
"""
import numpy as np
import xgboost as xgb

from ml.db import load
from ml.features import build_features
from ml.model import encode, load_bundle

b = load_bundle()
m = b["model"]
attempts, gateways, _ = load("csv", "data")
df = build_features(attempts, gateways)
X = encode(df[df["split"] == "test"].iloc[:500], b["levels"])
booster = m.get_booster()
dm = xgb.DMatrix(X, enable_categorical=True)
n_trees = booster.num_boosted_rounds()
best = getattr(m, "best_iteration", None)
print(f"xgboost {xgb.__version__}; rounds in booster: {n_trees}; best_iteration: {best}")

margin_best = m.predict(X, output_margin=True)
margin_all = booster.predict(dm, output_margin=True)
print("margin (sklearn predict) vs margin (all rounds), max abs diff:",
      float(np.abs(margin_best - margin_all).max()))

for label, rng in (("all rounds", (0, 0)), ("up to best_iteration", (0, int(best) + 1) if best is not None else (0, 0))):
    c = booster.predict(dm, pred_contribs=True, iteration_range=rng)
    total = c.sum(axis=1)
    ref = booster.predict(dm, output_margin=True, iteration_range=rng)
    print(f"{label}: sum(contribs) vs booster margin, max abs diff = {np.abs(total - ref).max():.6f}")

tree_df = booster.trees_to_dataframe()
cat = [f for f in X.columns if str(X[f].dtype) == "category"]
used = set(tree_df.loc[tree_df["Feature"] != "Leaf", "Feature"])
print("categorical features used in splits:", sorted(used & set(cat)))
print("split types:", dict(tree_df.loc[tree_df["Feature"] != "Leaf", "Split"].astype(str).str[:0].value_counts()) if False else "n/a")
