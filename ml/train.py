"""Train the success-probability model on the chronological training split.

    python -m ml.train                  # XGBoost, data from PostgreSQL
    python -m ml.train --source csv     # data from data/*.csv
    python -m ml.train --model hgb      # scikit-learn stand-in (only if XGBoost is unavailable)

The validation split is used for early stopping. The test split is touched only for the final
report. Writes ml/artifacts/model.joblib and ml/artifacts/metrics.json.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from .db import load
from .features import FEATURE_COLUMNS, TARGET, build_features
from .model import (
    ARTIFACT_PATH, MODEL_KINDS, encode, fit_model, learn_levels, make_model, predict_proba,
    save_bundle, score,
)


def calibration_table(y, p, bins: int = 10) -> pd.DataFrame:
    t = pd.DataFrame({"y": y, "p": p})
    t["bin"] = pd.qcut(t["p"], bins, duplicates="drop")
    return (t.groupby("bin", observed=True)
            .agg(attempts=("y", "size"), predicted=("p", "mean"), observed=("y", "mean"))
            .round(4).reset_index(drop=True))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", choices=["db", "csv"], default="db")
    ap.add_argument("--data", default="data")
    ap.add_argument("--model", choices=MODEL_KINDS, default="xgboost")
    ap.add_argument("--out", default=ARTIFACT_PATH)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    attempts, gateways, _ = load(args.source, args.data)
    df = build_features(attempts, gateways)
    parts = {s: df[df["split"] == s] for s in ("train", "valid", "test")}
    print(f"features: {len(FEATURE_COLUMNS)}; rows train/valid/test = "
          f"{len(parts['train']):,} / {len(parts['valid']):,} / {len(parts['test']):,}")

    levels = learn_levels(parts["train"])
    X = {s: encode(p, levels) for s, p in parts.items()}
    y = {s: p[TARGET].to_numpy() for s, p in parts.items()}

    model = make_model(args.model, args.seed)
    fit_model(model, args.model, X["train"], y["train"], X["valid"], y["valid"])
    if args.model == "xgboost":
        print(f"trained XGBoost: {model.best_iteration + 1} trees (early stopping on validation)")

    prior = parts["train"].groupby("gateway_id")[TARGET].mean()
    metrics = {}
    print(f"\n{'split':<6} {'model':<14} {'AUC':>7} {'logloss':>9} {'brier':>8} {'ECE':>7}")
    for s in ("valid", "test"):
        p_model = predict_proba(model, X[s])
        p_prior = parts[s]["gateway_id"].map(prior).to_numpy()
        metrics[s] = {"model": score(y[s], p_model), "gateway_prior": score(y[s], p_prior)}
        for name, m in (("gateway prior", metrics[s]["gateway_prior"]), (args.model, metrics[s]["model"])):
            print(f"{s:<6} {name:<14} {m['auc']:>7.4f} {m['logloss']:>9.4f} "
                  f"{m['brier']:>8.4f} {m['ece']:>7.4f}")

    print("\nCalibration on the test split (predicted vs observed success, by decile)")
    print(calibration_table(y["test"], predict_proba(model, X["test"])).to_string(index=False))

    if args.model == "xgboost":
        gain = model.get_booster().get_score(importance_type="gain")
        top = sorted(gain.items(), key=lambda kv: -kv[1])[:10]
        total = sum(gain.values())
        print("\nTop features by gain")
        for name, g in top:
            print(f"  {name:<24} {100 * g / total:5.1f}%")

    bundle = {
        "model": model, "kind": args.model, "levels": levels,
        "feature_columns": FEATURE_COLUMNS,
        "version": f"{args.model}-{pd.Timestamp.now():%Y%m%d%H%M}",
        "metrics": metrics,
        "latency_fallback": parts["train"].groupby("gateway_id")["latency_ms"].mean().to_dict(),
        "trained_on": {"from": str(parts["train"]["ts"].min()), "to": str(parts["train"]["ts"].max())},
    }
    save_bundle(args.out, bundle)
    Path(args.out).with_name("metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved {args.out} (version {bundle['version']})")


if __name__ == "__main__":
    main()
