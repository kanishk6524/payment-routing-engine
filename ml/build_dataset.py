"""Build the model-ready feature table.

    python -m ml.build_dataset                 # read from PostgreSQL (default)
    python -m ml.build_dataset --source csv    # read data/*.csv instead

Writes data/features.pkl (git-ignored) with one row per attempt, the features, the label
`success` and a chronological `split` column (train / valid / test).
"""
import argparse
from pathlib import Path

from .db import load
from .features import FEATURE_COLUMNS, TARGET, build_features


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", choices=["db", "csv"], default="db")
    ap.add_argument("--data", default="data", help="CSV directory (for --source csv)")
    ap.add_argument("--out", default="data/features.pkl")
    args = ap.parse_args()

    attempts, gateways, _ = load(args.source, args.data)
    print(f"Loaded {len(attempts):,} attempts from {args.source}")

    df = build_features(attempts, gateways)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(args.out)

    print(f"Wrote {args.out}: {len(df):,} rows, {len(FEATURE_COLUMNS)} features\n")
    print(f"{'split':<6} {'rows':>9} {'from':>12} {'to':>12} {'success':>8}")
    for name in ("train", "valid", "test"):
        part = df[df["split"] == name]
        print(f"{name:<6} {len(part):>9,} {part['ts'].min():%Y-%m-%d} "
              f"{part['ts'].max():%Y-%m-%d} {part[TARGET].mean():>8.4f}")
    empty = df[FEATURE_COLUMNS].isna().mean()
    empty = empty[empty > 0]
    if len(empty):
        print("\nShare of rows with no history in the window (stays NaN, XGBoost handles it):")
        print(empty.round(4).to_string())


if __name__ == "__main__":
    main()
