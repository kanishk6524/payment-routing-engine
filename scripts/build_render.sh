#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python -m pip install -r requirements-deploy.txt
python -m simulator.generate --n 300000 --days 60 --seed 42 --out data
python -m ml.train --source csv

# Training uses the full history. Serving only needs the latest 75 minutes,
# which HealthTracker.seed shifts to the present on every startup.
python - <<'PY'
import pandas as pd

path = "data/transactions.csv"
history = pd.read_csv(path, parse_dates=["ts"])
cutoff = history["ts"].max() - pd.Timedelta(minutes=75)
recent = history.loc[history["ts"] >= cutoff]
recent.to_csv(path, index=False)
print(f"Bundled {len(recent):,} recent attempts for demo startup")
PY
