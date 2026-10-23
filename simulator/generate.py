"""Generate simulated historical transactions.

Run from the project root:

    python -m simulator.generate --n 300000 --days 60 --seed 42 --out data

Outputs (CSV, git-ignored):
    data/transactions.csv  one row per payment attempt
    data/gateways.csv      public gateway attributes (fees, supported methods)
    data/outages.csv       ground-truth incident windows (NOT for use as features)

Logging policy: each historical transaction is routed to a uniformly random
gateway among those supporting its payment method. This mimics an exploration
phase and avoids selection bias, so the model can learn every gateway's
behaviour. Counterfactual outcomes for any policy can be obtained from
`simulator.model.success_probability`.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    BANK_PROBS, BANKS, CATEGORIES, CATEGORY_AMOUNT_SCALE, CATEGORY_PROBS,
    DOW_WEIGHTS, GATEWAYS, HOUR_WEIGHTS, MAX_AMOUNT_INR, METHOD_PROBS, METHODS,
    MIN_AMOUNT_INR, UPI_LIMIT_INR, resolve_outages,
)
from .model import fee_inr, sample_outcomes


def _timestamps(n: int, start: pd.Timestamp, days: int, rng: np.random.Generator):
    day_idx = np.arange(days)
    day_w = np.array(DOW_WEIGHTS)[(start.day_of_week + day_idx) % 7]
    day_w = day_w / day_w.sum()
    hour_w = np.array(HOUR_WEIGHTS) / np.sum(HOUR_WEIGHTS)

    d = rng.choice(days, n, p=day_w)
    h = rng.choice(24, n, p=hour_w)
    sec = rng.uniform(0, 3600, n)
    return (start + pd.to_timedelta(d * 86400 + h * 3600 + sec, unit="s")).floor("s")


def generate(n: int = 300_000, days: int = 60, seed: int = 42, start: str = "2026-01-01"):
    """Return (transactions DataFrame, resolved outages)."""
    rng = np.random.default_rng(seed)
    start_ts = pd.Timestamp(start)
    outages = resolve_outages(start_ts, days)

    category = rng.choice(CATEGORIES, n, p=CATEGORY_PROBS)
    scale = pd.Series(category).map(CATEGORY_AMOUNT_SCALE).to_numpy()
    amount = rng.lognormal(np.log(900.0), 1.1, n) * scale
    amount = np.clip(amount, MIN_AMOUNT_INR, MAX_AMOUNT_INR).round(2)

    method = rng.choice(METHODS, n, p=METHOD_PROBS)
    method = np.where((method == "upi") & (amount > UPI_LIMIT_INR), "card", method)

    df = pd.DataFrame(
        {
            "ts": _timestamps(n, start_ts, days, rng),
            "amount_inr": amount,
            "method": method,
            "bank": rng.choice(BANKS, n, p=BANK_PROBS),
            "merchant_category": category,
        }
    ).sort_values("ts").reset_index(drop=True)
    df.insert(0, "txn_id", [f"txn_{i:07d}" for i in range(n)])

    # Uniform random routing among gateways that support the method
    gateway_id = np.empty(n, dtype=object)
    for m in METHODS:
        idx = np.flatnonzero(df["method"].to_numpy() == m)
        supported = [g.id for g in GATEWAYS if m in g.methods]
        gateway_id[idx] = rng.choice(supported, len(idx))

    success = np.zeros(n, dtype=bool)
    latency = np.zeros(n, dtype=int)
    code = np.full(n, "", dtype=object)
    fee = np.zeros(n)
    for gw in GATEWAYS:
        idx = np.flatnonzero(gateway_id == gw.id)
        if len(idx) == 0:
            continue
        sub = df.iloc[idx]
        s, l, c = sample_outcomes(gw, sub, outages, rng)
        success[idx], latency[idx], code[idx] = s, l, c
        fee[idx] = fee_inr(gw, sub["amount_inr"].to_numpy())

    df["gateway_id"] = gateway_id
    df["success"] = success
    df["latency_ms"] = latency
    df["failure_code"] = pd.Series(code).replace("", None).to_numpy()
    df["fee_inr"] = fee
    df["attempt"] = 1
    return df, outages


def _summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("gateway_id")
    return pd.DataFrame(
        {
            "txns": g.size(),
            "success_rate": g["success"].mean().round(4),
            "avg_latency_ms": g["latency_ms"].mean().round(0),
            "avg_fee_inr": g["fee_inr"].mean().round(2),
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=300_000, help="number of transactions")
    ap.add_argument("--days", type=int, default=60, help="length of the simulated period")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start", default="2026-01-01", help="first day (YYYY-MM-DD)")
    ap.add_argument("--out", default="data", help="output directory")
    args = ap.parse_args()

    df, outages = generate(args.n, args.days, args.seed, args.start)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "transactions.csv", index=False)
    pd.DataFrame(
        [
            {
                "gateway_id": g.id, "name": g.name, "fee_pct": g.fee_pct,
                "fee_fixed": g.fee_fixed, "supported_methods": "|".join(g.methods),
            }
            for g in GATEWAYS
        ]
    ).to_csv(out / "gateways.csv", index=False)
    pd.DataFrame(
        [
            {"gateway_id": o.gateway_id, "start": o.start, "end": o.end,
             "success_prob": o.success_prob, "kind": o.kind}
            for o in outages
        ]
    ).to_csv(out / "outages.csv", index=False)

    print(f"Wrote {len(df):,} transactions to {out}/ "
          f"({df['ts'].min():%Y-%m-%d} to {df['ts'].max():%Y-%m-%d}), seed={args.seed}")
    print(_summary(df).to_string())
    print(f"overall success rate: {df['success'].mean():.4f}")


if __name__ == "__main__":
    main()
