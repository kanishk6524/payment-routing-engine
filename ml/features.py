"""Leakage-safe feature engineering (Stage 4).

One training row per payment attempt. The label is `success`.

Leakage rule
------------
A rolling feature for an attempt at time `t` may only use earlier attempts of the SAME
gateway whose outcome was already known at `t`. An attempt's outcome becomes known at
`ts + latency_ms`, so an attempt still in flight at `t` is not counted, and an attempt
never counts toward its own features. This matches what Redis can hold at serving time.

The feature set deliberately does not encode the simulator's hidden patterns (for example
there is no "is_peak_hour" or "is_high_value_card" flag). The model has to find them from
`hour`, `amount_inr`, `method` and friends, which is what makes the Stage 5 uplift credible.
"""
import numpy as np
import pandas as pd

NS_PER_MIN = 60 * 1_000_000_000
NS_PER_MS = 1_000_000

WINDOWS_MIN = (5, 15, 60)
P95_WINDOW_MIN = 15
GATEWAY_SIDE_FAILURES = ("GATEWAY_ERROR", "TIMEOUT")  # bank declines are not the gateway's fault

AMOUNT_BINS = [0, 100, 500, 1_000, 2_500, 5_000, 10_000, 25_000, 50_000, np.inf]

TARGET = "success"

STATIC_FEATURES = [
    "amount_inr", "amount_bucket", "method", "bank", "merchant_category", "gateway_id",
    "hour", "day_of_week", "is_weekend",
]
ROLLING_FEATURES = (
    [f"gw_n_{w}m" for w in WINDOWS_MIN]
    + [f"gw_success_rate_{w}m" for w in WINDOWS_MIN]
    + [f"gw_fail_rate_{w}m" for w in WINDOWS_MIN]
    + [f"gw_avg_latency_{w}m" for w in WINDOWS_MIN]
    + [f"gw_p95_latency_{P95_WINDOW_MIN}m"]
)
FEATURE_COLUMNS = STATIC_FEATURES + ROLLING_FEATURES
CATEGORICAL_FEATURES = ["method", "bank", "merchant_category", "gateway_id"]

# Columns kept next to the features: identifiers, label, and fields needed for routing evaluation
KEEP_COLUMNS = [
    "txn_id", "attempt", "ts", TARGET, "latency_ms", "failure_code",
    "gw_fee_pct", "gw_fee_fixed", "fee_inr", "fee_ratio",
]


def add_static_features(df: pd.DataFrame, gateways: pd.DataFrame) -> pd.DataFrame:
    """Transaction, time and fee features. Uses only the row itself."""
    out = df.copy()
    out["hour"] = out["ts"].dt.hour
    out["day_of_week"] = out["ts"].dt.day_of_week
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["amount_bucket"] = pd.cut(
        out["amount_inr"], AMOUNT_BINS, labels=False, right=False
    ).astype(int)

    g = gateways.set_index("gateway_id")
    out["gw_fee_pct"] = out["gateway_id"].map(g["fee_pct"]).astype(float)
    out["gw_fee_fixed"] = out["gateway_id"].map(g["fee_fixed"]).astype(float)
    out["fee_inr"] = (out["gw_fee_pct"] * out["amount_inr"] + out["gw_fee_fixed"]).round(2)
    out["fee_ratio"] = out["fee_inr"] / out["amount_inr"]
    return out


def rolling_stats(events: pd.DataFrame, queries: pd.DataFrame) -> pd.DataFrame:
    """Rolling gateway stats at arbitrary (gateway, time) query points.

    `events` are attempts with an outcome: gateway_id, ts, latency_ms, success, failure_code.
    `queries` hold gateway_id and ts. For a query at time `t` on gateway `g`, window `w`
    covers events of `g` that completed in [t - w, t), so an event never counts toward its own
    features and attempts still in flight at `t` are ignored. Counted per window: attempts,
    success rate, gateway-side failure rate and mean latency, plus a 15-minute p95 latency.
    Values are NaN when the window holds no completed events. The result is aligned to
    `queries.index`.

    This is exactly what serving needs: score any candidate gateway at the moment of decision.
    """
    nq = len(queries)
    feats = {c: np.full(nq, np.nan) for c in ROLLING_FEATURES}
    for w in WINDOWS_MIN:
        feats[f"gw_n_{w}m"] = np.zeros(nq)

    ev_ts = events["ts"].to_numpy("datetime64[ns]").astype("int64")
    ev_lat_all = events["latency_ms"].to_numpy(dtype=float)
    ev_done_all = ev_ts + events["latency_ms"].to_numpy("int64") * NS_PER_MS  # outcome known
    ev_success_all = events[TARGET].to_numpy(dtype=float)
    ev_fail_all = events["failure_code"].isin(GATEWAY_SIDE_FAILURES).to_numpy(dtype=float)
    q_ts = queries["ts"].to_numpy("datetime64[ns]").astype("int64")

    event_rows = events.groupby("gateway_id").indices
    for gw, qpos in queries.groupby("gateway_id").indices.items():
        if gw not in event_rows:
            continue
        epos = event_rows[gw]
        order = np.argsort(ev_done_all[epos], kind="stable")
        ev_done = ev_done_all[epos][order]
        zero = np.zeros(1)
        cs_success = np.concatenate([zero, np.cumsum(ev_success_all[epos][order])])
        cs_fail = np.concatenate([zero, np.cumsum(ev_fail_all[epos][order])])
        cs_lat = np.concatenate([zero, np.cumsum(ev_lat_all[epos][order])])
        ev_lat = ev_lat_all[epos][order]

        query = q_ts[qpos]
        hi = np.searchsorted(ev_done, query, side="left")  # outcomes known strictly before t

        for w in WINDOWS_MIN:
            lo = np.searchsorted(ev_done, query - w * NS_PER_MIN, side="left")
            cnt = hi - lo
            has = cnt > 0
            denom = np.where(has, cnt, 1)
            feats[f"gw_n_{w}m"][qpos] = cnt
            feats[f"gw_success_rate_{w}m"][qpos] = np.where(
                has, (cs_success[hi] - cs_success[lo]) / denom, np.nan)
            feats[f"gw_fail_rate_{w}m"][qpos] = np.where(
                has, (cs_fail[hi] - cs_fail[lo]) / denom, np.nan)
            feats[f"gw_avg_latency_{w}m"][qpos] = np.where(
                has, (cs_lat[hi] - cs_lat[lo]) / denom, np.nan)
            if w == P95_WINDOW_MIN:
                p95 = np.full(len(qpos), np.nan)
                for k in np.flatnonzero(has):
                    window = np.sort(ev_lat[lo[k]:hi[k]])
                    p95[k] = window[max(int(np.ceil(0.95 * len(window))) - 1, 0)]  # nearest rank
                feats[f"gw_p95_latency_{w}m"][qpos] = p95

    return pd.DataFrame(feats, index=queries.index)


def add_gateway_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling stats of each attempt's own gateway, using only earlier completed attempts."""
    stats = rolling_stats(df, df[["gateway_id", "ts"]])
    return pd.concat([df, stats], axis=1)


def time_split(df: pd.DataFrame, train_frac: float = 0.6, valid_frac: float = 0.2) -> pd.Series:
    """Chronological split by timestamp: train < valid < test. Never random."""
    ts = df["ts"]
    cut_valid = ts.sort_values().iloc[int(len(df) * train_frac)]
    cut_test = ts.sort_values().iloc[int(len(df) * (train_frac + valid_frac))]
    split = np.where(ts < cut_valid, "train", np.where(ts < cut_test, "valid", "test"))
    return pd.Series(split, index=df.index, name="split")


def build_features(attempts: pd.DataFrame, gateways: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline: static features, rolling features, label and time split."""
    df = attempts.sort_values(["ts", "txn_id", "attempt"]).reset_index(drop=True)
    df = add_static_features(df, gateways)
    df = add_gateway_rolling_features(df)
    df[TARGET] = df[TARGET].astype(int)
    df["split"] = time_split(df)
    return df[KEEP_COLUMNS + FEATURE_COLUMNS + ["split"]]
