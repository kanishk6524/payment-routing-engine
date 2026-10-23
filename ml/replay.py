"""Counterfactual replay of routing policies (Stage 5).

Real logs only show the outcome of the gateway that was actually used. The simulator knows the
true success probability of every gateway for every transaction, so any routing policy can be
scored on the same held-out transactions by replaying it against the ground truth.

Replay assumptions, stated plainly:
  * Gateway health is exogenous: it depends on time and incidents, not on our traffic, so the
    rolling gateway stats shown to a policy come from the logged history before each decision.
  * Policies are scored by expected outcome (true probabilities), so there is no sampling noise.

As a cross-check that works on real data too, `ips_success` re-estimates a policy's success rate
from the logged outcomes alone. The historical log routed uniformly at random, so inverse
propensity weighting is unbiased, and it should agree with the replay up to statistical noise.
"""
import numpy as np
import pandas as pd

from simulator.config import GATEWAYS_BY_ID, ResolvedOutage
from simulator.model import expected_latency_ms, success_probability

from .features import TARGET, add_static_features, rolling_stats
from .model import encode, predict_proba

TXN_COLUMNS = ["txn_id", "ts", "amount_inr", "method", "bank", "merchant_category"]
LAMBDA_COST_GRID = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2)
LAMBDA_LATENCY_GRID = (0.0, 0.005, 0.01, 0.02, 0.05)


def outages_from_df(outages: pd.DataFrame) -> list:
    return [ResolvedOutage(r.gateway_id, r.start, r.end, float(r.success_prob), r.kind)
            for r in outages.itertuples()]


# ----------------------------------------------------------------------------- candidates
def build_candidates(txns: pd.DataFrame, gateways: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """One row per (transaction, supported gateway) with the model's features at decision time.

    `events` is the logged history. Rolling stats only use events completed before each
    transaction, so nothing from the future leaks into the decision.
    """
    support = (gateways.assign(method=gateways["supported_methods"].str.split("|"))
               .explode("method")[["gateway_id", "method"]])
    cands = (txns[TXN_COLUMNS].merge(support, on="method")
             .sort_values(["txn_id", "gateway_id"]).reset_index(drop=True))
    cands = add_static_features(cands, gateways)
    cands = pd.concat([cands, rolling_stats(events, cands[["gateway_id", "ts"]])], axis=1)
    cands["n_candidates"] = cands.groupby("txn_id")["gateway_id"].transform("size")
    return cands


def add_ground_truth(cands: pd.DataFrame, outages: list) -> pd.DataFrame:
    """True success probability and expected latency of every candidate (evaluation only)."""
    out = cands.copy()
    out["p_true"] = np.nan
    out["lat_true"] = np.nan
    for gw_id, gw in GATEWAYS_BY_ID.items():
        m = (out["gateway_id"] == gw_id).to_numpy()
        if m.any():
            sub = out.loc[m]
            out.loc[m, "p_true"] = success_probability(gw, sub, outages)
            out.loc[m, "lat_true"] = expected_latency_ms(gw, sub, outages)
    return out


def latency_estimate(cands: pd.DataFrame, fallback: dict) -> pd.Series:
    """Latency the router can observe: recent mean, else the longer window, else the training mean."""
    return (cands["gw_avg_latency_15m"].fillna(cands["gw_avg_latency_60m"])
            .fillna(cands["gateway_id"].map(fallback)))


def add_model_scores(cands: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    out = cands.copy()
    out["p_pred"] = predict_proba(bundle["model"], encode(out, bundle["levels"]))
    out["lat_est"] = latency_estimate(out, bundle["latency_fallback"])
    return out


# ----------------------------------------------------------------------------- policies
def minmax_by_txn(cands: pd.DataFrame, col: str) -> pd.Series:
    """Scale `col` to 0..1 across the candidate gateways of each transaction."""
    g = cands.groupby("txn_id")[col]
    lo, hi = g.transform("min"), g.transform("max")
    span = (hi - lo).where(hi > lo, 1.0)
    return (cands[col] - lo) / span


def ml_score(cands: pd.DataFrame, lam_cost: float = 0.0, lam_latency: float = 0.0) -> pd.Series:
    """score = P(success) - lam_cost * normalised fee - lam_latency * normalised latency."""
    return (cands["p_pred"]
            - lam_cost * minmax_by_txn(cands, "fee_ratio")
            - lam_latency * minmax_by_txn(cands, "lat_est"))


def choose(cands: pd.DataFrame, score) -> pd.DataFrame:
    """The highest-scoring candidate for each transaction."""
    s = pd.Series(np.asarray(score, dtype=float), index=cands.index)
    return cands.loc[s.groupby(cands["txn_id"]).idxmax()]


def baseline_scores(cands: pd.DataFrame, train: pd.DataFrame) -> dict:
    """Rule-based policies. Their statistics come from the training period only."""
    per_gateway = train.groupby("gateway_id")[TARGET].mean()
    per_method = (train.groupby(["gateway_id", "method"])[TARGET].mean()
                  .rename("rate").reset_index())
    best_per_method = cands[["gateway_id", "method"]].merge(
        per_method, on=["gateway_id", "method"], how="left")["rate"].to_numpy()
    return {
        "cheapest": -cands["fee_ratio"],
        "best_overall": cands["gateway_id"].map(per_gateway),
        "best_per_method": pd.Series(best_per_method, index=cands.index),
    }


# ----------------------------------------------------------------------------- metrics
def policy_metrics(rows: pd.DataFrame, weight=None) -> dict:
    """Expected outcome of a policy, from the true probabilities (needs add_ground_truth)."""
    w = np.ones(len(rows)) if weight is None else np.asarray(weight, dtype=float)
    p = rows["p_true"].to_numpy()
    amount = rows["amount_inr"].to_numpy()
    fee = rows["fee_inr"].to_numpy()
    lat = rows["lat_true"].to_numpy()
    return {
        "transactions": float(w.sum()),
        "success_rate": float((w * p).sum() / w.sum()),
        "value_success_rate": float((w * p * amount).sum() / (w * amount).sum()),
        "effective_fee_pct": float(100 * (w * p * fee).sum() / (w * p * amount).sum()),
        "avg_latency_ms": float((w * lat).sum() / w.sum()),
    }


def predicted_metrics(rows: pd.DataFrame) -> dict:
    """The same quantities from the model's own predictions (no ground truth: used for tuning)."""
    p, amount, fee = rows["p_pred"].to_numpy(), rows["amount_inr"].to_numpy(), rows["fee_inr"].to_numpy()
    return {
        "success_rate": float(p.mean()),
        "effective_fee_pct": float(100 * (p * fee).sum() / (p * amount).sum()),
        "avg_latency_ms": float(rows["lat_est"].mean()),
    }


def tune_weights(valid_cands: pd.DataFrame, tolerance: float = 0.001):
    """Choose the cost and latency weights on the validation period, without ground truth.

    Rule: among all weight pairs whose predicted success rate is within `tolerance` of the
    success-only router, pick the one with the lowest combined fee-and-latency index (each
    normalised by the success-only router's value, equal weight).
    Returns (lambda_cost, lambda_latency, grid DataFrame).
    """
    rows = []
    for lc in LAMBDA_COST_GRID:
        for ll in LAMBDA_LATENCY_GRID:
            m = predicted_metrics(choose(valid_cands, ml_score(valid_cands, lc, ll)))
            rows.append({"lambda_cost": lc, "lambda_latency": ll, **m})
    grid = pd.DataFrame(rows)
    base = grid[(grid["lambda_cost"] == 0) & (grid["lambda_latency"] == 0)].iloc[0]
    grid["cost_latency_index"] = 0.5 * grid["effective_fee_pct"] / base["effective_fee_pct"] \
        + 0.5 * grid["avg_latency_ms"] / base["avg_latency_ms"]
    allowed = grid[grid["success_rate"] >= base["success_rate"] - tolerance]
    best = allowed.loc[allowed["cost_latency_index"].idxmin()]
    return float(best["lambda_cost"]), float(best["lambda_latency"]), grid


def ips_success(chosen: pd.DataFrame, logged: pd.DataFrame):
    """Success rate of a deterministic policy estimated from the logged outcomes alone (self-normalised IPS).

    `logged` has txn_id, gateway_id (the gateway actually used) and success. The log chose
    uniformly among `n_candidates` gateways, so each match is weighted by n_candidates.
    Only transactions where the log happened to use the policy's choice contribute.
    Returns (estimate, standard error).
    """
    lg = logged.set_index("txn_id")
    ids = chosen["txn_id"].to_numpy()
    matched = lg["gateway_id"].reindex(ids).to_numpy() == chosen["gateway_id"].to_numpy()
    outcome = lg[TARGET].reindex(ids).to_numpy(dtype=float)
    w = matched * chosen["n_candidates"].to_numpy(dtype=float)
    # Self-normalised (Hajek) estimator: divides by the realised weight total instead of N, so
    # chance fluctuations in how often the log picked a gateway do not leak into the estimate.
    est = float((w * outcome).sum() / w.sum())
    se = float(np.sqrt((((w * (outcome - est)) ** 2).sum())) / w.sum())
    return est, se


def incident_time_mask(ts: pd.Series, outages: list) -> np.ndarray:
    mask = np.zeros(len(ts), dtype=bool)
    for o in outages:
        mask |= ((ts >= o.start) & (ts < o.end)).to_numpy()
    return mask
