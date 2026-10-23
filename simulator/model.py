"""Ground-truth outcome model.

`success_probability` is the true probability that a gateway succeeds for a
transaction at a given time. Stage 5 can call it to evaluate routing policies by
replay (counterfactual outcomes), which real data would not allow.
"""
import numpy as np
import pandas as pd

from .config import HIGH_VALUE_INR, PEAK_HOURS, Gateway

LATENCY_SIGMA = 0.35            # log-normal spread of a normal response time
TIMEOUT_LATENCY_MS = 8000       # a timeout always takes about this long
TIMEOUT_SHARE_OUTAGE = 0.50     # share of failures that are timeouts during an incident
TIMEOUT_SHARE_NORMAL = 0.15     # ... and otherwise (matches the thresholds in sample_outcomes)


def outage_mask(gateway_id: str, ts: pd.Series, outages) -> np.ndarray:
    """True for rows whose timestamp falls inside any outage of this gateway."""
    mask = np.zeros(len(ts), dtype=bool)
    for o in outages:
        if o.gateway_id == gateway_id:
            mask |= ((ts >= o.start) & (ts < o.end)).to_numpy()
    return mask


def success_probability(gw: Gateway, df: pd.DataFrame, outages) -> np.ndarray:
    """True success probability of `gw` for each row of `df`.

    Expected columns: ts, amount_inr, method, bank, merchant_category.
    """
    amount = df["amount_inr"].to_numpy(dtype=float)
    hour = df["ts"].dt.hour.to_numpy()

    p = np.full(len(df), gw.base_success, dtype=float)
    p += df["method"].map(gw.method_adj).fillna(0.0).to_numpy()
    p += df["bank"].map(gw.bank_adj).fillna(0.0).to_numpy()
    p += df["merchant_category"].map(gw.category_adj).fillna(0.0).to_numpy()

    # Every gateway gets a little worse as the amount grows
    p -= 0.01 * np.clip(np.log10(amount / 1000.0), 0.0, None)

    high_value_card = (df["method"].to_numpy() == "card") & (amount >= HIGH_VALUE_INR)
    p += np.where(high_value_card, gw.high_value_card_adj, 0.0)
    p += np.where(np.isin(hour, PEAK_HOURS), gw.peak_adj, 0.0)

    for o in outages:
        if o.gateway_id == gw.id:
            m = ((df["ts"] >= o.start) & (df["ts"] < o.end)).to_numpy()
            p[m] = o.success_prob

    return np.clip(p, 0.01, 0.995)


def expected_latency_ms(gw: Gateway, df: pd.DataFrame, outages) -> np.ndarray:
    """True expected latency (ms) of `gw` for each row, including slow timeouts on failures."""
    p = success_probability(gw, df, outages)
    in_outage = outage_mask(gw.id, df["ts"], outages)
    peak = np.isin(df["ts"].dt.hour.to_numpy(), PEAK_HOURS)
    mean_normal = gw.median_latency_ms * np.exp(LATENCY_SIGMA ** 2 / 2)
    mean_normal = mean_normal * np.where(peak, gw.peak_latency_mult, 1.0)
    timeout_share = np.where(in_outage, TIMEOUT_SHARE_OUTAGE, TIMEOUT_SHARE_NORMAL)
    mean_failure = timeout_share * TIMEOUT_LATENCY_MS + (1 - timeout_share) * mean_normal
    return p * mean_normal + (1 - p) * mean_failure


def fee_inr(gw: Gateway, amount: np.ndarray) -> np.ndarray:
    """Fee charged if the payment succeeds."""
    return np.round(gw.fee_pct * amount + gw.fee_fixed, 2)


def sample_outcomes(gw: Gateway, df: pd.DataFrame, outages, rng: np.random.Generator):
    """Draw success, latency (ms) and failure code for each row routed to `gw`."""
    n = len(df)
    p = success_probability(gw, df, outages)
    success = rng.random(n) < p
    in_outage = outage_mask(gw.id, df["ts"], outages)
    peak = np.isin(df["ts"].dt.hour.to_numpy(), PEAK_HOURS)

    latency = rng.lognormal(np.log(gw.median_latency_ms), LATENCY_SIGMA, n)
    latency = latency * np.where(peak, gw.peak_latency_mult, 1.0)

    u = rng.random(n)
    code = np.where(
        in_outage,
        np.where(u < 0.5, "TIMEOUT", "GATEWAY_ERROR"),
        np.where(u < 0.60, "BANK_DECLINE", np.where(u < 0.85, "GATEWAY_ERROR", "TIMEOUT")),
    )
    code = np.where(success, "", code)

    # Timeouts are slow by definition
    latency = np.where(code == "TIMEOUT", TIMEOUT_LATENCY_MS + rng.normal(0, 400, n), latency)
    latency = np.clip(latency, 50, None).round().astype(int)
    return success, latency, code
