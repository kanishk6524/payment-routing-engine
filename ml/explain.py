"""Per-decision explanations (Stage 6).

XGBoost computes exact TreeSHAP values itself (`pred_contribs=True`), so the explanations need no
extra dependency. Contributions are in log-odds of success: positive pushes the predicted
success probability up, negative pushes it down. Only the XGBoost model supports this; the
scikit-learn stand-in returns no drivers.
"""
import numpy as np
import pandas as pd

LABELS = {
    "amount_inr": "amount", "amount_bucket": "amount band", "method": "payment method",
    "bank": "bank", "merchant_category": "merchant category", "gateway_id": "gateway",
    "hour": "hour of day", "day_of_week": "day of week", "is_weekend": "weekend",
    "gw_n_5m": "attempts in last 5 min", "gw_n_15m": "attempts in last 15 min",
    "gw_n_60m": "attempts in last 60 min",
    "gw_success_rate_5m": "5-min success rate", "gw_success_rate_15m": "15-min success rate",
    "gw_success_rate_60m": "60-min success rate",
    "gw_fail_rate_5m": "5-min gateway failure rate", "gw_fail_rate_15m": "15-min gateway failure rate",
    "gw_fail_rate_60m": "60-min gateway failure rate",
    "gw_avg_latency_5m": "5-min avg latency", "gw_avg_latency_15m": "15-min avg latency",
    "gw_avg_latency_60m": "60-min avg latency", "gw_p95_latency_15m": "15-min p95 latency",
}
PERCENT_FEATURES = ("success_rate", "fail_rate")


def contributions(model, X: pd.DataFrame):
    """Log-odds contribution of every feature to every row. Returns None if unsupported."""
    if not hasattr(model, "get_booster"):
        return None
    import xgboost as xgb

    raw = model.get_booster().predict(xgb.DMatrix(X, enable_categorical=True), pred_contribs=True)
    out = pd.DataFrame(raw[:, :-1], columns=list(X.columns), index=X.index)
    out.attrs["bias"] = raw[:, -1]  # contributions + bias = the model's raw log-odds
    return out


def describe_value(feature: str, value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "no recent data"
    if any(k in feature for k in PERCENT_FEATURES):
        return f"{100 * float(value):.1f}%"
    if "latency" in feature:
        return f"{float(value):.0f} ms"
    if feature == "amount_inr":
        return f"INR {float(value):,.0f}"
    if feature in ("is_weekend",):
        return "yes" if int(value) else "no"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):g}"
    return str(value)


def top_drivers(contrib_row: pd.Series, feature_row: pd.Series, k: int = 3) -> list:
    """The k features that moved this prediction most, with their value and effect."""
    order = contrib_row.abs().sort_values(ascending=False).index[:k]
    return [{
        "feature": f, "label": LABELS.get(f, f), "value": describe_value(f, feature_row[f]),
        "effect": round(float(contrib_row[f]), 3),
        "direction": "raises success odds" if contrib_row[f] > 0 else "lowers success odds",
    } for f in order]
