"""Data access and aggregation for the dashboard (Stage 10).

Everything here is plain pandas, so it is testable without Streamlit. The screens in
`app.py` call these functions through the module (`data.load_events()`), which lets tests
replace a source with sample data.
"""
import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
API_URL = os.getenv("DASHBOARD_API", "http://localhost:8000")
HEALTHY_MIN_SUCCESS = 0.80


# ------------------------------------------------------------------ sources
def api_get(path: str):
    """GET from the routing API. Returns (json, None) or (None, error message)."""
    import httpx

    try:
        r = httpx.get(f"{API_URL}{path}", timeout=4)
        r.raise_for_status()
        return r.json(), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def api_post(path: str, body: dict = None):
    import httpx

    try:
        r = httpx.post(f"{API_URL}{path}", json=body or {}, timeout=10)
        r.raise_for_status()
        return r.json(), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def load_events(gateway_ids=("gw_a", "gw_b", "gw_c", "gw_d", "gw_e")):
    """Recent attempt outcomes from Redis. Returns (DataFrame, error or None)."""
    from api.redis_health import RedisHealthTracker, connect_redis

    try:
        tracker = RedisHealthTracker(connect_redis(), gateway_ids, os.getenv("REDIS_PREFIX", "router:"))
        return tracker.snapshot(), None
    except Exception as exc:
        return pd.DataFrame(columns=["gateway_id", "ts", "latency_ms", "success", "failure_code"]), \
            f"{type(exc).__name__}: {exc}"


DECISIONS_SQL = """
SELECT d.decision_id, d.txn_id, d.decided_at, d.policy, d.model_version, d.chosen_gateway,
       d.scores, d.explanation, t.amount_inr, t.method, t.bank
FROM routing_decisions d JOIN transactions t USING (txn_id)
WHERE d.model_version IS NOT NULL
ORDER BY d.decided_at DESC, d.decision_id DESC
LIMIT %s
"""


def load_decisions(limit: int = 500):
    """Latest live routing decisions from PostgreSQL. Returns (DataFrame, error or None)."""
    try:
        from ml.db import _connect

        conn = _connect()
        try:
            with conn.cursor() as cur:
                cur.execute(DECISIONS_SQL, (limit,))
                cols = [c[0] for c in cur.description]
                df = pd.DataFrame(cur.fetchall(), columns=cols)
        finally:
            conn.close()
        df["amount_inr"] = df["amount_inr"].astype(float)
        df["decided_at"] = pd.to_datetime(df["decided_at"])
        return df, None
    except Exception as exc:
        return pd.DataFrame(columns=["decision_id", "txn_id", "decided_at", "policy", "model_version",
                                     "chosen_gateway", "scores", "explanation", "amount_inr", "method",
                                     "bank"]), f"{type(exc).__name__}: {exc}"


def load_results(path: str = "reports/results.json"):
    p = ROOT / path
    return json.loads(p.read_text()) if p.exists() else None


def load_metrics(path: str = "ml/artifacts/metrics.json"):
    p = ROOT / path
    return json.loads(p.read_text()) if p.exists() else None


# ------------------------------------------------------------------ aggregation
def classify(breaker: str, success_rate, attempts) -> str:
    """Display state of a gateway: 'healthy', 'degraded', 'down' or 'unknown' (no recent data)."""
    if breaker == "open":
        return "down"
    if breaker == "half_open":
        return "degraded"
    if attempts is None or not attempts or success_rate is None or pd.isna(success_rate):
        return "unknown"
    return "healthy" if success_rate >= HEALTHY_MIN_SUCCESS else "degraded"


def gateway_cards(gateways: list) -> list:
    """Turn the API's /gateways response into card data (state, success %, attempts, latency)."""
    cards = []
    for g in gateways:
        recent = g.get("last_15_min") or {}
        sr, n = recent.get("success_rate"), recent.get("attempts")
        cards.append({
            "gateway_id": g["gateway_id"], "enabled": g.get("enabled", True), "breaker": g.get("breaker", "closed"),
            "state": classify(g.get("breaker", "closed"), sr, n),
            "success_pct": None if sr is None else round(100 * sr, 1), "attempts": n or 0,
            "avg_latency_ms": recent.get("avg_latency_ms"),
        })
    return cards


def bucket_series(events: pd.DataFrame, now=None, window_min: int = 60, bucket_s: int = 60) -> pd.DataFrame:
    """Per gateway and time bucket: attempts, success rate (0-100) and mean latency."""
    cols = ["gateway_id", "bucket", "attempts", "success_pct", "avg_latency_ms"]
    if events.empty:
        return pd.DataFrame(columns=cols)
    now = pd.Timestamp(now) if now is not None else events["ts"].max()
    ev = events[(events["ts"] >= now - pd.Timedelta(minutes=window_min)) & (events["ts"] <= now)].copy()
    if ev.empty:
        return pd.DataFrame(columns=cols)
    ev["bucket"] = ev["ts"].dt.floor(f"{bucket_s}s")
    ev["success"] = ev["success"].astype(float)
    g = ev.groupby(["gateway_id", "bucket"]).agg(
        attempts=("success", "size"), success_pct=("success", "mean"), avg_latency_ms=("latency_ms", "mean"))
    g["success_pct"] = (100 * g["success_pct"]).round(2)
    g["avg_latency_ms"] = g["avg_latency_ms"].round(0)
    return g.reset_index()[cols]


def traffic_share(decisions: pd.DataFrame, bucket_s: int = 60) -> pd.DataFrame:
    """Share of routed transactions per gateway in each time bucket (shares sum to 100 per bucket)."""
    cols = ["bucket", "gateway", "share_pct", "n"]
    if decisions.empty:
        return pd.DataFrame(columns=cols)
    d = decisions.copy()
    d["bucket"] = d["decided_at"].dt.floor(f"{bucket_s}s")
    counts = d.groupby(["bucket", "chosen_gateway"]).size().rename("n").reset_index()
    counts["share_pct"] = (100 * counts["n"] / counts.groupby("bucket")["n"].transform("sum")).round(2)
    return counts.rename(columns={"chosen_gateway": "gateway"})[cols]


def drivers_frame(explanation: dict) -> pd.DataFrame:
    rows = (explanation or {}).get("drivers") or []
    return pd.DataFrame(rows, columns=["label", "value", "effect", "direction"]) if rows else \
        pd.DataFrame(columns=["label", "value", "effect", "direction"])


def scores_frame(scores: dict, chosen: str) -> pd.DataFrame:
    df = pd.DataFrame({"gateway": list((scores or {}).keys()), "score": list((scores or {}).values())})
    df["chosen"] = df["gateway"] == chosen
    return df.sort_values("score", ascending=True).reset_index(drop=True)


def alternatives_frame(alternatives: list, chosen: str) -> pd.DataFrame:
    df = pd.DataFrame(alternatives)
    if df.empty:
        return df
    df["chosen"] = df["gateway"] == chosen
    return df


POLICY_ORDER = ["random", "cheapest", "best_overall", "best_per_method", "ml", "ml_cost_aware", "oracle"]
POLICY_LABELS = {"random": "Random (logged)", "cheapest": "Cheapest", "best_overall": "Best overall",
                 "best_per_method": "Best per method", "ml": "ML router", "ml_cost_aware": "ML cost-aware",
                 "oracle": "Oracle (ceiling)"}


def policy_frame(results: dict) -> pd.DataFrame:
    rows = []
    for key in POLICY_ORDER:
        p = (results or {}).get("policies", {}).get(key)
        if p:
            rows.append({"policy": POLICY_LABELS[key], "key": key,
                         "success_pct": round(100 * p["all"]["success_rate"], 2),
                         "incident_pct": round(100 * p["incident"]["success_rate"], 2),
                         "fee_pct": round(p["all"]["effective_fee_pct"], 3),
                         "latency_ms": round(p["all"]["avg_latency_ms"])})
    return pd.DataFrame(rows, columns=["policy", "key", "success_pct", "incident_pct", "fee_pct", "latency_ms"])
