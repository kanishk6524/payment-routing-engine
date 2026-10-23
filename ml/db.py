"""Load attempts and gateways from PostgreSQL, or from the simulator CSVs."""
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

ATTEMPT_COLUMNS = [
    "txn_id", "attempt", "ts", "amount_inr", "method", "bank", "merchant_category",
    "gateway_id", "success", "latency_ms", "failure_code", "fee_inr",
]


def _connect():
    import psycopg2
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5433")),
        dbname=os.getenv("POSTGRES_DB", "payment_router"),
        user=os.getenv("POSTGRES_USER", "router"),
        password=os.getenv("POSTGRES_PASSWORD", "change_me"),
        connect_timeout=5,
    )


def _query(sql: str) -> pd.DataFrame:
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            cols = [d[0] for d in cur.description]
            return pd.DataFrame(cur.fetchall(), columns=cols)
    finally:
        conn.close()


def load_from_db():
    """Return (attempts, gateways, outages) from PostgreSQL."""
    attempts = _query(
        f"SELECT {', '.join(ATTEMPT_COLUMNS)} FROM v_attempt_facts ORDER BY ts, txn_id, attempt"
    )
    gateways = _query(
        "SELECT g.gateway_id, g.fee_pct, g.fee_fixed, "
        "       string_agg(gm.method, '|' ORDER BY gm.method) AS supported_methods "
        "FROM gateways AS g JOIN gateway_methods AS gm USING (gateway_id) "
        "GROUP BY g.gateway_id, g.fee_pct, g.fee_fixed ORDER BY g.gateway_id"
    )
    outages = _query(
        "SELECT gateway_id, start_ts AS start, end_ts AS \"end\", success_prob, kind "
        "FROM sim_outages ORDER BY start_ts"
    )
    attempts["amount_inr"] = attempts["amount_inr"].astype(float)
    attempts["fee_inr"] = attempts["fee_inr"].astype(float)
    attempts["ts"] = pd.to_datetime(attempts["ts"])
    gateways[["fee_pct", "fee_fixed"]] = gateways[["fee_pct", "fee_fixed"]].astype(float)
    outages[["start", "end"]] = outages[["start", "end"]].apply(pd.to_datetime)
    return attempts, gateways, outages


def load_from_csv(data_dir="data"):
    """Return (attempts, gateways, outages) from the simulator CSVs."""
    d = Path(data_dir)
    attempts = pd.read_csv(d / "transactions.csv", parse_dates=["ts"])
    attempts = attempts[ATTEMPT_COLUMNS].sort_values(["ts", "txn_id", "attempt"])
    gateways = pd.read_csv(d / "gateways.csv")[
        ["gateway_id", "fee_pct", "fee_fixed", "supported_methods"]
    ]
    outages = pd.read_csv(d / "outages.csv", parse_dates=["start", "end"])
    return attempts.reset_index(drop=True), gateways, outages


def load(source: str = "db", data_dir: str = "data"):
    if source == "csv":
        return load_from_csv(data_dir)
    if source == "db":
        return load_from_db()
    raise ValueError("source must be 'db' or 'csv'")
