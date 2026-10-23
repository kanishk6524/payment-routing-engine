"""Create the schema and load simulator output into PostgreSQL.

Run from the project root, after `docker compose up -d` and
`python -m simulator.generate`:

    python scripts/load_data.py            # create schema and load (idempotent)
    python scripts/load_data.py --reset    # wipe all tables first, then reload

Everything runs in a single transaction: either the whole load succeeds or nothing changes.
"""
import argparse
import os
import sys
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SQL = ROOT / "sql"

load_dotenv(ROOT / ".env")

STAGING_DDL = """
CREATE TEMP TABLE stg_transactions (
    txn_id text, ts timestamp, amount_inr numeric(12,2), method text, bank text,
    merchant_category text, gateway_id text, success boolean, latency_ms integer,
    failure_code text, fee_inr numeric(10,2), attempt smallint
) ON COMMIT DROP;
CREATE TEMP TABLE stg_gateways (
    gateway_id text, name text, fee_pct numeric(6,4), fee_fixed numeric(8,2),
    supported_methods text
) ON COMMIT DROP;
CREATE TEMP TABLE stg_outages (
    gateway_id text, start_ts timestamp, end_ts timestamp, success_prob numeric(4,3),
    kind text
) ON COMMIT DROP;
"""

RESET = """
TRUNCATE routing_decisions, gateway_metrics, payment_attempts, transactions,
         sim_outages, gateway_methods, gateways, banks, merchant_categories,
         payment_methods RESTART IDENTITY;
"""

SUMMARY = """
SELECT gateway_id,
       count(*)                                   AS attempts,
       round(avg(success::int)::numeric, 4)       AS success_rate,
       round(avg(latency_ms))                     AS avg_latency_ms,
       round(avg(fee_inr), 2)                     AS avg_fee_inr
FROM v_attempt_facts
GROUP BY gateway_id
ORDER BY gateway_id;
"""

TABLES = [
    "payment_methods", "banks", "merchant_categories", "gateways", "gateway_methods",
    "transactions", "payment_attempts", "routing_decisions", "gateway_metrics", "sim_outages",
]


def connect():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5433")),
        dbname=os.getenv("POSTGRES_DB", "payment_router"),
        user=os.getenv("POSTGRES_USER", "router"),
        password=os.getenv("POSTGRES_PASSWORD", "change_me"),
        connect_timeout=5,
    )


def read_sql(name: str) -> str:
    return (SQL / name).read_text()


def copy_csv(cur, table: str, columns: str, path: Path) -> None:
    with path.open() as f:
        cur.copy_expert(
            f"COPY {table} ({columns}) FROM STDIN WITH (FORMAT csv, HEADER true)", f
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", default="data", help="directory with the simulator CSVs")
    ap.add_argument("--reset", action="store_true", help="truncate all tables before loading")
    args = ap.parse_args()

    data = Path(args.data)
    files = {n: data / f"{n}.csv" for n in ("transactions", "gateways", "outages")}
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        print(f"Missing {', '.join(missing)}. Run: python -m simulator.generate --out {data}")
        return 1
    csv_rows = sum(1 for _ in files["transactions"].open()) - 1

    conn = connect()
    try:
        with conn, conn.cursor() as cur:  # `with conn` commits on success, rolls back on error
            cur.execute(read_sql("001_schema.sql"))
            cur.execute(read_sql("002_views.sql"))
            if args.reset:
                cur.execute(RESET)
            cur.execute(STAGING_DDL)
            copy_csv(
                cur, "stg_transactions",
                "txn_id, ts, amount_inr, method, bank, merchant_category, gateway_id, "
                "success, latency_ms, failure_code, fee_inr, attempt",
                files["transactions"],
            )
            copy_csv(
                cur, "stg_gateways",
                "gateway_id, name, fee_pct, fee_fixed, supported_methods", files["gateways"],
            )
            copy_csv(
                cur, "stg_outages",
                "gateway_id, start_ts, end_ts, success_prob, kind", files["outages"],
            )
            cur.execute(read_sql("003_load_from_staging.sql"))
            cur.execute(read_sql("004_backfill_gateway_metrics.sql"))

        with conn, conn.cursor() as cur:
            cur.execute("ANALYZE")
            print("Row counts")
            counts = {}
            for t in TABLES:
                cur.execute(f"SELECT count(*) FROM {t}")
                counts[t] = cur.fetchone()[0]
                print(f"  {t:<20} {counts[t]:>9,}")

            print("\nPer-gateway summary (from v_attempt_facts)")
            cur.execute(SUMMARY)
            print(f"  {'gateway':<8} {'attempts':>9} {'success':>8} {'latency':>8} {'fee':>7}")
            for gw, n, sr, lat, fee in cur.fetchall():
                print(f"  {gw:<8} {n:>9,} {float(sr):>8.4f} {int(lat):>8} {float(fee):>7.2f}")
    finally:
        conn.close()

    if counts["transactions"] != csv_rows or counts["payment_attempts"] != csv_rows:
        print(f"\nWARNING: CSV has {csv_rows:,} rows but the database has "
              f"{counts['transactions']:,} transactions / {counts['payment_attempts']:,} attempts. "
              "If the database holds data from an earlier run, use --reset.")
        return 1
    print(f"\nLoaded {csv_rows:,} transactions OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
