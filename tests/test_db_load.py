"""Integration tests for the loaded database.

Skipped automatically when PostgreSQL is unreachable or empty. To run them:

    docker compose up -d
    python -m simulator.generate --out data
    python scripts/load_data.py --reset
    pytest tests/test_db_load.py
"""
import os
from pathlib import Path

import pandas as pd
import pytest

psycopg2 = pytest.importorskip("psycopg2")
import psycopg2.errors  # noqa: E402  (not imported automatically by `import psycopg2`)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cur():
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    try:
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5433")),
            dbname=os.getenv("POSTGRES_DB", "payment_router"),
            user=os.getenv("POSTGRES_USER", "router"),
            password=os.getenv("POSTGRES_PASSWORD", "change_me"),
            connect_timeout=3,
        )
    except psycopg2.OperationalError:
        pytest.skip("PostgreSQL is not reachable")
    c = conn.cursor()
    try:
        c.execute("SELECT count(*) FROM transactions")
    except psycopg2.Error:
        pytest.skip("schema not created: run scripts/load_data.py")
    if c.fetchone()[0] == 0:
        pytest.skip("database is empty: run scripts/load_data.py")
    yield c
    conn.close()


def scalar(cur, sql):
    cur.execute(sql)
    return cur.fetchone()[0]


def test_row_counts_match_csv(cur):
    csv_rows = len(pd.read_csv(ROOT / "data" / "transactions.csv", usecols=["txn_id"]))
    assert scalar(cur, "SELECT count(*) FROM transactions") == csv_rows
    assert scalar(cur, "SELECT count(*) FROM payment_attempts") == csv_rows


def test_every_attempt_uses_a_supported_method(cur):
    bad = scalar(
        cur,
        """SELECT count(*) FROM payment_attempts a
           JOIN transactions t USING (txn_id)
           LEFT JOIN gateway_methods gm
                  ON gm.gateway_id = a.gateway_id AND gm.method = t.method
           WHERE gm.gateway_id IS NULL""",
    )
    assert bad == 0


def test_per_gateway_success_rate_matches_csv(cur):
    df = pd.read_csv(ROOT / "data" / "transactions.csv", usecols=["gateway_id", "success"])
    expected = df.groupby("gateway_id")["success"].mean()
    cur.execute(
        "SELECT gateway_id, avg(success::int)::float8 FROM payment_attempts GROUP BY gateway_id"
    )
    actual = dict(cur.fetchall())
    assert set(actual) == set(expected.index)
    for gw, rate in expected.items():
        assert actual[gw] == pytest.approx(rate, abs=1e-9)


def test_gateway_metrics_cover_every_attempt(cur):
    assert scalar(cur, "SELECT sum(attempts) FROM gateway_metrics") == scalar(
        cur, "SELECT count(*) FROM payment_attempts"
    )
    assert scalar(cur, "SELECT sum(successes) FROM gateway_metrics") == scalar(
        cur, "SELECT count(*) FROM payment_attempts WHERE success"
    )


def test_outages_loaded(cur):
    assert scalar(cur, "SELECT count(*) FROM sim_outages") == 4


def test_check_constraint_rejects_inconsistent_attempt(cur):
    # success=true with a failure_code must be rejected by the CHECK constraint
    cur.execute("SELECT txn_id FROM transactions LIMIT 1")
    txn_id = cur.fetchone()[0]
    with pytest.raises(psycopg2.errors.CheckViolation):
        cur.execute(
            """INSERT INTO payment_attempts
               (txn_id, attempt, gateway_id, attempted_at, success, latency_ms, failure_code, fee_inr)
               VALUES (%s, 99, 'gw_a', now(), true, 100, 'TIMEOUT', 1)""",
            (txn_id,),
        )
    cur.connection.rollback()
