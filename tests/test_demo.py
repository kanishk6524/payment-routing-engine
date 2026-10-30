"""Demo traffic feed, public-deployment protections and the dashboard fallbacks."""
import collections
import time

import pandas as pd
import pytest

from api.demo import DemoFeed
from api.events import decision_event
from dashboards import data
from tests.test_breaker import TXN, make_router, parts  # noqa: F401  (fixture reuse)


def fake_stream(n_ok=60, n_bad=60):
    rows = []
    for i in range(n_ok + n_bad + n_ok):
        bad = n_ok <= i < n_ok + n_bad
        gw = "gw_d" if i % 2 == 0 else "gw_a"
        fail = bad and gw == "gw_d"
        rows.append({"amount_inr": 1000.0, "method": "card", "bank": "HDFC", "merchant_category": "retail",
                     "gateway_id": gw, "success": not fail, "latency_ms": 5000 if fail else 200,
                     "failure_code": "GATEWAY_ERROR" if fail else None, "ts": pd.Timestamp("2026-02-04 16:00") + pd.Timedelta(seconds=i)})
    return pd.DataFrame(rows)


def test_feed_trips_the_breaker_and_records_decisions(parts, monkeypatch):  # noqa: F811
    router = make_router(parts)
    recent = collections.deque(maxlen=50)
    feed = DemoFeed(router, lambda d, t, x: recent.append(decision_event(d, t, x)), rate=2000, decide_every=2)
    monkeypatch.setattr(feed, "_rows", lambda: fake_stream())
    feed.start()
    deadline, seen_open = time.time() + 20, False
    while time.time() < deadline and not seen_open:
        seen_open = router.breakers.state("gw_d", pd.Timestamp.utcnow().tz_localize(None)) == "open"
        time.sleep(0.01)
    feed.stop()
    feed.join(5)
    assert seen_open, "the recorded gateway failures should trip gw_d's breaker"
    assert recent and {"txn_id", "chosen_gateway", "scores", "explanation"} <= set(recent[-1])
    assert not feed.is_alive()


def test_feed_survives_missing_incident(parts, monkeypatch):  # noqa: F811
    feed = DemoFeed(make_router(parts), lambda *a: None, incident="gw_zzz", data_dir="/nonexistent")
    feed.run()  # logs and returns instead of raising


def test_decisions_from_api_matches_the_postgres_shape():
    ev = {"txn_id": "t1", "decided_at": "2026-10-30 12:00:00.5", "amount_inr": 100, "method": "upi", "bank": "SBI",
          "chosen_gateway": "gw_a", "mode": "failover", "model_version": "v", "scores": {"gw_a": 0.9},
          "explanation": {"summary": "x", "drivers": []}}
    df = data.decisions_from_api([ev])
    assert list(df.columns) == data.DECISION_COLUMNS and df.loc[0, "policy"] == "xgboost-failover"
    assert df["decided_at"].dt.floor("s").iloc[0] == pd.Timestamp("2026-10-30 12:00:00")
    assert data.decisions_from_api([]).empty


def test_load_decisions_falls_back_to_the_api(monkeypatch):
    ev = {"txn_id": "t1", "decided_at": "2026-10-30 12:00:00", "amount_inr": 100, "method": "upi", "bank": "SBI",
          "chosen_gateway": "gw_a", "model_version": "v", "scores": {"gw_a": 0.9}, "explanation": {}}
    monkeypatch.setattr(data, "load_decisions_pg", lambda limit=500: (pd.DataFrame(columns=data.DECISION_COLUMNS), "pg down"))
    monkeypatch.setattr(data, "api_get", lambda path: ([ev], None))
    df, err = data.load_decisions()
    assert err is None and len(df) == 1
    monkeypatch.setattr(data, "api_get", lambda path: (None, "api down"))
    df, err = data.load_decisions()
    assert df.empty and err == "pg down"


def test_load_series_prefers_the_api(monkeypatch):
    rows = [{"gateway_id": "gw_a", "bucket": "2026-10-30 12:00:00", "attempts": 5, "success_pct": 80.0, "avg_latency_ms": 300}]
    monkeypatch.setattr(data, "api_get", lambda path: (rows, None))
    df, err = data.load_series()
    assert err is None and df["bucket"].dtype.kind == "M" and df.loc[0, "attempts"] == 5


def test_http_endpoints_and_admin_token(parts, monkeypatch):  # noqa: F811
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    import api.main as main

    monkeypatch.setenv("EVENT_BUS", "none")
    monkeypatch.setenv("ADMIN_TOKEN", "s3cret")
    monkeypatch.delenv("DEMO_MODE", raising=False)
    monkeypatch.setattr(main.Router, "from_artifacts", classmethod(lambda cls, **kw: make_router(parts)))
    with TestClient(main.app) as client:
        assert client.get("/health").json()["demo"] is False
        assert client.get("/decisions").json() == []
        client.post("/route", json=TXN)
        assert len(client.get("/decisions").json()) == 1
        series = client.get("/series?minutes=60&bucket_s=60")
        assert series.status_code == 200 and isinstance(series.json(), list)
        out = {"gateway_id": "gw_a", "success": True, "latency_ms": 90}
        assert client.post("/outcome", json=out).status_code == 401
        assert client.post("/gateways/gw_d/disable").status_code == 401
        assert client.post("/outcome", json=out, headers={"X-Admin-Token": "wrong"}).status_code == 401
        assert client.post("/outcome", json=out, headers={"X-Admin-Token": "s3cret"}).status_code == 202
        assert client.post("/gateways/gw_d/disable", headers={"X-Admin-Token": "s3cret"}).status_code == 200
        assert client.post("/route", json=TXN).status_code == 200  # routing stays public
