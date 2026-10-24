"""Stage 8 tests: Redis health store, Kafka events, decision sink.

The store contract runs against memory, a Redis fake and (when reachable) a real Redis.
Kafka and PostgreSQL integration tests skip themselves when the service is not running.
"""
import json
import os
import time
import uuid

import pandas as pd
import pytest

from api.breaker import BreakerConfig, CircuitBreakers
from api.events import (
    DECISIONS_TOPIC, OUTCOMES_TOPIC, apply_outcome_event, decision_event, outcome_event,
)
from api.health import HealthTracker, utc_now
from api.redis_health import RedisHealthTracker, RedisSet
from api.service import Router
from pipeline.bus import LocalBus
from pipeline.decision_sink import to_rows
from simulator.config import GATEWAYS
from tests._fakes import FakeRedis
from tests.test_breaker import CFG, GATEWAY_TABLE, TXN, parts  # noqa: F401  (fixture reuse)

GW_IDS = [g.id for g in GATEWAYS]


def real_redis():
    try:
        import redis

        c = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=int(os.getenv("REDIS_PORT", "6379")),
                        decode_responses=True, socket_connect_timeout=1)
        c.ping()
        return c
    except Exception:
        return None


def make_store(kind):
    if kind == "memory":
        return HealthTracker()
    if kind == "fake":
        return RedisHealthTracker(FakeRedis(), GW_IDS, f"t{uuid.uuid4().hex[:6]}:")
    client = real_redis()
    if client is None:
        pytest.skip("Redis is not running")
    return RedisHealthTracker(client, GW_IDS, f"test{uuid.uuid4().hex[:8]}:")


KINDS = ["memory", "fake", "redis"]


def cleanup(store):
    if isinstance(store, RedisHealthTracker):
        store.clear()


# ------------------------------------------------------------------ health store contract
@pytest.mark.parametrize("kind", KINDS)
def test_store_roundtrip_and_ordering(kind):
    s = make_store(kind)
    now = utc_now().floor("s")
    s.add("gw_a", True, 300, None, ts=now - pd.Timedelta(seconds=20))
    s.add("gw_b", False, 8000, "TIMEOUT", ts=now - pd.Timedelta(seconds=50))
    s.add("gw_a", False, 120, "BANK_DECLINE", ts=now - pd.Timedelta(seconds=5))
    ev = s.snapshot()
    assert list(ev["ts"]) == sorted(ev["ts"]) and len(ev) == 3
    a = ev[ev["gateway_id"] == "gw_a"]
    assert a["success"].tolist() == [True, False] and a["failure_code"].tolist() == [None, "BANK_DECLINE"]
    assert ev[ev["gateway_id"] == "gw_b"].iloc[0]["latency_ms"] == 8000
    cleanup(s)


@pytest.mark.parametrize("kind", KINDS)
def test_store_summary_and_empty(kind):
    s = make_store(kind)
    assert s.is_empty() and s.snapshot().empty
    now = utc_now()
    for i in range(10):
        s.add("gw_c", i < 8, 200, None if i < 8 else "TIMEOUT", ts=now - pd.Timedelta(seconds=30 + i))
    assert not s.is_empty()
    out = s.summary(now)
    assert out["gw_c"]["attempts"] == 10 and out["gw_c"]["success_rate"] == 0.8
    cleanup(s)


@pytest.mark.parametrize("kind", KINDS)
def test_store_forget_removes_only_that_gateway_and_window(kind):
    s = make_store(kind)
    t = utc_now().floor("s")
    for i in range(6):
        s.add("gw_a", False, 100, "TIMEOUT", ts=t - pd.Timedelta(minutes=10 - i))
    s.add("gw_b", False, 100, "TIMEOUT", ts=t - pd.Timedelta(minutes=8))
    removed = s.forget("gw_a", t - pd.Timedelta(minutes=9, seconds=30), t - pd.Timedelta(minutes=7, seconds=30))
    assert removed == 2
    ev = s.snapshot()
    assert len(ev[ev["gateway_id"] == "gw_a"]) == 4 and len(ev[ev["gateway_id"] == "gw_b"]) == 1
    cleanup(s)


@pytest.mark.parametrize("kind", KINDS)
def test_store_trims_events_older_than_retention(kind):
    s = make_store(kind)
    t = utc_now()
    s.add("gw_a", True, 100, None, ts=t - pd.Timedelta(hours=3))
    s.add("gw_a", True, 100, None, ts=t)
    assert len(s.snapshot()) == 1
    cleanup(s)


@pytest.mark.parametrize("kind", KINDS)
def test_store_disabled_set(kind):
    s = make_store(kind)
    assert not s.disabled and "gw_a" not in s.disabled
    s.disabled.add("gw_a")
    s.disabled.update(["gw_b", "gw_c"])
    assert "gw_a" in s.disabled and set(s.disabled) == {"gw_a", "gw_b", "gw_c"} and len(s.disabled) == 3
    s.disabled.discard("gw_b")
    assert sorted(s.disabled) == ["gw_a", "gw_c"]
    cleanup(s)
    if kind != "memory":
        assert not s.disabled


@pytest.mark.parametrize("kind", KINDS)
def test_store_seed_shifts_history_to_now(kind):
    s = make_store(kind)
    hist = pd.DataFrame({
        "gateway_id": ["gw_a", "gw_b", "gw_a"], "ts": pd.to_datetime(["2026-01-01 10:00", "2026-01-01 10:20", "2026-01-01 10:30"]),
        "latency_ms": [100, 200, 300], "success": [True, False, True], "failure_code": [None, "TIMEOUT", None]})
    assert s.seed(hist) == 3
    ev = s.snapshot()
    assert abs((ev["ts"].max() - utc_now()).total_seconds()) < 5
    assert not s.is_empty()
    cleanup(s)


# ------------------------------------------------------------------ the router on a Redis-backed store
def redis_router(parts):
    attempts, bundle = parts
    store = RedisHealthTracker(FakeRedis(), GW_IDS, "r:")
    store.seed(attempts, end=utc_now())
    return Router(bundle, GATEWAY_TABLE, store, (0.01, 0.05), CircuitBreakers(CFG, seed=3))


def test_full_breaker_cycle_works_on_the_redis_store(parts):
    r = redis_router(parts)
    now = utc_now().floor("s")
    victim = r.route(TXN, now=now)["gateway"]
    for i in range(12):
        r.record_outcome(victim, False, 8000, "TIMEOUT", ts=now + pd.Timedelta(seconds=1 + i))
    after = r.route(TXN, now=now + pd.Timedelta(seconds=20))
    assert after["gateway"] != victim and victim in after["tripped_gateways"]
    probe = r.route(TXN, now=now + pd.Timedelta(seconds=60))
    assert probe["mode"] == "probe" and probe["gateway"] == victim
    for i in range(3):
        change = r.record_outcome(victim, True, 250, ts=now + pd.Timedelta(seconds=61 + i))
    assert change == "closed"
    ev = r.tracker.snapshot()
    assert ev[(ev["gateway_id"] == victim) & (ev["failure_code"] == "TIMEOUT")].empty


def test_manual_disable_is_shared_through_redis(parts):
    r = redis_router(parts)
    top = r.route(TXN)["gateway"]
    r.tracker.disabled.add(top)
    other = Router(r.bundle, r.gateways, RedisHealthTracker(r.tracker._r, GW_IDS, "r:"), (0.01, 0.05))
    assert other.route(TXN)["gateway"] != top  # a second instance sees the same kill switch


# ------------------------------------------------------------------ events
def test_apply_outcome_event_updates_health_and_skips_own_messages(parts):
    r = redis_router(parts)
    n0 = len(r.tracker.snapshot())
    ev = outcome_event("gw_a", False, 8000, "TIMEOUT", origin="stream")
    apply_outcome_event(r, ev, own_origin="me")
    assert len(r.tracker.snapshot()) == n0 + 1
    apply_outcome_event(r, outcome_event("gw_a", False, 8000, "TIMEOUT", origin="me"), own_origin="me")
    assert len(r.tracker.snapshot()) == n0 + 1  # own message ignored


def test_bad_events_are_dropped_not_raised(parts):
    r = redis_router(parts)
    n0 = len(r.tracker.snapshot())
    for bad in ({}, {"gateway_id": "gw_zzz", "success": True, "latency_ms": 5},
                {"gateway_id": "gw_a", "success": True, "latency_ms": "slow"}, {"gateway_id": "gw_a"}):
        assert apply_outcome_event(r, bad) is None
    assert len(r.tracker.snapshot()) == n0


def test_decision_event_and_sink_rows(parts):
    r = redis_router(parts)
    d = r.route(TXN)
    ev = json.loads(json.dumps(decision_event(d, TXN, exclude=["gw_a"])))  # survives JSON
    assert ev["attempt"] == 2 and ev["chosen_gateway"] == d["gateway"]
    assert set(ev["scores"]) == {a["gateway"] for a in d["alternatives"]}
    row = to_rows(ev)
    assert row["policy"] == "xgboost" and row["attempt"] == 2
    assert json.loads(row["scores"]) == ev["scores"] and "drivers" in json.loads(row["explanation"])
    assert to_rows({**ev, "mode": "probe"})["policy"] == "xgboost-probe"


def test_sink_rejects_bad_events():
    good = {"txn_id": "t", "decided_at": "2026-03-01 12:00:00", "amount_inr": 10, "method": "upi",
            "bank": "HDFC", "merchant_category": "retail", "chosen_gateway": "gw_a", "scores": {}, "explanation": {}}
    to_rows(good)
    with pytest.raises(ValueError):
        to_rows({k: v for k, v in good.items() if k != "bank"})
    with pytest.raises(ValueError):
        to_rows({**good, "amount_inr": 0})


# ------------------------------------------------------------------ HTTP layer publishes events
def test_api_publishes_decisions_and_outcomes(parts, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    import api.main as main

    monkeypatch.setenv("EVENT_BUS", "none")
    monkeypatch.setattr(main.Router, "from_artifacts", classmethod(lambda cls, **kw: redis_router(parts)))
    with TestClient(main.app) as client:
        health = client.get("/health").json()
        assert health["event_bus"] == "none"
        d = client.post("/route", json=TXN).json()
        client.post("/outcome", json={"gateway_id": "gw_a", "success": True, "latency_ms": 90})
        bus = main.app.state.bus
        topics = [t for t, _, _ in bus.published]
        assert topics == [DECISIONS_TOPIC, OUTCOMES_TOPIC]
        assert bus.published[0][2]["txn_id"] == d["txn_id"]
        assert bus.published[1][2]["origin"] == main.app.state.instance


# ------------------------------------------------------------------ integration (skip when not running)
def test_kafka_roundtrip_through_the_real_broker():
    pytest.importorskip("confluent_kafka")
    from pipeline.bus import ConsumerThread, KafkaBus, bootstrap_servers

    try:
        from confluent_kafka.admin import AdminClient

        AdminClient({"bootstrap.servers": bootstrap_servers(), "log_level": 0}).list_topics(timeout=3)
    except Exception:
        pytest.skip("Kafka is not running")
    topic = f"test.outcomes.{uuid.uuid4().hex[:8]}"
    got = []
    consumer = ConsumerThread(topic, f"g-{uuid.uuid4().hex[:6]}", got.append, offset_reset="earliest")
    bus = KafkaBus()
    bus.publish(topic, "gw_a", outcome_event("gw_a", True, 120))   # creates the topic
    bus.flush()
    consumer.start()
    consumer.ready.wait(20)
    bus.publish(topic, "gw_b", outcome_event("gw_b", False, 8000, "TIMEOUT"))
    bus.flush()
    deadline = time.time() + 20
    while len(got) < 2 and time.time() < deadline:
        time.sleep(0.2)
    consumer.stop()
    assert sorted(m["gateway_id"] for m in got) == ["gw_a", "gw_b"]


def test_decision_sink_writes_to_postgres():
    psycopg2 = pytest.importorskip("psycopg2")
    from ml.db import _connect
    from pipeline.decision_sink import write_decisions

    try:
        conn = _connect()
    except Exception:
        pytest.skip("PostgreSQL is not running")
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM gateways")
        if cur.fetchone()[0] == 0:
            conn.close()
            pytest.skip("gateways table is empty: run scripts/load_data.py first")
    txn_id = f"test-{uuid.uuid4().hex[:10]}"
    ev = {"txn_id": txn_id, "attempt": 1, "decided_at": "2026-03-01 12:00:00", "amount_inr": 499.0,
          "method": "upi", "bank": "TESTBANK", "merchant_category": "testcat", "chosen_gateway": "gw_a",
          "mode": "normal", "model_version": "t", "scores": {"gw_a": 0.9}, "explanation": {"drivers": []}}
    try:
        assert write_decisions(conn, [ev, ev]) == 2     # duplicate is ignored by the database
        with conn.cursor() as cur:
            cur.execute("SELECT count(*), max(chosen_gateway) FROM routing_decisions WHERE txn_id=%s", (txn_id,))
            assert cur.fetchone() == (1, "gw_a")
    finally:
        with conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM routing_decisions WHERE txn_id=%s", (txn_id,))
                cur.execute("DELETE FROM transactions WHERE txn_id=%s", (txn_id,))
                cur.execute("DELETE FROM banks WHERE bank='TESTBANK'")
                cur.execute("DELETE FROM merchant_categories WHERE category='testcat'")
        conn.close()
