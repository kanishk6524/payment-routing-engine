"""Stage 9 tests: circuit breaker state machine, probing, failover and recovery."""
import pandas as pd
import pytest

from api.breaker import (
    CLOSED, HALF_OPEN, OPEN, BreakerConfig, CircuitBreakers, is_gateway_failure,
)
from api.health import HealthTracker
from api.service import NoGatewayAvailable, Router
from ml.features import TARGET, build_features
from ml.model import encode, fit_model, learn_levels, make_model
from simulator.config import GATEWAYS
from simulator.generate import generate

T0 = pd.Timestamp("2026-03-01 12:00:00")
CFG = BreakerConfig(window_s=120, min_attempts=10, failure_rate=0.5, cooldown_s=30,
                    probe_share=1.0, probe_max_amount=2000, probe_successes=3)


def at(seconds):
    return T0 + pd.Timedelta(seconds=seconds)


def trip(b, gw="gw_a", n=10):
    for i in range(n):
        b.observe(gw, False, "TIMEOUT", at(i))


# ------------------------------------------------------------------ state machine
def test_only_gateway_side_failures_count():
    assert is_gateway_failure(False, "TIMEOUT") and is_gateway_failure(False, "GATEWAY_ERROR")
    assert is_gateway_failure(False, None)
    assert not is_gateway_failure(False, "INSUFFICIENT_FUNDS")
    assert not is_gateway_failure(True, None)


def test_trips_at_threshold_but_not_below_min_attempts():
    b = CircuitBreakers(CFG)
    for i in range(9):
        assert b.observe("gw_a", False, "TIMEOUT", at(i)) is None
    assert b.state("gw_a", at(9)) == CLOSED
    assert b.observe("gw_a", False, "TIMEOUT", at(9)) == "opened"
    assert b.state("gw_a", at(10)) == OPEN


def test_bank_declines_never_trip_the_breaker():
    b = CircuitBreakers(CFG)
    for i in range(50):
        b.observe("gw_a", False, "INSUFFICIENT_FUNDS", at(i))
    assert b.state("gw_a", at(60)) == CLOSED


def test_healthy_traffic_with_some_failures_stays_closed():
    b = CircuitBreakers(CFG)
    for i in range(60):
        b.observe("gw_a", i % 5 != 0, "TIMEOUT", at(i))  # 20% failures
    assert b.state("gw_a", at(60)) == CLOSED


def test_old_failures_age_out_of_the_window():
    b = CircuitBreakers(CFG)
    for i in range(9):
        b.observe("gw_a", False, "TIMEOUT", at(i))
    assert b.observe("gw_a", False, "TIMEOUT", at(500)) is None  # the nine are long gone


def test_cooldown_then_half_open():
    b = CircuitBreakers(CFG)
    trip(b)
    assert b.state("gw_a", at(9 + 29)) == OPEN
    assert b.state("gw_a", at(9 + 30)) == HALF_OPEN


def test_probe_failure_reopens_and_restarts_cooldown():
    b = CircuitBreakers(CFG)
    trip(b)
    assert b.state("gw_a", at(60)) == HALF_OPEN
    assert b.observe("gw_a", False, "TIMEOUT", at(61)) == "reopened"
    assert b.state("gw_a", at(70)) == OPEN
    assert b.state("gw_a", at(61 + 30)) == HALF_OPEN


def test_enough_probe_successes_close_it_and_reports_the_incident_window():
    b = CircuitBreakers(CFG)
    trip(b)
    assert b.state("gw_a", at(60)) == HALF_OPEN
    assert b.observe("gw_a", True, None, at(61)) is None
    assert b.observe("gw_a", True, None, at(62)) is None
    assert b.observe("gw_a", True, None, at(63)) == "closed"
    assert b.state("gw_a", at(64)) == CLOSED
    start, end = b.incident_window("gw_a")
    assert start == at(9) - pd.Timedelta(seconds=120) and end == at(60)


def test_results_that_started_before_probing_are_not_probes():
    b = CircuitBreakers(CFG)
    trip(b)
    b.state("gw_a", at(60))
    assert b.observe("gw_a", False, "TIMEOUT", at(40)) is None  # stale in-flight result
    assert b.state("gw_a", at(61)) == HALF_OPEN


def test_probes_only_for_low_value_transactions_in_half_open():
    b = CircuitBreakers(CFG)
    assert not b.should_probe("gw_a", at(0), 100)          # closed
    trip(b)
    assert not b.should_probe("gw_a", at(20), 100)         # open
    assert b.should_probe("gw_a", at(60), 100)             # half open, small amount
    assert not b.should_probe("gw_a", at(60), 50_000)      # too valuable to risk


def test_probe_share_is_respected():
    b = CircuitBreakers(BreakerConfig(**{**CFG.__dict__, "probe_share": 0.1}), seed=1)
    trip(b)
    hits = sum(b.should_probe("gw_a", at(60), 100) for _ in range(2000))
    assert 140 < hits < 260


# ------------------------------------------------------------------ routing integration
GATEWAY_TABLE = pd.DataFrame(
    [{"gateway_id": g.id, "fee_pct": g.fee_pct, "fee_fixed": g.fee_fixed,
      "supported_methods": "|".join(g.methods)} for g in GATEWAYS]
)
TXN = {"amount_inr": 1000, "method": "card", "bank": "HDFC", "merchant_category": "retail"}


@pytest.fixture(scope="module")
def parts():
    attempts, _ = generate(30_000, 10, seed=11)
    attempts = attempts.sort_values(["ts", "txn_id"]).reset_index(drop=True)
    df = build_features(attempts, GATEWAY_TABLE)
    tr, va = df[df["split"] == "train"], df[df["split"] == "valid"]
    lv = learn_levels(tr)
    m = make_model("hgb", 1)
    fit_model(m, "hgb", encode(tr, lv), tr[TARGET], encode(va, lv), va[TARGET])
    bundle = {"model": m, "kind": "hgb", "levels": lv, "version": "hgb-test",
              "latency_fallback": tr.groupby("gateway_id")["latency_ms"].mean().to_dict()}
    return attempts, bundle


def make_router(parts, cfg=CFG):
    attempts, bundle = parts
    tracker = HealthTracker()
    tracker.seed(attempts, end=T0)
    return Router(bundle, GATEWAY_TABLE, tracker, (0.01, 0.05), CircuitBreakers(cfg, seed=3))


def fail(router, gw, n=12, start=1):
    for i in range(n):
        router.record_outcome(gw, False, 8000, "TIMEOUT", ts=at(start + i))


def test_tripped_gateway_receives_no_normal_traffic(parts):
    r = make_router(parts)
    victim = r.route(TXN, now=at(0))["gateway"]
    fail(r, victim)
    d = r.route(TXN, now=at(20))
    assert d["mode"] == "normal" and d["gateway"] != victim
    assert victim in d["tripped_gateways"]
    row = next(a for a in d["alternatives"] if a["gateway"] == victim)
    assert row["breaker"] == OPEN and not row["eligible"]


def test_half_open_gateway_gets_only_small_probe_transactions(parts):
    r = make_router(parts)
    victim = r.route(TXN, now=at(0))["gateway"]
    fail(r, victim)
    big = r.route({**TXN, "amount_inr": 40_000}, now=at(60))
    assert big["gateway"] != victim and big["mode"] == "normal"
    small = r.route(TXN, now=at(60))
    assert small["gateway"] == victim and small["mode"] == "probe"
    assert "Probe" in small["summary"]


def test_full_cycle_trip_probe_recover_and_forget_the_incident(parts):
    r = make_router(parts)
    victim = r.route(TXN, now=at(0))["gateway"]
    fail(r, victim)
    assert r.breakers.state(victim, at(20)) == OPEN
    assert r.route(TXN, now=at(60))["mode"] == "probe"
    assert r.record_outcome(victim, True, 250, ts=at(61)) is None
    assert r.record_outcome(victim, True, 250, ts=at(62)) is None
    assert r.record_outcome(victim, True, 250, ts=at(63)) == "closed"
    ev = r.tracker.snapshot()
    gone = ev[(ev["gateway_id"] == victim) & (ev["failure_code"] == "TIMEOUT")]
    assert gone.empty  # the incident no longer drags the gateway's health down
    after = r.route(TXN, now=at(70))
    assert after["mode"] == "normal" and victim not in after["tripped_gateways"]


def test_fail_open_when_every_supporting_gateway_is_tripped(parts):
    r = make_router(parts)
    for gw in ("gw_a", "gw_b", "gw_c", "gw_d"):   # netbanking is supported by these four
        fail(r, gw)
    d = r.route({**TXN, "method": "netbanking", "amount_inr": 40_000}, now=at(20))
    assert d["mode"] == "fail_open" and d["gateway"] in {"gw_a", "gw_b", "gw_c", "gw_d"}


def test_exclude_lets_a_client_retry_on_the_next_gateway(parts):
    r = make_router(parts)
    first = r.route(TXN, now=at(0))
    second = r.route(TXN, now=at(0), exclude=[first["gateway"]])
    assert second["gateway"] != first["gateway"]
    assert first["gateway"] not in {a["gateway"] for a in second["alternatives"]}
    with pytest.raises(NoGatewayAvailable):
        r.route(TXN, now=at(0), exclude=[g.id for g in GATEWAYS])


def test_bank_declines_do_not_trip_a_gateway_through_the_service(parts):
    r = make_router(parts)
    gw = r.route(TXN, now=at(0))["gateway"]
    for i in range(40):
        r.record_outcome(gw, False, 300, "INSUFFICIENT_FUNDS", ts=at(1 + i))
    assert r.breakers.state(gw, at(50)) == CLOSED
