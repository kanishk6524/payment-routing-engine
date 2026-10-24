"""Routing service tests (no web server needed) and the HTTP layer when FastAPI is installed."""
import pandas as pd
import pytest

from api.health import HealthTracker
from api.service import NoGatewayAvailable, Router
from ml.explain import describe_value, top_drivers
from ml.features import TARGET, build_features
from ml.model import encode, fit_model, learn_levels, make_model
from simulator.config import GATEWAYS
from simulator.generate import generate

GATEWAY_TABLE = pd.DataFrame(
    [{"gateway_id": g.id, "fee_pct": g.fee_pct, "fee_fixed": g.fee_fixed,
      "supported_methods": "|".join(g.methods)} for g in GATEWAYS]
)
TXN = {"amount_inr": 2500, "method": "card", "bank": "HDFC", "merchant_category": "retail"}


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


def make_router(parts):
    attempts, bundle = parts
    tracker = HealthTracker()
    tracker.seed(attempts)
    return Router(bundle, GATEWAY_TABLE, tracker, (0.01, 0.05))


def test_route_returns_a_supported_gateway_with_ranked_alternatives(parts):
    d = make_router(parts).route(TXN)
    assert d["gateway"] in {g.id for g in GATEWAYS}
    alts = d["alternatives"]
    assert alts[0]["gateway"] == d["gateway"]
    assert [a["score"] for a in alts] == sorted((a["score"] for a in alts), reverse=True)
    assert 0 <= d["predicted_success"] <= 1 and d["decision_ms"] >= 0


def test_gateway_without_netbanking_is_never_offered_for_netbanking(parts):
    d = make_router(parts).route({**TXN, "method": "netbanking"})
    assert "gw_e" not in {a["gateway"] for a in d["alternatives"]}


def test_disabled_gateway_is_never_chosen_and_all_disabled_raises(parts):
    r = make_router(parts)
    top = r.route(TXN)["gateway"]
    r.tracker.disabled.add(top)
    assert r.route(TXN)["gateway"] != top
    r.tracker.disabled.update(r.gateway_ids)
    with pytest.raises(NoGatewayAvailable):
        r.route(TXN)


def test_router_moves_traffic_away_from_a_failing_gateway(parts):
    r = make_router(parts)
    now = pd.Timestamp.now("UTC").tz_localize(None)
    victim = r.route(TXN, now=now)["gateway"]
    for _ in range(80):
        r.record_outcome(victim, False, 8000, "TIMEOUT", ts=now - pd.Timedelta(minutes=2))
    after = r.route(TXN, now=now)
    assert after["gateway"] != victim
    p = {a["gateway"]: a["predicted_success"] for a in after["alternatives"]}
    assert p[victim] < p[after["gateway"]]


def test_unknown_gateway_outcome_is_rejected(parts):
    with pytest.raises(KeyError):
        make_router(parts).record_outcome("gw_zzz", True, 100)


def test_health_tracker_trims_old_events():
    t = HealthTracker()
    now = pd.Timestamp("2026-01-01 12:00:00")
    t.add("gw_a", True, 100, ts=now - pd.Timedelta(hours=3))
    t.add("gw_a", True, 100, ts=now)
    assert len(t.snapshot()) == 1


def test_driver_formatting():
    assert describe_value("gw_fail_rate_15m", 0.0123) == "1.2%"
    assert describe_value("gw_avg_latency_60m", 301.4) == "301 ms"
    assert describe_value("gw_success_rate_5m", float("nan")) == "no recent data"
    c = pd.Series({"a": 0.1, "b": -0.9, "c": 0.4})
    f = pd.Series({"a": 1, "b": 2, "c": 3})
    out = top_drivers(c, f, k=2)
    assert [d["feature"] for d in out] == ["b", "c"] and out[0]["direction"] == "lowers success odds"


def test_xgboost_contributions_add_up_to_the_prediction(parts):
    try:
        import xgboost  # noqa: F401
    except Exception as exc:
        pytest.skip(f"XGBoost unavailable: {type(exc).__name__}")
    import numpy as np

    from ml.explain import contributions
    from ml.model import predict_proba

    attempts, bundle = parts
    df = build_features(attempts, GATEWAY_TABLE)
    tr, va = df[df["split"] == "train"], df[df["split"] == "valid"]
    lv = bundle["levels"]
    m = make_model("xgboost", 1)
    fit_model(m, "xgboost", encode(tr, lv), tr[TARGET], encode(va, lv), va[TARGET])
    X = encode(va.iloc[:200], lv)
    c = contributions(m, X)
    logit = c.sum(axis=1).to_numpy() + c.attrs["bias"]
    np.testing.assert_allclose(1 / (1 + np.exp(-logit)), predict_proba(m, X), atol=1e-4)


def test_http_layer(parts, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    import api.main as main

    monkeypatch.setenv("EVENT_BUS", "none")
    monkeypatch.setattr(main.Router, "from_artifacts", classmethod(lambda cls, **kw: make_router(parts)))
    with TestClient(main.app) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert len(client.get("/gateways").json()) == 5
        ok = client.post("/route", json=TXN)
        assert ok.status_code == 200 and "gateway" in ok.json()
        assert client.post("/route", json={**TXN, "method": "cheque"}).status_code == 422
        assert client.post("/route", json={**TXN, "amount_inr": -5}).status_code == 422
        assert client.post("/outcome", json={"gateway_id": "gw_a", "success": True, "latency_ms": 90}).status_code == 202
        assert client.post("/outcome", json={"gateway_id": "nope", "success": True, "latency_ms": 90}).status_code == 404
        assert client.post("/gateways/gw_d/disable").json()["enabled"] is False
        assert client.post("/gateways/nope/disable").status_code == 404
