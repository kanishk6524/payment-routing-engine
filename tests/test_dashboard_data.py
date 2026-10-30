"""Dashboard data layer tests (pure pandas, no Streamlit needed)."""
import json
from pathlib import Path

import pandas as pd
import pytest

from dashboards import data

T = pd.Timestamp("2026-03-01 12:00:30")


def events(rows):
    return pd.DataFrame(rows, columns=["gateway_id", "ts", "latency_ms", "success", "failure_code"])


def test_classify_states():
    assert data.classify("open", 0.99, 50) == "down"
    assert data.classify("half_open", 0.99, 50) == "degraded"
    assert data.classify("closed", 0.95, 50) == "healthy"
    assert data.classify("closed", 0.60, 50) == "degraded"
    assert data.classify("closed", None, 0) == "unknown"
    assert data.classify("closed", float("nan"), 12) == "unknown"


def test_gateway_cards_from_api_payload():
    cards = data.gateway_cards([
        {"gateway_id": "gw_a", "enabled": True, "breaker": "closed",
         "last_15_min": {"attempts": 40, "success_rate": 0.9, "avg_latency_ms": 310}},
        {"gateway_id": "gw_d", "enabled": False, "breaker": "open", "last_15_min": None}])
    assert cards[0]["state"] == "healthy" and cards[0]["success_pct"] == 90.0 and cards[0]["attempts"] == 40
    assert cards[1]["state"] == "down" and cards[1]["success_pct"] is None and cards[1]["enabled"] is False


def test_bucket_series_counts_and_rates():
    ev = events([
        ("gw_a", T - pd.Timedelta(seconds=70), 200, True, None),
        ("gw_a", T - pd.Timedelta(seconds=65), 400, False, "TIMEOUT"),
        ("gw_a", T - pd.Timedelta(seconds=5), 300, True, None),
        ("gw_b", T - pd.Timedelta(seconds=5), 100, True, None),
        ("gw_b", T - pd.Timedelta(hours=3), 100, False, "TIMEOUT"),   # outside the window
    ])
    out = data.bucket_series(ev, now=T, window_min=60, bucket_s=60)
    a = out[out["gateway_id"] == "gw_a"].sort_values("bucket")
    assert a["attempts"].tolist() == [2, 1] and a["success_pct"].tolist() == [50.0, 100.0]
    assert a["avg_latency_ms"].tolist() == [300.0, 300.0]
    assert out[out["gateway_id"] == "gw_b"]["attempts"].sum() == 1


def test_bucket_series_handles_empty():
    assert data.bucket_series(events([])).empty
    old = events([("gw_a", T - pd.Timedelta(days=2), 100, True, None)])
    assert data.bucket_series(old, now=T).empty


def test_traffic_share_sums_to_100_per_bucket():
    d = pd.DataFrame({"decided_at": [T, T, T, T + pd.Timedelta(seconds=40), T + pd.Timedelta(minutes=2)],
                      "chosen_gateway": ["gw_a", "gw_a", "gw_d", "gw_d", "gw_b"]})
    s = data.traffic_share(d)
    assert s.groupby("bucket")["share_pct"].sum().round(1).tolist() == [100.0, 100.0, 100.0]
    first = s[s["bucket"] == s["bucket"].min()].set_index("gateway")["share_pct"]
    assert first["gw_a"] == 66.67 and first["gw_d"] == 33.33
    assert data.traffic_share(d.iloc[0:0]).empty


def test_frames_from_explanation_and_scores():
    expl = {"drivers": [{"label": "gateway", "value": "gw_d", "effect": 0.4, "direction": "raises success odds"}]}
    assert data.drivers_frame(expl)["effect"].tolist() == [0.4]
    assert data.drivers_frame({}).empty and data.drivers_frame(None).empty
    sc = data.scores_frame({"gw_a": 0.91, "gw_d": 0.97, "gw_b": 0.9}, "gw_d")
    assert sc["gateway"].tolist() == ["gw_b", "gw_a", "gw_d"] and sc[sc["chosen"]]["gateway"].tolist() == ["gw_d"]
    alts = data.alternatives_frame([{"gateway": "gw_a", "score": 0.9}, {"gateway": "gw_b", "score": 0.8}], "gw_b")
    assert alts["chosen"].tolist() == [False, True]


def test_policy_frame_orders_and_converts():
    res = {"policies": {k: {"all": {"success_rate": 0.9 + i / 100, "effective_fee_pct": 2.0 + i / 10,
                                    "avg_latency_ms": 400 + i}, "incident": {"success_rate": 0.5 + i / 100}}
                        for i, k in enumerate(data.POLICY_ORDER)}}
    pf = data.policy_frame(res)
    assert pf["key"].tolist() == data.POLICY_ORDER and pf.iloc[0]["success_pct"] == 90.0
    assert data.policy_frame({}).empty and data.policy_frame(None).empty


def test_real_results_file_if_present():
    path = Path(__file__).resolve().parents[1] / "reports" / "results.json"
    if not path.exists():
        pytest.skip("run python -m ml.evaluate first")
    pf = data.policy_frame(json.loads(path.read_text()))
    ml = pf[pf["key"] == "ml"].iloc[0]
    assert len(pf) == 7 and ml["success_pct"] > pf[pf["key"] == "cheapest"].iloc[0]["success_pct"]


def test_sources_fail_soft_when_services_are_down(monkeypatch):
    pytest.importorskip("httpx")
    monkeypatch.setattr(data, "API_URL", "http://127.0.0.1:1")
    out, err = data.api_get("/health")
    assert out is None and err
