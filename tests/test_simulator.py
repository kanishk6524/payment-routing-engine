import numpy as np
import pandas as pd
import pytest

from simulator.config import GATEWAYS_BY_ID, HIGH_VALUE_INR, PEAK_HOURS
from simulator.generate import generate
from simulator.model import outage_mask

N, DAYS = 200_000, 60


@pytest.fixture(scope="module")
def data():
    return generate(N, DAYS, seed=7)


def test_deterministic_for_same_seed():
    a, _ = generate(3_000, 10, seed=1)
    b, _ = generate(3_000, 10, seed=1)
    pd.testing.assert_frame_equal(a, b)


def test_schema_and_ranges(data):
    df, _ = data
    expected = {
        "txn_id", "ts", "amount_inr", "method", "bank", "merchant_category",
        "gateway_id", "success", "latency_ms", "failure_code", "fee_inr", "attempt",
    }
    assert expected == set(df.columns)
    assert len(df) == N
    assert df["txn_id"].is_unique
    assert df["ts"].is_monotonic_increasing
    assert df["amount_inr"].between(10, 200_000).all()
    assert df.loc[df["method"] == "upi", "amount_inr"].max() <= 100_000
    assert (df["latency_ms"] >= 50).all()


def test_gateway_supports_method(data):
    df, _ = data
    for gw_id, sub in df.groupby("gateway_id"):
        assert set(sub["method"]) <= set(GATEWAYS_BY_ID[gw_id].methods)


def test_failures_have_codes_and_successes_do_not(data):
    df, _ = data
    assert df.loc[df["success"], "failure_code"].isna().all()
    assert df.loc[~df["success"], "failure_code"].notna().all()


def test_outage_windows_drop_success_rate(data):
    df, outages = data
    for o in outages:
        sub = df[df["gateway_id"] == o.gateway_id]
        inside = outage_mask(o.gateway_id, sub["ts"], [o])
        assert inside.sum() > 30, "outage window should contain enough rows"
        assert sub.loc[inside, "success"].mean() < sub.loc[~inside, "success"].mean() - 0.3


def _clean(df, outages, gw_id):
    sub = df[df["gateway_id"] == gw_id]
    return sub[~outage_mask(gw_id, sub["ts"], outages)]


def test_gw_a_weak_on_high_value_cards(data):
    df, outages = data
    sub = _clean(df, outages, "gw_a")
    card = sub[sub["method"] == "card"]
    high = card[card["amount_inr"] >= HIGH_VALUE_INR]["success"].mean()
    low = card[card["amount_inr"] < HIGH_VALUE_INR]["success"].mean()
    assert low - high > 0.08


def test_gw_b_degrades_in_peak_hours(data):
    df, outages = data
    sub = _clean(df, outages, "gw_b")
    peak = sub[sub["ts"].dt.hour.isin(PEAK_HOURS)]["success"].mean()
    off = sub[~sub["ts"].dt.hour.isin(PEAK_HOURS)]["success"].mean()
    assert off - peak > 0.08


def test_gw_d_weak_on_sbi(data):
    df, outages = data
    sub = _clean(df, outages, "gw_d")
    sbi = sub[sub["bank"] == "SBI"]["success"].mean()
    other = sub[sub["bank"] != "SBI"]["success"].mean()
    assert other - sbi > 0.05


def test_expected_latency_matches_sampled_latency(data):
    from simulator.model import expected_latency_ms

    df, outages = data
    for gw_id, gw in GATEWAYS_BY_ID.items():
        sub = df[df["gateway_id"] == gw_id]
        expected = expected_latency_ms(gw, sub, outages).mean()
        assert expected == pytest.approx(sub["latency_ms"].mean(), rel=0.03)


def test_cost_and_reliability_tradeoff(data):
    df, outages = data
    rate = {g: _clean(df, outages, g)["success"].mean() for g in GATEWAYS_BY_ID}
    assert max(rate, key=rate.get) == "gw_d"
    assert min(rate, key=rate.get) == "gw_e"
    assert GATEWAYS_BY_ID["gw_e"].fee_pct == min(g.fee_pct for g in GATEWAYS_BY_ID.values())
