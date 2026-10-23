import numpy as np
import pandas as pd
import pytest

from ml.features import (
    FEATURE_COLUMNS, GATEWAY_SIDE_FAILURES, ROLLING_FEATURES, STATIC_FEATURES, TARGET,
    add_gateway_rolling_features, build_features,
)
from simulator.config import GATEWAYS
from simulator.generate import generate

GATEWAY_TABLE = pd.DataFrame(
    [{"gateway_id": g.id, "fee_pct": g.fee_pct, "fee_fixed": g.fee_fixed} for g in GATEWAYS]
)


@pytest.fixture(scope="module")
def attempts():
    df, _ = generate(15_000, 10, seed=3)
    return df


def _row(ts, gw, latency_ms, success, code):
    return {"ts": pd.Timestamp(ts), "gateway_id": gw, "latency_ms": latency_ms,
            "success": success, "failure_code": code}


def test_rolling_features_exact_values_on_a_hand_built_case():
    df = pd.DataFrame([
        _row("2026-01-01 10:00:00.000", "gw_a", 1000, True, None),            # A, done 10:00:01.0
        _row("2026-01-01 10:00:00.500", "gw_a", 8000, False, "TIMEOUT"),      # B, done 10:00:08.5
        _row("2026-01-01 10:00:02.000", "gw_a", 200, False, "BANK_DECLINE"),  # C, done 10:00:02.2
        _row("2026-01-01 10:00:10.000", "gw_a", 300, True, None),             # D, done 10:00:10.3
        _row("2026-01-01 10:00:10.000", "gw_b", 400, True, None),             # other gateway
        _row("2026-01-01 10:06:02.000", "gw_a", 250, True, None),             # F
    ])
    out = add_gateway_rolling_features(df)
    A, B, C, D, other, F = (out.iloc[i] for i in range(6))

    # A is the first attempt: no history. B starts before A has completed (still in flight).
    assert A["gw_n_5m"] == 0 and np.isnan(A["gw_success_rate_5m"])
    assert B["gw_n_5m"] == 0
    # C: only A has completed before 10:00:02
    assert C["gw_n_5m"] == 1 and C["gw_success_rate_5m"] == 1.0
    # D: A, B and C are all done; B is a gateway-side failure, C is a bank decline
    assert D["gw_n_5m"] == 3
    assert D["gw_success_rate_5m"] == pytest.approx(1 / 3)
    assert D["gw_fail_rate_5m"] == pytest.approx(1 / 3)
    assert D["gw_avg_latency_5m"] == pytest.approx((1000 + 8000 + 200) / 3)
    assert D["gw_p95_latency_15m"] == 8000
    # another gateway never sees gw_a history
    assert other["gw_n_60m"] == 0
    # F at 10:06:02: everything is older than 5 minutes, but inside 15 and 60
    assert F["gw_n_5m"] == 0 and np.isnan(F["gw_success_rate_5m"])
    assert F["gw_n_15m"] == 4 and F["gw_n_60m"] == 4
    assert F["gw_success_rate_15m"] == pytest.approx(2 / 4)


def test_rolling_features_match_a_brute_force_recomputation(attempts):
    feats = add_gateway_rolling_features(attempts.sort_values(["ts", "txn_id"]).reset_index(drop=True))
    done = feats["ts"] + pd.to_timedelta(feats["latency_ms"], unit="ms")
    rng = np.random.default_rng(0)
    for i in rng.choice(len(feats), 300, replace=False):
        row = feats.iloc[i]
        for w in (5, 15, 60):
            same = (feats["gateway_id"] == row["gateway_id"]) & (done < row["ts"]) \
                & (done >= row["ts"] - pd.Timedelta(minutes=w))
            win = feats[same]
            assert row[f"gw_n_{w}m"] == len(win)
            if len(win):
                assert row[f"gw_success_rate_{w}m"] == pytest.approx(win[TARGET].mean())
                assert row[f"gw_fail_rate_{w}m"] == pytest.approx(
                    win["failure_code"].isin(GATEWAY_SIDE_FAILURES).mean())
                assert row[f"gw_avg_latency_{w}m"] == pytest.approx(win["latency_ms"].mean())
            else:
                assert np.isnan(row[f"gw_success_rate_{w}m"])
        win15 = feats[(feats["gateway_id"] == row["gateway_id"]) & (done < row["ts"])
                      & (done >= row["ts"] - pd.Timedelta(minutes=15))]["latency_ms"].sort_values()
        if len(win15):
            expected = win15.iloc[max(int(np.ceil(0.95 * len(win15))) - 1, 0)]
            assert row["gw_p95_latency_15m"] == expected


def test_features_never_peek_into_the_future(attempts):
    """Dropping every row after a cutoff must not change any earlier row's features."""
    full = build_features(attempts, GATEWAY_TABLE).set_index("txn_id")
    cutoff = attempts["ts"].quantile(0.5)
    early = attempts[attempts["ts"] < cutoff]
    truncated = build_features(early, GATEWAY_TABLE).set_index("txn_id")
    pd.testing.assert_frame_equal(
        full.loc[truncated.index, ROLLING_FEATURES], truncated[ROLLING_FEATURES]
    )


def test_split_is_chronological_and_roughly_60_20_20(attempts):
    df = build_features(attempts, GATEWAY_TABLE)
    parts = {s: df[df["split"] == s] for s in ("train", "valid", "test")}
    assert sum(len(p) for p in parts.values()) == len(df)
    assert parts["train"]["ts"].max() < parts["valid"]["ts"].min()
    assert parts["valid"]["ts"].max() < parts["test"]["ts"].min()
    for name, share in (("train", 0.6), ("valid", 0.2), ("test", 0.2)):
        assert abs(len(parts[name]) / len(df) - share) < 0.02


def test_feature_table_shape_and_columns(attempts):
    df = build_features(attempts, GATEWAY_TABLE)
    assert len(df) == len(attempts)
    assert df["txn_id"].is_unique
    assert set(FEATURE_COLUMNS) <= set(df.columns)
    assert df[STATIC_FEATURES].notna().all().all()
    assert df["amount_bucket"].between(0, 8).all()
    assert set(df[TARGET].unique()) <= {0, 1}


def test_outcome_columns_and_simulator_hints_are_not_features():
    forbidden = {"success", "latency_ms", "failure_code", "split", "txn_id", "fee_inr",
                 "is_peak_hour", "is_high_value_card", "in_outage"}
    assert forbidden.isdisjoint(FEATURE_COLUMNS)
