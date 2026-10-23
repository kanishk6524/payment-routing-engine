"""Stage 5 tests: model utilities and the counterfactual replay."""
import numpy as np
import pandas as pd
import pytest

from ml.features import FEATURE_COLUMNS, TARGET, build_features, rolling_stats
from ml.model import (
    encode, expected_calibration_error, fit_model, learn_levels, make_model, predict_proba, score,
)
from ml.replay import (
    TXN_COLUMNS, add_ground_truth, add_model_scores, baseline_scores, build_candidates, choose,
    ips_success, ml_score, policy_metrics, tune_weights,
)
from simulator.config import GATEWAYS
from simulator.generate import generate

GATEWAY_TABLE = pd.DataFrame(
    [{"gateway_id": g.id, "fee_pct": g.fee_pct, "fee_fixed": g.fee_fixed,
      "supported_methods": "|".join(g.methods)} for g in GATEWAYS]
)


@pytest.fixture(scope="module")
def world():
    attempts, outages = generate(40_000, 14, seed=5)
    attempts = attempts.sort_values(["ts", "txn_id"]).reset_index(drop=True)
    df = build_features(attempts, GATEWAY_TABLE)
    parts = {s: df[df["split"] == s] for s in ("train", "valid", "test")}
    levels = learn_levels(parts["train"])
    model = make_model("hgb", 1)
    fit_model(model, "hgb", encode(parts["train"], levels), parts["train"][TARGET],
              encode(parts["valid"], levels), parts["valid"][TARGET])
    bundle = {"model": model, "levels": levels,
              "latency_fallback": parts["train"].groupby("gateway_id")["latency_ms"].mean().to_dict()}
    test = parts["test"].iloc[:6000]
    cands = add_model_scores(build_candidates(test[TXN_COLUMNS], GATEWAY_TABLE, attempts), bundle)
    cands = add_ground_truth(cands, outages)
    return {"attempts": attempts, "parts": parts, "bundle": bundle, "test": test,
            "cands": cands, "outages": outages}


# ------------------------------------------------------------------ model utilities
def test_score_perfect_and_uninformative():
    y = np.array([0, 1] * 500)
    assert score(y, y.astype(float))["auc"] == 1.0
    assert score(y, np.full(1000, 0.5))["auc"] == 0.5


def test_calibration_error_zero_when_calibrated():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.1, 0.9, 200_000)
    y = rng.uniform(size=p.size) < p
    assert expected_calibration_error(y, p) < 0.01
    assert expected_calibration_error(y, np.clip(p + 0.2, 0, 1)) > 0.1


def test_encode_unseen_category_becomes_missing(world):
    row = world["parts"]["test"].iloc[:1].copy()
    row["bank"] = "NOT_A_BANK"
    X = encode(row, world["bundle"]["levels"])
    assert X["bank"].isna().all()
    assert list(X.columns) == FEATURE_COLUMNS
    assert 0 <= predict_proba(world["bundle"]["model"], X)[0] <= 1


def test_xgboost_trains_when_installed(world):
    xgb = pytest.importorskip("xgboost")  # skipped where XGBoost is unavailable
    del xgb
    tr, va = world["parts"]["train"], world["parts"]["valid"]
    lv = world["bundle"]["levels"]
    m = make_model("xgboost", 1)
    fit_model(m, "xgboost", encode(tr, lv), tr[TARGET], encode(va, lv), va[TARGET])
    assert score(va[TARGET], predict_proba(m, encode(va, lv)))["auc"] > 0.6


# ------------------------------------------------------------------ rolling stats on arbitrary queries
def test_rolling_stats_query_at_attempt_time_matches_pipeline(world):
    ev = world["attempts"].iloc[:3000]
    q = ev[["gateway_id", "ts"]]
    out = rolling_stats(ev, q)
    assert out.index.equals(q.index)
    full = build_features(world["attempts"], GATEWAY_TABLE)
    cols = [c for c in out.columns if c in full.columns]
    a = full.set_index(["txn_id", "attempt"]).loc[list(zip(ev["txn_id"], ev["attempt"])), cols]
    np.testing.assert_allclose(out[cols].to_numpy(dtype=float), a.to_numpy(dtype=float),
                               equal_nan=True, rtol=1e-9, atol=1e-9)


def test_rolling_stats_ignores_events_not_yet_completed():
    ts = pd.Timestamp("2026-01-01 10:00:00")
    ev = pd.DataFrame([{"ts": ts, "gateway_id": "gw_a", "latency_ms": 5000, "success": True,
                        "failure_code": None}])
    q = pd.DataFrame({"gateway_id": ["gw_a", "gw_a"],
                      "ts": [ts + pd.Timedelta(seconds=2), ts + pd.Timedelta(seconds=6)]})
    out = rolling_stats(ev, q)
    assert out["gw_n_5m"].tolist() == [0, 1]  # finished at +5s


# ------------------------------------------------------------------ candidates and policies
def test_candidates_cover_every_supported_gateway_once(world):
    c, t = world["cands"], world["test"]
    assert c["txn_id"].nunique() == len(t)
    assert not c.duplicated(["txn_id", "gateway_id"]).any()
    support = {g.id: set(g.methods) for g in GATEWAYS}
    assert all(m in support[g] for g, m in zip(c["gateway_id"], c["method"]))
    assert (c.groupby("txn_id").size() == c.groupby("txn_id")["n_candidates"].first()).all()


def test_probabilities_are_valid(world):
    c = world["cands"]
    assert c["p_pred"].between(0, 1).all() and c["p_true"].between(0, 1).all()
    assert c["lat_true"].gt(0).all() and c["lat_est"].notna().all()


def test_cheapest_picks_minimum_fee(world):
    c = world["cands"]
    pick = choose(c, baseline_scores(c, world["parts"]["train"])["cheapest"])
    assert len(pick) == c["txn_id"].nunique()
    np.testing.assert_allclose(pick.set_index("txn_id")["fee_inr"],
                               c.groupby("txn_id")["fee_inr"].min())


def test_random_policy_equals_candidate_average(world):
    c = world["cands"]
    m = policy_metrics(c, 1.0 / c["n_candidates"])
    direct = c.groupby("txn_id")["p_true"].mean().mean()
    assert m["success_rate"] == pytest.approx(direct)


def test_oracle_is_upper_bound_and_ml_beats_simple_policies(world):
    c, train = world["cands"], world["parts"]["train"]
    rate = lambda s: policy_metrics(choose(c, s))["success_rate"]  # noqa: E731
    base = baseline_scores(c, train)
    oracle, ml = rate(c["p_true"]), rate(ml_score(c))
    random_rate = policy_metrics(c, 1.0 / c["n_candidates"])["success_rate"]
    assert oracle >= ml - 1e-12
    for s in base.values():
        assert oracle >= rate(s) - 1e-12
    assert ml > random_rate + 0.01
    assert ml > rate(base["cheapest"]) + 0.01
    assert ml > rate(base["best_overall"])


def test_cost_weight_lowers_fee_and_costs_success(world):
    c = world["cands"]
    plain, costly = choose(c, ml_score(c)), choose(c, ml_score(c, 0.2, 0.0))
    a, b = policy_metrics(plain), policy_metrics(costly)
    assert b["effective_fee_pct"] < a["effective_fee_pct"]
    assert b["success_rate"] <= a["success_rate"] + 1e-9


def test_ips_agrees_with_replay_within_noise(world):
    c, t = world["cands"], world["test"]
    pick = choose(c, ml_score(c))
    est, se = ips_success(pick, t[["txn_id", "gateway_id", TARGET]])
    assert abs(est - policy_metrics(pick)["success_rate"]) < 4 * se + 0.01


def test_tune_weights_respects_tolerance(world):
    lc, ll, grid = tune_weights(world["cands"], tolerance=0.001)
    base = grid[(grid["lambda_cost"] == 0) & (grid["lambda_latency"] == 0)].iloc[0]
    chosen = grid[(grid["lambda_cost"] == lc) & (grid["lambda_latency"] == ll)].iloc[0]
    assert chosen["success_rate"] >= base["success_rate"] - 0.001
    assert chosen["cost_latency_index"] <= 1.0 + 1e-9
