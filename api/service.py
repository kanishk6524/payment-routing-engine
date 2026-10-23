"""The routing decision: score every supported gateway and pick the best (Stage 7 core).

Plain Python so it can be tested without a web server. `api/main.py` exposes it over HTTP.
"""
import json
import time
import uuid
from pathlib import Path

import pandas as pd

from ml.db import load
from ml.explain import contributions, top_drivers
from ml.features import FEATURE_COLUMNS
from ml.model import ARTIFACT_PATH, encode, load_bundle
from ml.replay import TXN_COLUMNS, add_model_scores, build_candidates, ml_score

from .health import HealthTracker, utc_now

DEFAULT_LAMBDAS = (0.01, 0.05)


def read_lambdas(results_path: str = "reports/results.json") -> tuple:
    """Cost and latency weights tuned in Stage 5, else sensible defaults."""
    try:
        r = json.loads(Path(results_path).read_text())
        return float(r["lambda_cost"]), float(r["lambda_latency"])
    except Exception:
        return DEFAULT_LAMBDAS


class NoGatewayAvailable(Exception):
    pass


class Router:
    def __init__(self, bundle: dict, gateways: pd.DataFrame, tracker: HealthTracker,
                 lambdas: tuple = DEFAULT_LAMBDAS):
        self.bundle, self.gateways, self.tracker = bundle, gateways, tracker
        self.lam_cost, self.lam_latency = lambdas

    @classmethod
    def from_artifacts(cls, source: str = "db", data_dir: str = "data",
                       model_path: str = ARTIFACT_PATH, seed_demo: bool = True):
        bundle = load_bundle(model_path)
        try:
            attempts, gateways, _ = load(source, data_dir)
        except Exception:  # database not reachable: fall back to the simulator CSVs
            attempts, gateways, _ = load("csv", data_dir)
        tracker = HealthTracker()
        if seed_demo:
            tracker.seed(attempts)
        return cls(bundle, gateways, tracker, read_lambdas())

    @property
    def gateway_ids(self) -> list:
        return sorted(self.gateways["gateway_id"])

    def route(self, txn: dict, now=None) -> dict:
        started = time.perf_counter()
        ts = pd.Timestamp(now) if now is not None else utc_now()
        row = pd.DataFrame([{
            "txn_id": txn.get("txn_id") or uuid.uuid4().hex[:12], "ts": ts,
            "amount_inr": float(txn["amount_inr"]), "method": txn["method"],
            "bank": txn.get("bank") or "OTHER", "merchant_category": txn.get("merchant_category") or "other",
        }])
        cands = build_candidates(row[TXN_COLUMNS], self.gateways, self.tracker.snapshot())
        if self.tracker.disabled:
            cands = cands[~cands["gateway_id"].isin(self.tracker.disabled)]
        if cands.empty:
            raise NoGatewayAvailable(f"no enabled gateway supports {txn['method']}")

        cands = add_model_scores(cands, self.bundle)
        cands["score"] = ml_score(cands, self.lam_cost, self.lam_latency)
        ranked = cands.sort_values("score", ascending=False).reset_index()
        best, runner = ranked.iloc[0], (ranked.iloc[1] if len(ranked) > 1 else None)

        X = encode(cands, self.bundle["levels"])
        contrib = contributions(self.bundle["model"], X)
        drivers = []
        if contrib is not None:
            pos = ranked.loc[0, "index"]
            drivers = top_drivers(contrib.loc[pos], X.loc[pos])

        why = (f"{best['gateway_id']}: predicted success {100 * best['p_pred']:.1f}%, "
               f"fee INR {best['fee_inr']:.2f}, expected latency {best['lat_est']:.0f} ms")
        if runner is not None:
            why += (f". Next best {runner['gateway_id']} at {100 * runner['p_pred']:.1f}% "
                    f"(score gap {best['score'] - runner['score']:.3f})")
        return {
            "txn_id": best["txn_id"], "gateway": best["gateway_id"],
            "predicted_success": round(float(best["p_pred"]), 4),
            "fee_inr": float(best["fee_inr"]), "expected_latency_ms": round(float(best["lat_est"])),
            "summary": why, "drivers": drivers,
            "alternatives": [{
                "gateway": r["gateway_id"], "predicted_success": round(float(r["p_pred"]), 4),
                "fee_inr": float(r["fee_inr"]), "expected_latency_ms": round(float(r["lat_est"])),
                "score": round(float(r["score"]), 4),
            } for _, r in ranked.iterrows()],
            "model_version": self.bundle["version"],
            "decision_ms": round(1000 * (time.perf_counter() - started), 1),
        }

    def record_outcome(self, gateway_id: str, success: bool, latency_ms: int,
                       failure_code: str = None, ts=None) -> None:
        if gateway_id not in set(self.gateways["gateway_id"]):
            raise KeyError(gateway_id)
        self.tracker.add(gateway_id, success, latency_ms, failure_code, ts)
