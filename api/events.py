"""Event messages exchanged over Kafka, and how the router reacts to them (Stage 8)."""
import logging

import pandas as pd

log = logging.getLogger("router.events")

OUTCOMES_TOPIC = "payment.outcomes"
DECISIONS_TOPIC = "payment.decisions"


def outcome_event(gateway_id, success, latency_ms, failure_code=None, origin="api", ts=None) -> dict:
    return {"gateway_id": gateway_id, "success": bool(success), "latency_ms": int(latency_ms),
            "failure_code": failure_code, "origin": origin, "ts": ts}


def decision_event(decision: dict, txn: dict, exclude=(), ts=None) -> dict:
    """What the sink stores for one routing decision."""
    return {
        "txn_id": decision["txn_id"], "attempt": len(exclude) + 1,
        "decided_at": str(pd.Timestamp(ts) if ts is not None else pd.Timestamp.now("UTC").tz_localize(None)),
        "amount_inr": float(txn["amount_inr"]), "method": txn["method"],
        "bank": txn.get("bank") or "OTHER", "merchant_category": txn.get("merchant_category") or "other",
        "chosen_gateway": decision["gateway"], "mode": decision["mode"],
        "model_version": decision["model_version"],
        "scores": {a["gateway"]: a["score"] for a in decision["alternatives"]},
        "explanation": {"summary": decision["summary"], "drivers": decision["drivers"],
                        "predicted_success": decision["predicted_success"],
                        "tripped_gateways": decision["tripped_gateways"]},
    }


def apply_outcome_event(router, event: dict, own_origin: str = None):
    """Apply an outcome that arrived over Kafka. Returns the breaker transition, or None.

    Messages this instance published itself are skipped (it applied them directly), and
    malformed or unknown-gateway messages are logged and dropped, never raised.
    """
    try:
        if own_origin is not None and event.get("origin") == own_origin:
            return None
        return router.record_outcome(
            event["gateway_id"], bool(event["success"]), int(event["latency_ms"]),
            event.get("failure_code"), event.get("ts"))
    except KeyError as exc:
        log.warning("dropping outcome event (missing or unknown field %s): %s", exc, event)
    except (TypeError, ValueError) as exc:
        log.warning("dropping malformed outcome event (%s): %s", exc, event)
    return None
