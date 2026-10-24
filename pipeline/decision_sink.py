"""Store routing decisions from Kafka in PostgreSQL (`routing_decisions`).

    python -m pipeline.decision_sink            # runs until Ctrl+C

Each decision also creates its `transactions` row (live transactions are not in the table yet),
and the lookup rows (payment method, bank, merchant category) if they are new.
"""
import json
import logging
import os
import signal
from datetime import datetime

from api.events import DECISIONS_TOPIC

log = logging.getLogger("router.sink")

INSERT_LOOKUPS = (
    "INSERT INTO payment_methods (method) VALUES (%s) ON CONFLICT DO NOTHING",
    "INSERT INTO banks (bank) VALUES (%s) ON CONFLICT DO NOTHING",
    "INSERT INTO merchant_categories (category) VALUES (%s) ON CONFLICT DO NOTHING",
)
INSERT_TXN = """
INSERT INTO transactions (txn_id, created_at, amount_inr, method, bank, merchant_category)
VALUES (%(txn_id)s, %(decided_at)s, %(amount_inr)s, %(method)s, %(bank)s, %(merchant_category)s)
ON CONFLICT (txn_id) DO NOTHING
"""
INSERT_DECISION = """
INSERT INTO routing_decisions (txn_id, attempt, decided_at, policy, model_version, chosen_gateway,
                               scores, explanation)
VALUES (%(txn_id)s, %(attempt)s, %(decided_at)s, %(policy)s, %(model_version)s, %(chosen_gateway)s,
        %(scores)s::jsonb, %(explanation)s::jsonb)
ON CONFLICT (txn_id, attempt, policy) DO NOTHING
"""


def to_rows(event: dict) -> dict:
    """Validate a decision event and shape it for the INSERT statements. Raises on bad input."""
    required = ("txn_id", "decided_at", "amount_inr", "method", "bank", "merchant_category",
                "chosen_gateway", "scores", "explanation")
    missing = [k for k in required if k not in event]
    if missing:
        raise ValueError(f"decision event is missing {missing}")
    if float(event["amount_inr"]) <= 0:
        raise ValueError("amount_inr must be positive")
    return {
        "txn_id": str(event["txn_id"]), "attempt": int(event.get("attempt", 1)),
        "decided_at": datetime.fromisoformat(str(event["decided_at"])),
        "amount_inr": float(event["amount_inr"]), "method": event["method"], "bank": event["bank"],
        "merchant_category": event["merchant_category"],
        "policy": "xgboost" + ("" if event.get("mode", "normal") == "normal" else f"-{event['mode']}"),
        "model_version": event.get("model_version"), "chosen_gateway": event["chosen_gateway"],
        "scores": json.dumps(event["scores"]), "explanation": json.dumps(event["explanation"]),
    }


def write_decisions(conn, events: list) -> int:
    """Insert decisions in one transaction. Duplicates are ignored, so replays are safe."""
    rows = []
    for e in events:
        try:
            rows.append(to_rows(e))
        except (ValueError, TypeError, KeyError) as exc:
            log.warning("skipping bad decision event (%s): %s", exc, e)
    with conn:
        with conn.cursor() as cur:
            for r in rows:
                cur.execute(INSERT_LOOKUPS[0], (r["method"],))
                cur.execute(INSERT_LOOKUPS[1], (r["bank"],))
                cur.execute(INSERT_LOOKUPS[2], (r["merchant_category"],))
                cur.execute(INSERT_TXN, r)
                cur.execute(INSERT_DECISION, r)
    return len(rows)


def main() -> None:
    from confluent_kafka import Consumer

    from ml.db import _connect
    from .bus import bootstrap_servers

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    conn = _connect()
    c = Consumer({"bootstrap.servers": bootstrap_servers(), "group.id": "decision-sink", "log_level": 0,
                  "auto.offset.reset": "earliest", "enable.auto.commit": False})
    c.subscribe([DECISIONS_TOPIC])
    stop = {"now": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))
    log.info("decision sink running (topic %s)", DECISIONS_TOPIC)
    total = 0
    try:
        while not stop["now"]:
            msgs = [m for m in c.consume(num_messages=200, timeout=1.0) if m.error() is None]
            if not msgs:
                continue
            events = []
            for m in msgs:
                try:
                    events.append(json.loads(m.value()))
                except ValueError:
                    log.warning("skipping a message that is not JSON")
            total += write_decisions(conn, events)
            c.commit(asynchronous=False)
            log.info("stored %d decisions (%d total)", len(events), total)
    finally:
        c.close()
        conn.close()


if __name__ == "__main__":
    main()
