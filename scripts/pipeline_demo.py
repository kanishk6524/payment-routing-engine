"""End-to-end Stage 8 demo: Kafka stream -> API -> Redis health -> breaker -> Kafka -> PostgreSQL.

    python scripts/pipeline_demo.py                 # needs: docker compose up -d, data loaded, model trained

It starts the API, the decision sink and a replay of recorded outcomes around gw_d's incident,
then shows how the circuit breaker reacted and what reached the database.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
URL = "http://localhost:8000"


def preflight():
    from scripts.check_services import check_kafka, check_postgres, check_redis

    for name, fn in (("PostgreSQL", check_postgres), ("Redis", check_redis), ("Kafka", check_kafka)):
        try:
            print(f"  {name}: {fn()}")
        except Exception as exc:
            raise SystemExit(f"{name} is not reachable ({exc}). Run: docker compose up -d")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--incident", default="gw_d")
    ap.add_argument("--seconds", type=int, default=40)
    ap.add_argument("--rate", type=int, default=80)
    a = ap.parse_args()

    print("Checking services")
    preflight()
    env = {**os.environ, "BREAKER_COOLDOWN_S": "5", "ROUTER_HEALTH": "redis", "EVENT_BUS": "kafka",
           "REDIS_PREFIX": "demo:"}
    procs = []

    def spawn(*cmd):
        p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        procs.append(p)
        return p

    try:
        api = spawn(sys.executable, "-m", "uvicorn", "api.main:app", "--port", "8000", "--log-level", "warning")
        spawn(sys.executable, "-m", "pipeline.decision_sink")
        with httpx.Client(base_url=URL, timeout=20) as c:
            for _ in range(120):
                try:
                    health = c.get("/health").json()
                    break
                except httpx.HTTPError:
                    if api.poll() is not None:
                        raise SystemExit("the API exited early:\n" + api.stderr.read().decode()[-1500:])
                    time.sleep(0.5)
            else:
                raise SystemExit("the API did not start")
            print(f"\nAPI up | health store: {health['health_store']} | event bus: {health['event_bus']}"
                  f" | breaker cooldown: {health['breaker_cooldown_s']} s")
            if health["event_bus"] != "kafka" or health["health_store"] != "redis":
                raise SystemExit("expected Redis and Kafka to be in use; check docker compose ps")
            time.sleep(3)  # let the API's consumer join its Kafka group before the stream starts

            print(f"\nReplaying recorded outcomes around {a.incident}'s incident for {a.seconds} s "
                  f"({a.rate}/s through Kafka)")
            spawn(sys.executable, "-m", "pipeline.stream", "--incident", a.incident,
                  "--rate", str(a.rate), "--seconds", str(a.seconds))
            t0, last, routed = time.time(), None, 0
            while time.time() - t0 < a.seconds + 4:
                gws = {g["gateway_id"]: g for g in c.get("/gateways").json()}
                line = " ".join(f"{k}={v['breaker']}" for k, v in sorted(gws.items()))
                if line != last:
                    print(f"  t={time.time() - t0:4.0f}s  {line}")
                    last = line
                for _ in range(3):
                    c.post("/route", json={"amount_inr": 800 + routed % 500, "method": ["upi", "card", "netbanking"][routed % 3],
                                           "bank": "HDFC", "merchant_category": "retail"})
                    routed += 1
                time.sleep(0.5)
            print(f"\n{routed} routing requests were made while the stream ran")
            time.sleep(4)  # sink flush

        from ml.db import _connect

        conn = _connect()
        with conn.cursor() as cur:
            cur.execute("SELECT count(*), count(DISTINCT chosen_gateway) FROM routing_decisions "
                        "WHERE model_version IS NOT NULL")
            n, g = cur.fetchone()
            cur.execute("SELECT chosen_gateway, policy, explanation->>'summary' FROM routing_decisions "
                        "WHERE model_version IS NOT NULL ORDER BY decision_id DESC LIMIT 3")
            rows = cur.fetchall()
        conn.close()
        print(f"PostgreSQL now holds {n} live routing decisions across {g} gateways. Latest:")
        for gw, pol, summary in rows:
            print(f"  [{pol}] {summary}")
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(5)
            except subprocess.TimeoutExpired:
                p.kill()


if __name__ == "__main__":
    main()
