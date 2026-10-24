"""Watch the router trip a failing gateway, probe it, and let it back in. Needs a running API.

    BREAKER_COOLDOWN_S=5 uvicorn api.main:app --port 8000      (in another terminal)
    python scripts/demo_live.py

Or run everything with one command:  bash scripts/run_demo.sh
"""
import argparse
import random
import time
from collections import Counter

import httpx

BANKS = ["HDFC", "ICICI", "SBI", "AXIS", "KOTAK"]
METHODS = ["upi", "card", "netbanking"]


def txn_body(rng):
    return {"amount_inr": round(rng.lognormvariate(6.8, 1.0), 2), "method": rng.choice(METHODS),
            "bank": rng.choice(BANKS), "merchant_category": "retail"}


def traffic(client, n, rng):
    picks = Counter()
    for _ in range(n):
        picks[client.post("/route", json=txn_body(rng)).json()["gateway"]] += 1
    return picks


def show(title, picks):
    total = sum(picks.values())
    print(f"{title}: " + ", ".join(f"{g} {100 * c / total:.0f}%" for g, c in sorted(picks.items())))


def states(client):
    return ", ".join(f"{g['gateway_id']}={g['breaker']}" for g in client.get("/gateways").json())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=150)
    a = ap.parse_args()
    rng = random.Random(7)
    with httpx.Client(base_url=a.url, timeout=30) as c:
        health = c.get("/health").json()
        print("model:", health["model_version"], "| breaker cooldown:", health["breaker_cooldown_s"], "s")
        before = traffic(c, a.n, rng)
        victim = before.most_common(1)[0][0]
        show("1. normal conditions      ", before)
        print("   breakers:", states(c))

        print(f"\n2. Incident: {victim} starts timing out (60 failures reported)")
        changes = [c.post("/outcome", json={"gateway_id": victim, "success": False, "latency_ms": 8000,
                                            "failure_code": "TIMEOUT"}).json()["breaker_change"]
                   for _ in range(60)]
        print(f"   breaker transitions: {[x for x in changes if x]}")
        show(f"   traffic with {victim} tripped", traffic(c, 60, rng))  # short, so the cooldown has not ended
        print("   breakers:", states(c))

        wait = health["breaker_cooldown_s"] + 0.5
        print(f"\n3. Waiting {wait:.1f} s for the cooldown, then probing {victim} with small transactions")
        time.sleep(wait)
        probes, closed = 0, False
        for _ in range(1500):
            d = c.post("/route", json=txn_body(rng)).json()
            if d["mode"] == "probe":
                probes += 1   # the gateway has recovered, so every probe succeeds
                change = c.post("/outcome", json={"gateway_id": d["gateway"], "success": True,
                                                  "latency_ms": 300}).json()["breaker_change"]
                if change == "closed":
                    closed = True
                    break
        print(f"   {probes} probe transactions sent; breaker {'closed' if closed else 'still not closed'}")
        print("   breakers:", states(c))

        print()
        show("4. after recovery         ", traffic(c, a.n, rng))


if __name__ == "__main__":
    main()
