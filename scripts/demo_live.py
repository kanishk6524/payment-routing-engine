"""Watch the router react to a gateway incident, against a running API.

    uvicorn api.main:app --port 8000        (in another terminal)
    python scripts/demo_live.py
"""
import argparse
import random
from collections import Counter

import httpx

BANKS = ["HDFC", "ICICI", "SBI", "AXIS", "KOTAK"]
METHODS = ["upi", "card", "netbanking"]


def traffic(client, n, rng):
    picks = Counter()
    for _ in range(n):
        body = {"amount_inr": round(rng.lognormvariate(6.8, 1.0), 2), "method": rng.choice(METHODS),
                "bank": rng.choice(BANKS), "merchant_category": "retail"}
        picks[client.post("/route", json=body).json()["gateway"]] += 1
    return picks


def show(title, picks):
    total = sum(picks.values())
    print(f"{title}: " + ", ".join(f"{g} {100 * c / total:.0f}%" for g, c in sorted(picks.items())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=150)
    a = ap.parse_args()
    rng = random.Random(7)
    with httpx.Client(base_url=a.url, timeout=30) as c:
        print("model:", c.get("/health").json()["model_version"])
        before = traffic(c, a.n, rng)
        busiest = before.most_common(1)[0][0]
        show("normal conditions      ", before)

        print(f"\nIncident: {busiest} starts failing (60 gateway errors reported)...")
        for _ in range(60):
            c.post("/outcome", json={"gateway_id": busiest, "success": False, "latency_ms": 8000,
                                     "failure_code": "TIMEOUT"})
        show(f"during {busiest} incident   ", traffic(c, a.n, rng))

        print(f"\nRecovery: {busiest} succeeds again (150 good outcomes reported)...")
        for _ in range(150):
            c.post("/outcome", json={"gateway_id": busiest, "success": True, "latency_ms": 300})
        show("after recovery         ", traffic(c, a.n, rng))


if __name__ == "__main__":
    main()
