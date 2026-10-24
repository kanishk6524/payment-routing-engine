"""Replay recorded gateway outcomes into Kafka as a live stream.

    python -m pipeline.stream --incident gw_d            # replay around gw_d's incident
    python -m pipeline.stream --rate 100 --seconds 60

Takes attempts from the test period of data/*.csv in time order and publishes each result to
`payment.outcomes`. With --incident the replay starts two hours (simulated) before that
gateway's outage, so the API's circuit breaker trips on real incident traffic.
"""
import argparse
import time

import pandas as pd

from api.events import OUTCOMES_TOPIC, outcome_event
from ml.db import load
from .bus import KafkaBus


def pick_start(attempts: pd.DataFrame, outages: pd.DataFrame, incident: str = None):
    test = attempts[attempts["ts"] >= attempts["ts"].quantile(0.8)]
    if incident:
        o = outages[outages["gateway_id"] == incident].sort_values("start")
        o = o[o["start"] >= test["ts"].min()]
        if o.empty:
            raise SystemExit(f"no incident for {incident} in the test period; try one of: "
                             f"{sorted(outages['gateway_id'].unique())}")
        test = test[test["ts"] >= o.iloc[0]["start"] - pd.Timedelta(hours=2)]
    return test.sort_values("ts").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", choices=["db", "csv"], default="csv")
    ap.add_argument("--data", default="data")
    ap.add_argument("--rate", type=float, default=50, help="events per second")
    ap.add_argument("--seconds", type=float, default=90)
    ap.add_argument("--incident", help="gateway whose incident to replay, for example gw_d")
    a = ap.parse_args()

    attempts, _, outages = load(a.source, a.data)
    stream = pick_start(attempts, outages, a.incident)
    bus = KafkaBus()
    print(f"replaying {min(len(stream), int(a.rate * a.seconds)):,} outcomes at {a.rate:g}/s "
          f"from simulated time {stream['ts'].iloc[0]}")
    sent, t0 = 0, time.time()
    for r in stream.itertuples():
        if time.time() - t0 > a.seconds:
            break
        bus.publish(OUTCOMES_TOPIC, r.gateway_id, outcome_event(
            r.gateway_id, r.success, r.latency_ms,
            None if pd.isna(r.failure_code) else r.failure_code, origin="stream"))
        sent += 1
        if sent % int(max(a.rate, 1)) == 0:
            print(f"  {sent:,} sent; simulated time {r.ts}", flush=True)
        time.sleep(max(0.0, t0 + sent / a.rate - time.time()))
    bus.flush()
    print(f"done: {sent:,} outcomes published")


if __name__ == "__main__":
    main()
