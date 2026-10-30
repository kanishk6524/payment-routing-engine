"""Demo traffic for a hosted instance (DEMO_MODE=1).

A public deployment has no real payments, so this thread replays recorded simulator attempts
(starting two simulated hours before a gateway incident) through the live router, in a loop:
every attempt's result is reported to the circuit breaker, and every few attempts the router
also makes a routing decision, which is stored for the dashboard. The breaker trips on the
recorded incident, probes, and recovers, repeatedly, on real wall-clock time.
It is a replay of simulated data and is labelled as such in the README and dashboard.
"""
import logging
import threading
import time

import pandas as pd

from ml.db import load
from pipeline.stream import pick_start

from .service import NoGatewayAvailable

log = logging.getLogger("router.demo")


class DemoFeed(threading.Thread):
    def __init__(self, router, remember, rate: float = 8.0, incident: str = "gw_d",
                 data_dir: str = "data", decide_every: int = 3):
        super().__init__(daemon=True, name="demo-feed")
        self.router, self.remember = router, remember
        self.rate, self.incident, self.data_dir, self.decide_every = rate, incident, data_dir, decide_every
        self._halt = threading.Event()
        self.sent = 0

    def stop(self) -> None:
        self._halt.set()

    def _rows(self):
        attempts, _, outages = load("csv", self.data_dir)
        stream = pick_start(attempts, outages, self.incident)
        # keep the incident plus an hour of recovery, so the loop (and the breaker cycle) is short
        end = outages[(outages["gateway_id"] == self.incident) & (outages["start"] >= stream["ts"].min())] \
            .sort_values("start").iloc[0]["end"]
        return stream[stream["ts"] <= end + pd.Timedelta(hours=1)].reset_index(drop=True)

    def run(self) -> None:
        try:
            stream = self._rows()
        except BaseException as exc:  # SystemExit from pick_start included
            log.error("demo feed disabled: %s", exc)
            return
        log.info("demo feed: %d attempts, %.1f/s, incident %s", len(stream), self.rate, self.incident)
        while not self._halt.is_set():
            t0, n = time.time(), 0
            for r in stream.itertuples():
                if self._halt.is_set():
                    return
                n += 1
                self.sent += 1
                try:
                    if n % self.decide_every == 0:
                        txn = {"amount_inr": float(r.amount_inr), "method": r.method, "bank": r.bank,
                               "merchant_category": r.merchant_category}
                        try:
                            self.remember(self.router.route(txn), txn, ())
                        except NoGatewayAvailable:
                            pass
                    code = None if pd.isna(r.failure_code) else r.failure_code
                    self.router.record_outcome(r.gateway_id, bool(r.success), int(r.latency_ms), code)
                except Exception:  # a bad row must never kill the feed
                    log.exception("demo feed row failed")
                self._halt.wait(max(0.0, t0 + n / self.rate - time.time()))
