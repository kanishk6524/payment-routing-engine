"""Per-gateway circuit breaker (Stage 9).

States
------
CLOSED     normal. Gateway-side failures in a sliding window are counted; too many trips it.
OPEN       the gateway gets no traffic until the cooldown ends.
HALF_OPEN  the gateway gets a small share of low-value probe transactions. A run of successes
           closes the breaker; a single failure reopens it and restarts the cooldown.

Only gateway-side failures (timeouts, gateway errors) count. Bank declines are the customer's
problem, not the gateway's. Every method takes the current time so tests control the clock.
"""
import os
import random
import threading
from collections import deque
from dataclasses import dataclass, field

import pandas as pd

from ml.features import GATEWAY_SIDE_FAILURES

CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"


@dataclass
class BreakerConfig:
    window_s: float = 120.0          # look-back window for failures
    min_attempts: int = 20           # fewer attempts than this never trip the breaker
    failure_rate: float = 0.5        # gateway-side failure share that trips it
    cooldown_s: float = 30.0         # time spent OPEN before probing starts
    probe_share: float = 0.10        # share of eligible transactions sent as probes
    probe_max_amount: float = 2_000  # only low-value transactions are used as probes
    probe_successes: int = 5         # consecutive probe successes needed to close

    @classmethod
    def from_env(cls) -> "BreakerConfig":
        c = cls()
        for name in c.__dataclass_fields__:
            raw = os.getenv(f"BREAKER_{name.upper()}")
            if raw is not None:
                setattr(c, name, type(getattr(c, name))(raw))
        return c


def is_gateway_failure(success: bool, failure_code) -> bool:
    """A failed attempt counts against the gateway unless it carries a non-gateway reason."""
    if success:
        return False
    return failure_code is None or failure_code in GATEWAY_SIDE_FAILURES


@dataclass
class _State:
    state: str = CLOSED
    recent: deque = field(default_factory=deque)   # (ts, is_failure)
    opened_at: pd.Timestamp = None
    trip_start: pd.Timestamp = None
    half_open_at: pd.Timestamp = None
    probe_ok: int = 0


class CircuitBreakers:
    def __init__(self, config: BreakerConfig = None, seed: int = None):
        self.cfg = config or BreakerConfig()
        self._s = {}
        self._lock = threading.Lock()
        self._rng = random.Random(seed)

    def _get(self, gw) -> _State:
        return self._s.setdefault(gw, _State())

    def _refresh(self, st: _State, now: pd.Timestamp) -> None:
        if st.state == OPEN and (now - st.opened_at).total_seconds() >= self.cfg.cooldown_s:
            st.state, st.half_open_at, st.probe_ok = HALF_OPEN, now, 0

    def state(self, gw: str, now: pd.Timestamp) -> str:
        with self._lock:
            st = self._get(gw)
            self._refresh(st, now)
            return st.state

    def should_probe(self, gw: str, now: pd.Timestamp, amount: float) -> bool:
        if self.state(gw, now) != HALF_OPEN or amount > self.cfg.probe_max_amount:
            return False
        return self._rng.random() < self.cfg.probe_share

    def observe(self, gw: str, success: bool, failure_code, now: pd.Timestamp):
        """Record an outcome. Returns "opened", "reopened", "closed" or None."""
        bad = is_gateway_failure(success, failure_code)
        with self._lock:
            st = self._get(gw)
            self._refresh(st, now)
            if st.state == CLOSED:
                st.recent.append((now, bad))
                while st.recent and (now - st.recent[0][0]).total_seconds() > self.cfg.window_s:
                    st.recent.popleft()
                n = len(st.recent)
                if n >= self.cfg.min_attempts and sum(b for _, b in st.recent) / n >= self.cfg.failure_rate:
                    st.state, st.opened_at = OPEN, now
                    st.trip_start = now - pd.Timedelta(seconds=self.cfg.window_s)
                    st.recent.clear()
                    return "opened"
            elif st.state == HALF_OPEN:
                if now < st.half_open_at:  # an old in-flight result, not a probe
                    return None
                if bad:
                    st.state, st.opened_at, st.probe_ok = OPEN, now, 0
                    return "reopened"
                st.probe_ok += 1
                if st.probe_ok >= self.cfg.probe_successes:
                    st.state = CLOSED
                    return "closed"
        return None

    def incident_window(self, gw: str):
        """(start, end) of the last incident: from the start of the failure window to the first probe."""
        st = self._get(gw)
        return st.trip_start, st.half_open_at

    def snapshot(self, now: pd.Timestamp) -> dict:
        out = {}
        with self._lock:
            for gw, st in self._s.items():
                self._refresh(st, now)
                out[gw] = {"state": st.state, "probe_successes": st.probe_ok if st.state == HALF_OPEN else 0}
        return out
