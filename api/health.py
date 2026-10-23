"""Rolling record of recent gateway outcomes (the serving-time stand-in for Redis)."""
import threading

import pandas as pd

EVENT_COLUMNS = ["gateway_id", "ts", "latency_ms", "success", "failure_code"]
RETAIN_MINUTES = 75  # longest feature window is 60 minutes


def utc_now() -> pd.Timestamp:
    return pd.Timestamp.now("UTC").tz_localize(None)


class HealthTracker:
    """Holds the last ~75 minutes of attempt outcomes and a manual kill switch per gateway."""

    def __init__(self):
        self._lock = threading.Lock()
        self._events = pd.DataFrame({c: pd.Series(dtype=object) for c in EVENT_COLUMNS})
        self.disabled = set()

    def seed(self, history: pd.DataFrame, end: pd.Timestamp = None) -> int:
        """Load historical attempts, shifted in time so the newest ends at `end` (default now).

        Demo mode: the bundled data is from the past, so it is moved to the present to give the
        router recent health to read. Real deployments feed live outcomes through `add`.
        """
        end = end or utc_now()
        h = history[EVENT_COLUMNS].copy()
        h["ts"] = h["ts"] + (end - h["ts"].max())
        h = h[h["ts"] >= end - pd.Timedelta(minutes=RETAIN_MINUTES)]
        with self._lock:
            self._events = h.reset_index(drop=True)
        return len(h)

    def add(self, gateway_id, success, latency_ms, failure_code=None, ts=None):
        ts = pd.Timestamp(ts) if ts is not None else utc_now() - pd.Timedelta(milliseconds=latency_ms)
        row = pd.DataFrame([{"gateway_id": gateway_id, "ts": ts, "latency_ms": int(latency_ms),
                             "success": bool(success), "failure_code": failure_code}])
        with self._lock:
            ev = pd.concat([self._events, row], ignore_index=True)
            self._events = ev[ev["ts"] >= ev["ts"].max() - pd.Timedelta(minutes=RETAIN_MINUTES)]

    def snapshot(self) -> pd.DataFrame:
        with self._lock:
            return self._events.copy()

    def summary(self, now=None, minutes: int = 15) -> dict:
        """Success rate and volume per gateway over the last `minutes`."""
        now = now or utc_now()
        ev = self.snapshot()
        ev = ev[(ev["ts"] >= now - pd.Timedelta(minutes=minutes)) & (ev["ts"] <= now)]
        out = {}
        for gw, g in ev.groupby("gateway_id"):
            out[gw] = {"attempts": int(len(g)), "success_rate": round(float(g["success"].astype(float).mean()), 4),
                       "avg_latency_ms": round(float(g["latency_ms"].astype(float).mean()))}
        return out
