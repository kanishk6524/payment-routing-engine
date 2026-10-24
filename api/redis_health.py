"""Gateway health kept in Redis (Stage 8).

Same interface as `HealthTracker`, but the recent outcomes live in Redis sorted sets, one per
gateway (`router:events:<gateway>`, score = attempt start in epoch ms). Health therefore survives
an API restart and can be read by other processes such as the dashboard. The manual kill
switch is a Redis set. Circuit-breaker state stays inside the API process.
"""
import json
import os
import uuid

import pandas as pd

from .health import EVENT_COLUMNS, RETAIN_MINUTES, HealthTracker, summarise, utc_now

RETAIN_MS = RETAIN_MINUTES * 60 * 1000
KEY_TTL_S = 3 * 3600


def to_ms(ts) -> int:
    return int(pd.Timestamp(ts).value // 1_000_000)


def empty_events() -> pd.DataFrame:
    return pd.DataFrame({
        "gateway_id": pd.Series(dtype=object), "ts": pd.Series(dtype="datetime64[ns]"),
        "latency_ms": pd.Series(dtype="int64"), "success": pd.Series(dtype=bool),
        "failure_code": pd.Series(dtype=object)})


class RedisSet:
    """The slice of the `set` interface the router uses, backed by a Redis set."""

    def __init__(self, client, key: str):
        self._r, self._key = client, key

    def add(self, item):
        self._r.sadd(self._key, item)

    def discard(self, item):
        self._r.srem(self._key, item)

    def update(self, items):
        items = list(items)
        if items:
            self._r.sadd(self._key, *items)

    def __contains__(self, item):
        return bool(self._r.sismember(self._key, item))

    def __iter__(self):
        return iter(self._r.smembers(self._key))

    def __len__(self):
        return int(self._r.scard(self._key))

    def __bool__(self):
        return len(self) > 0


class RedisHealthTracker:
    def __init__(self, client, gateway_ids, prefix: str = "router:"):
        self._r, self._gateways, self._prefix = client, list(gateway_ids), prefix
        self.disabled = RedisSet(client, f"{prefix}disabled")

    def _key(self, gw: str) -> str:
        return f"{self._prefix}events:{gw}"

    def clear(self) -> None:
        self._r.delete(*[self._key(g) for g in self._gateways], f"{self._prefix}disabled")

    def _put(self, gw, members: dict) -> None:
        if not members:
            return
        key = self._key(gw)
        self._r.zadd(key, members)
        newest = max(members.values())
        self._r.zremrangebyscore(key, "-inf", newest - RETAIN_MS)
        self._r.expire(key, KEY_TTL_S)

    @staticmethod
    def _member(success, latency_ms, failure_code) -> str:
        return json.dumps({"id": uuid.uuid4().hex[:10], "lat": int(latency_ms), "ok": bool(success),
                           "code": failure_code})

    def add(self, gateway_id, success, latency_ms, failure_code=None, ts=None):
        ts = pd.Timestamp(ts) if ts is not None else utc_now() - pd.Timedelta(milliseconds=latency_ms)
        self._put(gateway_id, {self._member(success, latency_ms, failure_code): to_ms(ts)})

    def seed(self, history: pd.DataFrame, end: pd.Timestamp = None) -> int:
        """Load history shifted so the newest event ends at `end` (default now). Demo mode only."""
        end = end or utc_now()
        h = history[EVENT_COLUMNS].copy()
        h["ts"] = h["ts"] + (end - h["ts"].max())
        h = h[h["ts"] >= end - pd.Timedelta(minutes=RETAIN_MINUTES)]
        self.clear()
        for gw, g in h.groupby("gateway_id"):
            self._put(gw, {self._member(r.success, r.latency_ms, r.failure_code): to_ms(r.ts)
                           for r in g.itertuples()})
        return len(h)

    def forget(self, gateway_id: str, start, end) -> int:
        return int(self._r.zremrangebyscore(self._key(gateway_id), to_ms(start), to_ms(end)))

    def snapshot(self) -> pd.DataFrame:
        rows = []
        for gw in self._gateways:
            for member, score in self._r.zrangebyscore(self._key(gw), "-inf", "+inf", withscores=True):
                m = json.loads(member)
                rows.append((gw, int(score), m["lat"], m["ok"], m["code"]))
        if not rows:
            return empty_events()
        df = pd.DataFrame(rows, columns=["gateway_id", "ms", "latency_ms", "success", "failure_code"])
        df["ts"] = pd.to_datetime(df.pop("ms"), unit="ms").astype("datetime64[ns]")
        codes = df["failure_code"].astype(object)
        df["failure_code"] = codes.where(codes.notna(), None)
        return df[EVENT_COLUMNS].sort_values("ts", kind="stable").reset_index(drop=True)

    def is_empty(self) -> bool:
        floor = to_ms(utc_now()) - RETAIN_MS
        return all(int(self._r.zcount(self._key(g), floor, "+inf")) == 0 for g in self._gateways)

    def summary(self, now=None, minutes: int = 15) -> dict:
        return summarise(self.snapshot(), now or utc_now(), minutes)


def connect_redis():
    import redis

    client = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=int(os.getenv("REDIS_PORT", "6379")),
                         decode_responses=True, socket_connect_timeout=2, socket_timeout=5)
    client.ping()
    return client


def make_tracker(gateway_ids, backend: str = None):
    """Redis when reachable (or demanded with ROUTER_HEALTH=redis), otherwise in memory."""
    backend = backend or os.getenv("ROUTER_HEALTH", "auto")
    if backend != "memory":
        try:
            return RedisHealthTracker(connect_redis(), gateway_ids, os.getenv("REDIS_PREFIX", "router:")), "redis"
        except Exception:
            if backend == "redis":
                raise
    return HealthTracker(), "memory"
