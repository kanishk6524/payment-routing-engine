"""A small in-memory Redis with the commands the router uses (strings in, strings out)."""


def _bound(v, default):
    if v in ("-inf", "+inf", "inf"):
        return float(v.replace("+", ""))
    return float(v)


class FakeRedis:
    def __init__(self):
        self.z, self.sets = {}, {}

    def ping(self):
        return True

    def delete(self, *keys):
        n = 0
        for k in keys:
            n += (self.z.pop(k, None) is not None) + (self.sets.pop(k, None) is not None)
        return n

    def expire(self, key, seconds):
        return True

    # sorted sets
    def zadd(self, key, mapping):
        d = self.z.setdefault(key, {})
        added = sum(m not in d for m in mapping)
        d.update({m: float(s) for m, s in mapping.items()})
        return added

    def _range(self, key, lo, hi):
        lo, hi = _bound(lo, 0), _bound(hi, 0)
        return sorted(((m, s) for m, s in self.z.get(key, {}).items() if lo <= s <= hi),
                      key=lambda x: (x[1], x[0]))

    def zrangebyscore(self, key, lo, hi, withscores=False):
        rows = self._range(key, lo, hi)
        return rows if withscores else [m for m, _ in rows]

    def zcount(self, key, lo, hi):
        return len(self._range(key, lo, hi))

    def zcard(self, key):
        return len(self.z.get(key, {}))

    def zremrangebyscore(self, key, lo, hi):
        gone = [m for m, _ in self._range(key, lo, hi)]
        for m in gone:
            del self.z[key][m]
        return len(gone)

    # sets
    def sadd(self, key, *items):
        s = self.sets.setdefault(key, set())
        n = len(s)
        s.update(items)
        return len(s) - n

    def srem(self, key, *items):
        s = self.sets.get(key, set())
        n = len(s)
        s.difference_update(items)
        return n - len(s)

    def sismember(self, key, item):
        return item in self.sets.get(key, set())

    def smembers(self, key):
        return set(self.sets.get(key, set()))

    def scard(self, key):
        return len(self.sets.get(key, set()))
