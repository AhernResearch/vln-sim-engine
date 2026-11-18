"""Bounded exact memoization for immutable geometry during one planning task."""

from collections import OrderedDict

MISSING = object()


class BoundedMemo:
    def __init__(self, capacity=65536):
        if type(capacity) is not int or capacity < 0:
            raise ValueError("cache capacity must be a nonnegative integer")
        self.capacity = capacity
        self.values = OrderedDict()
        self.hits = self.misses = self.evictions = 0

    def get(self, key):
        if key not in self.values:
            self.misses += 1
            return MISSING
        self.hits += 1
        self.values.move_to_end(key)
        return self.values[key]

    def put(self, key, value):
        if self.capacity == 0:
            return
        self.values[key] = value
        self.values.move_to_end(key)
        if len(self.values) > self.capacity:
            self.values.popitem(last=False)
            self.evictions += 1

    def clear(self):
        self.values.clear()
        self.hits = self.misses = self.evictions = 0

    def stats(self):
        return {
            "entries": len(self.values),
            "capacity": self.capacity,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
        }
