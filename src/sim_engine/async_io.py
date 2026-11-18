"""Bounded disk work without blocking the service loop or abandoning evidence writes."""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor


class AsyncDisk:
    def __init__(self, workers=4):
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="evidence-io")
        self.slots = asyncio.Semaphore(workers * 4)
        self.pending = {}
        self.completed = 0
        self.max_wall_s = 0.0

    async def run(self, function, *args):
        async with self.slots:
            return await self._run(function, *args)

    async def _run(self, function, *args):
        began = time.monotonic()
        future = asyncio.get_running_loop().run_in_executor(self.executor, function, *args)
        self.pending[future] = (began, getattr(function, "__name__", type(function).__name__))
        try:
            try:
                return await asyncio.shield(future)
            except asyncio.CancelledError:
                # Do not release a claim/transition lock while its write is still running.
                # Shutdown must never seal a receipt ahead of an abandoned disk operation.
                while not future.done():
                    try:
                        await asyncio.shield(future)
                    except asyncio.CancelledError:
                        continue
                if not future.cancelled():
                    future.exception()  # Retrieve any failure before propagating cancellation.
                raise
        finally:
            self.pending.pop(future, None)
            self.completed += 1
            self.max_wall_s = max(self.max_wall_s, time.monotonic() - began)

    def health(self):
        oldest = min(self.pending.values(), default=None)
        return {
            "pending": len(self.pending), "completed": self.completed,
            "oldest_operation": None if oldest is None else oldest[1],
            "oldest_wall_s": 0.0 if oldest is None else time.monotonic() - oldest[0],
            "max_wall_s": self.max_wall_s,
        }

    async def close(self):
        await asyncio.to_thread(self.executor.shutdown, wait=True, cancel_futures=False)
