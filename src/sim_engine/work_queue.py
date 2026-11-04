"""Bounded asynchronous work admission with independent per-item completion."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass


class QueueBusy(RuntimeError):
    pass


@dataclass
class Work:
    value: object
    future: asyncio.Future
    arrived: float


class AsyncWorkQueue:
    """Workers resolve each future immediately; there is no batch completion barrier."""

    def __init__(self, handler, *, concurrency, capacity, name):
        if concurrency < 1 or capacity < 1:
            raise ValueError("queue concurrency and capacity must be positive")
        self.handler = handler
        self.concurrency = concurrency
        self.queue = asyncio.Queue(maxsize=capacity)
        self.name = name
        self.workers = []
        self.closed = False
        self.active = self.completed = self.failed = self.rejected = 0
        self.peak_active = self.peak_queued = 0
        self.queue_seconds = self.service_seconds = 0.0

    def start(self):
        if not self.workers and not self.closed:
            self.workers = [asyncio.create_task(self._worker(), name=self.name + str(i))
                            for i in range(self.concurrency)]

    async def submit(self, value):
        if self.closed:
            raise RuntimeError(self.name + " queue is closed")
        self.start()
        future = asyncio.get_running_loop().create_future()
        try:
            self.queue.put_nowait(Work(value, future, time.monotonic()))
        except asyncio.QueueFull as error:
            self.rejected += 1
            raise QueueBusy(self.name + " queue is full") from error
        self.peak_queued = max(self.peak_queued, self.queue.qsize())
        # Cancellation of a client must not interrupt an already accepted simulator action.
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            future.add_done_callback(lambda f: f.exception() if not f.cancelled() else None)
            raise

    async def _worker(self):
        while True:
            work = await self.queue.get()
            began = time.monotonic()
            self.queue_seconds += began - work.arrived
            self.active += 1
            self.peak_active = max(self.peak_active, self.active)
            try:
                result = await self.handler(work.value)
            except asyncio.CancelledError:
                work.future.cancel()
                raise
            except Exception as error:
                self.failed += 1
                work.future.set_exception(error)
            else:
                self.completed += 1
                work.future.set_result(result)
            finally:
                self.active -= 1
                self.service_seconds += time.monotonic() - began
                self.queue.task_done()

    def snapshot(self):
        return {"name": self.name, "capacity": self.queue.maxsize,
                "concurrency": self.concurrency, "queued": self.queue.qsize(),
                "active": self.active, "completed": self.completed, "failed": self.failed,
                "rejected": self.rejected, "peak_active": self.peak_active,
                "peak_queued": self.peak_queued, "queue_seconds": self.queue_seconds,
                "service_seconds": self.service_seconds}

    async def close(self):
        self.closed = True
        for task in self.workers:
            task.cancel()
        await asyncio.gather(*self.workers, return_exceptions=True)
        while not self.queue.empty():
            work = self.queue.get_nowait()
            work.future.cancel()
            self.queue.task_done()
