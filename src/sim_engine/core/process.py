"""Spawned child processes behind one serialized request/reply pipe.

The parent sends ``(key, payload)``; the child replies ``(key, status, result)``. Each
exchange runs on the worker's single RPC thread and is shielded from caller
cancellation, so a reply is always consumed by the request that produced it. A
timeout, interruption or lost correspondence terminates the child: partially
executed work is never replayed.
"""

from __future__ import annotations

import asyncio
import functools
import multiprocessing as mp
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Protocol

SPAWN = mp.get_context("spawn")
READY = "ready"
READY_TIMEOUT_S = 60.0
POLL_S = 0.05


class WorkerError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class WorkerTimeout(WorkerError):
    def __init__(self, message: str):
        super().__init__(503, message)


class CancellationFlag:
    """A byte in shared memory survives a hard kill without an abandoned lock."""

    def __init__(self):
        self.value = SPAWN.RawValue("b", 0)

    def set(self):
        self.value.value = 1

    def clear(self):
        self.value.value = 0

    def is_set(self) -> bool:
        return bool(self.value.value)


class Host(Protocol):
    """Process-local state owned by one child."""

    def handle(self, payload) -> tuple[int, Any]: ...
    def close(self) -> None: ...


def serve(connection, host_factory: Callable[..., Host], *args):
    """Child request loop. ``None`` ends it; the host is always closed."""
    host = None
    try:
        host = host_factory(*args)
        connection.send(READY)
        while (message := connection.recv()) is not None:
            key, payload = message
            connection.send((key, *host.handle(payload)))
    except EOFError:
        pass  # The parent closed its end; shut down quietly.
    finally:
        if host is not None:
            host.close()
        connection.close()


class ProcessWorker:
    """Parent-side handle of one spawned child running ``host_factory(stop, *args)``."""

    kind = "worker"

    def __init__(self, host_factory: Callable[..., Host], *args):
        self.host_factory, self.host_args = host_factory, args
        self.stop = CancellationFlag()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"{self.kind}-rpc")
        self.generation = 0
        self.spawn()

    def spawn(self):
        """Start a fresh child; an earlier generation must already have exited."""
        if self.generation:
            self.connection.close()
        self.connection, child = SPAWN.Pipe()
        self.instance = uuid.uuid4().hex
        self.process = SPAWN.Process(
            target=serve, args=(child, self.host_factory, self.stop, *self.host_args),
            name=f"{self.kind}-{self.instance[:8]}", daemon=True)
        self.process.start()
        child.close()
        self.ready = False
        self.generation += 1

    @property
    def alive(self) -> bool:
        return self.process.is_alive()

    def exchange(self, key, payload, *, timeout: float,
                 interrupted: Callable[[], bool] | None = None) -> tuple[int, Any]:
        """Blocking request/reply. ``timeout`` starts once the child is ready."""
        if not self.alive:
            raise WorkerError(503, f"{self.kind} worker unavailable; no automatic replay")
        try:
            if not self.ready:
                self._wait(time.monotonic() + READY_TIMEOUT_S, interrupted)
                if self.connection.recv() != READY:
                    raise self._abandon(f"{self.kind} worker initialization mismatch")
                self.ready = True
            self.connection.send((key, payload))
            self._wait(time.monotonic() + timeout, interrupted)
            reply_key, status, result = self.connection.recv()
        except (EOFError, OSError) as error:
            raise WorkerError(503, f"{self.kind} worker disconnected; no automatic replay") from error
        if reply_key != key:
            raise self._abandon(f"{self.kind} pipe lost request correspondence")
        return status, result

    async def submit(self, key, payload, **options) -> tuple[int, Any]:
        loop = asyncio.get_running_loop()
        call = functools.partial(self.exchange, key, payload, **options)
        return await asyncio.shield(loop.run_in_executor(self.executor, call))

    def _wait(self, deadline, interrupted):
        while not self.connection.poll(POLL_S):
            if not self.alive:
                raise WorkerError(503, f"{self.kind} worker exited; no automatic replay")
            if time.monotonic() > deadline:
                raise self._abandon(f"{self.kind} operation timed out", WorkerTimeout)
            if interrupted is not None and interrupted():
                raise self._abandon(f"{self.kind} operation interrupted", WorkerTimeout)

    def _abandon(self, message, error=WorkerError):
        self.stop.set()
        self.terminate()
        return error(message) if error is WorkerTimeout else error(503, message)

    def terminate(self, grace: float = 2.0):
        for signal in ("terminate", "kill"):
            if self.process.is_alive():
                getattr(self.process, signal)()
                self.process.join(grace)

    def close(self, grace: float = 2.0):
        self.stop.set()
        if self.alive:
            try:
                self.connection.send(None)
            except (OSError, EOFError):
                pass  # The child may already have exited.
            self.process.join(grace)
        self.terminate(grace)
        self.connection.close()
        self.executor.shutdown(wait=False, cancel_futures=True)
