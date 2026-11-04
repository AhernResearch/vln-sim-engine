"""One EGL owner per process; HTTP concurrency never migrates graphics contexts."""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

from .habitat import HabitatEnvironment
from .session import Conflict, Session
from .work_queue import AsyncWorkQueue, QueueBusy


def _worker(connection, settings, factory):
    environment = None
    session = None
    try:
        while True:
            method, payload = connection.recv()
            if method == "shutdown":
                break
            try:
                if method == "reset":
                    if session is not None:
                        raise Conflict("worker already owns an active session")
                    if environment is None:
                        environment = factory(settings)
                    session = Session.open(environment=environment, settings=settings, **payload)
                    result = session.reset_reply()
                elif session is None or payload["session_id"] != session.id:
                    raise KeyError("session is not active on this worker")
                elif method == "step":
                    result = session.step(payload["request"])
                elif method == "frame":
                    index = payload["event_index"]
                    if not 0 <= index < len(session.frames):
                        raise KeyError("observation is not available")
                    result = (
                        *session.frames[index],
                        session.camera_name,
                        session.camera.width,
                        session.camera.height,
                    )
                elif method == "metrics":
                    result = session.metrics()
                elif method == "finish":
                    if session.termination is None:
                        session.termination = payload["reason"]
                    result = session.metrics()
                elif method == "close":
                    environment.close()
                    session = None
                    result = {"closed": True}
                else:
                    raise ValueError("unknown worker operation")
                connection.send((200, result))
            except Conflict as error:
                connection.send((409, str(error)))
            except KeyError as error:
                connection.send((404, str(error)))
            except Exception:
                # Partial simulator failures make the session unusable; never
                # let a retry continue from an unknown partially applied step.
                traceback.print_exc()
                if environment is not None:
                    environment.close()
                session = None
                connection.send((500, "simulator operation failed; session invalidated"))
    except EOFError:
        pass
    finally:
        if environment is not None:
            environment.close()
        connection.close()


class WorkerError(RuntimeError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class Worker:
    def __init__(self, settings, factory):
        ctx = mp.get_context("spawn")
        self.connection, child = ctx.Pipe()
        self.process = ctx.Process(target=_worker, args=(child, settings, factory), daemon=True)
        self.process.start()
        child.close()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.timeout = settings.simulation.rpc_timeout_seconds
        self.session_id = None
        self.queue = AsyncWorkQueue(self._queued_call, concurrency=1,
            capacity=settings.simulation.queue_capacity, name="simulation")

    def _call(self, method, payload):
        if not self.process.is_alive():
            raise WorkerError(503, "simulator worker is unavailable")
        try:
            self.connection.send((method, payload))
            if not self.connection.poll(self.timeout):
                self.process.terminate()
                raise WorkerError(503, "simulator worker timed out; session invalidated")
            status, result = self.connection.recv()
        except (OSError, EOFError) as error:
            raise WorkerError(503, "simulator worker disconnected; session invalidated") from error
        if status != 200:
            raise WorkerError(status, result)
        return result

    async def call(self, method, payload):
        try:
            return await self.queue.submit((method, payload))
        except QueueBusy as error:
            raise WorkerError(429, str(error)) from error

    async def _queued_call(self, operation):
        method, payload = operation
        # The single-thread executor finishes consuming each reply even if its
        # HTTP caller disconnects, preserving pipe/request correspondence.
        loop = asyncio.get_running_loop()
        return await asyncio.shield(loop.run_in_executor(self.executor, self._call, method, payload))

    def close(self):
        self.queue.closed = True
        for task in self.queue.workers:
            task.cancel()
        if self.process.is_alive():
            try:
                self.connection.send(("shutdown", {}))
            except (OSError, EOFError):
                pass  # The service manager may already have stopped the child.
            self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
        self.connection.close()
        self.executor.shutdown(wait=False, cancel_futures=True)


class SessionPool:
    def __init__(self, settings, factory=HabitatEnvironment):
        self.settings = settings
        self.workers = [Worker(settings, factory) for _ in range(settings.simulation.workers)]
        self.sessions = {}
        self.lock = asyncio.Lock()

    async def reset(self, episode, camera_name, seed):
        async with self.lock:
            worker = next(
                (w for w in self.workers if w.session_id is None and w.process.is_alive()), None
            )
            if worker is None:
                raise WorkerError(429, "all simulator workers are occupied")
            session_id = uuid.uuid4().hex
            worker.session_id = session_id
            self.sessions[session_id] = worker
        try:
            return await worker.call(
                "reset",
                {
                    "session_id": session_id,
                    "episode": episode,
                    "camera_name": camera_name,
                    "camera": self.settings.cameras[camera_name],
                    "seed": seed,
                },
            )
        except BaseException:
            # A disconnected reset may have succeeded. Release it on the same
            # worker queue before making the capacity available again.
            try:
                await worker.call("close", {"session_id": session_id})
            except Exception:
                pass  # Preserve the original reset failure, including its HTTP status.
            finally:
                self.sessions.pop(session_id, None)
                worker.session_id = None
            raise

    async def call(self, session_id, method, **payload):
        worker = self.sessions.get(session_id)
        if worker is None:
            raise WorkerError(404, "unknown or closed session")
        try:
            return await worker.call(method, {"session_id": session_id, **payload})
        except WorkerError as error:
            if error.status >= 500:
                self.sessions.pop(session_id, None)
                worker.session_id = None
            raise

    async def release(self, session_id):
        result = await self.call(session_id, "close")
        self.sessions.pop(session_id, None)
        for worker in self.workers:
            if worker.session_id == session_id:
                worker.session_id = None
        return result

    def close(self):
        for worker in self.workers:
            worker.close()

    async def aclose(self):
        await asyncio.gather(*(worker.queue.close() for worker in self.workers))
        self.close()
