"""One EGL owner per process; HTTP concurrency never migrates graphics contexts."""

from __future__ import annotations

import asyncio
import hashlib
import io
import traceback
import uuid

import numpy as np

from .backends.habitat import HabitatEnvironment
from .controllers import CommandRejected
from .core.process import ProcessWorker, WorkerError
from .core.queue import AsyncWorkQueue, QueueBusy
from .expert import expert_job
from .navigation.planners import NavMeshPlanners
from .session import Conflict, Session


def encode_sensor(capture, sensor):
    """One captured sensor with its calibration and world pose; depth is metres in an npz."""
    if sensor not in capture.images:
        raise KeyError("sensor was not captured at this event")
    value = capture.images[sensor]
    if "depth" in sensor:
        stream = io.BytesIO()
        np.savez_compressed(stream, depth_m=np.asarray(value, dtype=np.float32))
        raw, encoding = stream.getvalue(), "depth_npz"
    else:
        raw, encoding = value.tobytes(), "rgb8"
    return {"raw": raw, "sha256": hashlib.sha256(raw).hexdigest(), "encoding": encoding,
            "calibration": capture.calibration, "pose": capture.sensors[sensor]}


class SessionHost:
    """Child-side owner of one environment and at most one active session."""

    ERRORS = ((Conflict, 409), (CommandRejected, 422), (KeyError, 404))

    def __init__(self, stop, settings, factory):
        self.settings, self.factory = settings, factory
        self.environment = None
        self.session = None
        self.planners = None

    def handle(self, payload):
        method, data = payload
        try:
            return 200, self._dispatch(method, data)
        except Exception as error:
            status = next((code for kind, code in self.ERRORS if isinstance(error, kind)), None)
            if status is not None:
                return status, str(error)
            # Partial simulator failures make the session unusable; never let a
            # retry continue from an unknown partially applied step.
            traceback.print_exc()
            self.close()
            return 500, "simulator operation failed; session invalidated"

    def _env(self):
        if self.environment is None:
            self.environment = self.factory(self.settings)
        return self.environment

    def _dispatch(self, method, data):
        if method in ("reset", "expert") and self.session is not None:
            raise Conflict("worker already owns an active session")
        if method == "reset":
            self.session = Session.open(environment=self._env(), settings=self.settings,
                                        geometry=self._planners, **data)
            return self.session.reset_reply()
        if method == "expert":
            return self._expert(data["episode"], data["camera"])
        if method == "runtime":
            return self._env().identity()
        if self.session is None or data["session_id"] != self.session.id:
            raise KeyError("session is not active on this worker")
        handler = getattr(self, "_" + method, None)
        if handler is None:
            raise ValueError("unknown worker operation")
        return handler(self.session, data)

    def _planners(self):
        """One planner set per loaded NavMesh; a scene change replaces the pathfinder."""
        pathfinder = self._env().pathfinder
        if self.planners is None or self.planners.pf is not pathfinder:
            self.planners = NavMeshPlanners(
                pathfinder, self.settings.simulation.legs.minimum_clearance_m)
        return self.planners

    def _expert(self, episode, camera):
        if not episode.get("goals"):
            raise CommandRejected("expert requires an explicit goal")
        if self.settings.simulation.expert_root is None:
            raise CommandRejected("server has no expert_root configured")
        environment = self._env()
        try:
            return expert_job(environment, episode, camera, self.settings.simulation.expert_root,
                              self.settings.simulation.max_actions)
        except FileExistsError as error:
            raise Conflict(str(error)) from error
        except ValueError as error:
            environment.close()
            raise CommandRejected(f"expert failed: {error}") from error

    @staticmethod
    def _step(session, data):
        return session.step(data["request"])

    @staticmethod
    def _observe(session, data):
        return session.observe(data["request_id"])

    @staticmethod
    def _observation(session, data):
        if data.get("sensor", "rgb") == "rgb":
            return session.observation_image(data["observation_id"], data["view"])
        capture = session.observation_capture(data["observation_id"], data["view"])
        return encode_sensor(capture, data["sensor"])

    @staticmethod
    def _frame(session, data):
        index = data["event_index"]
        if not 0 <= index < len(session.frames):
            raise KeyError("observation is not available")
        capture = session.sensor_frames[index]
        pose = capture and {"position": list(capture.position), "rotation_xyzw": list(capture.rotation_xyzw)}
        return (*session.frames[index], session.camera_name, session.camera.width,
                session.camera.height, pose)

    @staticmethod
    def _sensor(session, data):
        index, sensor = data["event_index"], data["sensor"]
        if not 0 <= index < len(session.sensor_frames):
            raise KeyError("observation is not available")
        snapshot = session.sensor_frames[index]
        if snapshot is None:
            raise KeyError("sensor was not captured at this event")
        return encode_sensor(snapshot, sensor)

    @staticmethod
    def _metrics(session, data):
        return session.metrics()

    @staticmethod
    def _finish(session, data):
        if session.termination is None:
            session.termination = data["reason"]
        return session.metrics()

    def _close(self, session, data):
        # Session state is discarded; the compatible scene stays warm.
        self.session = None
        return {"closed": True}

    def close(self):
        if self.environment is not None:
            self.environment.close()
        self.session = None


class Worker(ProcessWorker):
    kind = "simulator"

    def __init__(self, settings, factory):
        super().__init__(SessionHost, settings, factory)
        self.timeout = settings.simulation.rpc_timeout_seconds
        self.expert_timeout = settings.simulation.expert_timeout_seconds
        self.session_id = None
        self.queue = AsyncWorkQueue(self._queued_call, concurrency=1,
                                    capacity=settings.simulation.queue_capacity, name="simulation")

    async def call(self, method, payload):
        try:
            return await self.queue.submit((method, payload))
        except QueueBusy as error:
            raise WorkerError(429, str(error)) from error

    async def _queued_call(self, operation):
        method, _ = operation
        timeout = self.expert_timeout if method == "expert" else self.timeout
        status, result = await self.submit(uuid.uuid4().hex, operation, timeout=timeout)
        if status != 200:
            raise WorkerError(status, result)
        return result

    def close(self):
        self.queue.closed = True
        for task in self.queue.workers:
            task.cancel()
        super().close(grace=5)


class SessionPool:
    def __init__(self, settings, factory=HabitatEnvironment):
        self.settings = settings
        self.workers = [Worker(settings, factory) for _ in range(settings.simulation.workers)]
        self.sessions = {}
        self.lock = asyncio.Lock()

    async def _reserve(self):
        async with self.lock:
            worker = next(
                (w for w in self.workers if w.session_id is None and w.alive), None
            )
            if worker is None:
                raise WorkerError(429, "all simulator workers are occupied")
            session_id = uuid.uuid4().hex
            worker.session_id = session_id
            self.sessions[session_id] = worker
        return worker, session_id

    def _unreserve(self, worker, session_id):
        self.sessions.pop(session_id, None)
        worker.session_id = None

    async def expert(self, episode, camera_name):
        """Whole-episode job: the worker is held for its duration, then freed."""
        worker, session_id = await self._reserve()
        try:
            return await worker.call(
                "expert", {"episode": episode, "camera": self.settings.cameras[camera_name]}
            )
        finally:
            self._unreserve(worker, session_id)

    async def reset(self, episode, camera_name, seed):
        worker, session_id = await self._reserve()
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
                self._unreserve(worker, session_id)
            raise

    async def call(self, session_id, method, **payload):
        worker = self.sessions.get(session_id)
        if worker is None:
            raise WorkerError(404, "unknown or closed session")
        try:
            return await worker.call(method, {"session_id": session_id, **payload})
        except WorkerError as error:
            if error.status >= 500:
                self._unreserve(worker, session_id)
            raise

    async def release(self, session_id):
        result = await self.call(session_id, "close")
        self.sessions.pop(session_id, None)
        for worker in self.workers:
            if worker.session_id == session_id:
                worker.session_id = None
        return result

    async def start(self):
        """Each worker reports the runtime identity it actually loaded."""
        self.runtimes = await asyncio.gather(*(w.call("runtime", {}) for w in self.workers))
        return self

    async def close(self):
        await asyncio.gather(*(worker.queue.close() for worker in self.workers))
        for worker in self.workers:
            worker.close()
