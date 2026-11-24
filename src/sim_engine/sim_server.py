import json
import uuid

from fastapi import HTTPException, Response

from . import __version__
from .core.http import service_app
from .core.io import implementation_sources
from .data import EpisodeStore
from .pool import SessionPool
from .protocol import ExpertRequest, FinishRequest, ObserveRequest, ResetRequest, StepRequest


def sensor_response(data):
    """A raw sensor payload; calibration and world pose travel in one metadata header."""
    return Response(data["raw"], media_type="application/octet-stream", headers={
        "X-Content-SHA256": data["sha256"], "X-Encoding": data["encoding"],
        "X-Sensor-Metadata": json.dumps({"calibration": data["calibration"], "pose": data["pose"]}),
    })


def create_app(settings, *, store=None, pool_factory=SessionPool):
    store = store or EpisodeStore(settings)

    async def open_pool():
        return await pool_factory(settings).start()

    app = service_app("VLN Sim Engine", "pool", open_pool)
    startup_id = uuid.uuid4().hex
    protocol = {
        "max_actions": settings.simulation.max_actions,
        "dataset_physics": {
            name: data.physics.model_dump() for name, data in settings.simulation.datasets.items()
        },
        "reward_mode": settings.simulation.reward_mode,
        "cameras": {key: c.model_dump() for key, c in settings.cameras.items()},
        "scene_hashes": store.scene_hashes,
        "implementation_sha256": implementation_sources(),
    }

    @app.get("/v1/health")
    async def health():
        live = sum(worker.alive for worker in app.state.pool.workers)
        runtimes = app.state.pool.runtimes
        return {
            "status": "ok" if live == settings.simulation.workers else "degraded",
            "backend": settings.simulation.backend,
            "runtime": runtimes[0],
            "habitat_version": runtimes[0].get("habitat_version"),
            "worker_runtimes": runtimes,
            "capabilities": {"sensors": settings.simulation.sensors,
                             "commands": ["waypoints", "velocity", "point_goal", "retrace"],
                             "observe": True,
                             "retrace_targets": ["leg_id", "event_index"],
                             "task_types": ["navigation", "development"]},
            "version": __version__,
            "startup_id": startup_id,
            "workers": settings.simulation.workers,
            "live_workers": live,
            "protocol": protocol,
            "active_sessions": len(app.state.pool.sessions),
            "queues": [worker.queue.snapshot() for worker in app.state.pool.workers],
        }

    @app.get("/v1/datasets")
    async def datasets():
        return store.metadata

    def resolve(request):
        if request.camera not in settings.cameras:
            raise HTTPException(404, "unknown camera profile")
        try:
            return store.get(request.dataset, request.episode_id)
        except KeyError as error:
            raise HTTPException(404, "unknown dataset or episode") from error

    @app.post("/v1/sessions")
    async def reset(request: ResetRequest):
        return await app.state.pool.reset(resolve(request), request.camera, request.seed)

    @app.post("/v1/experts")
    async def expert(request: ExpertRequest):
        return await app.state.pool.expert(resolve(request), request.camera)

    @app.post("/v1/sessions/{session_id}/step")
    async def step(session_id: str, request: StepRequest):
        return await app.state.pool.call(session_id, "step", request=request.model_dump())

    @app.post("/v1/sessions/{session_id}/observe")
    async def observe(session_id: str, request: ObserveRequest):
        return await app.state.pool.call(session_id, "observe", request_id=request.request_id)

    @app.get("/v1/sessions/{session_id}/observations/{observation_id}/{view}")
    async def observation(session_id: str, observation_id: str, view: str, sensor: str = "rgb"):
        if sensor != "rgb":
            data = await app.state.pool.call(session_id, "observation", observation_id=observation_id,
                                             view=view, sensor=sensor)
            return sensor_response(data)
        png, digest = await app.state.pool.call(
            session_id, "observation", observation_id=observation_id, view=view)
        return Response(png, media_type="image/png", headers={"X-Content-SHA256": digest})

    @app.get("/v1/sessions/{session_id}/frames/{event_index}")
    async def frame(session_id: str, event_index: int, sensor: str = "rgb"):
        if sensor != "rgb":
            data = await app.state.pool.call(session_id, "sensor", event_index=event_index, sensor=sensor)
            return sensor_response(data)
        raw, digest, camera, width, height, pose = await app.state.pool.call(
            session_id, "frame", event_index=event_index
        )
        return Response(
            raw,
            media_type="application/octet-stream",
            headers={
                "X-Content-SHA256": digest,
                "X-Camera-Profile": camera,
                "X-Width": str(width),
                "X-Height": str(height),
                # Chassis pose at this event, for the caller's records only, never for a model.
                "X-Pose": json.dumps(pose),
            },
        )

    @app.get("/v1/sessions/{session_id}/metrics")
    async def metrics(session_id: str):
        return await app.state.pool.call(session_id, "metrics")

    @app.post("/v1/sessions/{session_id}/finish")
    async def finish(session_id: str, request: FinishRequest):
        return await app.state.pool.call(session_id, "finish", reason=request.reason)

    @app.delete("/v1/sessions/{session_id}")
    async def close(session_id: str):
        return await app.state.pool.release(session_id)

    return app
