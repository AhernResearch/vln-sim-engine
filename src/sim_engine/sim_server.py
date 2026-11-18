import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response

from . import __version__
from .core.process import WorkerError
from .data import EpisodeStore
from .pool import SessionPool
from .protocol import ExpertRequest, FinishRequest, ResetRequest, StepRequest


def create_app(settings, *, store=None, pool_factory=SessionPool):
    store = store or EpisodeStore(settings)

    @asynccontextmanager
    async def lifespan(app):
        app.state.pool = pool_factory(settings)
        try:
            import asyncio

            app.state.runtimes = await asyncio.gather(
                *(worker.call("runtime", {}) for worker in app.state.pool.workers))
            app.state.runtime = app.state.runtimes[0]
            yield
        finally:
            await app.state.pool.aclose()

    app = FastAPI(title="VLN Sim Engine", version=__version__, lifespan=lifespan)
    from .planning.implementation import implementation_sources

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

    @app.exception_handler(WorkerError)
    async def worker_error(request, error):
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=error.status, content={"detail": str(error)})

    @app.get("/v1/health")
    async def health():
        live = sum(worker.alive for worker in app.state.pool.workers)
        return {
            "status": "ok" if live == settings.simulation.workers else "degraded",
            "backend": settings.simulation.backend,
            "runtime": app.state.runtime,
            "habitat_version": app.state.runtime.get("habitat_version"),
            "worker_runtimes": app.state.runtimes,
            "capabilities": {"sensors": settings.simulation.sensors,
                             "commands": ["waypoints", "velocity"],
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

    @app.get("/v1/sessions/{session_id}/frames/{event_index}")
    async def frame(session_id: str, event_index: int, sensor: str = "rgb"):
        if sensor != "rgb":
            import json

            data = await app.state.pool.call(session_id, "sensor", event_index=event_index, sensor=sensor)
            return Response(data["raw"], media_type="application/octet-stream", headers={
                "X-Content-SHA256": data["sha256"], "X-Encoding": data["encoding"],
                "X-Sensor-Metadata": json.dumps({"calibration": data["calibration"],
                                                 "pose": data["pose"]}),
            })
        raw, digest, camera, width, height = await app.state.pool.call(
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
