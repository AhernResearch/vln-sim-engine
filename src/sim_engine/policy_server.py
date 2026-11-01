from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from . import __version__
from .policy import PolicyBusy, VllmPolicy
from .protocol import GenerateRequest


def create_app(settings, *, backend_factory=VllmPolicy):
    @asynccontextmanager
    async def lifespan(app):
        app.state.backend = backend_factory(settings)
        try:
            await app.state.backend.start()
            yield
        finally:
            await app.state.backend.close()

    app = FastAPI(title="VLN vLLM Policy", version=__version__, lifespan=lifespan)

    @app.get("/v1/health")
    async def health():
        import importlib.metadata

        return {
            "status": "ok",
            "backend": "vllm",
            "vllm_version": importlib.metadata.version("vllm"),
            "model": str(settings.policy.model.resolve()),
            "policy_version": settings.policy_version,
            "cameras": {key: c.robot() for key, c in settings.cameras.items()},
            "camera_specs": {key: c.model_dump() for key, c in settings.cameras.items()},
            "weight_updates": False,
            "max_num_seqs": settings.policy.max_num_seqs,
            **app.state.backend.identity(),
        }

    @app.post("/v1/policy/generate")
    async def generate(request: GenerateRequest):
        if request.camera not in settings.cameras:
            raise HTTPException(404, "unknown camera profile")
        try:
            return await app.state.backend.generate(request)
        except PolicyBusy as error:
            raise HTTPException(429, str(error)) from error
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    return app
