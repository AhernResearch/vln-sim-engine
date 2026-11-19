import importlib.metadata

from fastapi import HTTPException

from .core.http import service_app
from .policy import PolicyBusy, VllmPolicy
from .protocol import GenerateRequest, RecordedGenerateRequest


def create_app(settings, *, backend_factory=VllmPolicy):
    async def open_backend():
        backend = backend_factory(settings)
        try:
            await backend.start()
        except BaseException:
            await backend.close()
            raise
        return backend

    app = service_app("VLN vLLM Policy", "backend", open_backend)

    @app.get("/v1/health")
    async def health():
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
        return await execute(request)

    @app.post("/v1/policy/generate-recorded")
    async def generate_recorded(request: RecordedGenerateRequest):
        return await execute(request)

    async def execute(request):
        try:
            return await app.state.backend.generate(request)
        except PolicyBusy as error:
            raise HTTPException(429, str(error)) from error
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    return app
