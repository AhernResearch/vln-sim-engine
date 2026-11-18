"""Common FastAPI shell for services that own long-lived workers."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Awaitable, Callable, Protocol

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from sim_engine import __version__

from .process import WorkerError


class Service(Protocol):
    async def close(self) -> None: ...


def service_app(title: str, state: str, open_service: Callable[[], Awaitable[Service]]) -> FastAPI:
    """The lifespan opens one service as ``app.state.<state>`` and always closes it.

    A ``WorkerError`` raised by any route becomes its HTTP status with a ``detail`` body.
    """

    @asynccontextmanager
    async def lifespan(app):
        service = await open_service()
        setattr(app.state, state, service)
        try:
            yield
        finally:
            await service.close()

    app = FastAPI(title=title, version=__version__, lifespan=lifespan)

    @app.exception_handler(WorkerError)
    async def worker_error(request, error):
        return JSONResponse(status_code=error.status, content={"detail": str(error)})

    return app
