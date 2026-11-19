"""Collection HTTP boundary: task descriptions and receipts, never executable code."""

from __future__ import annotations

import asyncio

from sim_engine.collection.protocol import StatusRequest
from sim_engine.core.http import service_app
from sim_engine.data import EpisodeStore
from sim_engine.planning.tasks import RegionRequest, TaskRequest

from .backend import CollectionActor
from .service import CollectionService


def create_app(settings, *, store=None, actor_factory=CollectionActor):
    store = store or EpisodeStore(settings)

    async def open_service():
        return CollectionService(settings, store, actor_factory)

    app = service_app("Privileged training collection service", "collection", open_service)

    @app.get("/v1/health")
    async def health():
        return app.state.collection.health()

    @app.post("/v1/collections")
    async def submit(request: TaskRequest):
        return await app.state.collection.submit(request)

    @app.post("/v1/collections/status")
    async def statuses(request: StatusRequest):
        return await asyncio.gather(*(app.state.collection.status(cid) for cid in request.request_ids))

    @app.get("/v1/collections/{cid}")
    async def status(cid: str):
        return await app.state.collection.status(cid)

    @app.post("/v1/collections/{cid}/target")
    async def target(cid: str, request: RegionRequest):
        return await app.state.collection.target(cid, request)

    @app.delete("/v1/collections/{cid}")
    async def cancel(cid: str):
        return await app.state.collection.cancel(cid)

    return app
