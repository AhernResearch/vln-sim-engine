import asyncio
import hashlib
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from helpers import TestEnvironment, episode, request, settings

from sim_engine.policy import PolicyBusy, VllmPolicy
from sim_engine.pool import SessionPool
from sim_engine.protocol import GenerateRequest
from sim_engine.sim_server import create_app


def test_http_frame_contract_and_retry_conflicts(tmp_path):
    config = settings(tmp_path)
    store = SimpleNamespace(
        metadata={"test": {"episodes": 1}}, scene_hashes={}, get=lambda dataset, key: episode(key)
    )
    app = create_app(
        config, store=store, pool_factory=lambda cfg: SessionPool(cfg, factory=TestEnvironment)
    )
    with TestClient(app) as client:
        reset = client.post("/v1/sessions", json={"dataset": "test", "episode_id": "a"})
        assert reset.status_code == 200
        value = reset.json()
        assert not {"goal", "positions", "reference_path"} & value.keys()
        sid = value["session_id"]
        original = client.get(f"/v1/sessions/{sid}/frames/0")
        assert hashlib.sha256(original.content).hexdigest() == original.headers["X-Content-SHA256"]
        result = client.post(f"/v1/sessions/{sid}/step", json=request())
        assert result.status_code == 200
        assert client.post(f"/v1/sessions/{sid}/step", json=request()).json() == result.json()
        assert client.post(f"/v1/sessions/{sid}/step", json=request("stale")).status_code == 409
        assert client.get(f"/v1/sessions/{sid}/frames/0").content == original.content
        assert client.get(f"/v1/sessions/{sid}/frames/99").status_code == 404
        assert client.delete(f"/v1/sessions/{sid}").status_code == 200
        assert client.get(f"/v1/sessions/{sid}/metrics").status_code == 404


async def test_inference_duplicates_share_work_and_admission_is_bounded(tmp_path):
    config = settings(tmp_path)
    config.policy.max_pending_requests = 1
    backend = VllmPolicy(config)
    started, complete = asyncio.Event(), asyncio.Event()
    calls = []

    async def generate(req):
        calls.append(req.request_id)
        started.set()
        await complete.wait()
        return {"text": "same result"}

    backend._generate = generate
    req = GenerateRequest(
        request_id="one", instruction="forward", observations=[{"session_id": "test", "event_index": 0}]
    )
    first = asyncio.create_task(backend.generate(req))
    await started.wait()
    duplicate = asyncio.create_task(backend.generate(req))
    with pytest.raises(PolicyBusy):
        await backend.generate(req.model_copy(update={"request_id": "two"}))
    with pytest.raises(ValueError):
        await backend.generate(req.model_copy(update={"seed": 1}))
    complete.set()
    assert await first == await duplicate
    assert calls == ["one"]
