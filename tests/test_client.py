import asyncio
import json

import httpx
import pytest

from sim_engine import client


async def test_rollout_failure_drains_other_session_before_closing_http(tmp_path, monkeypatch):
    started = asyncio.Event()
    state = {"cleaned": False, "closed": False}

    class StubClient:
        def __init__(self, *args):
            pass

        async def request(self, method, url, payload=None):
            if url.endswith("datasets"):
                return {"test": {"episodes": 2}}
            if "sim/" in url:
                return {"workers": 2, "protocol": {"cameras": {"camera": {}}}}
            return {"backend": "vllm", "camera_specs": {"camera": {}}}

        async def run_episode(self, dataset, key, *args):
            if key == "failure":
                await started.wait()
                raise RuntimeError("inference unavailable")
            try:
                started.set()
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0)
                assert not state["closed"]
                state["cleaned"] = True

        async def close(self):
            assert state["cleaned"]
            state["closed"] = True

    monkeypatch.setattr(client, "EngineClient", StubClient)
    with pytest.raises(RuntimeError, match="inference unavailable"):
        await client.rollout(
            "http://sim", "http://policy", "test", ["failure", "active"], "camera", tmp_path
        )
    assert state == {"cleaned": True, "closed": True}
    assert not (tmp_path / "summary.json").exists()


async def test_explicit_stop_finishes_without_executing_an_action():
    calls = []

    def handler(request):
        calls.append((request.method, request.url.path))
        path = request.url.path
        if path == "/v1/sessions":
            value = {
                "session_id": "s",
                "instruction": "Walk.",
                "observations": [{"session_id": "s", "event_index": 0}],
            }
        elif path.endswith("/generate"):
            value = {
                "text": '{"stop":true,"trajectory":[]}',
                "valid": True,
                "stop": True,
                "policy_version": "v2",
                "token_ids": [1],
                "token_logprobs": [-0.1],
                "prompt_tokens": 10,
                "timing_seconds": {},
            }
        elif path.endswith("/finish"):
            assert json.loads(request.content) == {"reason": "policy_stop"}
            value = {"done": True}
        elif path.endswith("/metrics"):
            value = {"predictions": [], "actions": [], "termination": "policy_stop"}
        elif request.method == "DELETE":
            value = {"closed": True}
        else:
            pytest.fail("STOP must not execute a step")
        return httpx.Response(200, json=value)

    backend = client.EngineClient("http://sim", "http://policy")
    await backend.http.aclose()
    backend.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        result = await backend.run_episode("test", "a", "trained_ce30")
    finally:
        await backend.close()
    assert result["termination"] == "policy_stop"
    assert result["actions"] == [] and result["predictions"][0]["stop"]
    assert calls[-1] == ("DELETE", "/v1/sessions/s")
