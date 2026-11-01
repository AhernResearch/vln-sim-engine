import asyncio

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
