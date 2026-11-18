import asyncio

import pytest
from helpers import TestEnvironment, episode, request, settings

from sim_engine.core.process import WorkerError
from sim_engine.pool import SessionPool
from sim_engine.session import Conflict, Session


def test_step_retry_is_exactly_once_and_observations_are_immutable(tmp_path):
    config = settings(tmp_path)
    session = Session.open(
        "one",
        TestEnvironment(config),
        episode(),
        "trained_ce30",
        config.cameras["trained_ce30"],
        0,
        config,
    )
    initial = session.frames[0]
    result = session.step(request())
    assert session.step(request()) == result
    assert len(session.actions) == 4
    assert result["event_index"] == 4
    assert session.frames[0] == initial
    assert session.frames[0] != session.frames[4]
    with pytest.raises(Conflict, match="different content"):
        session.step(request(expected=4))
    with pytest.raises(Conflict, match="out-of-order"):
        session.step(request("other", 0))
    session.step(request("second", 4))
    assert session.termination == "timeout"
    assert session.metrics()["metrics"]["success"] == 0


async def test_worker_processes_isolate_sessions_and_release_capacity(tmp_path):
    config = settings(tmp_path)
    pool = SessionPool(config, factory=TestEnvironment)
    try:
        a, b = await asyncio.gather(
            pool.reset(episode("a"), "trained_ce30", 0), pool.reset(episode("b"), "trained_ce30", 0)
        )
        with pytest.raises(WorkerError) as error:
            await pool.reset(episode("c"), "trained_ce30", 0)
        assert error.value.status == 429
        await pool.call(a["session_id"], "step", request=request())
        assert len((await pool.call(b["session_id"], "metrics"))["actions"]) == 0
        await pool.release(a["session_id"])
        c = await pool.reset(episode("c"), "trained_ce30", 0)
        assert c["session_id"] != a["session_id"]
        with pytest.raises(WorkerError):
            await pool.call(a["session_id"], "metrics")
    finally:
        await pool.close()


async def test_dead_worker_invalidates_session_and_other_worker_remains_usable(tmp_path):
    pool = SessionPool(settings(tmp_path), factory=TestEnvironment)
    try:
        reset = await pool.reset(episode(), "trained_ce30", 0)
        worker = pool.sessions[reset["session_id"]]
        worker.process.terminate()
        worker.process.join(timeout=5)
        with pytest.raises(WorkerError) as error:
            await pool.call(reset["session_id"], "metrics")
        assert error.value.status == 503
        assert reset["session_id"] not in pool.sessions
        other = await pool.reset(episode("other"), "trained_ce30", 0)
        assert other["episode_id"] == "other"
    finally:
        await pool.close()
