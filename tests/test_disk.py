"""Cancellation may not release transition ownership before evidence I/O settles."""

import asyncio
import threading

import pytest

from sim_engine.core.disk import AsyncDisk


@pytest.mark.asyncio
async def test_repeated_cancellation_waits_for_inflight_write_and_keeps_lock(tmp_path):
    disk = AsyncDisk(workers=1)
    entered, release = threading.Event(), threading.Event()
    lock = asyncio.Lock()
    path = tmp_path / "evidence"

    def blocked_write():
        entered.set()
        assert release.wait(5)
        path.write_text("durable")

    async def transition():
        async with lock:
            await disk.run(blocked_write)

    pending = asyncio.create_task(transition())
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        pending.cancel()
        await asyncio.sleep(0)
        pending.cancel()
        await asyncio.sleep(0)
        assert not pending.done() and lock.locked()
        assert disk.health()["pending"] == 1
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert path.read_text() == "durable" and not lock.locked()
        assert disk.health()["pending"] == 0
    finally:
        release.set()
        await asyncio.gather(pending, return_exceptions=True)
        await disk.close()
