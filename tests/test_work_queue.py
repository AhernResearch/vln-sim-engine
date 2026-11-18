import asyncio

import pytest

from sim_engine.core.queue import AsyncWorkQueue, QueueBusy


async def test_fast_completion_reaches_simulation_before_slow_inference_finishes():
    slow_started, release_slow, simulated_fast = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def infer(value):
        if value == 'slow':
            slow_started.set()
            await release_slow.wait()
        return value

    async def simulate(value):
        simulated_fast.set()
        return value

    inference = AsyncWorkQueue(infer, concurrency=2, capacity=2, name='inference')
    simulation = AsyncWorkQueue(simulate, concurrency=1, capacity=2, name='simulation')

    async def flow(value):
        return await simulation.submit(await inference.submit(value))

    try:
        slow = asyncio.create_task(flow('slow'))
        await slow_started.wait()
        fast = asyncio.create_task(flow('fast'))
        await asyncio.wait_for(simulated_fast.wait(), 1)
        assert await fast == 'fast'
        assert not slow.done()
        release_slow.set()
        assert await slow == 'slow'
        assert inference.snapshot()['peak_active'] == 2
    finally:
        await inference.close()
        await simulation.close()


async def test_backpressure_cancellation_and_worker_error_do_not_corrupt_other_items():
    started, release = asyncio.Event(), asyncio.Event()
    completed = []

    async def handle(value):
        if value == 0:
            started.set()
            await release.wait()
        if value == 'bad':
            raise ValueError('invalid work')
        completed.append(value)
        return value

    queue = AsyncWorkQueue(handle, concurrency=1, capacity=1, name='test')
    try:
        first = asyncio.create_task(queue.submit(0))
        await started.wait()
        second = asyncio.create_task(queue.submit(1))
        await asyncio.sleep(0)
        with pytest.raises(QueueBusy):
            await queue.submit(2)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        release.set()
        assert await second == 1
        assert completed == [0, 1]
        with pytest.raises(ValueError, match='invalid work'):
            await queue.submit('bad')
        assert await queue.submit(3) == 3
        assert queue.snapshot()['rejected'] == 1
    finally:
        await queue.close()
