"""ProcessWorker: correspondence, deadlines, interruption and replacement of spawned children."""

import os
import time

import pytest

from sim_engine.core.process import ProcessWorker, WorkerError, WorkerTimeout


class EchoHost:
    def __init__(self, stop, scale):
        self.stop, self.scale = stop, scale

    def handle(self, payload):
        action, value = payload
        if action == "sleep":
            time.sleep(value)
        elif action == "wait_for_stop":
            while not self.stop.is_set():
                time.sleep(0.01)
        elif action == "exit":
            os._exit(1)
        elif action == "reject":
            return 422, "rejected"
        return 200, value * self.scale

    def close(self):
        pass


@pytest.fixture
def worker():
    value = ProcessWorker(EchoHost, 3)
    yield value
    value.close()


def test_reply_matches_request_and_child_state_persists(worker):
    assert worker.exchange("a", ("echo", 2), timeout=30) == (200, 6)
    assert worker.exchange("b", ("reject", 0), timeout=5) == (422, "rejected")
    assert worker.exchange("c", ("echo", 5), timeout=5) == (200, 15)
    assert worker.generation == 1 and worker.alive


async def test_async_submit_uses_the_serialized_rpc_thread(worker):
    assert await worker.submit("a", ("echo", 1), timeout=30) == (200, 3)


def test_deadline_terminates_the_child_without_replay(worker):
    worker.exchange("warm", ("echo", 0), timeout=30)
    with pytest.raises(WorkerTimeout):
        worker.exchange("slow", ("sleep", 5), timeout=0.2)
    assert not worker.alive and worker.stop.is_set()
    with pytest.raises(WorkerError) as error:
        worker.exchange("next", ("echo", 1), timeout=1)
    assert error.value.status == 503


def test_interruption_is_observed_while_waiting(worker):
    worker.exchange("warm", ("echo", 0), timeout=30)
    began = time.monotonic()
    with pytest.raises(WorkerTimeout):
        worker.exchange("slow", ("sleep", 5), timeout=30, interrupted=lambda: time.monotonic() - began > 0.2)
    assert time.monotonic() - began < 4 and not worker.alive


def test_child_sees_the_shared_cancellation_flag(worker):
    worker.exchange("warm", ("echo", 0), timeout=30)
    worker.stop.set()
    assert worker.exchange("cancel", ("wait_for_stop", 4), timeout=5) == (200, 12)


def test_exited_child_is_reported_and_capacity_can_be_respawned(worker):
    with pytest.raises(WorkerError) as error:
        worker.exchange("crash", ("exit", 0), timeout=30)
    assert error.value.status == 503 and not worker.alive
    previous = worker.instance
    worker.spawn()
    assert worker.generation == 2 and worker.instance != previous
    assert worker.exchange("again", ("echo", 1), timeout=30) == (200, 3)
