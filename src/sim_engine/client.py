"""Async clients share a connection pool; environment sessions stay independent."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from pathlib import Path

import httpx

from .atomic import atomic_json


def history_indices(anchor):
    if anchor < 8:
        return list(range(anchor + 1))
    recent = anchor - 3
    middle = [1 + i * (recent - 2) // 2 for i in range(3)]
    return [0, *middle, *range(recent, anchor), anchor]


class EngineClient:
    def __init__(self, sim_url, policy_url, *, timeout=180):
        self.sim_url, self.policy_url = sim_url.rstrip("/"), policy_url.rstrip("/")
        self.http = httpx.AsyncClient(timeout=timeout)

    async def close(self):
        await self.http.aclose()

    async def request(self, method, url, payload=None):
        response = await self.http.request(method, url, json=payload)
        response.raise_for_status()
        return response.json()

    async def run_episode(self, dataset, episode_id, camera, seed=0, max_plans=None):
        reset = await self.request(
            "POST",
            self.sim_url + "/v1/sessions",
            {"dataset": dataset, "episode_id": str(episode_id), "camera": camera, "seed": seed},
        )
        sid = reset["session_id"]
        history = reset["observations"]
        plans = []
        failed = False
        try:
            while True:
                indices = history_indices(history[-1]["event_index"])
                observations = [{"session_id": sid, "event_index": i} for i in indices]
                request_id = uuid.uuid4().hex
                started = time.monotonic()
                result = await self.request(
                    "POST",
                    self.policy_url + "/v1/policy/generate",
                    {
                        "request_id": request_id,
                        "instruction": reset["instruction"],
                        "camera": camera,
                        "observations": observations,
                        "seed": seed,
                    },
                )
                plans.append(
                    {
                        "output": result["text"],
                        "valid": result["valid"],
                        "stop": result.get("stop", False),
                        "seconds": time.monotonic() - started,
                        "request_id": request_id,
                        "policy_version": result["policy_version"],
                        "token_ids": result["token_ids"],
                        "token_logprobs": result["token_logprobs"],
                        "prompt_tokens": result["prompt_tokens"],
                        "timing_seconds": result["timing_seconds"],
                    }
                )
                if not result["valid"]:
                    await self.request(
                        "POST", f"{self.sim_url}/v1/sessions/{sid}/finish", {"reason": "invalid_output"}
                    )
                    break
                if result.get("stop", False):
                    await self.request(
                        "POST", f"{self.sim_url}/v1/sessions/{sid}/finish", {"reason": "policy_stop"}
                    )
                    break
                execution_started = time.monotonic()
                executed = await self.request(
                    "POST",
                    f"{self.sim_url}/v1/sessions/{sid}/step",
                    {
                        "request_id": request_id,
                        "expected_event_index": history[-1]["event_index"],
                        "command": {"kind": "waypoints", "trajectory": result["trajectory"],
                                    "action_budget": 4},
                        "policy_version": result["policy_version"],
                    },
                )
                history.extend(executed["observations"])
                plans[-1]["simulation_seconds"] = time.monotonic() - execution_started
                plans[-1]["execution"] = {
                    key: executed[key]
                    for key in ("event_index", "executed_actions", "collisions", "termination")
                }
                if "training_reward" in executed:
                    plans[-1]["training_reward"] = executed["training_reward"]
                if executed["done"]:
                    break
                if max_plans is not None and len(plans) >= max_plans:
                    await self.request(
                        "POST", f"{self.sim_url}/v1/sessions/{sid}/finish", {"reason": "cancelled"}
                    )
                    break
            metrics = await self.request("GET", f"{self.sim_url}/v1/sessions/{sid}/metrics")
            metrics["execution_trace"] = metrics.pop("predictions")
            metrics["predictions"] = plans
            metrics["diagnostic_only"] = max_plans is not None
            return metrics
        except BaseException:
            failed = True
            raise
        finally:
            try:
                await self.request("DELETE", f"{self.sim_url}/v1/sessions/{sid}")
            except Exception:
                if not failed:
                    raise


async def rollout(
    sim_url, policy_url, dataset, episode_ids, camera, output, concurrency=2, seed=0, max_plans=None
):
    if (
        concurrency < 1
        or not episode_ids
        or len(set(episode_ids)) != len(episode_ids)
        or (max_plans is not None and max_plans < 1)
    ):
        raise ValueError("invalid concurrency or duplicate/empty episode selection")
    output = Path(output)
    client = EngineClient(sim_url, policy_url)
    try:
        sim_health = await client.request("GET", sim_url.rstrip("/") + "/v1/health")
        policy_health = await client.request("GET", policy_url.rstrip("/") + "/v1/health")
        datasets = await client.request("GET", sim_url.rstrip("/") + "/v1/datasets")
        if concurrency > sim_health.get("live_workers", sim_health["workers"]):
            raise ValueError("rollout concurrency exceeds simulation worker capacity")
        if policy_health["backend"] != "vllm":
            raise ValueError("this rollout requires the vLLM core")
        if policy_health["camera_specs"][camera] != sim_health["protocol"]["cameras"][camera]:
            raise ValueError("policy camera configuration differs from actual simulation")
        run = {
            "dataset": dataset,
            "dataset_identity": datasets[dataset],
            "episode_ids": list(map(str, episode_ids)),
            "camera": camera,
            "seed": seed,
            "concurrency": concurrency,
            "max_plans": max_plans,
            "policy": policy_health,
            "simulation_protocol": sim_health.get("protocol", {}),
            "stop_policy": (
                "explicit_model_stop; forced_end_is_failure"
                if policy_health.get("stop_enabled")
                else "absent_in_q8_baseline; forced_end_is_failure"
            ),
            "rollout_implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        run["fingerprint"] = hashlib.sha256(json.dumps(run, sort_keys=True).encode()).hexdigest()
        manifest = output / "run.json"
        if manifest.exists() and json.loads(manifest.read_text()) != run:
            raise ValueError("resume model/config/assets mismatch")
        atomic_json(manifest, run)
        semaphore = asyncio.Semaphore(concurrency)

        async def one(key):
            async with semaphore:
                path = output / "episodes" / (hashlib.sha256(str(key).encode()).hexdigest() + ".json")
                if path.exists():
                    record = json.loads(path.read_text())
                    if record["fingerprint"] != run["fingerprint"] or record["episode_id"] != str(key):
                        raise ValueError("existing episode receipt mismatch")
                    return record
                result = await client.run_episode(dataset, str(key), camera, seed, max_plans)
                result["fingerprint"] = run["fingerprint"]
                atomic_json(path, result)
                print(json.dumps({"episode": key, "termination": result["termination"]}), flush=True)
                return result

        tasks = [asyncio.create_task(one(key)) for key in episode_ids]
        try:
            records = await asyncio.gather(*tasks)
        except BaseException:
            # Drain session cleanup before closing the shared HTTP connection pool.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        names = sorted(records[0]["metrics"])
        if any(sorted(row["metrics"]) != names for row in records):
            raise ValueError("inconsistent metric fields")
        summary = {
            "complete": True,
            "diagnostic_only": max_plans is not None,
            "episodes": len(records),
            "fingerprint": run["fingerprint"],
            "metrics": {k: sum(row["metrics"][k] for row in records) / len(records) for k in names},
            "termination_counts": {
                name: sum(r["termination"] == name for r in records)
                for name in sorted({r["termination"] for r in records})
            },
        }
        atomic_json(output / "summary.json", summary)
        return summary
    finally:
        await client.close()
