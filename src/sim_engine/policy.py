"""Asynchronous vLLM core with bounded admission and immutable observation caching."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import OrderedDict
from pathlib import Path

import httpx
from PIL import Image

from .data import model_identity, sha256
from .outputs import parse_prediction
from .prompt import PromptBuilder, prepare_training_image


class PolicyBusy(RuntimeError):
    pass


class VllmPolicy:
    def __init__(self, settings):
        self.settings = settings
        self.builder = PromptBuilder(settings.policy.prompt_profile)
        self.cache = OrderedDict()
        self.requests = OrderedDict()
        self.admission = asyncio.Semaphore(settings.policy.max_pending_requests)
        self.engine = self.http = None

    async def start(self):
        from transformers import AutoProcessor
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        config = self.settings.policy
        self.model_identity = model_identity(config.model)
        self.processor = AutoProcessor.from_pretrained(str(config.model), local_files_only=True)
        arguments = AsyncEngineArgs(
            model=str(config.model),
            dtype="bfloat16",
            max_model_len=config.max_model_len,
            max_num_seqs=config.max_num_seqs,
            max_num_batched_tokens=config.max_num_batched_tokens,
            gpu_memory_utilization=config.gpu_memory_utilization,
            enforce_eager=config.enforce_eager,
            limit_mm_per_prompt={"image": {"count": 8, "width": 512, "height": 512}},
            mm_processor_kwargs={
                "do_resize": False,
                "size": {
                    "longest_edge": config.max_pixels,
                    "shortest_edge": min(65536, config.max_pixels),
                },
            },
            enable_prefix_caching=True,
        )
        self.engine = AsyncLLM.from_engine_args(arguments)
        self.http = httpx.AsyncClient(base_url=config.sim_url.rstrip("/"), timeout=60)

    async def close(self):
        if self.http is not None:
            await self.http.aclose()
        if self.engine is not None:
            self.engine.shutdown()

    async def _image(self, observation, camera_name):
        key = (observation.session_id, observation.event_index)
        if key in self.cache:
            image, camera = self.cache[key]
            if camera != camera_name:
                raise ValueError("observation camera does not match requested profile")
            self.cache.move_to_end(key)
            return image
        result = await self.http.get(f"/v1/sessions/{key[0]}/frames/{key[1]}")
        result.raise_for_status()
        if result.headers["X-Camera-Profile"] != camera_name:
            raise ValueError("observation camera does not match requested profile")
        if hashlib.sha256(result.content).hexdigest() != result.headers["X-Content-SHA256"]:
            raise ValueError("observation content hash mismatch")
        size = int(result.headers["X-Width"]), int(result.headers["X-Height"])
        image = Image.frombytes("RGB", size, result.content)
        image = prepare_training_image(
            image, self.settings.policy.max_pixels, self.processor.image_processor.patch_size
        )
        self.cache[key] = image, camera_name
        while len(self.cache) > self.settings.policy.image_cache_entries:
            self.cache.popitem(last=False)
        return image

    async def generate(self, request):
        fingerprint = hashlib.sha256(
            json.dumps(request.model_dump(), sort_keys=True).encode()
        ).hexdigest()
        if request.request_id in self.requests:
            previous, task = self.requests[request.request_id]
            if fingerprint != previous:
                raise ValueError("request_id was already used with different inference inputs")
            self.requests.move_to_end(request.request_id)
            return await asyncio.shield(task)
        if (
            sum(not task.done() for _, task in self.requests.values())
            >= self.settings.policy.max_pending_requests
        ):
            raise PolicyBusy("policy request queue is full")
        task = asyncio.create_task(self._generate(request))
        self.requests[request.request_id] = fingerprint, task
        while len(self.requests) > 1024:
            oldest = next((key for key, (_, value) in self.requests.items() if value.done()), None)
            if oldest is None:
                break
            self.requests.pop(oldest)
        return await asyncio.shield(task)

    def identity(self):
        import importlib.metadata

        return {
            "model_identity": self.model_identity,
            "prompt_profile": self.builder.profile,
            "stop_enabled": self.builder.stop_enabled,
            "implementation_sha256": {
                name: sha256(Path(__file__).parent / name)
                for name in ("policy.py", "prompt.py", "outputs.py", "protocol.py", "config.py")
            },
            "processor_versions": {
                name: importlib.metadata.version(name)
                for name in ("transformers", "qwen-vl-utils", "pillow", "torch")
            },
            "generation_contract": {
                "max_model_len": self.settings.policy.max_model_len,
                "max_pixels": self.settings.policy.max_pixels,
                "dtype": "bfloat16",
                "max_num_seqs": self.settings.policy.max_num_seqs,
                "enforce_eager": self.settings.policy.enforce_eager,
            },
            "prompt_hashes": {
                name: sha256(self.builder.template_root / (name + ".jinja"))
                for name in ("system", "user")
            },
        }

    async def _generate(self, request):
        from vllm import SamplingParams

        arrival = time.monotonic()
        async with self.admission:
            admitted = time.monotonic()
            images = await asyncio.gather(
                *(self._image(o, request.camera) for o in request.observations)
            )
            indices = [o.event_index for o in request.observations]
            prompt = self.builder.native_prompt(
                self.processor,
                request.instruction,
                self.settings.cameras[request.camera],
                indices,
                images,
            )
            prepared = time.monotonic()
            inputs = {
                "prompt": prompt,
                "multi_modal_data": {"image": images},
                "multi_modal_uuids": {
                    "image": [f"{o.session_id}:{o.event_index}" for o in request.observations]
                },
            }
            sampling = SamplingParams(
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                seed=request.seed,
                logprobs=1,
            )
            final = None
            try:
                async for output in self.engine.generate(
                    inputs, sampling, request_id=request.request_id
                ):
                    final = output
            except BaseException:
                await self.engine.abort(request.request_id)
                raise
            if final is None or not final.outputs:
                raise RuntimeError("vLLM returned no completion")
            completion = final.outputs[0]
            result = {
                "request_id": request.request_id,
                "backend": "vllm",
                "policy_version": self.settings.policy_version,
                "text": completion.text,
                "token_ids": list(completion.token_ids),
                "token_logprobs": [
                    float(entry[token].logprob)
                    for token, entry in zip(completion.token_ids, completion.logprobs, strict=True)
                ]
                if completion.logprobs
                else None,
                "prompt_tokens": len(final.prompt_token_ids or []),
                "timing_seconds": {
                    "admission_queue": admitted - arrival,
                    "images_and_prompt": prepared - admitted,
                    "engine_including_queue": time.monotonic() - prepared,
                },
                "valid": False,
                "stop": False,
                "trajectory": None,
            }
            try:
                parsed = parse_prediction(completion.text, self.builder.profile)
                result.update(valid=True, stop=parsed.stop, trajectory=parsed.trajectory)
            except (ValueError, TypeError):
                result["output_error"] = "invalid output for " + self.builder.profile
            return result
