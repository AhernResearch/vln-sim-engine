"""Asynchronous vLLM core with bounded admission and immutable observation caching."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import importlib.metadata
import io
import json
import time
from collections import OrderedDict
from pathlib import Path

import httpx
from PIL import Image

from sim_engine.core.io import file_sha256
from sim_engine.core.optional import optional_module

from .core.queue import AsyncWorkQueue, QueueBusy
from .data import model_identity
from .outputs import parse_prediction
from .prompt import PromptBuilder, native_chat_prompt, prepare_training_image
from .protocol import RecordedGenerateRequest


class PolicyBusy(RuntimeError):
    pass


def decode_recorded_image(value):
    raw = base64.b64decode(value.data_base64, validate=True)
    if hashlib.sha256(raw).hexdigest() != value.sha256:
        raise ValueError("recorded image content hash mismatch")
    with Image.open(io.BytesIO(raw)) as image:
        if image.format not in ("JPEG", "PNG") or image.width * image.height > 4_000_000:
            raise ValueError("recorded image must be PNG/JPEG with at most four million pixels")
        return image.convert("RGB")


class VllmPolicy:
    def __init__(self, settings):
        self.settings = settings
        self.builder = PromptBuilder(settings.policy.prompt_profile)
        self.cache = OrderedDict()
        self.requests = OrderedDict()
        self.in_flight = 0
        self.queue = AsyncWorkQueue(
            lambda request: self._generate(request),
            concurrency=settings.policy.max_num_seqs,
            capacity=settings.policy.max_pending_requests,
            name="inference",
        )
        self.engine = self.http = None

    async def start(self):
        AutoProcessor = optional_module("transformers").AutoProcessor
        AsyncEngineArgs = optional_module("vllm.engine.arg_utils").AsyncEngineArgs
        AsyncLLM = optional_module("vllm.v1.engine.async_llm").AsyncLLM

        config = self.settings.policy
        self.model_identity = model_identity(config.model)
        self.processor = AutoProcessor.from_pretrained(str(config.model), local_files_only=True)
        arguments = AsyncEngineArgs(
            model=str(config.model),
            seed=config.seed,
            dtype="bfloat16",
            max_model_len=config.max_model_len,
            max_num_seqs=config.max_num_seqs,
            max_num_batched_tokens=config.max_num_batched_tokens,
            gpu_memory_utilization=config.gpu_memory_utilization,
            enforce_eager=config.enforce_eager,
            limit_mm_per_prompt={"image": {"count": config.max_images, "width": 576, "height": 512}},
            mm_processor_kwargs={
                "do_resize": False,
                "size": {
                    "longest_edge": config.max_pixels,
                    "shortest_edge": min(65536, config.max_pixels),
                },
            },
            enable_prefix_caching=True,
            enable_lora=config.lora_adapter is not None,
            max_lora_rank=config.max_lora_rank,
        )
        self.engine = AsyncLLM.from_engine_args(arguments)
        self.adapter = None
        if config.lora_adapter is not None:
            LoRARequest = optional_module("vllm.lora.request").LoRARequest

            self.adapter = LoRARequest("navigation", 1, str(config.lora_adapter))
        self.image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.http = httpx.AsyncClient(base_url=config.sim_url.rstrip("/"), timeout=60)

    async def close(self):
        await self.queue.close()
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
        if self.in_flight >= self.settings.policy.max_pending_requests:
            raise PolicyBusy("policy request queue is full")
        task = asyncio.create_task(self._enqueue(request))
        self.in_flight += 1
        task.add_done_callback(self._finished)
        self.requests[request.request_id] = fingerprint, task
        while len(self.requests) > 1024:
            oldest = next((key for key, (_, value) in self.requests.items() if value.done()), None)
            if oldest is None:
                break
            self.requests.pop(oldest)
        return await asyncio.shield(task)

    def _finished(self, _task):
        self.in_flight -= 1

    async def _enqueue(self, request):
        began = time.monotonic()
        try:
            result = await self.queue.submit(request)
        except QueueBusy as error:
            raise PolicyBusy(str(error)) from error
        timing = result.get("timing_seconds")
        if timing is not None:
            timing["service_queue"] = max(0.0, time.monotonic() - began - sum(timing.values()))
        return result

    def _recorded_inputs(self, request):
        """PNG decoding and chat rendering run outside the engine's event loop."""
        if len(request.images) > self.settings.policy.max_images:
            raise ValueError("image count exceeds configured policy limit")
        images = [decode_recorded_image(value) for value in request.images]
        if request.prepared:
            factor = self.processor.image_processor.patch_size * 2
            if any(im.width % factor or im.height % factor for im in images):
                raise ValueError("prepared images must align with the visual patch grid")
            if any(im.width * im.height > self.settings.policy.max_pixels for im in images):
                raise ValueError("prepared image exceeds configured pixel limit")
        else:
            images = [
                prepare_training_image(
                    image,
                    self.settings.policy.max_pixels,
                    self.processor.image_processor.patch_size,
                )
                for image in images
            ]
        prompt = native_chat_prompt(
            self.processor, [message.model_dump() for message in request.messages], images
        )
        return images, prompt, [value.sha256 for value in request.images]

    def identity(self):
        return {
            "model_identity": self.model_identity,
            "prompt_profile": self.builder.profile,
            "stop_enabled": self.builder.stop_enabled,
            "implementation_sha256": {
                name: file_sha256(Path(__file__).parent / name)
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
                "max_images": self.settings.policy.max_images,
                "lora_adapter": str(self.settings.policy.lora_adapter)
                if self.settings.policy.lora_adapter
                else None,
                "seed": self.settings.policy.seed,
            },
            "queue": self.queue.snapshot(),
            "prompt_hashes": {
                name: file_sha256(self.builder.template_root / (name + ".jinja"))
                for name in ("system", "user")
            },
        }

    async def _live_inputs(self, request):
        images = await asyncio.gather(
            *(self._image(observation, request.camera) for observation in request.observations)
        )
        indices = [observation.event_index for observation in request.observations]
        prompt = self.builder.native_prompt(
            self.processor,
            request.instruction,
            self.settings.cameras[request.camera],
            indices,
            images,
        )
        return (
            images,
            prompt,
            [
                f"{observation.session_id}:{observation.event_index}"
                for observation in request.observations
            ],
        )

    async def _inputs(self, request):
        """Native prompt and stable image identities for either request route."""
        if isinstance(request, RecordedGenerateRequest):
            return await asyncio.to_thread(self._recorded_inputs, request)
        return await self._live_inputs(request)

    async def _generate(self, request):
        SamplingParams = optional_module("vllm").SamplingParams

        arrival = time.monotonic()
        images, prompt, image_ids = await self._inputs(request)
        prepared = time.monotonic()
        inputs = {
            "prompt": prompt,
            "multi_modal_data": {"image": images},
            "multi_modal_uuids": {"image": image_ids},
        }
        sampling = SamplingParams(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            seed=request.seed,
            logprobs=1 if self.settings.policy.return_logprobs else None,
        )
        final = None
        try:
            async for output in self.engine.generate(
                inputs, sampling, request_id=request.request_id, lora_request=self.adapter
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
            "actual_visual_tokens": list(final.prompt_token_ids or []).count(self.image_token_id),
            "finish_reason": completion.finish_reason,
            "timing_seconds": {
                "images_and_prompt": prepared - arrival,
                "engine_including_queue": time.monotonic() - prepared,
            },
            "valid": False,
            "stop": False,
            "trajectory": None,
        }
        if (
            isinstance(request, RecordedGenerateRequest)
            and request.expected_visual_tokens is not None
            and result["actual_visual_tokens"] != request.expected_visual_tokens
        ):
            raise ValueError("vLLM changed the prepared visual-token allocation")
        try:
            parsed = parse_prediction(completion.text, self.builder.profile)
            result.update(valid=True, stop=parsed.stop, trajectory=parsed.trajectory)
        except (ValueError, TypeError):
            result["output_error"] = "invalid output for " + self.builder.profile
        return result
