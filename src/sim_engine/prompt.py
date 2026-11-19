"""Frozen training prompt content rendered into the model's native chat template."""

import json
import math
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from PIL import Image

from sim_engine.core.optional import optional_module


def rescale_training_image(image, max_pixels):
    """Match Swift 4.5.3's pre-processor torchvision Resize on PIL RGB images."""
    if image.width * image.height <= max_pixels:
        return image
    ratio = image.width / image.height
    height = math.sqrt(max_pixels / ratio)
    return image.resize((int(height * ratio), int(height)), resample=Image.Resampling.BILINEAR)


def prepare_training_image(image, max_pixels, patch_size=16):
    fetch_image = optional_module("qwen_vl_utils").fetch_image

    resized = rescale_training_image(image, max_pixels)
    return fetch_image(
        {"image": resized, "max_pixels": max_pixels, "min_pixels": 4 * (patch_size * 2) ** 2},
        image_patch_size=patch_size,
    )


def json_value(value):
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def native_chat_prompt(processor, messages, images):
    """Serialize a causal text/image prefix without changing its text."""
    messages = [dict(m) for m in messages]
    pieces = messages[-1]["content"].split("<image>")
    if len(pieces) - 1 != len(images):
        raise ValueError("prompt/image count mismatch")
    content = []
    for i, text in enumerate(pieces):
        if text:
            content.append({"type": "text", "text": text})
        if i < len(images):
            content.append({"type": "image", "image": images[i]})
    messages[-1]["content"] = content
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


class PromptBuilder:
    def __init__(self, profile="trajectory_v1"):
        if profile not in ("trajectory_v1", "navigation_v2"):
            raise ValueError("unknown prompt profile")
        self.profile = profile
        self.stop_enabled = profile == "navigation_v2"
        self.template_root = Path(__file__).parent / "templates"
        if self.stop_enabled:
            self.template_root /= "navigation_v2"
        self.jinja = Environment(
            loader=FileSystemLoader(self.template_root),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
        )
        self.jinja.filters["json"] = json_value

    def messages(self, instruction, camera, indices):
        context = {
            "instruction": instruction,
            "robot": camera.robot(),
            "profile_id": "progress_se2_q8_complete_v1",
            "observations": [{"event_index": i} for i in indices],
        }
        if self.stop_enabled:
            robot = {"robot": "ground", "camera_height_m": camera.height_m}
            if camera.pitch_down_rad:
                degrees = math.degrees(camera.pitch_down_rad)
                robot["camera_pitch_deg"] = (
                    round(degrees) if abs(degrees - round(degrees)) < 1e-8 else degrees
                )
            else:
                robot["camera_view"] = camera.view
            context["robot_camera"] = robot
        return [
            {"role": role, "content": self.jinja.get_template(name + ".jinja").render(**context).strip()}
            for role, name in [("system", "system"), ("user", "user")]
        ]

    def native_prompt(self, processor, instruction, camera, indices, images):
        messages = self.messages(instruction, camera, indices)
        return native_chat_prompt(processor, messages, images)
