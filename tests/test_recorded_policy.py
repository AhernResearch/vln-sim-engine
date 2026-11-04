import base64
import hashlib
import io
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from helpers import settings
from PIL import Image
from pydantic import ValidationError

from sim_engine.policy import VllmPolicy, decode_recorded_image
from sim_engine.policy_server import create_app
from sim_engine.prompt import native_chat_prompt
from sim_engine.protocol import RecordedGenerateRequest


def payload():
    stream = io.BytesIO()
    Image.new('RGB', (32, 32), 'red').save(stream, format='PNG')
    raw = stream.getvalue()
    return {'request_id': 'test', 'messages': [{'role': 'user', 'content': 'Look <image> now'}],
            'images': [{'data_base64': base64.b64encode(raw).decode(), 'sha256': hashlib.sha256(raw).hexdigest()}]}


def test_recorded_requests_exclude_answers_and_verify_content():
    request = RecordedGenerateRequest.model_validate(payload())
    assert decode_recorded_image(request.images[0]).getpixel((0, 0)) == (255, 0, 0)
    with pytest.raises(ValueError, match='hash'):
        decode_recorded_image(request.images[0].model_copy(update={'sha256': '0'*64}))
    value = payload()
    value['messages'].append({'role': 'assistant', 'content': 'answer'})
    with pytest.raises(ValidationError):
        RecordedGenerateRequest.model_validate(value)
    value = payload()
    value['messages'][0]['content'] = 'no image'
    with pytest.raises(ValidationError):
        RecordedGenerateRequest.model_validate(value)


def test_native_recorded_prompt_preserves_text_and_image_order():
    messages = [{'role': 'system', 'content': 'system'}, {'role': 'user', 'content': 'A<image>B<image>C'}]
    processor = SimpleNamespace(apply_chat_template=lambda value, **kwargs: (value, kwargs))
    value, options = native_chat_prompt(processor, messages, ['first', 'second'])
    assert messages[-1]['content'] == 'A<image>B<image>C'
    assert [p.get('image') for p in value[-1]['content'] if p['type'] == 'image'] == ['first', 'second']
    assert options['add_generation_prompt'] is True


def test_recorded_route_reuses_policy_admission_and_response(tmp_path):
    class Backend:
        def __init__(self, config): pass
        async def start(self): pass
        async def close(self): pass
        async def generate(self, request):
            return {'request_id': request.request_id, 'text': 'result'}
    with TestClient(create_app(settings(tmp_path), backend_factory=Backend)) as client:
        response = client.post('/v1/policy/generate-recorded', json=payload())
        assert response.status_code == 200
        assert response.json() == {'request_id': 'test', 'text': 'result'}


def test_prepared_sixteen_image_inputs_preserve_pixels_and_skip_resizing(tmp_path, monkeypatch):
    config = settings(tmp_path)
    config.policy.max_images = 16
    backend = VllmPolicy(config)
    backend.processor = SimpleNamespace(
        image_processor=SimpleNamespace(patch_size=16),
        apply_chat_template=lambda messages, **kwargs: messages,
    )
    value = payload()
    value.update(prepared=True, expected_visual_tokens=16)
    value['images'] *= 16
    value['messages'][0]['content'] = '<image>' * 16
    request = RecordedGenerateRequest.model_validate(value)
    def forbidden(*args, **kwargs):
        raise AssertionError('prepared images must not be resized a second time')
    monkeypatch.setattr('sim_engine.policy.prepare_training_image', forbidden)
    images, prompt, identities = backend._recorded_inputs(request)
    assert len(images) == len(identities) == 16
    assert all(image.size == (32, 32) and image.getpixel((0, 0)) == (255, 0, 0) for image in images)
    assert len(prompt[0]['content']) == 16
    config.policy.max_images = 8
    with pytest.raises(ValueError, match='image count'):
        backend._recorded_inputs(request)
