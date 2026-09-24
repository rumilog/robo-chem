"""
Shared LLM access: structured output, vision input, honest key handling.

Merged in from robomail_Aliyah/agents/llm_client.py, which is where the project's
ChatGPT call lives. Every agent in this package goes through here so that model
selection, JSON-schema enforcement and missing-key behaviour are defined once.

Two additions for this repo:

  * :func:`image_block_from_array` -- the cage and the simulated cage hand back
    numpy frames, not files on disk, and they do not agree on channel order:
    RealSense streams bgr8 and MuJoCo renders RGB. Encoding one as the other
    sends the model a picture with its reds and blues swapped, which is exactly
    the kind of thing a scene-understanding agent quietly gets wrong rather than
    failing on. The order is therefore explicit at every call site.
  * ``max_output_tokens`` handling for models that reject ``temperature``.

Key policy: no API key is hardcoded. With none set, :class:`MissingAPIKeyError`
names the agent that needed it, and the orchestrator records which stage could
not run rather than skipping it silently.
"""

from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any, List

import numpy as np

from . import config


class MissingAPIKeyError(RuntimeError):
    """Raised when an LLM call is attempted with no API key configured."""

    def __init__(self, agent: str) -> None:
        self.agent = agent
        super().__init__(
            f"{agent}: no API key. Set {config.ENV_API_KEY} in the environment or in "
            f"the repo's .env (which may spell it 'openai_api_key'). "
            f"Optionally set {config.ENV_BASE_URL} for an OpenAI-compatible endpoint. "
            "No key is read from anywhere else or hardcoded."
        )


class LLMResponseError(RuntimeError):
    """The provider returned something that did not satisfy the requested schema."""


def encode_image(image_path) -> tuple:
    """Return ``(base64_payload, mime_type)`` for an image on disk."""
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"image not found: {path}")
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return base64.b64encode(path.read_bytes()).decode("utf-8"), mime


def image_block(image_path, detail: str = "high") -> dict:
    """Vision content block for an image file."""
    payload, mime = encode_image(image_path)
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{payload}", "detail": detail},
    }


def image_block_from_array(
    frame: np.ndarray,
    *,
    color_order: str = "bgr",
    detail: str = "high",
    max_width: int = 1024,
) -> dict:
    """
    Vision content block for a camera frame held in memory.

    Args:
        frame: HxWx3 uint8 image.
        color_order: ``"bgr"`` for a RealSense frame, ``"rgb"`` for a rendered
            one. See :attr:`robochem.vision.VisionSystem.frame_color_order`.
        detail: OpenAI image detail tier.
        max_width: Downscale wider frames. 848-wide cage frames go through
            untouched; this only guards against a caller handing over something
            much larger than the model needs.
    """
    import cv2

    if frame is None:
        raise ValueError("frame is None")
    image = np.asarray(frame)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if str(color_order).lower() == "rgb":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)   # imencode expects BGR
    if image.shape[1] > max_width:
        scale = max_width / float(image.shape[1])
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise ValueError("failed to JPEG-encode the frame")
    payload = base64.b64encode(buffer).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{payload}", "detail": detail},
    }


def text_block(text: str) -> dict:
    """Plain-text content block."""
    return {"type": "text", "text": text}


def get_client(agent: str):
    """Construct an OpenAI-compatible client, or raise a clear, actionable error."""
    key = config.api_key()
    if not key:
        raise MissingAPIKeyError(agent)
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - environment, not logic
        raise RuntimeError(
            "The openai package is not installed. Run: pip install -r requirements.txt"
        ) from exc
    kwargs: dict = {"api_key": key}
    url = config.base_url()
    if url:
        kwargs["base_url"] = url
    return OpenAI(**kwargs)


def structured_completion(
    *,
    agent: str,
    model: str,
    system_prompt: str,
    user_content: List[dict],
    schema: dict,
    schema_name: str,
    temperature=0.0,
    client: Any = None,
) -> dict:
    """
    Call the model and return a dict matching ``schema``.

    Uses strict JSON-schema structured output. If the provider rejects
    ``response_format=json_schema`` we retry in JSON-object mode with the schema
    inlined in the system prompt; either way the result is re-checked by the
    caller, so a model that invents a skill name produces a reported failure
    rather than a silent one.

    ``client`` exists so tests can inject a transport double. Production callers
    leave it None.
    """
    client = client or get_client(agent)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    request: dict = {
        "model": model,
        "messages": messages,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "schema": schema, "strict": True},
        },
    }
    if temperature is not None:
        request["temperature"] = temperature

    try:
        completion = client.chat.completions.create(**request)
    except Exception as exc:
        if _mentions(exc, "temperature"):
            # Reasoning models reject a temperature override. Drop it and retry
            # rather than making the whole tier unusable.
            request.pop("temperature", None)
            completion = client.chat.completions.create(**request)
        elif _looks_like_schema_unsupported(exc):
            request["response_format"] = {"type": "json_object"}
            request["messages"] = [
                {
                    "role": "system",
                    "content": (
                        system_prompt
                        + "\n\nRespond with a single JSON object matching this schema "
                          "exactly:\n"
                        + json.dumps(schema, indent=2)
                    ),
                },
                messages[1],
            ]
            completion = client.chat.completions.create(**request)
        else:
            raise

    raw = completion.choices[0].message.content
    if not raw:
        raise LLMResponseError(f"{agent}: model returned an empty response")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMResponseError(
            f"{agent}: model response was not valid JSON: {raw[:400]}"
        ) from exc
    if not isinstance(parsed, dict):
        raise LLMResponseError(
            f"{agent}: expected a JSON object, got {type(parsed).__name__}"
        )
    return parsed


def _mentions(exc: Exception, word: str) -> bool:
    return word in str(exc).lower()


def _looks_like_schema_unsupported(exc: Exception) -> bool:
    text = str(exc).lower()
    return "response_format" in text or "json_schema" in text or "unsupported" in text
