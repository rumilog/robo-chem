"""
Client for the open-vocabulary grounding service.

The arm runs under Python 3.8 (frankapy), which is too old for the grounding
models, so they live in a separate Python 3.10 process. This client speaks to
it over localhost and returns masks in the same shape SceneAnalyzer produced
before, so the rest of the vision pipeline is unchanged.

The service backend (GroundingDINO or SAM 3) is chosen when the service is
started; nothing here needs to know which is running.
"""

import base64
import json
from typing import Dict, Tuple

import cv2
import numpy as np

from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError


class GroundingUnavailable(RuntimeError):
    """Raised when the grounding service cannot be reached."""


class GroundingClient:
    """
    Thin HTTP client for the grounding service.
    """

    def __init__(self, url: str = "http://127.0.0.1:5005", timeout: float = 180.0):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict:
        try:
            with urlopen(f"{self.url}/health", timeout=5) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (URLError, HTTPError, OSError) as e:
            raise GroundingUnavailable(
                f"Grounding service not reachable at {self.url}: {e}. "
                f"Start it with: perception_env/bin/python "
                f"perception_service/grounding_service.py "
                f"--backend sam3 --model weights/sam3.pt"
            )

    def segment(
        self,
        images: Dict[int, np.ndarray],
        prompt: str,
        conf: float = None,
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, float]]:
        """
        Segment a concept across several camera images.

        Args:
            images: Camera ID -> BGR image
            prompt: Short noun phrase, e.g. "white paper cup"
            conf: Optional confidence override

        Returns:
            (masks, confidences) keyed by camera ID. Cameras where the concept
            was not found are absent from both dicts.
        """
        encoded = {}
        for cam_id, img in images.items():
            ok, buf = cv2.imencode(".jpg", img)
            if not ok:
                continue
            encoded[str(cam_id)] = base64.b64encode(buf.tobytes()).decode("ascii")

        body = {"prompt": prompt, "images": encoded}
        if conf is not None:
            body["conf"] = conf

        req = Request(
            f"{self.url}/segment",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        try:
            with urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except (URLError, HTTPError, OSError) as e:
            raise GroundingUnavailable(f"Grounding request failed: {e}")

        masks: Dict[int, np.ndarray] = {}
        confidences: Dict[int, float] = {}

        for cam_key, entry in (payload.get("results") or {}).items():
            cam_id = int(cam_key)
            if not entry.get("found"):
                reason = entry.get("reason") or entry.get("error") or "not found"
                print(f"[Grounding] cam {cam_id}: {prompt!r} -> {reason}")
                continue

            masks[cam_id] = self._decode_mask(entry["mask"])
            confidences[cam_id] = entry.get("score")

            if entry.get("num_instances", 1) > 1:
                print(f"[Grounding] cam {cam_id}: {entry['num_instances']} candidates for "
                      f"{prompt!r}; using the highest-scoring one")

        return masks, confidences

    @staticmethod
    def _decode_mask(b64: str) -> np.ndarray:
        buf = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("could not decode mask")
        return img > 127
