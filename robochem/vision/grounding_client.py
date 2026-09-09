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
from typing import Dict, List, Optional, Tuple

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

    def _post_segment(
        self,
        images: Dict[int, np.ndarray],
        prompt: str,
        conf: float = None,
        return_all: bool = False,
    ) -> dict:
        encoded = {}
        for cam_id, img in images.items():
            ok, buf = cv2.imencode(".jpg", img)
            if not ok:
                continue
            encoded[str(cam_id)] = base64.b64encode(buf.tobytes()).decode("ascii")

        body = {"prompt": prompt, "images": encoded, "return_all": bool(return_all)}
        if conf is not None:
            body["conf"] = conf

        req = Request(
            f"{self.url}/segment",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        try:
            with urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (URLError, HTTPError, OSError) as e:
            raise GroundingUnavailable(f"Grounding request failed: {e}")

    def segment(
        self,
        images: Dict[int, np.ndarray],
        prompt: str,
        conf: float = None,
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, float]]:
        """
        Segment a concept across several camera images (best mask per camera).

        Args:
            images: Camera ID -> BGR image
            prompt: Short noun phrase, e.g. "white paper cup"
            conf: Optional confidence override

        Returns:
            (masks, confidences) keyed by camera ID. Cameras where the concept
            was not found are absent from both dicts.
        """
        payload = self._post_segment(images, prompt, conf=conf, return_all=False)
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

    def segment_instances(
        self,
        images: Dict[int, np.ndarray],
        prompt: str,
        conf: float = None,
        iou_nms: float = 0.5,
    ) -> Dict[int, List[dict]]:
        """
        Segment every instance of a concept (not just the top score).

        Returns:
            camera_id -> list of {"mask", "score", "box", "area_px"}
        """
        payload = self._post_segment(images, prompt, conf=conf, return_all=True)
        out: Dict[int, List[dict]] = {}

        for cam_key, entry in (payload.get("results") or {}).items():
            cam_id = int(cam_key)
            if not entry.get("found"):
                reason = entry.get("reason") or entry.get("error") or "not found"
                print(f"[Grounding] cam {cam_id}: {prompt!r} -> {reason}")
                continue

            raw = entry.get("instances")
            if not raw:
                # Older service without instances: fall back to the single mask.
                raw = [{
                    "mask": entry["mask"],
                    "score": entry.get("score"),
                    "box": entry.get("box"),
                    "area_px": entry.get("area_px"),
                }]

            instances = []
            for inst in raw:
                instances.append({
                    "mask": self._decode_mask(inst["mask"]),
                    "score": inst.get("score"),
                    "box": inst.get("box"),
                    "area_px": inst.get("area_px"),
                })
            instances = _nms_instances(instances, iou_thresh=iou_nms)
            print(f"[Grounding] cam {cam_id}: {len(instances)} instance(s) for {prompt!r}")
            out[cam_id] = instances

        return out

    @staticmethod
    def _decode_mask(b64: str) -> np.ndarray:
        buf = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("could not decode mask")
        return img > 127


def _box_iou(a, b) -> float:
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _nms_instances(instances: List[dict], iou_thresh: float = 0.5) -> List[dict]:
    """Drop overlapping duplicate masks, keeping higher score."""
    if len(instances) <= 1:
        return instances
    order = sorted(
        range(len(instances)),
        key=lambda i: (instances[i].get("score") or 0.0),
        reverse=True,
    )
    keep = []
    suppressed = set()
    for i in order:
        if i in suppressed:
            continue
        keep.append(instances[i])
        for j in order:
            if j == i or j in suppressed:
                continue
            if _box_iou(instances[i].get("box"), instances[j].get("box")) >= iou_thresh:
                suppressed.add(j)
    return keep
