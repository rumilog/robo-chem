"""
Associate printed / handwritten reagent labels with cup instances.

SAM only gives category masks ("white paper cup"). Scoop targets are cups
sitting on labelled paper squares (CITRIC ACID, BAKING SODA, …). This module:

  1. Takes every cup instance from the grounding service
  2. Asks GPT-4o which label sits under each numbered cup (one call per camera)
  3. Picks the instance whose label fuzzy-matches the requested reagent name

Skills then fuse that cup's mask in 3D the same way as any other object.
"""

from __future__ import annotations

import base64
import json
import re
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# Category phrases that should go straight to SAM, not through label matching.
DIRECT_SAM_QUERIES = {
    "white paper cup",
    "paper cup",
    "plastic beaker",
    "beaker",
    "clear cup",
    "clear plastic cup",
    "larger spoon",
    "large spoon",
    "small spoon",
    "spoon",
    "scoop",
}


def normalize_label(text: str) -> str:
    """Lowercase, strip punctuation/spaces for fuzzy compare."""
    if not text:
        return ""
    t = text.lower().strip()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def labels_match(requested: str, observed: str) -> bool:
    """
    True if the observed label names the same reagent as requested.

    Allows substring either way so "citric acid" matches "CITRIC ACID"
    and "red cabbage" matches "RED CABBAGE POWDER".
    """
    a, b = normalize_label(requested), normalize_label(observed)
    if not a or not b:
        return False
    if a == b:
        return True
    if a in b or b in a:
        return True
    # Token overlap: at least half of the longer phrase's tokens.
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return False
    overlap = len(ta & tb)
    return overlap >= max(1, (min(len(ta), len(tb)) + 1) // 2)


# Utensil words: any query containing one of these is a physical-object
# query, never a reagent label. Unlike "cup"/"beaker", these never appear
# in "cup labeled X" phrasing, so matching on the word alone is safe.
_DIRECT_SAM_KEYWORDS = {"spoon", "scoop"}


def looks_like_label_query(object_name: str) -> bool:
    """
    Heuristic: reagent / content names go through label matching; raw
    SAM category phrases do not.
    """
    key = normalize_label(object_name)
    if not key:
        return False
    if key in {normalize_label(q) for q in DIRECT_SAM_QUERIES}:
        return False
    if set(key.split()) & _DIRECT_SAM_KEYWORDS:
        return False
    # "cup labeled X" / "labelled X" still count as label queries.
    return True


def _encode_jpeg(image_bgr: np.ndarray, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("could not encode jpeg")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _ensure_box(inst: dict, h: int, w: int) -> List[float]:
    box = inst.get("box")
    if box is not None and len(box) == 4:
        return [float(v) for v in box]
    mask = inst["mask"]
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0.0, 0.0, float(w - 1), float(h - 1)]
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def annotate_instances(image_bgr: np.ndarray, instances: List[dict]) -> np.ndarray:
    """Draw numbered boxes on a copy of the image for the VLM."""
    vis = image_bgr.copy()
    h, w = vis.shape[:2]
    for i, inst in enumerate(instances):
        x1, y1, x2, y2 = _ensure_box(inst, h, w)
        x1, y1 = int(max(0, x1)), int(max(0, y1))
        x2, y2 = int(min(w - 1, x2)), int(min(h - 1, y2))
        color = (0, 255, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        tag = f"{i}"
        cv2.putText(
            vis, tag, (x1 + 4, max(18, y1 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA,
        )
        cv2.putText(
            vis, tag, (x1 + 4, max(18, y1 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA,
        )
    return vis


def read_labels_vlm(vlm_client, image_bgr: np.ndarray,
                    instances: List[dict], model: str = "gpt-4o") -> List[Optional[str]]:
    """
    Ask the VLM what handwritten/printed label sits under each numbered cup.

    Returns a list parallel to ``instances`` (None if unread).
    """
    if not instances:
        return []

    annotated = annotate_instances(image_bgr, instances)
    b64 = _encode_jpeg(annotated)
    n = len(instances)

    prompt = f"""This lab photo has {n} white paper cups marked with yellow boxes numbered 0..{n - 1}.
Each cup sits on a white paper square with a handwritten reagent label
(e.g. BAKING SODA, CITRIC ACID, RED CABBAGE POWDER, Water).

Read the label associated with EACH numbered cup. The text is on the paper
under/beside the cup, not on the cup wall.

Return ONLY JSON:
{{"cups": [{{"id": 0, "label": "BAKING SODA"}}, ...]}}

Use null for label if you cannot read it. Include every id from 0 to {n - 1}.
"""

    response = vlm_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                ],
            }
        ],
        max_tokens=400,
        temperature=0,
    )
    text = response.choices[0].message.content or ""
    return _parse_cup_labels(text, n)


def _parse_cup_labels(text: str, n: int) -> List[Optional[str]]:
    labels: List[Optional[str]] = [None] * n
    # Strip markdown fences if present.
    cleaned = text.strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                cleaned = part
                break
    try:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]
        data = json.loads(cleaned)
    except Exception:
        print(f"[LabelResolver] Could not parse VLM labels: {text[:200]!r}")
        return labels

    cups = data.get("cups") if isinstance(data, dict) else None
    if not isinstance(cups, list):
        return labels

    for entry in cups:
        if not isinstance(entry, dict):
            continue
        try:
            idx = int(entry.get("id"))
        except (TypeError, ValueError):
            continue
        if 0 <= idx < n:
            lab = entry.get("label")
            if lab is None:
                labels[idx] = None
            else:
                s = str(lab).strip()
                labels[idx] = None if (not s or s.lower() == "null") else s
    return labels


def pick_matching_instance(
    instances: List[dict],
    labels: List[Optional[str]],
    requested: str,
) -> Tuple[Optional[dict], Optional[str]]:
    """Return (instance, observed_label) for the best label match, or (None, None)."""
    matches = []
    for inst, lab in zip(instances, labels):
        if lab is None:
            continue
        if labels_match(requested, lab):
            matches.append((inst, lab, inst.get("score") or 0.0))
    if not matches:
        return None, None
    matches.sort(key=lambda t: t[2], reverse=True)
    inst, lab, _ = matches[0]
    return inst, lab


class LabelResolver:
    """
    Resolve a reagent name to per-camera cup masks via SAM instances + VLM OCR.
    """

    def __init__(self, vlm_client=None, model: str = "gpt-4o"):
        self.vlm_client = vlm_client
        self.model = model

    def resolve(
        self,
        images: Dict[int, np.ndarray],
        instances_by_cam: Dict[int, List[dict]],
        requested_label: str,
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, float], Dict[int, str]]:
        """
        Args:
            images: cam_id -> BGR frame
            instances_by_cam: cam_id -> list of instance dicts from grounding
            requested_label: e.g. "citric acid"

        Returns:
            (masks, confidences, observed_labels) for cameras that matched
        """
        if self.vlm_client is None:
            from openai import OpenAI
            self.vlm_client = OpenAI()

        masks: Dict[int, np.ndarray] = {}
        confidences: Dict[int, float] = {}
        observed: Dict[int, str] = {}

        for cam_id, instances in instances_by_cam.items():
            if cam_id not in images or not instances:
                continue
            try:
                labels = read_labels_vlm(
                    self.vlm_client, images[cam_id], instances, model=self.model
                )
            except Exception as e:
                print(f"[LabelResolver] cam {cam_id}: VLM label read failed: {e}")
                continue

            readable = [
                f"{i}:{lab!r}" for i, lab in enumerate(labels) if lab
            ]
            print(f"[LabelResolver] cam {cam_id}: labels [{', '.join(readable) or 'none'}]")

            inst, lab = pick_matching_instance(instances, labels, requested_label)
            if inst is None:
                print(f"[LabelResolver] cam {cam_id}: no cup matched "
                      f"{requested_label!r}")
                continue

            masks[cam_id] = inst["mask"]
            confidences[cam_id] = float(inst.get("score") or 0.0)
            observed[cam_id] = lab
            print(f"[LabelResolver] cam {cam_id}: matched {requested_label!r} "
                  f"-> {lab!r} (score={confidences[cam_id]:.2f})")

        return masks, confidences, observed
