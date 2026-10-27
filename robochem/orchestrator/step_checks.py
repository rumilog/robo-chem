"""
Did the step do what it said? A look at the target after each transfer.

A skill's ``success`` means its motions ran: pour reached its angle, dump
tipped the bowl. Nothing in it says the liquid or the powder ended up in the
container. This module takes one camera frame of the target after ``pour``
and ``dump`` and shows it, next to the frame of the same camera from the
skill's own pre-scan, to the strong model: did the contents change, and did
the container move or tip?

Cost per check, by design: one camera (not four), no SAM, one model call.
The "before" frame is free -- the scan the skill already made, kept by
``VisionSystem.scans``. After a dump that did not end at home the arm is
first sent home, since it is hovering over the cup; the next skill homes
before its own scan anyway.

When the cup did not move, the after frame becomes that scan's reference, so
the stir that usually follows can reuse the scan instead of rescanning.

Verdicts: ``ok`` (contents changed), ``suspect`` (no visible change, or the
container moved), ``unclear``, ``skipped``. Only with ``strict`` does a
``suspect`` fail the step; by default it is reported and the run goes on.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

from robochem.agents import config as agent_config
from robochem.agents.llm_client import image_block_from_array, structured_completion, text_block

#: Skills whose result can be seen in the target container afterwards.
CHECK_AFTER = ("pour", "dump")

_WHAT = {
    "pour": "liquid was poured into it",
    "dump": "a scoop of powder was tipped into it",
}

_SYSTEM = """You check one step of a robot's chemistry experiment from two photos
taken by the same fixed camera, before and after the step. Both are crops
around one container. Judge only that container. Small powder amounts and
clear liquids can be subtle: look for a raised liquid level, new colour,
cloudiness, powder on the surface or at the bottom, or splashes. Say "unclear"
when the view does not let you tell, rather than guessing."""

_SCHEMA = {
    "type": "object",
    "properties": {
        "contents_changed": {"type": "string", "enum": ["yes", "no", "unclear"]},
        "container_moved_or_tipped": {"type": "string", "enum": ["yes", "no", "unclear"]},
        "observation": {"type": "string"},
    },
    "required": ["contents_changed", "container_moved_or_tipped", "observation"],
    "additionalProperties": False,
}


def _crop(image, box, grow: float = 1.6):
    """``box`` grown about its centre, for some context around the container."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    hw, hh = (x2 - x1) * grow / 2.0, (y2 - y1) * grow / 2.0
    h, w = image.shape[:2]
    crop = image[int(max(0, cy - hh)):int(min(h, cy + hh)),
                 int(max(0, cx - hw)):int(min(w, cx + hw))]
    if 0 < crop.shape[1] < 384:
        # A cup is ~100 px across in these cameras; enlarge it for the model.
        import cv2
        scale = 384.0 / crop.shape[1]
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return crop


class StepChecker:
    """Looks at the target of a pour or dump once the skill has finished."""

    def __init__(self, vision, skills, *, strict: bool = False,
                 photo_dir: Optional[str] = None, client=None):
        self.vision = vision
        self.skills = skills
        self.strict = bool(strict)
        self.photo_dir = Path(photo_dir) if photo_dir else None
        self.client = client

    def check(self, call, result) -> Optional[dict]:
        """A verdict dict for ``call``, or None for a skill this does not check."""
        if call.skill not in CHECK_AFTER:
            return None
        started = time.time()
        verdict = self._check(call, result or {})
        verdict["seconds"] = round(time.time() - started, 1)
        verdict["fails_step"] = bool(self.strict and verdict["status"] == "suspect")
        return verdict

    def _check(self, call, result) -> dict:
        target = call.params.get("target_container")
        scans = getattr(self.vision, "scans", None)
        if not isinstance(target, str) or scans is None:
            return {"status": "skipped", "detail": "no named target to look at"}
        key = self.vision._scan_key(target)
        scan = scans.get(key)
        if scan is None:
            return {"status": "skipped",
                    "detail": f"no kept scan of {target!r} to compare against"}

        if call.skill == "dump" and not result.get("homed"):
            # The arm stops just above the rim, in the camera's way.
            if not self.skills.go_home():
                return {"status": "skipped", "detail": "could not home the arm to look"}

        cam = self.vision.best_scan_camera(scan)
        try:
            now = self.vision.object_localizer.capture_pointclouds(cam_ids=[cam])
        except Exception as exc:
            return {"status": "skipped", "detail": f"cam {cam} capture failed: {exc}"}
        after = now["images"].get(cam)
        if after is None:
            return {"status": "skipped", "detail": f"cam {cam} returned no frame"}

        box = scan["boxes"][cam]
        before_crop, after_crop = _crop(scan["images"][cam], box), _crop(after, box)
        photo = self._save(call, before_crop, after_crop)
        order = getattr(self.vision, "frame_color_order", "bgr")
        try:
            reply = structured_completion(
                agent="step_check",
                model=agent_config.strong_model(),
                system_prompt=_SYSTEM,
                user_content=[
                    text_block(f"The container is the {target}. During this step "
                               f"{_WHAT[call.skill]}. First photo: BEFORE the step."),
                    image_block_from_array(before_crop, color_order=order, detail="high"),
                    text_block("Second photo: AFTER the step, same camera."),
                    image_block_from_array(after_crop, color_order=order, detail="high"),
                ],
                schema=_SCHEMA,
                schema_name="step_check",
                client=self.client,
            )
        except Exception as exc:
            return {"status": "skipped", "camera": cam, "photo": photo,
                    "detail": f"model call failed: {type(exc).__name__}: {exc}"}

        changed = reply.get("contents_changed")
        moved = reply.get("container_moved_or_tipped")
        if moved == "yes":
            status = "suspect"
        elif changed == "yes":
            status = "ok"
        elif changed == "no":
            status = "suspect"
        else:
            status = "unclear"

        if moved == "no":
            # Same place, new contents: the after frame is now what the scan
            # should look like, so the next skill on this cup can reuse it.
            scan["images"] = {cam: after}
            scan["boxes"] = {cam: box}
            scan["time"] = time.time()
        else:
            scans.pop(key, None)

        return {"status": status, "camera": cam, "photo": photo,
                "contents_changed": changed, "container_moved_or_tipped": moved,
                "detail": str(reply.get("observation") or "")}

    def _save(self, call, before, after) -> Optional[str]:
        """Before and after side by side, for the trial record and the report."""
        if self.photo_dir is None:
            return None
        try:
            import cv2
            h = max(before.shape[0], after.shape[0])
            pad = lambda im: cv2.copyMakeBorder(im, 0, h - im.shape[0], 0, 4,
                                                cv2.BORDER_CONSTANT, value=(255, 255, 255))
            pair = np.hstack([pad(before), pad(after)])
            if getattr(self.vision, "frame_color_order", "bgr") == "rgb":
                pair = cv2.cvtColor(pair, cv2.COLOR_RGB2BGR)
            self.photo_dir.mkdir(parents=True, exist_ok=True)
            path = self.photo_dir / (f"s{call.subtask_index:02d}_{call.skill}_"
                                     f"before_after.jpg")
            cv2.imwrite(str(path), pair)
            return str(path)
        except Exception:
            return None
