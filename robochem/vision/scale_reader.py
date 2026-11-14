"""
Read a digital scale's display from the cage cameras.

The cage frames are 848x480; the scale's LCD is ~15 px tall digits in them, and
from cam 5 it is turned ~70 deg. Handing gpt-4.1 the whole scale crop gave a
confident "174.00" for a display showing -1140.0 (2026-11-11,
run_20261111_162957). So, per camera:

  1. a COLOUR-ONLY 1920x1080 stream (one camera at a time: the USB wedge,
     progress.md), cropped to the located scale's box (the 848x480 scan box,
     scaled; same field of view);
  2. SAM 3 finds the LCD inside that crop: prompt "digital display" (0.94 on
     cam 2, 0.96 on cam 5; "lcd screen" missed cam 5, "scale display" took the
     whole scale on cam 2);
  3. the LCD's quadrilateral is warped flat with its long side horizontal, 15%
     margin (the minus sign sits at the edge), upscaled to 640 px wide;
  4. the model gets it both ways up (the warp cannot tell 0 from 180 deg; from
     cam 5 it came out upside down), each with a contrast-stretched copy, says
     which is upright and returns the DIGITS only;
  5. the minus sign is a separate yes/no call on the left strip of the upright
     display (the whole-display call kept dismissing it), and the decimal point
     is placed by the scale's known format (``decimals``): it is ~1 px here.
     The number is composed from those, not taken from the model;
  6. ``frames`` frames per camera, majority vote. A camera that sees no display,
     or a blank one, is not asked again. No majority -> no reading.

The model is the repo's strong model (``ROBOCHEM_MODEL_STRONG``, gpt-4.1) via
the same OpenAI key as the label reader.
"""

from __future__ import annotations

import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

SCAN_WH = (848, 480)
READ_WH = (1920, 1080)
DISPLAY_PROMPT = "digital display"

_SYSTEM = (
    "You read a 7-segment LCD on a digital scale, photographed by a fixed camera "
    "and digitally straightened. You get the same display twice: FIRST as "
    "straightened and SECOND rotated 180 degrees; exactly one of them is upright "
    "(digits not upside down). The scale RIGHT-ALIGNS its number: on the upright "
    "image the last digit is at the RIGHT end of the display and any unused cells "
    "are blank on the LEFT. If the digits sit at the left end of an image, that "
    "image is the upside-down one. Each image is the display as captured (top) and "
    "contrast-stretched (bottom). Segments are dark on a grey background and may "
    "be faint. The display can show a minus sign (a single short horizontal bar "
    "left of the digits), a decimal point, and up to 5-6 digits; a unit (g, oz, "
    "lb, ml) only if printed on the display. Decide which image is upright, then "
    "list the characters left to right as you see them, then the number. Only "
    "report what you can see. If the display is blank, off or unreadable, say so "
    "and give reading null. Never guess a digit. "
    "The MINUS SIGN: on the upright image it is a short horizontal bar at "
    "mid-height LEFT of the first digit (on the upside-down one it shows at the "
    "right). It is part of the reading -- do not dismiss it as a stray segment; "
    "report it in minus_sign. Report digits only (no point, no sign) in digits. "
    "0 versus 8: an 8 has a complete horizontal MIDDLE BAR running across from the "
    "left side to the right side at mid-height. A 0 has none: the left and right "
    "sides of a 0 are each two vertical segments with a small gap where they meet "
    "at mid-height, and those gaps are NOT a bar. For every digit, look at the "
    "middle and report middle_bar in digit_cells."
)

_SCHEMA = {
    "type": "object",
    "properties": {
        "display_state": {"type": "string",
                          "enum": ["showing_digits", "blank_or_off", "unreadable"]},
        "upright": {"type": "string", "enum": ["first", "second", "unclear"]},
        "characters": {"type": "string",
                       "description": "Characters left to right separated by spaces, "
                                      "e.g. '- 1 1 4 0 . 0'"},
        "minus_sign": {"type": "string", "enum": ["yes", "no", "unclear"]},
        "digits": {"type": ["string", "null"],
                   "description": "Only the digits, left to right, e.g. '11400'"},
        "digit_cells": {
            "type": "array",
            "description": "One entry per digit, left to right in the upright image",
            "items": {
                "type": "object",
                "properties": {
                    "digit": {"type": "string"},
                    "middle_bar": {"type": "string", "enum": ["yes", "no", "unclear"],
                                   "description": "A dark horizontal bar across the middle of this digit"},
                },
                "required": ["digit", "middle_bar"],
                "additionalProperties": False,
            },
        },
        "decimal_point_after_digit": {"type": ["integer", "null"],
                   "description": "How many digits come before the decimal point, if a point is visible"},
        "reading": {"type": ["string", "null"],
                    "description": "The display as one string, e.g. '-1140.0'"},
        "value": {"type": ["number", "null"]},
        "unit": {"type": ["string", "null"]},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "notes": {"type": "string"},
    },
    "required": ["display_state", "upright", "characters", "minus_sign", "digits",
                 "digit_cells", "decimal_point_after_digit", "reading", "value", "unit",
                 "confidence", "notes"],
    "additionalProperties": False,
}


def capture_colors(cam_id: int, n: int = 3, gap: float = 0.4,
                   width: int = READ_WH[0], height: int = READ_WH[1],
                   warmup: int = 30) -> List[np.ndarray]:
    """``n`` BGR colour frames, ``gap`` s apart, from cage camera ``cam_id``; no depth."""
    import pyrealsense2 as rs
    from robomail.vision.cam_utils import get_cam_info

    serial = get_cam_info(cam_id)[2]
    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
    pipe.start(cfg)
    try:
        # Auto-exposure needs ~1 s to settle after the stream starts.
        for _ in range(warmup):
            pipe.wait_for_frames()
        frames = []
        for i in range(max(1, n)):
            if i:
                t_end = time.time() + gap
                while time.time() < t_end:      # keep draining, or frames go stale
                    pipe.wait_for_frames()
            frame = pipe.wait_for_frames().get_color_frame()
            frames.append(np.asanyarray(frame.get_data()).copy())
        return frames
    finally:
        pipe.stop()


def crop_box(image: np.ndarray, box, grow: float = 1.2, box_wh=SCAN_WH) -> np.ndarray:
    """``box`` (x1, y1, x2, y2 in ``box_wh`` pixels) grown about its centre."""
    h, w = image.shape[:2]
    sx, sy = w / float(box_wh[0]), h / float(box_wh[1])
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0 * sx, (y1 + y2) / 2.0 * sy
    hw, hh = (x2 - x1) * sx * grow / 2.0, (y2 - y1) * sy * grow / 2.0
    return image[int(max(0, cy - hh)):int(min(h, cy + hh)),
                 int(max(0, cx - hw)):int(min(w, cx + hw))].copy()


def enhance(image: np.ndarray) -> np.ndarray:
    """CLAHE on lightness, for faint dark segments on a grey LCD."""
    import cv2

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def find_display(image: np.ndarray, grounding, cam_id: int = 0):
    """The LCD's quadrilateral (4x2 float32, image px) in ``image``, or None."""
    import cv2

    masks, scores = grounding.segment({cam_id: image}, DISPLAY_PROMPT)
    mask = masks.get(cam_id)
    if mask is None:
        return None, None
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 200:
        return None, None
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    quad = None
    # A perspective quad if the outline simplifies to 4 corners, else the
    # min-area rectangle.
    for eps in (0.02, 0.03, 0.04, 0.06, 0.08):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(np.float32)
            break
    if quad is None:
        quad = cv2.boxPoints(cv2.minAreaRect(cnt)).astype(np.float32)
    return quad, scores.get(cam_id)


def rectify(image: np.ndarray, quad: np.ndarray, margin: float = 0.15,
            width: int = 640) -> np.ndarray:
    """Warp ``quad`` flat, long side horizontal, ``margin`` added on every side."""
    import cv2

    ctr = quad.mean(axis=0)
    quad = quad[np.argsort(np.arctan2(quad[:, 1] - ctr[1], quad[:, 0] - ctr[0]))]
    edges = [np.linalg.norm(quad[(i + 1) % 4] - quad[i]) for i in range(4)]
    if edges[0] + edges[2] < edges[1] + edges[3]:
        quad = np.roll(quad, -1, axis=0)
        edges = edges[1:] + edges[:1]
    quad = ctr + (quad - ctr) * (1.0 + 2 * margin)
    long_side, short_side = (edges[0] + edges[2]) / 2.0, (edges[1] + edges[3]) / 2.0
    w, h = int(width), int(round(width * short_side / max(long_side, 1.0)))
    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    m = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(image, m, (w, h), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)


def _panel(image: np.ndarray) -> np.ndarray:
    """As captured over contrast-stretched, with a white bar between."""
    bar = np.full((6, image.shape[1], 3), 255, np.uint8)
    return np.vstack([image, bar, enhance(image)])


def compose_value(reply: dict, decimals: Optional[int]) -> Optional[float]:
    """
    The number from the model's digits and sign, the point placed by ``decimals``.

    The decimal point is ~1 px in these cameras (2026-11-11: invisible on cam 5,
    where the digits themselves read fine), so the scale's fixed format places
    it. With ``decimals`` None, a point the model saw is used, else its value.
    """
    digits = "".join(ch for ch in str(reply.get("digits") or "") if ch.isdigit())
    if not digits:
        return reply.get("value")
    if decimals is not None:
        number = int(digits) / (10 ** int(decimals))
    elif reply.get("decimal_point_after_digit") is not None:
        k = int(reply["decimal_point_after_digit"])
        number = float(digits[:k] + "." + digits[k:]) if 0 < k < len(digits) else float(digits)
    else:
        return reply.get("value")
    return -number if reply.get("minus_sign") == "yes" else number


def fix_zero_eight(reply: dict) -> None:
    """
    Rebuild ``digits`` from the per-digit middle-bar answers: 0 or 8 is decided
    by middle_bar, not by the model's digit.

    2026-11-13 (run_20261113_182846, empty scale, display "0.0"): cam 5 read
    "8 8" on 3/3 frames. The two vertical segments on each side of a 0 meet at
    mid-height with a gap, and the model took the gaps for an 8's middle bar.
    Asked about the bar per digit, it separates the two.
    """
    cells = reply.get("digit_cells") or []
    if not cells:
        return
    rebuilt = []
    for cell in cells:
        d = "".join(ch for ch in str(cell.get("digit") or "") if ch.isdigit())[:1]
        if not d:
            continue
        bar = cell.get("middle_bar")
        if d in "08" and bar in ("yes", "no"):
            d = "8" if bar == "yes" else "0"
        rebuilt.append(d)
    if rebuilt:
        reply["digits_model"] = reply.get("digits")
        reply["digits"] = "".join(rebuilt)


def read_display(display: np.ndarray, *, label: str = "", client=None,
                 model: Optional[str] = None, decimals: Optional[int] = None) -> dict:
    """One model call on one straightened display. Returns the schema's fields."""
    import cv2
    from robochem.agents import config as agent_config
    from robochem.agents.llm_client import (image_block_from_array, structured_completion,
                                            text_block)

    if decimals is not None:
        label += (f". This scale always shows exactly {int(decimals)} digit(s) after "
                  f"the decimal point; the point may be too small to see")
    reply = structured_completion(
        agent="scale_reader",
        model=model or agent_config.strong_model(),
        system_prompt=_SYSTEM,
        user_content=[
            text_block(f"{label}. FIRST (as straightened):"),
            image_block_from_array(_panel(display), detail="high"),
            text_block("SECOND (rotated 180 degrees):"),
            image_block_from_array(_panel(cv2.rotate(display, cv2.ROTATE_180)),
                                   detail="high"),
        ],
        schema=_SCHEMA,
        schema_name="scale_reading",
        client=client,
    )
    if reply.get("display_state") == "showing_digits":
        fix_zero_eight(reply)
        if reply.get("upright") in ("first", "second"):
            upright = (display if reply["upright"] == "first"
                       else cv2.rotate(display, cv2.ROTATE_180))
            reply["minus_sign_whole_view"] = reply.get("minus_sign")
            reply["minus_sign"] = read_minus(upright, client=client, model=model)
        reply["model_value"] = reply.get("value")
        reply["value"] = compose_value(reply, decimals)
    return reply


_MINUS_SYSTEM = (
    "This is the LEFT end of an upright 7-segment LCD on a scale, up to its first "
    "digit. A negative reading shows a minus sign: one short horizontal dark bar "
    "at mid-height, left of the first digit, separate from it. Is there one? "
    "Answer no if you see only the display background and edges."
)
_MINUS_SCHEMA = {
    "type": "object",
    "properties": {"minus_sign": {"type": "string", "enum": ["yes", "no", "unclear"]},
                   "notes": {"type": "string"}},
    "required": ["minus_sign", "notes"],
    "additionalProperties": False,
}


def read_minus(upright: np.ndarray, *, client=None, model: Optional[str] = None,
               strip: float = 0.30) -> str:
    """
    "yes" / "no" / "unclear": is there a minus left of the first digit?

    Asked on its own, on the left ``strip`` of the upright display, because in
    the whole-display call gpt-4.1 dismissed the bar 4/5 times (2026-11-11, a
    display showing -1140.0: "a segment that could be mistaken for a minus
    sign"). This question: 3/3 yes on that strip, 3/3 no with the bar painted
    over with LCD background.
    """
    import cv2
    from robochem.agents import config as agent_config
    from robochem.agents.llm_client import image_block_from_array, structured_completion

    left = upright[:, :max(8, int(upright.shape[1] * strip))]
    left = cv2.resize(left, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    try:
        reply = structured_completion(
            agent="scale_reader", model=model or agent_config.strong_model(),
            system_prompt=_MINUS_SYSTEM,
            user_content=[image_block_from_array(left, detail="high")],
            schema=_MINUS_SCHEMA, schema_name="minus_sign", client=client)
        return str(reply.get("minus_sign") or "unclear")
    except Exception as exc:
        print(f"[Scale] minus-sign check failed ({exc})")
        return "unclear"


def _vote(replies: List[dict]):
    """(value, n, of) for the value most frames agree on, or (None, 0, n_read)."""
    values = [round(float(r["value"]), 3) for r in replies
              if r.get("display_state") == "showing_digits" and r.get("value") is not None]
    if not values:
        return None, 0, 0
    value, n = Counter(values).most_common(1)[0]
    return value, n, len(values)


def read_scale(cam_ids: List[int], boxes: Optional[Dict[int, tuple]] = None,
               out_dir: Optional[Path] = None, tag: str = "read", frames: int = 3,
               grounding=None, decimals: Optional[int] = None,
               stop_when_unanimous: bool = True) -> dict:
    """
    Read the display from each camera in turn and combine.

    ``cam_ids`` is an order of preference: with ``stop_when_unanimous`` the
    first camera whose frames all agree ends the read (each camera costs ~6.5 s
    of capture: a 1080p stream start, 30 warm-up frames, 3 frames).

    A camera's value needs a strict majority of its ``frames`` reads. Across
    cameras the value with the most agreeing frames wins. Returns
    {"value", "unit", "reading", "cameras": {cam: {...}}, "agree"}; value None
    if no camera reached a majority.
    """
    import cv2

    if grounding is None:
        from robochem.vision.grounding_client import GroundingClient
        grounding = GroundingClient(os.environ.get("GROUNDING_URL", "http://127.0.0.1:5005"))
    boxes = boxes or {}
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    per_cam = {}
    for cam in cam_ids:
        started = time.time()
        try:
            shots = capture_colors(cam, n=frames)
        except Exception as exc:
            per_cam[cam] = {"error": f"capture failed: {type(exc).__name__}: {exc}"}
            print(f"[Scale] cam {cam}: capture failed ({exc})")
            continue
        box = boxes.get(cam)
        crops = [crop_box(s, box) if box is not None else s for s in shots]
        if out_dir is not None:
            cv2.imwrite(str(out_dir / f"{tag}_cam{cam}_full.jpg"), shots[0],
                        [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            cv2.imwrite(str(out_dir / f"{tag}_cam{cam}_crop.png"), crops[0])

        # The display does not move between frames: find it once.
        try:
            quad, score = find_display(crops[0], grounding, cam)
        except Exception as exc:
            quad, score = None, None
            print(f"[Scale] cam {cam}: display segmentation failed ({exc})")
        if quad is None:
            per_cam[cam] = {"display_state": "not_visible", "value": None,
                            "detail": f"SAM found no {DISPLAY_PROMPT!r} in the scale crop"}
            print(f"[Scale] cam {cam}: no display found in the crop -- not read")
            continue

        # All frames' model calls at once: 3 x ~2.5 s in series was half of
        # every camera's 15 s (2026-11-13).
        displays = [rectify(crop, quad) for crop in crops]
        if out_dir is not None:
            for i, display in enumerate(displays):
                cv2.imwrite(str(out_dir / f"{tag}_cam{cam}_display{i}.png"), _panel(display))

        def one(i):
            try:
                return read_display(displays[i], label=f"Camera {cam}, frame {i + 1}",
                                    decimals=decimals)
            except Exception as exc:
                return {"error": f"model call failed: {type(exc).__name__}: {exc}"}

        with ThreadPoolExecutor(max_workers=len(displays)) as pool:
            replies = list(pool.map(one, range(len(displays))))
        for i, reply in enumerate(replies):
            print(f"[Scale] cam {cam} frame {i + 1}: {reply.get('display_state', reply.get('error'))}"
                  f" upright={reply.get('upright')} chars={reply.get('characters')!r} "
                  f"digits={reply.get('digits')!r} (model {reply.get('digits_model')!r}) "
                  f"minus={reply.get('minus_sign')} "
                  f"value={reply.get('value')} unit={reply.get('unit')} "
                  f"conf={reply.get('confidence')}")
        value, n, of = _vote(replies)
        majority = value is not None and n * 2 > len(replies)
        unit = next((r.get("unit") for r in replies if r.get("unit")), None)
        per_cam[cam] = {
            "value": value if majority else None, "votes": f"{n}/{len(replies)}",
            "unit": unit, "display_score": score, "quad": quad.tolist(),
            "replies": replies,
            "display_state": (replies[0].get("display_state") if replies else None),
        }
        print(f"[Scale] cam {cam} ({time.time() - started:.1f}s): "
              + (f"{value} {unit or ''} ({n}/{len(replies)} frames)" if majority
                 else f"no majority ({n}/{len(replies)} frames on {value})"))
        if stop_when_unanimous and value is not None and n == len(replies) >= 2:
            rest = [c for c in cam_ids if c not in per_cam]
            if rest:
                print(f"[Scale] cam {cam}: all {n} frames agree; not reading cams {rest}")
            break

    voted = [(c, r) for c, r in per_cam.items() if r.get("value") is not None]
    if not voted:
        return {"value": None, "unit": None, "reading": None, "cameras": per_cam,
                "agree": None}
    tally = Counter()
    for _, r in voted:
        tally[r["value"]] += int(str(r["votes"]).split("/")[0])
    value = tally.most_common(1)[0][0]
    cams = [c for c, r in voted if r["value"] == value]
    unit = next((per_cam[c]["unit"] for c in cams if per_cam[c].get("unit")), None)
    return {"value": value, "unit": unit, "reading": str(value), "cameras": per_cam,
            "agree": f"cams {cams} of {len(voted)} that read digits"}


# ------------------------------------------------------------------
# What lies on the platform, found by height (no segmentation).
#
# On 2026-11-13 the scoop on the scale could not be segmented: "white plastic
# tool" took the white scale and the white scoop as one mask (259 x 214 mm,
# centroid = the platform centre) in all four cameras, so the camera
# consensus could not reject it either. The platform's top is known from the
# scan of the empty scale, and whatever lies on it is the only thing above it
# inside its outline.

def platform_from_points(points: np.ndarray, band: float = 0.006,
                         bin_size: float = 0.002) -> dict:
    """
    The platform of a located scale: top z, centre and xy bounds of its top.

    The top is the most populated z level in the upper half of the cloud, not
    its 97th percentile: with the scoop already on the scale that percentile is
    the scoop (2026-11-13: 0.036 vs 0.027, and the "platform" came out
    95 x 57 mm, the scoop's patch).
    """
    pts = np.asarray(points, dtype=float)
    z = pts[:, 2]
    upper = z[z >= np.median(z)]
    edges = np.arange(upper.min(), upper.max() + bin_size, bin_size)
    if len(edges) < 2:
        top_z = float(np.percentile(z, 97))
    else:
        counts, edges = np.histogram(upper, bins=edges)
        k = int(np.argmax(counts))
        level = upper[(upper >= edges[k] - bin_size) & (upper <= edges[k + 1] + bin_size)]
        top_z = float(np.percentile(level, 90))
    top = pts[np.abs(z - top_z) <= band]
    lo, hi = np.percentile(top[:, :2], [2, 98], axis=0)
    return {"top_z": top_z, "centre": np.median(top[:, :2], axis=0),
            "xy_lo": lo, "xy_hi": hi}


def _largest_cluster(xy: np.ndarray, cell: float = 0.005, min_points: int = 15):
    """Indices of the largest 8-connected group of occupied ``cell``-sized squares."""
    from scipy import ndimage

    if len(xy) < min_points:
        return None
    ij = np.floor((xy - xy.min(axis=0)) / cell).astype(int)
    grid = np.zeros(tuple(ij.max(axis=0) + 1), dtype=bool)
    grid[ij[:, 0], ij[:, 1]] = True
    labels, n = ndimage.label(grid, structure=np.ones((3, 3), bool))
    if n == 0:
        return None
    per_point = labels[ij[:, 0], ij[:, 1]]
    counts = np.bincount(per_point, minlength=n + 1)
    counts[0] = 0
    keep = np.nonzero(per_point == int(np.argmax(counts)))[0]
    return keep if len(keep) >= min_points else None


def locate_on_platform(localizer, platform: dict, boxes: Dict[int, tuple],
                       above=(0.004, 0.06), margin: float = 0.01,
                       agree: float = 0.03, out_dir: Optional[Path] = None,
                       tag: str = "on_scale") -> Optional[dict]:
    """
    Find what lies on the scale's platform from depth alone.

    Every camera's depth inside the scale's box -> world points -> those inside
    the platform's outline (``margin`` grown) and ``above`` its top ->
    largest xy cluster -> that camera's centroid. Cameras more than ``agree``
    from the median are dropped. Returns {"centroid", "extent", "axis",
    "cameras": {cam: n_points}, "rejected"} or None.
    """
    import cv2

    data = localizer.capture_pointclouds()
    lo = np.asarray(platform["xy_lo"], dtype=float) - margin
    hi = np.asarray(platform["xy_hi"], dtype=float) + margin
    z0, z1 = platform["top_z"] + above[0], platform["top_z"] + above[1]
    per_cam, clouds = {}, {}
    for cam, depth in (data.get("depth_images") or {}).items():
        intr = (data.get("intrinsics") or {}).get(cam)
        box = boxes.get(cam)
        if intr is None or box is None or depth is None:
            continue
        # The box is the EMPTY scale's; what lies on it sticks out above.
        bx1, by1, bx2, by2 = [float(v) for v in box]
        gw, gh = (bx2 - bx1) * 0.3, (by2 - by1) * 0.3
        h, w = depth.shape[:2]
        x1, y1 = int(max(0, bx1 - gw)), int(max(0, by1 - gh))
        x2, y2 = int(min(w, bx2 + gw)), int(min(h, by2 + gh))
        sub = depth[y1:y2, x1:x2].astype(float) / 1000.0
        v, u = np.nonzero((sub > 0.1) & (sub < 2.0))
        if len(u) == 0:
            continue
        z = sub[v, u]
        u, v = u + x1, v + y1
        fx, fy = intr.fx, intr.fy
        cx = getattr(intr, "cx", getattr(intr, "ppx", None))
        cy = getattr(intr, "cy", getattr(intr, "ppy", None))
        cam_pts = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=-1)
        world = localizer._transform_points(cam_pts, localizer._extrinsics(cam))
        keep = ((world[:, 0] >= lo[0]) & (world[:, 0] <= hi[0])
                & (world[:, 1] >= lo[1]) & (world[:, 1] <= hi[1])
                & (world[:, 2] >= z0) & (world[:, 2] <= z1))
        pts, uv = world[keep], np.stack([u[keep], v[keep]], axis=-1)
        idx = _largest_cluster(pts[:, :2])
        print(f"[OnScale] cam {cam}: {int(keep.sum())} points above the platform"
              + ("" if idx is None else f", largest cluster {len(idx)}"))
        if idx is None:
            continue
        clouds[cam] = pts[idx]
        per_cam[cam] = np.median(pts[idx], axis=0)
        if out_dir is not None and cam in (data.get("images") or {}):
            out_dir.mkdir(parents=True, exist_ok=True)
            img = np.asarray(data["images"][cam]).copy()
            for x, y in uv[idx]:
                img[int(y), int(x)] = (0, 0, 255)
            cv2.imwrite(str(out_dir / f"{tag}_cam{cam}.jpg"), img)
    if not per_cam:
        print("[OnScale] nothing found above the platform")
        return None
    med = np.median(np.stack(list(per_cam.values()))[:, :2], axis=0)
    good = [c for c, p in per_cam.items() if np.linalg.norm(p[:2] - med) <= agree]
    rejected = [c for c in per_cam if c not in good]
    if not good:
        return None
    pts = np.concatenate([clouds[c] for c in good])
    centroid = np.median(np.stack([per_cam[c] for c in good]), axis=0)
    xy = pts[:, :2] - pts[:, :2].mean(axis=0)
    w, vecs = np.linalg.eigh(np.cov(xy.T))
    axis = vecs[:, int(np.argmax(w))]
    along = xy @ axis
    extent = float(np.percentile(along, 98) - np.percentile(along, 2))
    print(f"[OnScale] on the platform: centroid ({centroid[0]:.4f}, {centroid[1]:+.4f}, "
          f"{centroid[2]:.4f}), {extent * 1000:.0f} mm long, cams {good}"
          + (f" (rejected {rejected})" if rejected else ""))
    return {"centroid": centroid, "extent": extent, "axis": axis, "points": pts,
            "cameras": {c: len(clouds[c]) for c in good}, "rejected": rejected}
