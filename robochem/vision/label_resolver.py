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
    # The reagents moved into shallow white bowls on 2026-09-24; these are
    # container categories, not reagent labels, so they prompt SAM directly.
    "white bowl",
    "white ceramic bowl",
    "white glass bowl",
    "bowl",
    "plastic beaker",
    "beaker",
    "clear cup",
    "clear plastic cup",
    "larger spoon",
    "large spoon",
    "small spoon",
    "spoon",
    "scoop",
    # stir / dispense tools. Without these a pick_up of a stirrer or pipette
    # burns a full multi-camera capture plus one VLM label read per camera
    # trying to match the tool name against the reagent labels, before falling
    # back to the direct prompt anyway.
    "pipette",
    "dropper",
    "eye dropper",
    "stirrer",
    "stirring rod",
    "glass rod",
    "spatula",
    # The printed cranked scoop — see SAM_PROMPT_ALIASES for why this wording.
    "white plastic tool",
}


#: Names this project uses -> the phrase SAM 3 actually responds to.
#:
#: SAM 3 is open-vocabulary but not unlimited: a phrase it has no concept for
#: returns NOTHING, silently and on every camera. "rectangular scoop" found zero
#: masks across all four cages on 2026-09-18 — the description was accurate and
#: the model simply had no such concept.
#:
#: So the name a skill or an agent uses does not have to be the name SAM is
#: given. Put the readable name on the left and whatever the prompt sweep
#: actually found on the right. Both sides bypass the reagent-label path.
#:
#: Populate the right-hand side from scripts/sweep_prompts.py — do not guess.
SAM_PROMPT_ALIASES = {
    # The printed cranked scoop: an 8mm handle with a 20.5mm open box on the
    # end. Chosen by scripts/sweep_prompts.py against
    # scene_captures/new_scoop_20260918_151846 and confirmed by eye on the
    # overlays — cam5 masks the whole Z-shape, cam2 catches the handle only.
    #
    # The sweep is worth reading before changing this. SAM 3 found NOTHING for
    # "spoon", "spatula", "rectangular scoop" or "square spoon" — plausible
    # descriptions that simply are not concepts it has. "scoop" alone managed
    # 2/4 cameras. "white object" scored 0.95 on 4/4 but masked 2% of the
    # frame: it was matching the paper cups, which is the failure this list
    # exists to avoid. Do not pick a prompt on score alone.
    "rectangular scoop": "white plastic tool",
    "printed scoop": "white plastic tool",
    # robomail_Aliyah's robot profile names this position "measuring scoop";
    # the bench object it refers to is the same printed tool.
    "measuring scoop": "white plastic tool",
}


def resolve_sam_prompt(object_name: str) -> str:
    """
    Map a project name onto the phrase SAM 3 is actually given.

    Returns ``object_name`` unchanged when there is no alias, so this is safe to
    call on every query.
    """
    if not object_name:
        return object_name
    return SAM_PROMPT_ALIASES.get(normalize_label(object_name), object_name)


def normalize_label(text: str) -> str:
    """Lowercase, strip punctuation/spaces for fuzzy compare."""
    if not text:
        return ""
    t = text.lower().strip()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def labels_match(requested: str, observed: str) -> bool:
    """
    True if the observed label names the same container as requested.

    Rule: every token of the REQUESTED label must appear in the observed one.
    That keeps the useful partial matches — "red cabbage" still finds
    "RED CABBAGE POWDER" — while refusing to conflate labels that differ by a
    discriminator.

    The discriminator case is what forced this. Replicates get labelled
    "A 10 ML WATER" and "B 10 ML WATER", which differ by a single short token
    out of four. The previous rule accepted a match on half the tokens
    overlapping, so asking for A happily returned B — on a bench where both
    cups exist and mean different things. Any rule loose enough to ignore one
    token cannot distinguish replicates, and replicates are the whole point of
    running A against B.

    Note this is deliberately asymmetric: a request may be less specific than
    the label ("10 ml water" matches both A and B, and the caller is told the
    result is ambiguous), but it may never contradict it.
    """
    a, b = normalize_label(requested), normalize_label(observed)
    if not a or not b:
        return False
    if a == b:
        return True
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return False
    return ta.issubset(tb)


# Utensil words: any query containing one of these is a physical-object
# query, never a reagent label. Unlike "cup"/"beaker", these never appear
# in "cup labeled X" phrasing, so matching on the word alone is safe.
_DIRECT_SAM_KEYWORDS = {"spoon", "scoop", "pipette", "dropper", "stirrer",
                        "stirring", "spatula"}


def looks_like_label_query(object_name: str) -> bool:
    """
    Heuristic: reagent / content names go through label matching; raw
    SAM category phrases do not.
    """
    key = normalize_label(object_name)
    if not key:
        return False
    # Anything with an explicit SAM alias is a physical object we know how to
    # prompt for, never a reagent label.
    if key in SAM_PROMPT_ALIASES:
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
    """
    Draw numbered boxes on a copy of the image for the VLM.

    The number goes in a filled disc at the CENTRE of each container, not at
    the box's top-left corner. That corner is where the previous version put
    it, and with two lookalike cups side by side the tag for one sat visually
    over the other's paper: on 2026-09-21 the model read "A 10 ML WATER"
    correctly and attached it to the cup standing on the B paper. The reading
    was never the problem; the association was.

    Centring the tag on the container it belongs to removes that ambiguity
    while keeping ONE call per camera. Cropping each container into its own
    call also fixes it, but costs a call and a JPEG encode per container, and
    that is memory spent at the exact moment this host has none to spare.
    """
    vis = image_bgr.copy()
    h, w = vis.shape[:2]
    colour = (0, 255, 255)
    for i, inst in enumerate(instances):
        x1, y1, x2, y2 = _ensure_box(inst, h, w)
        x1, y1 = int(max(0, x1)), int(max(0, y1))
        x2, y2 = int(min(w - 1, x2)), int(min(h - 1, y2))
        cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 2)

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        tag = str(i)
        radius = 15
        cv2.circle(vis, (cx, cy), radius, (0, 0, 0), -1)
        cv2.circle(vis, (cx, cy), radius, colour, 2)
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.putText(vis, tag, (cx - tw // 2, cy + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2, cv2.LINE_AA)
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

    # The label must come back COMPLETE. An earlier version of this prompt
    # listed only plain reagent names as examples, and the model duly returned
    # the reagent and dropped everything else: two cups labelled "A 10 ML
    # WATER" and "B 10 ML WATER" both came back as "WATER", which is precisely
    # the distinction the experiment depends on. Transcribe, do not summarise.
    prompt = f"""This lab photo has {n} containers marked with yellow boxes numbered 0..{n - 1}.
Each sits on a white paper square with a handwritten label.

Each container has its index printed in a yellow circle ON the container
itself. The label for that index is on the paper under or beside THAT
container -- not the nearest paper to the circle, and never a neighbour's.

Transcribe the label for EACH numbered container, COMPLETELY and VERBATIM.

Critical rules:
- Copy the WHOLE label, every character, in the order written. Do NOT
  summarise it down to the reagent name.
- Labels often carry a leading identifier letter (A, B, C ...) and/or a
  volume. "A 10 ML WATER" and "B 10 ML WATER" are DIFFERENT containers that
  must never be reported as the same thing. Returning "WATER" for either is
  wrong and will cause the robot to use the wrong container.
- The handwriting is frequently ROTATED — sideways or upside down relative to
  the image. Read it at whatever orientation it is written.
- The text is on the paper under or beside the container, never on its wall.
- If part of a label is cut off or illegible, return the part you can read
  rather than guessing the rest.

Return ONLY JSON:
{{"cups": [{{"id": 0, "label": "A 10 ML WATER"}}, {{"id": 1, "label": "BAKING SODA"}}, ...]}}

Use null for label only if you can read nothing at all. Include every id from
0 to {n - 1}.
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


def crop_around_instance(image_bgr: np.ndarray, inst: dict,
                         pad_scale: float = 0.9, min_pad: int = 30,
                         neighbours: Optional[List[dict]] = None) -> np.ndarray:
    """
    Crop one container plus enough surroundings to include its own paper label.

    Reading every container from one annotated frame turns out to be the weak
    step: the model transcribes the handwriting correctly but attaches it to
    the wrong numbered box when two near-identical cups sit side by side. On
    2026-09-21 that put the "A 10 ML WATER" label on the cup standing on the
    "B 10 ML WATER" paper — the reading was right, the association was not.

    Cropping removes the association problem rather than pleading with the
    model about it: one container in frame means there is nothing to mix it up
    with. The padding still has to reach the paper *beside* the container, not
    just the container.

    But padding that reaches the neighbour's paper recreates the bug in a new
    form. At 1.8x the box, three of four cameras returned the SAME label for
    two different cups — a real label, read accurately, belonging to the cup
    next door. So the padding is now tight, and any neighbouring container in
    range is greyed out: whatever paper it is standing on stops looking like a
    candidate.
    """
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = _ensure_box(inst, h, w)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half_w = max((x2 - x1) * pad_scale, min_pad)
    half_h = max((y2 - y1) * pad_scale, min_pad)
    a = int(max(0, cx - half_w)); b = int(min(w, cx + half_w))
    c = int(max(0, cy - half_h)); d = int(min(h, cy + half_h))
    crop = image_bgr[c:d, a:b].copy()

    # Grey out any OTHER container caught in the crop, along with a margin
    # around it, so its paper cannot be mistaken for this one's.
    for other in (neighbours or []):
        if other is inst:
            continue
        ox1, oy1, ox2, oy2 = _ensure_box(other, h, w)
        if ox2 < a or ox1 > b or oy2 < c or oy1 > d:
            continue                      # not in this crop at all
        m = 18
        cv2.rectangle(crop,
                      (int(max(0, ox1 - a - m)), int(max(0, oy1 - c - m))),
                      (int(min(b - a, ox2 - a + m)), int(min(d - c, oy2 - c + m))),
                      (128, 128, 128), -1)

    # Mark the container itself, so "this one" is unambiguous inside the crop.
    cv2.rectangle(crop, (int(x1 - a), int(y1 - c)), (int(x2 - a), int(y2 - c)),
                  (0, 255, 255), 2)
    return crop


def read_one_label_vlm(vlm_client, image_bgr: np.ndarray, inst: dict,
                       model: str = "gpt-4o",
                       neighbours: Optional[List[dict]] = None) -> Optional[str]:
    """Transcribe the label belonging to a single container."""
    crop = crop_around_instance(image_bgr, inst, neighbours=neighbours)
    if crop.size == 0:
        return None
    b64 = _encode_jpeg(crop)

    prompt = """This crop shows ONE container, outlined in yellow, standing on or beside a
white paper square with a handwritten label.

Transcribe that label COMPLETELY and VERBATIM.

- Copy every character in the order written. Do NOT summarise to the reagent
  name: labels often carry a leading identifier letter (A, B, C ...) and a
  volume, and "A 10 ML WATER" and "B 10 ML WATER" are different containers.
- The handwriting is frequently ROTATED, sideways or upside down. Read it at
  whatever orientation it is written.
- Read ONLY the label on the paper the OUTLINED container is standing on or
  directly beside. Grey blocks are other containers; ignore them and any
  paper near them. If a label is nearer a grey block than the outlined
  container, it is not the one being asked for — return null instead.
- If you genuinely cannot read any label for this container, return null.

Return ONLY JSON: {"label": "A 10 ML WATER"}  or  {"label": null}
"""
    response = vlm_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]}],
        max_tokens=100,
        temperature=0,
    )
    text = (response.choices[0].message.content or "").strip()
    if "```" in text:
        for part in text.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                text = part
                break
    try:
        start, end = text.find("{"), text.rfind("}")
        data = json.loads(text[start:end + 1]) if start >= 0 < end else {}
    except Exception:
        print(f"[LabelResolver] could not parse single-label reply: {text[:120]!r}")
        return None
    lab = data.get("label")
    if lab is None:
        return None
    lab = str(lab).strip()
    return None if (not lab or lab.lower() == "null") else lab


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

    # Several containers answering to the same request is not something to
    # resolve by score: the highest-scoring mask is just the one SAM liked
    # best, which says nothing about which cup the caller meant. Say so.
    distinct = {normalize_label(lab) for _, lab, _ in matches}
    if len(distinct) > 1:
        print(f"[LabelResolver] AMBIGUOUS: {requested!r} matches "
              f"{sorted(distinct)} — be more specific, or the wrong container "
              f"may be used")
    elif len(matches) > 1:
        # Same label on two containers means the crop reached the neighbour's
        # paper: one of these is standing somewhere else entirely. Picking the
        # higher-scoring mask here is a coin flip, and on 2026-09-21 it landed
        # on a cup 31cm away.
        print(f"[LabelResolver] DUPLICATE: {len(matches)} containers all read "
              f"as {matches[0][1]!r}. Only one can be right — the crop has "
              f"probably picked up a neighbouring label. Position from this "
              f"camera is unreliable.")

    matches.sort(key=lambda t: t[2], reverse=True)
    inst, lab, _ = matches[0]
    return inst, lab


class LabelResolver:
    """
    Resolve a reagent name to per-camera cup masks via SAM instances + VLM OCR.
    """

    def __init__(self, vlm_client=None, model: str = "gpt-4o",
                 per_instance: bool = True):
        self.vlm_client = vlm_client
        self.model = model
        #: Read each container in its own cropped call. Default True.
        #:
        #: Measured on scene_captures/ab_water_20260921_102436, which has the
        #: A and B water cups side by side:
        #:
        #:   per-instance   "a 10 ml water" 4/4 cameras, "b" 4/4, no invented labels
        #:   whole-frame    "a" 2/4, "b" 0/4, and reads like "WEAK BASE",
        #:                  "C 10% NaCl", "GLITTER ADD" that are not on the bench
        #:
        #: Centring the index tag on each container (see annotate_instances)
        #: was tried as a cheaper fix and did not close that gap. The crops are
        #: ~100x100px, so the memory they cost is negligible -- the host freeze
        #: of 2026-09-25 was the grounding service's 8.9GB resident set against
        #: 3GB free, not this.
        self.per_instance = per_instance

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
                if self.per_instance:
                    # One call per container: slower, but it cannot mis-assign
                    # a label to the wrong cup, which reading the whole frame
                    # at once demonstrably does.
                    labels = [read_one_label_vlm(self.vlm_client, images[cam_id],
                                                 inst, model=self.model,
                                                 neighbours=instances)
                              for inst in instances]
                else:
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
