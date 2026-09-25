"""
One-shot exemplar matching: name an object by SHOWING it, not describing it.

SAM 3 is open-vocabulary, not unlimited. A printed part has no name it knows.
scripts/sweep_prompts.py spent twenty phrases on the cranked scoop and the best
honest hit was "scoop" on 2/4 cameras; diag_out/stirrer_prompts is the same
story with "black object", "black cylinder", "black holder". No wording fixes
that, because the concept is not in the model -- and the phrases that DO score
well ("white object", 0.95 on 4/4) are matching the cups.

So for these objects, drop text entirely:

  1. Segment EVERYTHING in the frame. SAM's automatic mask generator takes no
     prompt and is class-agnostic, so a nameless object is no harder for it
     than a cup.
  2. Embed every candidate crop with DINOv2.
  3. Keep the candidate that looks most like a reference crop you boxed by hand
     once, offline.

The reference embeddings ARE the vocabulary. scripts/register_object.py writes
them into exemplars/*.npz; this module reads them. Adding an object is one
minute of clicking, not another prompt sweep.

Scoring is deliberately discriminative: a candidate is returned for "stirrer"
only if it matches the stirrer references better than every OTHER registered
object. With two small white printed tools on the same bench, an absolute
similarity threshold on its own will cheerfully hand back the scoop.
"""

from __future__ import annotations

import glob
import hashlib
import logging
import os
import re
from collections import OrderedDict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger("grounding.exemplar")

#: Where registered objects live. Small .npz files (a few KB), meant to be
#: committed: they are cell calibration, same as calibration_out/.
DEFAULT_EXEMPLAR_DIR = "exemplars"

#: The embedding model. Recorded inside every .npz -- embeddings made by a
#: different model are not comparable, so a mismatch skips the file loudly
#: rather than silently scoring garbage.
EMBED_MODEL_ID = "facebook/dinov2-base"

#: Candidate generator. SAM 1 base, already cached here for the gdino backend.
SAM_MODEL_ID = "facebook/sam-vit-base"


# ---------------------------------------------------------------- naming


def normalize_name(text: str) -> str:
    """Lowercase, strip punctuation, collapse spaces (matches label_resolver)."""
    if not text:
        return ""
    t = re.sub(r"[^a-z0-9]+", " ", text.lower().strip())
    return re.sub(r"\s+", " ", t).strip()


def slugify(text: str) -> str:
    return normalize_name(text).replace(" ", "_") or "object"


# ---------------------------------------------------------------- cropping


def crop_for_embedding(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    pad: float = 0.15,
    min_pad: int = 8,
    grey: int = 128,
) -> Optional[Tuple[np.ndarray, List[float]]]:
    """
    Cut one candidate out of the frame, the same way on both sides of the match.

    The reference crop (from a box you drew) and the query crop (from a mask SAM
    found on its own) must be processed IDENTICALLY, or the cosine similarity
    ends up measuring the preprocessing rather than the object. That is the
    whole reason register_object.py imports this function instead of doing its
    own cropping -- keep it that way.

    Everything outside the mask is painted flat grey. The cage background is the
    same white table under every object, so leaving it in makes two different
    tools look alike to DINOv2, and makes the same tool look different when it
    moves to another part of the bench. The padding is not context (it gets
    greyed too); it just keeps the silhouette off the crop border, where the
    patch embeddings are worst.

    Returns (crop_bgr, box_xyxy) or None if the mask is empty.
    """
    mask = mask.astype(bool)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None

    h, w = image_bgr.shape[:2]
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())

    px = max(int(round((x2 - x1 + 1) * pad)), min_pad)
    py = max(int(round((y2 - y1 + 1) * pad)), min_pad)
    a, b = max(0, x1 - px), min(w, x2 + 1 + px)
    c, d = max(0, y1 - py), min(h, y2 + 1 + py)

    crop = image_bgr[c:d, a:b].copy()
    if crop.size == 0:
        return None
    crop[~mask[c:d, a:b]] = grey
    return crop, [float(x1), float(y1), float(x2), float(y2)]


def box_of(mask: np.ndarray) -> Optional[List[float]]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


# ---------------------------------------------------------------- models


class Dinov2Embedder:
    """DINOv2 CLS embeddings, L2-normalized so a dot product is cosine."""

    def __init__(self, model_id: str = EMBED_MODEL_ID, device: str = None,
                 batch_size: int = 32):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.torch = torch
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        log.info(f"loading embedder {model_id} on {self.device}")
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        self.dim = int(self.model.config.hidden_size)

    def embed(self, crops_bgr: List[np.ndarray]) -> np.ndarray:
        """(N, D) float32, unit norm. Empty input gives an empty (0, D)."""
        if not crops_bgr:
            return np.zeros((0, self.dim), dtype=np.float32)

        chunks = []
        for i in range(0, len(crops_bgr), self.batch_size):
            batch = [
                cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
                for c in crops_bgr[i : i + self.batch_size]
            ]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                feats = self.model(**inputs).last_hidden_state[:, 0]  # CLS token
            chunks.append(feats.float().cpu().numpy())

        emb = np.concatenate(chunks, axis=0)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (emb / norms).astype(np.float32)


class SamCandidates:
    """
    SAM automatic mask generation: every object in the frame, no prompt at all.

    This is the recall step, and it is the expensive one (a grid of points
    through the mask decoder, ~2-5s per 720p frame on the 2080 Ti). The results
    are cached per image by ExemplarBackend, so asking for the stirrer and then
    the scoop on the same capture only pays it once.
    """

    def __init__(
        self,
        model_id: str = SAM_MODEL_ID,
        device: str = None,
        points_per_crop: int = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.86,
        stability_score_thresh: float = 0.90,
    ):
        import torch
        from transformers import pipeline

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"loading candidate generator {model_id} on {self.device}")
        self.gen = pipeline("mask-generation", model=model_id, device=self.device)
        self.points_per_crop = points_per_crop
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh

    def __call__(self, image_bgr: np.ndarray,
                 points_per_crop: int = None) -> List[np.ndarray]:
        from PIL import Image

        pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        out = self.gen(
            pil,
            points_per_crop=points_per_crop or self.points_per_crop,
            points_per_batch=self.points_per_batch,
            crops_n_layers=0,
            pred_iou_thresh=self.pred_iou_thresh,
            stability_score_thresh=self.stability_score_thresh,
        )
        masks = []
        for m in out["masks"]:
            arr = m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)
            masks.append(arr.astype(bool))
        return masks


# ---------------------------------------------------------------- library


class ExemplarEntry:
    def __init__(self, name: str, aliases: List[str], embeddings: np.ndarray,
                 path: str, created: str = "", sources: List[str] = None):
        self.name = name
        self.aliases = aliases
        self.embeddings = embeddings
        self.path = path
        self.created = created
        self.sources = sources or []


class ExemplarLibrary:
    """
    The custom vocabulary: object name -> reference embeddings.

    One .npz per object, several reference views inside it (one per camera, and
    more if you register the tool again in a different pose). Matching takes the
    MAX over views, not the mean: cam2 sees the scoop's handle end-on and cam5
    sees the whole Z-shape, and averaging those two makes the object look like
    neither.
    """

    def __init__(self, entries: List[ExemplarEntry]):
        self.entries = entries
        self.names = [e.name for e in entries]

        self._lookup: Dict[str, int] = {}
        for i, e in enumerate(entries):
            for key in [e.name] + list(e.aliases):
                k = normalize_name(key)
                if not k:
                    continue
                if k in self._lookup and self._lookup[k] != i:
                    log.warning(
                        f"alias {k!r} claimed by both {self.names[self._lookup[k]]!r} "
                        f"and {e.name!r}; keeping the first"
                    )
                    continue
                self._lookup[k] = i

        if entries:
            self.matrix = np.concatenate([e.embeddings for e in entries], axis=0)
            self.owners = np.concatenate(
                [np.full(len(e.embeddings), i, dtype=int) for i, e in enumerate(entries)]
            )
        else:
            self.matrix = np.zeros((0, 0), dtype=np.float32)
            self.owners = np.zeros((0,), dtype=int)

    # -- io

    @classmethod
    def load(cls, directory: str, embed_model_id: str = EMBED_MODEL_ID) -> "ExemplarLibrary":
        entries = []
        for path in sorted(glob.glob(os.path.join(directory, "*.npz"))):
            try:
                entry = load_exemplar(path)
            except Exception as e:
                log.warning(f"skipping {path}: {e}")
                continue
            model = getattr(entry, "model", embed_model_id)
            if model != embed_model_id:
                # Embeddings from a different encoder are not comparable to
                # anything this process computes. Scoring them would produce
                # plausible-looking similarities that mean nothing.
                log.warning(
                    f"skipping {path}: registered with {model!r}, service runs "
                    f"{embed_model_id!r}. Re-run scripts/register_object.py."
                )
                continue
            entries.append(entry)
            log.info(
                f"exemplar {entry.name!r}: {len(entry.embeddings)} view(s)"
                + (f", aliases {entry.aliases}" if entry.aliases else "")
            )
        return cls(entries)

    # -- queries

    def knows(self, prompt: str) -> bool:
        return normalize_name(prompt) in self._lookup

    def index_of(self, prompt: str) -> Optional[int]:
        return self._lookup.get(normalize_name(prompt))

    def __len__(self) -> int:
        return len(self.entries)


def load_exemplar(path: str) -> ExemplarEntry:
    d = np.load(path, allow_pickle=False)
    emb = np.asarray(d["embeddings"], dtype=np.float32)
    if emb.ndim != 2 or len(emb) == 0:
        raise ValueError(f"bad embeddings array {emb.shape}")
    # Stored normalized, but re-normalize: a hand-edited or concatenated file
    # otherwise skews every similarity it takes part in.
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb = emb / norms

    entry = ExemplarEntry(
        name=str(d["name"].item()) if "name" in d else
             os.path.splitext(os.path.basename(path))[0],
        aliases=[str(a) for a in d["aliases"]] if "aliases" in d else [],
        embeddings=emb.astype(np.float32),
        path=path,
        created=str(d["created"].item()) if "created" in d else "",
        sources=[str(s) for s in d["sources"]] if "sources" in d else [],
    )
    entry.model = str(d["model"].item()) if "model" in d else EMBED_MODEL_ID
    return entry


def save_exemplar(path: str, name: str, embeddings: np.ndarray,
                  aliases: List[str] = None, sources: List[str] = None,
                  model: str = EMBED_MODEL_ID) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    np.savez(
        path,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        name=np.array(name),
        aliases=np.array(list(aliases or []), dtype=object).astype("U64"),
        sources=np.array(list(sources or []), dtype=object).astype("U256"),
        model=np.array(model),
        created=np.array(datetime.now().isoformat(timespec="seconds")),
    )


# ---------------------------------------------------------------- backend


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """
    Close the seam a union leaves between two parts of one object.

    Gluing the stirrer's cube to its body leaves the cube's shaded front face
    outside both masks, so the crop has a grey hole punched through the middle
    of the tool and scores 0.64 against a reference that has none. Refilling
    the outer contour costs nothing and is only ever applied to merged
    candidates, where an enclosed gap is a seam rather than a feature.
    """
    filled = np.zeros(mask.shape, dtype=np.uint8)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    return filled.astype(bool)


def _box_iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


class ExemplarBackend:
    """
    Segment-everything + embedding match, behind the same infer() contract as
    the text backends, so the HTTP layer and every skill above it are unchanged.
    """

    name = "exemplar"

    def __init__(
        self,
        library: ExemplarLibrary,
        embedder: Dinov2Embedder = None,
        candidates: SamCandidates = None,
        threshold: float = 0.60,
        points: int = 32,
        retry_points: int = 48,
        min_area_px: int = 300,
        max_area_frac: float = 0.15,
        max_candidates: int = 192,
        merge_gap: int = 16,
        max_grow_steps: int = 4,
        max_grow_options: int = 24,
        grow_seeds: int = 3,
        cache_size: int = 8,
    ):
        self.library = library
        self.embedder = embedder or Dinov2Embedder()
        self.candidates = candidates or SamCandidates()
        self.threshold = threshold
        self.points = points
        self.retry_points = retry_points
        self.min_area_px = min_area_px
        self.max_area_frac = max_area_frac
        self.max_candidates = max_candidates
        self.merge_gap = merge_gap
        self.max_grow_steps = max_grow_steps
        self.max_grow_options = max_grow_options
        self.grow_seeds = grow_seeds
        self._cache: "OrderedDict[str, dict]" = OrderedDict()
        self._cache_size = cache_size

    def knows(self, prompt: str) -> bool:
        return self.library.knows(prompt)

    # -- candidate generation (cached per image)

    def _propose(self, image_bgr: np.ndarray, points: int = None) -> dict:
        points = points or self.points
        key = (hashlib.md5(np.ascontiguousarray(image_bgr).tobytes()).hexdigest()
               + f"@{points}")
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            return hit

        h, w = image_bgr.shape[:2]
        frame_px = h * w
        raw = self.candidates(image_bgr, points_per_crop=points)

        kept_masks, kept_boxes, kept_crops = [], [], []
        for mask in raw:
            area = int(mask.sum())
            # A mask covering a sixth of the frame is the table, the bench or a
            # camera-filling blur, never a hand tool. Dropping these before the
            # embedder is most of the speed of this step.
            if area < self.min_area_px or area > self.max_area_frac * frame_px:
                continue
            got = crop_for_embedding(image_bgr, mask)
            if got is None:
                continue
            crop, box = got
            # AMG emits nested near-duplicates (whole object, then its lid, then
            # its shadow-free core). Keep one of each shape.
            if any(_box_iou(box, b) >= 0.9 for b in kept_boxes):
                continue
            kept_masks.append(mask)
            kept_boxes.append(box)
            kept_crops.append(crop)

        if len(kept_masks) > self.max_candidates:
            order = np.argsort([m.sum() for m in kept_masks])[: self.max_candidates]
            kept_masks = [kept_masks[i] for i in order]
            kept_boxes = [kept_boxes[i] for i in order]
            kept_crops = [kept_crops[i] for i in order]

        embeddings = self.embedder.embed(kept_crops)
        log.info(f"{len(raw)} raw masks -> {len(kept_masks)} candidates "
                 f"({points}x{points} points)")

        entry = {
            # Packed: 60 full-resolution bool masks per frame across an 8-frame
            # cache is half a gigabyte of RAM unpacked, and 7MB packed.
            "packed": [np.packbits(m) for m in kept_masks],
            "shape": (h, w),
            "boxes": kept_boxes,
            "embeddings": embeddings,
        }
        self._cache[key] = entry
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return entry

    def _grow(self, image_bgr, cand, seed: int, ref_cols, frame_px):
        """
        Glue neighbouring pieces onto a candidate for as long as that makes it
        look MORE like the reference.

        A two-tone object is several objects to the mask generator. The
        stirrer's black body, the white cube's top and the cube's shaded front
        face come back as three separate masks on cam2, and the best single one
        (the body) scores 0.63 against a reference of the whole tool, with the
        head cut off.

        Merging every adjacent pair blindly does not fix that -- body + cube-top
        is a lid floating above a mug, and it scored 0.64. What fixes it is
        letting the reference say which pieces belong together: add the
        neighbour that improves similarity most, repeat, stop when nothing
        does. Three pieces cost three steps instead of the combinatorial blowup
        of merging everything with everything, and an object SAM splits five
        ways is no harder.

        Returns (mask, crop) or None if the seed could not be improved.
        """
        masks = [self._unpack(p, cand["shape"]) for p in cand["packed"]]
        current = masks[seed]
        current_box = cand["boxes"][seed]
        current_score = float(
            (cand["embeddings"][seed : seed + 1] @ self.library.matrix[ref_cols].T).max()
        )
        best = None

        for _ in range(self.max_grow_steps):
            options = []
            # Be generous about what counts as "next to": similarity decides
            # whether a union survives, so the geometry test only has to avoid
            # considering the whole frame. A fixed 8px was too mean -- the two
            # halves of the scoop on cam5 sit 9px apart, because the thin
            # connector between them was never segmented, and growth refused to
            # try the one merge that would have made the tool whole. Scaling
            # with the current candidate keeps it sane across cameras: a tool
            # 25px wide at cam5 and 90px wide at cam2 need different absolutes.
            cw = current_box[2] - current_box[0]
            ch = current_box[3] - current_box[1]
            g = max(self.merge_gap, int(round(0.25 * max(cw, ch))))
            for i, box in enumerate(cand["boxes"]):
                a, b = current_box, box
                if (a[0] - g > b[2] or b[0] - g > a[2]
                        or a[1] - g > b[3] or b[1] - g > a[3]):
                    continue                        # not touching
                if (masks[i] & ~current).sum() < self.min_area_px:
                    continue                        # adds nothing new
                union = fill_holes(current | masks[i])
                if union.sum() > self.max_area_frac * frame_px:
                    continue
                got = crop_for_embedding(image_bgr, union)
                if got is not None:
                    options.append((union, got[0], got[1]))
                if len(options) >= self.max_grow_options:
                    break

            if not options:
                break

            scores = (self.embedder.embed([o[1] for o in options])
                      @ self.library.matrix[ref_cols].T).max(axis=1)
            k = int(np.argmax(scores))
            if scores[k] <= current_score + 1e-3:
                break                               # nothing helps any more

            current, crop, current_box = options[k]
            current_score = float(scores[k])
            best = (current, crop)

        return best

    @staticmethod
    def _unpack(packed: np.ndarray, shape) -> np.ndarray:
        h, w = shape
        return np.unpackbits(packed, count=h * w).reshape(h, w).astype(bool)

    # -- matching

    def infer(self, image_bgr: np.ndarray, prompt: str,
              return_all: bool = False, conf: float = None) -> dict:
        target = self.library.index_of(prompt)
        if target is None:
            return {"found": False,
                    "reason": f"no exemplar registered for {prompt!r}"}

        thresh = self.threshold if conf is None else float(conf)
        out = self._match(image_bgr, prompt, target, thresh, return_all, self.points)
        if out["found"] or self.retry_points <= self.points:
            return out

        # Pay for a denser point grid only on a miss.
        #
        # The grid is what finds candidates at all, and at 32x32 over an 848x480
        # frame the points land ~26px apart -- wider than the scoop's shank. On
        # cam5 of the holder_empty capture that produced HALF a scoop, and half a
        # scoop is a small white blob that genuinely resembles the stirrer cube
        # more than it resembles a scoop (0.56 vs 0.65), so the match was
        # correctly refused.
        #
        # Raising the density everywhere is the wrong trade: it costs 2-3x on
        # every frame, and it also shifts which candidates seed _grow, which cost
        # cam2 its perfect scoop match (1.00 -> 0.89) in the sweep. Retrying only
        # the frames that missed keeps the common path fast and leaves the seeds
        # alone.
        log.info(f"{prompt!r} not found at {self.points}x{self.points}; retrying "
                 f"at {self.retry_points}x{self.retry_points}")
        retry = self._match(image_bgr, prompt, target, thresh, return_all,
                            self.retry_points)
        if retry["found"]:
            retry["retried"] = True
            return retry
        out["reason"] = (f"{out['reason']}; still missing at "
                         f"{self.retry_points}x{self.retry_points}")
        return out

    def _match(self, image_bgr: np.ndarray, prompt: str, target: int,
               thresh: float, return_all: bool, points: int) -> dict:
        cand = self._propose(image_bgr, points)
        n = len(cand["boxes"])
        if n == 0:
            return {"found": False, "reason": "no candidate masks in frame"}

        h, w = image_bgr.shape[:2]
        ref_cols = np.where(self.library.owners == target)[0]

        # Grow the most promising few before scoring anything: an object the
        # mask generator split into pieces has no single candidate that is the
        # whole thing, and the grown shapes must compete on the same terms as
        # the originals (including against the other registered objects).
        embeddings = cand["embeddings"]
        packed = list(cand["packed"])
        boxes = list(cand["boxes"])
        if len(ref_cols) and self.grow_seeds > 0:
            seed_scores = (embeddings @ self.library.matrix[ref_cols].T).max(axis=1)
            grown_crops, grown_masks = [], []
            for seed in np.argsort(-seed_scores)[: self.grow_seeds]:
                got = self._grow(image_bgr, cand, int(seed), ref_cols, h * w)
                if got is None:
                    continue
                mask, crop = got
                box = box_of(mask)
                if box is None or any(_box_iou(box, b) >= 0.95 for b in boxes):
                    continue
                grown_masks.append(mask)
                grown_crops.append(crop)
                boxes.append(box)
                packed.append(np.packbits(mask))
            if grown_crops:
                embeddings = np.concatenate(
                    [embeddings, self.embedder.embed(grown_crops)], axis=0
                )
                log.info(f"grew {len(grown_crops)} candidate(s) for {prompt!r}")

        cand = {"packed": packed, "boxes": boxes, "shape": cand["shape"],
                "embeddings": embeddings}
        n = len(boxes)

        sims = embeddings @ self.library.matrix.T                  # (n, views)
        n_obj = len(self.library.entries)
        obj_sims = np.full((n, n_obj), -1.0, dtype=np.float32)
        for j in range(n_obj):
            cols = np.where(self.library.owners == j)[0]
            obj_sims[:, j] = sims[:, cols].max(axis=1)

        winner = obj_sims.argmax(axis=1)
        score = obj_sims[:, target]
        if n_obj > 1:
            others = np.delete(obj_sims, target, axis=1).max(axis=1)
        else:
            others = np.zeros(n, dtype=np.float32)

        for i in np.argsort(-score)[:3]:
            log.info(
                f"  cand box={[int(v) for v in cand['boxes'][i]]} "
                f"sim({self.library.names[target]})={score[i]:.3f} "
                f"best={self.library.names[winner[i]]}={obj_sims[i, winner[i]]:.3f}"
            )

        keep = np.where((winner == target) & (score >= thresh))[0]
        if len(keep) == 0:
            best = int(np.argmax(score))
            why = (
                f"best candidate scored {score[best]:.3f} < {thresh:.2f}"
                if winner[best] == target
                else f"best candidate for {prompt!r} ({score[best]:.3f}) looks "
                     f"more like {self.library.names[winner[best]]!r} "
                     f"({obj_sims[best, winner[best]]:.3f})"
            )
            return {"found": False, "reason": why}

        keep = keep[np.argsort(-score[keep])]

        def build(i: int) -> dict:
            return {
                "mask": self._unpack(cand["packed"][i], cand["shape"]),
                "score": float(score[i]),
                "box": [float(v) for v in cand["boxes"][i]],
                "margin": float(score[i] - others[i]),
            }

        if not return_all:
            out = build(int(keep[0]))
            out.update({"found": True, "num_instances": int(len(keep))})
            return out

        instances = [build(int(i)) for i in keep]
        best = instances[0]
        return {
            "found": True,
            "instances": instances,
            "num_instances": len(instances),
            "mask": best["mask"],
            "score": best["score"],
            "box": best["box"],
        }


class SamBoxSegmenter:
    """
    Box -> mask, for registration. You drag a rough rectangle, SAM tightens it.

    A rectangle is not good enough to embed directly: at the scoop's aspect
    ratio a tight box is still ~60% table, and the whole point of the grey
    background in crop_for_embedding is to keep the table out of the
    embedding. So the reference crop comes from a SAM mask, exactly like the
    query crops it will be compared against.
    """

    def __init__(self, model_id: str = SAM_MODEL_ID, device: str = None):
        import torch
        from transformers import SamModel, SamProcessor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"loading box segmenter {model_id} on {self.device}")
        self.processor = SamProcessor.from_pretrained(model_id)
        self.model = SamModel.from_pretrained(model_id).to(self.device).eval()

    def __call__(self, image_bgr: np.ndarray, box_xyxy) -> np.ndarray:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(
            rgb, input_boxes=[[[float(v) for v in box_xyxy]]], return_tensors="pt"
        ).to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs, multimask_output=True)
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0][0]
        scores = outputs.iou_scores[0][0].cpu().numpy()
        return masks[int(np.argmax(scores))].numpy().astype(bool)


def overlay_mask(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Red blend plus a yellow outline -- the blend alone is invisible on a
    dark object, which made the stirrer's mask look like a miss in the sweeps."""
    vis = image_bgr.copy()
    m = mask.astype(bool)
    vis[m] = (0.5 * vis[m] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
    contours, _ = cv2.findContours(
        m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(vis, contours, -1, (0, 255, 255), 2)
    return vis
