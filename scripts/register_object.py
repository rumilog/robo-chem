"""
Teach the cell an object by boxing it once, for things SAM cannot be told about.

Custom printed parts have no name SAM 3 knows -- see the sweeps in
diag_out/stirrer_prompts, where every plausible phrase either found nothing or
found the cups. This registers the object by appearance instead: you drag a box
around it in a saved capture, SAM tightens the box into a mask, DINOv2 embeds
the crop, and the embedding goes into exemplars/<name>.npz. The grounding
service then answers segment("stirrer") by matching that embedding against
every object it can find in the frame -- no text involved.

    # 1. put the tool on the bench and photograph it from all four cameras
    source scripts/env.sh
    python scripts/capture_scene.py --label stirrer

    # 2. box it once per camera (perception_env: this needs torch + a display)
    perception_env/bin/python scripts/register_object.py --name stirrer \
        --images-dir scene_captures/stirrer_<ts>

    # 3. restart the service so it picks the new exemplar up
    perception_env/bin/python perception_service/grounding_service.py \
        --backend sam3 --model weights/sam3.pt --exemplars exemplars

    # 4. check it on the same stills, and LOOK at the overlays
    python scripts/sweep_prompts.py --images-dir scene_captures/stirrer_<ts> \
        --only stirrer --save-overlays diag_out/stirrer_exemplar

Register from several captures (--images-dir is repeatable) if the tool will be
seen in very different poses: lying on the bench and standing in its holder are
not the same view, and one reference crop cannot cover both.

Controls per camera:
    drag a box, ENTER/SPACE to confirm, c to skip this camera
    then: y accept  r redraw  s skip  q stop here and save what you have
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from perception_service.exemplar_matcher import (  # noqa: E402
    DEFAULT_EXEMPLAR_DIR,
    EMBED_MODEL_ID,
    Dinov2Embedder,
    ExemplarLibrary,
    SamBoxSegmenter,
    crop_for_embedding,
    load_exemplar,
    normalize_name as normalize,
    overlay_mask,
    save_exemplar,
    slugify,
)


def load_color_images(images_dir: str):
    """cam id -> BGR frame, same layout scripts/sweep_prompts.py reads."""
    images = {}
    for path in sorted(glob.glob(os.path.join(images_dir, "cam*_color.png"))):
        m = re.search(r"cam(\d+)_color", os.path.basename(path))
        if not m:
            continue
        img = cv2.imread(path)
        if img is not None:
            images[int(m.group(1))] = (img, path)
    return images


def pick_one(window: str, image_bgr: np.ndarray, segmenter, label: str):
    """
    Drag-box -> SAM mask -> confirm. Returns (mask, action) where action is one
    of "accept", "skip", "quit".
    """
    while True:
        print(f"  [{label}] drag a box around the object, ENTER to confirm, c to skip")
        box = cv2.selectROI(window, image_bgr, showCrosshair=True, fromCenter=False)
        x, y, w, h = [int(v) for v in box]
        if w <= 2 or h <= 2:
            print(f"  [{label}] skipped (no box)")
            return None, "skip"

        mask = segmenter(image_bgr, [x, y, x + w, y + h])
        area = int(mask.sum())
        frac = area / float(image_bgr.shape[0] * image_bgr.shape[1])
        print(f"  [{label}] mask {area} px ({frac * 100:.2f}% of frame)")
        if frac > 0.25:
            # SAM answers a loose box with "the table, then", and a reference
            # crop of the table matches everything later.
            print(f"  [{label}] WARNING: that mask is huge. Draw the box tighter.")

        vis = overlay_mask(image_bgr, mask)
        cv2.putText(vis, "y accept   r redraw   s skip   q stop", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, "y accept   r redraw   s skip   q stop", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(window, vis)

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in (ord("y"), 13, 32):
                return mask, "accept"
            if key == ord("r"):
                break
            if key == ord("s"):
                return None, "skip"
            if key in (ord("q"), 27):
                return None, "quit"


def report_separation(name: str, embeddings: np.ndarray, exemplar_dir: str,
                      model_id: str) -> None:
    """
    Say how well this object stands apart -- the number that decides the
    threshold.

    Own views that agree at 0.9 and a neighbour at 0.5 is a clean object. Own
    views at 0.5 and a neighbour at 0.8 means the crops are mostly grey
    background or mostly table, and no threshold will save the match.
    """
    if len(embeddings) > 1:
        sims = embeddings @ embeddings.T
        off = sims[~np.eye(len(sims), dtype=bool)]
        print(f"\n  own views agree at {off.min():.3f}-{off.max():.3f} "
              f"(mean {off.mean():.3f})")
        if off.min() < 0.4:
            print("  NOTE: views disagree strongly. That is normal for a tool "
                  "seen end-on from one camera, but check the crops.")

    worst = 0.0
    for path in sorted(glob.glob(os.path.join(exemplar_dir, "*.npz"))):
        try:
            other = load_exemplar(path)
        except Exception:
            continue
        if slugify(other.name) == slugify(name) or getattr(other, "model", model_id) != model_id:
            continue
        best = float((embeddings @ other.embeddings.T).max())
        print(f"  vs {other.name!r}: {best:.3f}")
        worst = max(worst, best)

    if worst >= 0.85:
        print("\n  WARNING: another registered object scores nearly as high as "
              "this one's own views.\n  They are not reliably distinguishable. "
              "Register more views, or make the\n  objects visually different "
              "(colour is the cheap fix).")
    elif worst > 0.0:
        suggested = max(0.45, min(0.80, (worst + 0.95) / 2.0))
        print(f"\n  Suggested --exemplar-thresh: {suggested:.2f} "
              f"(above every other object, below this one's own views)")


def warn_about_prompt_aliases(name: str, registered: list) -> None:
    """
    Catch the silent miss: a skill asks for a name the client rewrites before
    the service ever sees it.

    GroundingClient._resolve applies SAM_PROMPT_ALIASES at the HTTP boundary, so
    a skill asking for "rectangular scoop" sends "white plastic tool". Dispatch
    in the service is by name, so an exemplar registered as "scoop" is never
    consulted for that request -- it falls through to the text backend and
    quietly misses, which is the failure this whole path exists to remove.
    """
    try:
        from robochem.vision.label_resolver import SAM_PROMPT_ALIASES
    except Exception:
        return

    known = {normalize(a) for a in [name] + list(registered)}
    tokens = set(normalize(name).split())
    missing = {
        target for src, target in SAM_PROMPT_ALIASES.items()
        # Any alias naming this object -- "rectangular scoop" shares a word
        # with "scoop" -- rewrites to a phrase that must resolve here too.
        if (tokens & set(normalize(src).split()) or normalize(src) in known)
        and normalize(target) not in known
    }
    if missing:
        flags = " ".join(f'--alias "{m}"' for m in sorted(missing))
        print(f"\n  NOTE: label_resolver rewrites some names to {sorted(missing)} "
              f"before the\n  service sees them. If those are this object, "
              f"re-run with:\n    ... --append {flags}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--name", required=True,
                    help="what skills will ask for, e.g. stirrer")
    ap.add_argument("--images-dir", action="append", required=True,
                    help="a scene_captures/<run> folder (repeatable)")
    ap.add_argument("--alias", action="append", default=[],
                    help="another name that should resolve here (repeatable). "
                         "Include any phrase label_resolver rewrites to.")
    ap.add_argument("--exemplar-dir", default=os.path.join(ROOT, DEFAULT_EXEMPLAR_DIR))
    ap.add_argument("--append", action="store_true",
                    help="add these views to an existing exemplar")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing exemplar")
    ap.add_argument("--preview-dir", default=os.path.join(ROOT, "diag_out", "exemplars"),
                    help="where the accepted crops are written for inspection")
    args = ap.parse_args()

    out_path = os.path.join(args.exemplar_dir, f"{slugify(args.name)}.npz")
    existing = None
    if os.path.exists(out_path):
        if not (args.append or args.overwrite):
            print(f"{out_path} already exists. Use --append to add views to it, "
                  f"or --overwrite to replace it.")
            return 1
        if args.append:
            existing = load_exemplar(out_path)
            if getattr(existing, "model", EMBED_MODEL_ID) != EMBED_MODEL_ID:
                print(f"{out_path} was registered with {existing.model!r}; cannot "
                      f"append {EMBED_MODEL_ID!r} views. Use --overwrite.")
                return 1
            print(f"Appending to {len(existing.embeddings)} existing view(s)")

    segmenter = SamBoxSegmenter()
    embedder = Dinov2Embedder()

    window = f"register: {args.name}"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    crops, sources = [], []
    stop = False
    for images_dir in args.images_dir:
        if stop:
            break
        images = load_color_images(images_dir)
        if not images:
            print(f"No cam*_color.png under {images_dir}")
            continue
        print(f"\n{images_dir}: cameras {sorted(images)}")
        for cam_id in sorted(images):
            image, path = images[cam_id]
            mask, action = pick_one(window, image, segmenter, f"cam{cam_id}")
            if action == "quit":
                stop = True
                break
            if action == "skip" or mask is None:
                continue
            got = crop_for_embedding(image, mask)
            if got is None:
                print(f"  [cam{cam_id}] empty crop, skipped")
                continue
            crop, _ = got
            crops.append(crop)
            sources.append(os.path.relpath(path, ROOT))

    cv2.destroyAllWindows()

    if not crops:
        print("\nNothing registered.")
        return 1

    print(f"\nEmbedding {len(crops)} view(s) with {EMBED_MODEL_ID}")
    embeddings = embedder.embed(crops)

    preview_dir = os.path.join(args.preview_dir, slugify(args.name))
    os.makedirs(preview_dir, exist_ok=True)
    for i, (crop, src) in enumerate(zip(crops, sources)):
        cv2.imwrite(os.path.join(preview_dir, f"view{i:02d}.png"), crop)
    print(f"Reference crops written to {preview_dir}/ -- look at them: a crop "
          f"that is mostly grey\nor mostly table is a reference that will match "
          f"anything.")

    aliases = list(args.alias)
    if existing is not None:
        embeddings = np.concatenate([existing.embeddings, embeddings], axis=0)
        sources = list(existing.sources) + sources
        aliases = list(dict.fromkeys(list(existing.aliases) + aliases))

    save_exemplar(out_path, args.name, embeddings, aliases=aliases,
                  sources=sources, model=EMBED_MODEL_ID)
    print(f"\nWrote {out_path}: {len(embeddings)} view(s)"
          + (f", aliases {aliases}" if aliases else ""))

    report_separation(args.name, embeddings, args.exemplar_dir, EMBED_MODEL_ID)
    warn_about_prompt_aliases(args.name, aliases)

    print(f"\nRestart the grounding service with --exemplars {args.exemplar_dir} "
          f"to use it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
