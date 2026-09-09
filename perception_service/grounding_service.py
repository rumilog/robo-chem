"""
Open-vocabulary grounding service.

Turns a short noun phrase into a segmentation mask per camera. Runs in the
Python 3.10 perception_env because these models need a newer Python and the
GPU; the Python 3.8 frankapy environment that drives the arm calls it over
localhost.

This replaces asking GPT-4o for pixel bounding boxes. GPT-4o recognizes
objects well but regresses coordinates poorly, and it reported high confidence
on boxes that landed on bare table, which silently corrupted the fused
pointcloud.

Two backends, same HTTP contract:

  sam3   SAM 3 promptable concept segmentation (text -> masks directly).
         Preferred. On the cage images it scores ~0.95 on "white paper cup" in
         every camera and correctly ignores the glass beaker beside it.

  gdino  GroundingDINO (text -> boxes) + SAM (boxes -> masks), both ungated.
         The architecture the implementation plan originally specified, kept as
         a fallback. Scores only ~0.3-0.6 here and tends to expand its box to
         surrounding structure, so prefer sam3 unless the weights are missing.

Usage:
    perception_env/bin/python perception_service/grounding_service.py \
        --backend sam3 --model weights/sam3.pt
    perception_env/bin/python perception_service/grounding_service.py --backend gdino
"""

import argparse
import base64
import logging
import time

import cv2
import numpy as np
from flask import Flask, jsonify, request

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("grounding")

app = Flask(__name__)

STATE = {"backend": None, "name": None, "conf": 0.25}


# ---------------------------------------------------------------- backends


class GroundingDinoSamBackend:
    """
    GroundingDINO for open-vocabulary detection, SAM for mask refinement.

    Both models come from transformers and are ungated, so this works without
    the SAM 3 license.
    """

    name = "gdino"

    def __init__(
        self,
        device: str = "cuda",
        detector_id: str = "IDEA-Research/grounding-dino-base",
        segmenter_id: str = "facebook/sam-vit-base",
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
    ):
        import torch
        from transformers import (
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
            SamModel,
            SamProcessor,
        )

        self.torch = torch
        self.device = device if torch.cuda.is_available() else "cpu"
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        log.info(f"loading detector {detector_id} on {self.device}")
        self.det_processor = AutoProcessor.from_pretrained(detector_id)
        self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(
            detector_id
        ).to(self.device).eval()

        log.info(f"loading segmenter {segmenter_id} on {self.device}")
        self.sam_processor = SamProcessor.from_pretrained(segmenter_id)
        self.sam = SamModel.from_pretrained(segmenter_id).to(self.device).eval()
        log.info("backend ready")

    def detect(self, image_bgr: np.ndarray, prompt: str):
        """Return (boxes_xyxy, scores) for a text prompt."""
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # GroundingDINO expects lowercase phrases terminated with a period.
        text = prompt.strip().lower()
        if not text.endswith("."):
            text += "."

        inputs = self.det_processor(images=rgb, text=text, return_tensors="pt").to(
            self.device
        )
        with self.torch.no_grad():
            outputs = self.detector(**inputs)

        results = self.det_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[rgb.shape[:2]],
        )[0]

        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        return boxes, scores

    def segment_box(self, image_bgr: np.ndarray, box) -> np.ndarray:
        """Refine one box into a mask with SAM."""
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.sam_processor(
            rgb, input_boxes=[[[float(v) for v in box]]], return_tensors="pt"
        ).to(self.device)

        with self.torch.no_grad():
            outputs = self.sam(**inputs, multimask_output=True)

        masks = self.sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0][0]

        scores = outputs.iou_scores[0][0].cpu().numpy()
        return masks[int(np.argmax(scores))].numpy().astype(bool)

    def infer(self, image_bgr: np.ndarray, prompt: str,
              return_all: bool = False) -> dict:
        boxes, scores = self.detect(image_bgr, prompt)
        if len(boxes) == 0:
            return {"found": False, "reason": "no detection above threshold"}

        if not return_all:
            best = int(np.argmax(scores))
            mask = self.segment_box(image_bgr, boxes[best])
            return {
                "found": True,
                "mask": mask,
                "score": float(scores[best]),
                "box": [float(v) for v in boxes[best]],
                "num_instances": int(len(boxes)),
            }

        instances = []
        for box, score in zip(boxes, scores):
            mask = self.segment_box(image_bgr, box)
            instances.append({
                "mask": mask,
                "score": float(score),
                "box": [float(v) for v in box],
            })
        return {
            "found": True,
            "instances": instances,
            "num_instances": int(len(instances)),
            # Keep best as the primary mask for backward-compatible clients.
            "mask": instances[int(np.argmax(scores))]["mask"],
            "score": float(np.max(scores)),
            "box": [float(v) for v in boxes[int(np.argmax(scores))]],
        }


class Sam3Backend:
    """SAM 3 promptable concept segmentation: text straight to masks."""

    name = "sam3"

    def __init__(self, model_path: str = "sam3.pt", conf: float = 0.25):
        from ultralytics.models.sam import SAM3SemanticPredictor

        log.info(f"loading SAM 3 from {model_path}")
        self.predictor = SAM3SemanticPredictor(
            overrides={
                "conf": conf,
                "task": "segment",
                "mode": "predict",
                "model": model_path,
                "save": False,
                "verbose": False,
            }
        )
        log.info("backend ready")

    def _resize_mask(self, mask: np.ndarray, h: int, w: int) -> np.ndarray:
        mask = mask.astype(bool)
        if mask.shape != (h, w):
            mask = cv2.resize(
                mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        return mask

    def infer(self, image_bgr: np.ndarray, prompt: str,
              return_all: bool = False) -> dict:
        self.predictor.set_image(image_bgr)
        out = self.predictor(text=[prompt])

        if out is None or len(out) == 0:
            return {"found": False, "reason": "no results"}

        res = out[0]
        masks = getattr(res, "masks", None)
        boxes = getattr(res, "boxes", None)

        if masks is None or masks.data is None or len(masks.data) == 0:
            return {"found": False, "reason": "no masks"}

        mask_data = masks.data.cpu().numpy()
        h, w = image_bgr.shape[:2]

        if boxes is not None and getattr(boxes, "conf", None) is not None and len(boxes.conf):
            scores = boxes.conf.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()
        else:
            scores = np.ones(len(mask_data), dtype=float)
            xyxy = None

        if not return_all:
            best = int(np.argmax(scores))
            score = float(scores[best])
            box = xyxy[best].tolist() if xyxy is not None else None
            mask = self._resize_mask(mask_data[best], h, w)
            return {
                "found": True,
                "mask": mask,
                "score": score,
                "box": box,
                "num_instances": int(len(mask_data)),
            }

        instances = []
        for i in range(len(mask_data)):
            mask = self._resize_mask(mask_data[i], h, w)
            box = xyxy[i].tolist() if xyxy is not None else _mask_to_box(mask)
            instances.append({
                "mask": mask,
                "score": float(scores[i]),
                "box": [float(v) for v in box] if box is not None else None,
            })
        best = int(np.argmax(scores))
        return {
            "found": True,
            "instances": instances,
            "num_instances": int(len(instances)),
            "mask": instances[best]["mask"],
            "score": instances[best]["score"],
            "box": instances[best]["box"],
        }


def _mask_to_box(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


# ---------------------------------------------------------------- transport


def decode_image(b64: str) -> np.ndarray:
    buf = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode image")
    return img


def encode_mask(mask: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", mask.astype(np.uint8) * 255)
    if not ok:
        raise ValueError("could not encode mask")
    return base64.b64encode(buf.tobytes()).decode("ascii")


@app.get("/health")
def health():
    return jsonify(
        {
            "ok": STATE["backend"] is not None,
            "backend": STATE["name"],
            "conf": STATE["conf"],
        }
    )


@app.post("/segment")
def segment():
    """
    Request JSON:
        {"prompt": "white paper cup", "images": {"2": "<b64 jpeg>", ...},
         "return_all": false}

    Response JSON (return_all=false, default):
        {"prompt": ..., "backend": ..., "results": {
            "2": {"found": true, "mask": "<b64 png>", "score": 0.71,
                  "box": [...], "num_instances": 1, "area_px": 8123},
            "3": {"found": false, "reason": "..."}
        }}

    With return_all=true, each found camera also includes:
        "instances": [{"mask": "<b64 png>", "score": ..., "box": [...],
                       "area_px": ...}, ...]
    """
    payload = request.get_json(force=True)
    prompt = payload.get("prompt")
    images = payload.get("images") or {}
    return_all = bool(payload.get("return_all", False))

    if not prompt:
        return jsonify({"error": "missing 'prompt'"}), 400
    if not images:
        return jsonify({"error": "missing 'images'"}), 400

    backend = STATE["backend"]
    if backend is None:
        return jsonify({"error": "backend not loaded"}), 503

    results = {}
    for cam_id, b64 in images.items():
        t0 = time.time()
        try:
            image = decode_image(b64)
        except Exception as e:
            results[cam_id] = {"found": False, "error": f"decode failed: {e}"}
            continue

        try:
            entry = backend.infer(image, prompt, return_all=return_all)
        except Exception as e:
            log.exception(f"inference failed on cam {cam_id}")
            results[cam_id] = {"found": False, "error": f"inference failed: {e}"}
            continue

        if entry.get("found"):
            instances = entry.pop("instances", None)
            mask = entry.pop("mask")
            entry["area_px"] = int(mask.sum())
            entry["shape"] = [int(mask.shape[0]), int(mask.shape[1])]
            entry["mask"] = encode_mask(mask)

            if instances is not None:
                encoded_instances = []
                for inst in instances:
                    m = inst["mask"]
                    encoded_instances.append({
                        "mask": encode_mask(m),
                        "score": inst.get("score"),
                        "box": inst.get("box"),
                        "area_px": int(m.sum()),
                    })
                entry["instances"] = encoded_instances

        entry["elapsed_s"] = round(time.time() - t0, 3)
        results[cam_id] = entry

        log.info(
            f"cam {cam_id}: {prompt!r} found={entry.get('found')} "
            f"score={entry.get('score')} instances={entry.get('num_instances')} "
            f"return_all={return_all} area={entry.get('area_px')} "
            f"({entry['elapsed_s']}s)"
        )

    return jsonify({
        "prompt": prompt,
        "backend": STATE["name"],
        "return_all": return_all,
        "results": results,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["gdino", "sam3"], default="gdino")
    parser.add_argument("--model", default="weights/sam3.pt", help="sam3 backend only")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--box-threshold", type=float, default=0.3, help="gdino only")
    parser.add_argument("--text-threshold", type=float, default=0.25, help="gdino only")
    args = parser.parse_args()

    if args.backend == "sam3":
        STATE["backend"] = Sam3Backend(args.model, args.conf)
    else:
        STATE["backend"] = GroundingDinoSamBackend(
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
        )

    STATE["name"] = STATE["backend"].name
    STATE["conf"] = args.conf

    log.info(f"serving {STATE['name']} on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=False)


if __name__ == "__main__":
    main()
