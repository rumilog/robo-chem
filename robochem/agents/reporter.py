"""
The experiment report, written by the model when the operator asks for one
(``scripts/write_report.py``); runs never write it by themselves.

What goes in:
  * the run record(s): the goal and the parsed instruction sheet, the bench
    the scene agent saw, the plan, and every step the robot executed with
    its parameters, outcome, timing and the values the skills measured. A run
    that was stopped and resumed (``--resume``) is several records chained by
    ``resumed_from.started``; :func:`load_run_chain` puts them back together,
    so the report covers the whole experiment, failures and restarts included;
  * the cameras, which the model may use while it writes (:class:`BenchCamera`):
    overview photos of the bench to start with, then ``look_at(object)`` for
    a close-up of one object -- say, the colour of the liquid in a cup -- and
    ``photograph_bench(cameras)`` for fresh views. They are the only source
    for what the chemistry did;
  * the console log of the run, when one was kept (``... | tee
    experiments/<name>.log``): for each failed step, the skill's own lines
    (``[Scoop] JAMMED at ...``), which the record keeps only as a one-line
    error. :func:`console_excerpts` finds them.

What comes out, in ``<out_dir>/report_<time>/``: ``report.md`` (always in
English), every photo the model looked at, and ``facts.json`` (what it was
given, and the camera calls it made).

The model writes the prose (summary, procedure, observations, comparison
with the expected outcome, issues, conclusion). The step table, timings and
measured values are rendered from the record itself, never by the model, so
the report cannot misstate them.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import config, llm_client

AGENT_NAME = "report-writer"
SCHEMA_NAME = "experiment_report"

_TEXT = {"type": "string"}

REPORT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["title", "summary", "objective", "materials", "procedure",
                 "observations", "result_vs_expected", "issues", "conclusion",
                 "next_steps"],
    "properties": {
        "title": _TEXT,
        "summary": _TEXT,
        "objective": _TEXT,
        "materials": {"type": "array", "items": _TEXT},
        "procedure": _TEXT,
        "observations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["subject", "observed", "source"],
                "properties": {
                    "subject": _TEXT,
                    "observed": _TEXT,
                    "source": {"type": "string", "enum": ["photos", "record"]},
                },
            },
        },
        "result_vs_expected": _TEXT,
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["subtask", "what_happened", "likely_cause"],
                "properties": {
                    "subtask": {"type": "integer"},
                    "what_happened": _TEXT,
                    "likely_cause": _TEXT,
                },
            },
        },
        "conclusion": _TEXT,
        "next_steps": {"type": "array", "items": _TEXT},
    },
}

_SYSTEM_PROMPT = """\
You write the laboratory report, in English, for an experiment that a robot
arm has carried out on a bench. You are given FACTS, the robot's own record of
the run as JSON; RUN PHOTOS, when the run took them, of the bench right after
some steps (each pour); and, when the cameras are available, PHOTOS of the
bench from the fixed cameras around it, taken now.

RUN PHOTOS show each container as it was at that stage, before later steps
changed it. A cup that was later poured out is empty in the photos taken now;
its colour at each stage is in the RUN PHOTOS. Use them for the colour every
solution reached, say which stage a photo shows ("after sub-task 24, clear cup
b held a blue liquid"), and use source "photos" for them too.

When the cameras are available you can use them while you write:
- look_at(object_name): close-up photos of one bench object from the cameras
  that see it best, e.g. to read the colour, the level, foam or bubbles of the
  liquid in a cup.
- photograph_bench(cameras): fresh photos of the whole bench.
Look at every container the experiment used before you describe it: the
colour of each solution is the result of this experiment. Do not repeat a
call that already answered the question; each takes a few seconds.

Rules:
- Report only what FACTS or the PHOTOS support. Never invent quantities,
  colours, timings, readings or outcomes. If something the instruction sheet
  asks for was not done, or cannot be judged, say so plainly.
- FACTS.steps lists what the robot actually executed, in order. A run can have
  several sessions: a session ends when a step fails or the operator stops it,
  and the next session resumes from a sub-task. Failed steps and resumptions
  are part of the experiment; describe them, do not hide them.
- Colours, liquid levels, foam or fizzing come only from the PHOTOS. The cups
  are clear plastic and stand on paper labels: describe what is visible in
  each container you can identify, and say "not visible in the photos" when a
  container or its contents cannot be judged. Use source "photos" for these
  and source "record" for facts taken from FACTS. With no photos, give no
  colour at all.
- An observation with source "photos" says only what the photos show
  ("clear, colourless liquid about 5 mm deep"); what the record implies
  about it ("baking soda was never added") is a separate observation with
  source "record". Never mix the two, and never write "presumed".
- Compare the outcome with the instruction sheet's expected outcome
  (FACTS.instruction.expected_outcome) and with the plan's expected
  observation, for the parts that were actually performed.
- issues: one entry per failed step, refusal or anomaly in FACTS, with its
  sub-task number (0 if it belongs to no sub-task), what happened, and the
  cause as FACTS states it; write "cause not recorded" rather than guess.
- A failure message is quoted, not reinterpreted: "did not lift clear" is
  about the arm rising out of the container, not about the powder.
- The photos were taken when this report was written (FACTS.photos_taken),
  which can be some time after the run ended (FACTS.sessions); say so if it
  matters, e.g. a colour that may have developed or faded since.
- FACTS.console_excerpts, when present, hold the skill's own console lines
  for each failed step, keyed "session.subtask". Use their numbers to say
  what happened (e.g. "the sweep stopped at 35 deg, 34 mm from its
  waypoint"); still state no cause they do not show.
- FACTS.operator_notes, when present, are what the operator added after the
  run: diagnoses, or what was seen at the bench. Use them where they apply
  and attribute them ("according to the operator, ..."); a note that
  explains a recorded failure goes into that issue's likely_cause.
- procedure: how the robot carried out the instruction sheet, mapping its
  steps onto the sub-tasks that were executed, and what was skipped, adapted
  or not reached.
- materials: the reagents, containers and tools that were used, with the
  amounts the record gives.
- Past tense, third person ("the robot ..."), concise and factual. Lengths in
  mm, angles in degrees, times in seconds.
- Write every text field in English. Keep object names exactly as in FACTS.
"""

#: Result keys that are poses, point sets or similar: long and not readable.
_SKIP_KEYS = ("pose", "points", "matrix", "transform", "path", "waypoints_xyz")


def measurements(result: Optional[dict], limit: int = 30) -> Dict[str, Any]:
    """A step result's readable values: scalars, short strings, short vectors."""
    out: Dict[str, Any] = {}
    for key, value in (result or {}).items():
        if len(out) >= limit or any(s in key for s in _SKIP_KEYS):
            continue
        if isinstance(value, bool):
            out[key] = value
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            out[key] = round(float(value), 4)
        elif isinstance(value, str) and value and len(value) <= 200:
            out[key] = value
        elif (isinstance(value, (list, tuple)) and 0 < len(value) <= 3
              and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)):
            out[key] = [round(float(v), 4) for v in value]
    return out


def load_run_chain(path) -> Tuple[List[dict], List[str]]:
    """
    A run record and every earlier record it resumed from, oldest first.

    A resumed record names its predecessor only by ``resumed_from.started``,
    so the predecessor is looked up among the records in the same folder.
    """
    path = Path(path)
    records = [json.loads(path.read_text(encoding="utf-8"))]
    paths = [str(path)]
    by_start = {}
    for other in path.parent.glob("agent_run_*.json"):
        try:
            started = json.loads(other.read_text(encoding="utf-8")).get("started")
        except (OSError, json.JSONDecodeError):
            continue
        if started:
            by_start.setdefault(started, other)
    seen = {records[0].get("started")}
    while True:
        prev = (records[0].get("resumed_from") or {}).get("started")
        if not prev or prev in seen or prev not in by_start:
            break
        seen.add(prev)
        older = by_start[prev]
        records.insert(0, json.loads(older.read_text(encoding="utf-8")))
        paths.insert(0, str(older))
    return records, paths


def latest_record(folder) -> Optional[Path]:
    """
    The newest run record that actually moved the robot: dry runs (plans
    only) and records with no executed step are skipped.
    """
    for path in sorted(Path(folder).glob("agent_run_*.json"), reverse=True):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not record.get("dry_run") and (record.get("steps") or record.get("resumed_steps")):
            return path
    return None


def _epoch(started: Optional[str]) -> Optional[float]:
    try:
        return time.mktime(time.strptime(str(started), "%Y-%m-%d %H:%M:%S"))
    except (TypeError, ValueError):
        return None


def find_console_logs(folder, records: Sequence[dict]) -> List[Path]:
    """
    Console logs (``*.log``) in ``folder`` written to since the experiment
    started: what ``tee`` kept of the run's terminal output.
    """
    since = _epoch(records[0].get("started")) if records else None
    logs = []
    for path in sorted(Path(folder).glob("*.log")):
        try:
            if since is None or path.stat().st_mtime >= since - 60:
                logs.append(path)
        except OSError:
            continue
    return logs


#: Console lines worth quoting: what the skills themselves said.
_SKILL_PREFIXES = ("[Scoop]", "[Stir]", "[Dump]", "[Pour]", "[PickUp]", "[Place]",
                   "[Safety]", "[Skills]", "[Dispense]", "[Gripper]")


def console_excerpts(steps: Sequence[dict], log_paths: Sequence, max_lines: int = 40
                     ) -> Dict[str, List[str]]:
    """
    For each failed step, the skill lines the console printed for it.

    ``run_experiment.py`` heads every sub-task with ``[step N] <sub-task>``
    (the ``--step`` prompt repeats ``[step N] Enter to run it ...`` on the
    next line, with the first skill line after it). The failed attempt is
    the last block under that heading that contains a failure.
    """
    import re

    lines: List[str] = []
    for path in log_paths:
        try:
            lines += Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
    heading = re.compile(r"^\[step (\d+)\] (?!Enter to run it)")
    out: Dict[str, List[str]] = {}
    for st in steps:
        if st.get("success"):
            continue
        n, desc = st.get("subtask"), str(st.get("description") or "")
        starts = [i for i, l in enumerate(lines)
                  if l.strip().startswith(f"[step {n}] {desc}")]
        chosen = None
        for i in reversed(starts):
            block = []
            for l in lines[i + 1:]:
                text = l.strip()
                if heading.match(text) or text.startswith("[failed]") or text.startswith("[success]"):
                    if text.startswith("[failed]"):
                        block.append(text)
                    break
                if "q to stop here:" in text:
                    text = text.split("q to stop here:", 1)[1].strip()
                block.append(text)
            if any("FAILED" in b or b.startswith("[failed]") for b in block):
                chosen = block
                break
        if chosen:
            keep = [b for b in chosen if b.startswith(_SKILL_PREFIXES)]
            out[f"{st.get('session')}.{n}"] = keep[-max_lines:]
    return out


class BenchCamera:
    """
    The report writer's eyes on the cage: photos of the bench, and close-ups
    of one object, taken when the model asks for them. Read-only; the robot
    is never commanded.

    A close-up locates the object as the skills do (``vision.locate``),
    projects its box into every camera and crops the two views that show it
    largest and whole, scaled up so a 4 cm cup fills the picture.
    """

    def __init__(self, vision, cameras=(2, 3, 4, 5), color_order: str = "bgr"):
        self.vision = vision
        self.cameras = [int(c) for c in cameras]
        self.color_order = color_order

    def names(self) -> List[str]:
        bench = getattr(self.vision, "bench", None)
        return [o.name for o in bench.objects] if bench is not None else []

    def _capture(self) -> dict:
        loc = self.vision.object_localizer
        loc.camera_ids = list(self.cameras)
        return loc.capture_pointclouds()

    def photos(self, cameras=None) -> List[Tuple[int, np.ndarray]]:
        data = self._capture()
        images = data.get("images") or {}
        return [(c, images[c]) for c in (cameras or self.cameras) if c in images]

    def close_up(self, name: str, views: int = 2, size: int = 512
                 ) -> Tuple[str, List[Tuple[int, np.ndarray]]]:
        import cv2

        located = self.vision.locate(name, force_refresh=True)
        points = None if not located else located.get("points")
        if points is None or len(points) == 0:
            return f"'{name}' was not found by the cameras.", []
        pts = np.asarray(points, dtype=float)
        lo, hi = pts.min(axis=0) - 0.01, pts.max(axis=0) + 0.01
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                            for z in (lo[2], hi[2])])
        data = self._capture()
        loc = self.vision.object_localizer
        found = []
        for cam in self.cameras:
            image = (data.get("images") or {}).get(cam)
            intr = (data.get("intrinsics") or {}).get(cam)
            if image is None or intr is None:
                continue
            T = loc._extrinsics(cam)
            T = np.asarray(T.matrix() if hasattr(T, "matrix") else T, dtype=float)
            Ti = np.linalg.inv(T)
            q = corners @ Ti[:3, :3].T + Ti[:3, 3]
            if np.any(q[:, 2] <= 0.05):
                continue
            cx = float(getattr(intr, "cx", getattr(intr, "ppx", 0.0)))
            cy = float(getattr(intr, "cy", getattr(intr, "ppy", 0.0)))
            u = float(intr.fx) * q[:, 0] / q[:, 2] + cx
            v = float(intr.fy) * q[:, 1] / q[:, 2] + cy
            h, w = image.shape[:2]
            x0, x1, y0, y1 = u.min(), u.max(), v.min(), v.max()
            area = max(1.0, (x1 - x0) * (y1 - y0))
            inside = (max(0.0, min(x1, w) - max(x0, 0.0)) * max(0.0, min(y1, h) - max(y0, 0.0))
                      / area)
            if inside < 0.6:
                continue
            pad = 0.25 * max(x1 - x0, y1 - y0) + 8
            X0, X1 = int(max(0, x0 - pad)), int(min(w, x1 + pad))
            Y0, Y1 = int(max(0, y0 - pad)), int(min(h, y1 + pad))
            if X1 - X0 < 16 or Y1 - Y0 < 16:
                continue
            crop = image[Y0:Y1, X0:X1]
            scale = float(size) / max(crop.shape[:2])
            if scale > 1.0:
                crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            found.append((area * inside, cam, crop))
        found.sort(key=lambda t: -t[0])
        chosen = [(cam, crop) for _, cam, crop in found[:views]]
        centre = np.asarray(located.get("centroid", pts.mean(axis=0)), dtype=float)
        how = located.get("localized_by") or ""
        info = (f"'{name}' located at x {centre[0]:.3f} m, y {centre[1]:+.3f} m"
                + (f" ({how})" if how else ""))
        info += (f"; close-ups from cameras {[cam for cam, _ in chosen]}." if chosen
                 else "; no camera has it whole in view.")
        return info, chosen


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(text).lower()).strip("_")


def camera_tools(camera: BenchCamera) -> List[dict]:
    """The function definitions the model is offered."""
    names = camera.names()
    object_name = {"type": "string", "enum": names} if names else {"type": "string"}
    return [
        {"type": "function", "function": {
            "name": "look_at",
            "description": ("Close-up photos of one bench object from the cameras that see it "
                            "best, e.g. to read the colour, level, foam or bubbles of the "
                            "liquid in a cup. Returns where it was found and the photos."),
            "parameters": {"type": "object", "additionalProperties": False,
                           "required": ["object_name"],
                           "properties": {"object_name": object_name}},
            "strict": True}},
        {"type": "function", "function": {
            "name": "photograph_bench",
            "description": ("Fresh photos of the whole bench from the given cameras; "
                            "an empty list means all of them."),
            "parameters": {"type": "object", "additionalProperties": False,
                           "required": ["cameras"],
                           "properties": {"cameras": {"type": "array", "items": {
                               "type": "integer", "enum": list(camera.cameras)}}}},
            "strict": True}},
    ]


def executed_steps(records: Sequence[dict]) -> List[dict]:
    """Every step executed across the sessions, in order, with its session number."""
    steps = []
    first = records[0] if records else {}
    if first.get("resumed_from") and first.get("resumed_steps"):
        # The earliest record we have is itself a resume and its predecessor
        # was not found: keep what it carried, marked as an earlier session.
        for st in first["resumed_steps"]:
            steps.append(dict(st, session=0))
    for n, record in enumerate(records, start=1):
        for st in record.get("steps") or []:
            steps.append(dict(st, session=n))
    return steps


def build_facts(records: Sequence[dict]) -> dict:
    """What the model is given: the run, compacted, and nothing else."""
    last = records[-1]
    first = records[0]
    plan = (last.get("plans") or [{}])[-1]
    steps = []
    for st in executed_steps(records):
        steps.append({
            "session": st.get("session"),
            "subtask": st.get("subtask_index"),
            "description": st.get("subtask"),
            "skill": st.get("skill"),
            "params": st.get("params"),
            "success": bool(st.get("success")),
            "failure": st.get("reason") or None,
            "seconds": st.get("seconds"),
            "measured": measurements(st.get("result")),
            "photos_after": len(st.get("photos") or []),
        })
    scene = first.get("scene") or last.get("scene") or {}
    return {
        "goal": last.get("goal"),
        "instruction": first.get("instruction") or last.get("instruction"),
        "bench": [{"name": o.get("name"), "contents": o.get("contents")}
                  for o in scene.get("objects") or []],
        "scene_notes": scene.get("scene_notes"),
        "plan": [{"subtask": s.get("index"), "description": s.get("description")}
                 for s in plan.get("subtasks") or []],
        "expected_observation": plan.get("expected_observation"),
        "sessions": [{"session": n, "started": r.get("started"),
                      "resumed_from_subtask": (r.get("resumed_from") or {}).get("from_step"),
                      "outcome": r.get("outcome"), "reason": r.get("reason"),
                      "seconds": r.get("seconds")}
                     for n, r in enumerate(records, start=1)],
        "steps": steps,
        "final_outcome": last.get("outcome"),
        "final_reason": last.get("reason"),
        "verifications": last.get("verifications") or [],
    }


def write_report(records: Sequence[dict], *, camera: Optional[BenchCamera] = None,
                 record_paths: Optional[Sequence[str]] = None, out_dir="experiments",
                 model: Optional[str] = None, notes: Sequence[str] = (),
                 console_logs: Sequence = (), max_tool_rounds: int = 8,
                 max_images: int = 24, max_run_photos: int = 24,
                 client: Any = None) -> Path:
    """
    Write the report for one experiment, in English, and return the path of
    ``report.md``.

    Args:
        records: the run record(s), oldest first (see :func:`load_run_chain`).
        camera: the cameras, or None to write from the records alone. With
            them, the model starts from overview photos and may call
            ``look_at`` / ``photograph_bench`` (up to ``max_tool_rounds``
            rounds, ``max_images`` photos) before it writes.
        max_run_photos: how many of the photos the run took after its steps
            (the step records' ``photos``) the model is shown; they are thinned
            evenly across the steps when there are more.
        record_paths: the record files, quoted in the report.
        out_dir: where ``report_<time>/`` is created.
        model: overrides the strong tier.
        notes: the operator's remarks, e.g. a diagnosis of a failure.
        console_logs: the run's console logs, quoted for failed steps.
        client: transport double, for tests.
    """
    import cv2

    model = model or config.strong_model()
    out = Path(out_dir) / f"report_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    order = camera.color_order if camera is not None else "bgr"
    shown: List[Tuple[str, str]] = []          # (caption, file) in the order seen
    budget = {"images": int(max_images)}

    def keep(image, name, caption):
        image = np.asarray(image)
        if str(order).lower() == "rgb":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out / name), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        shown.append((caption, name))

    facts = build_facts(records)
    if notes:
        facts["operator_notes"] = [str(n) for n in notes]
    excerpts = console_excerpts(facts["steps"], console_logs)
    if excerpts:
        facts["console_excerpts"] = excerpts

    overview = []
    if camera is not None:
        overview = camera.photos()
        facts["photos_taken"] = time.strftime("%Y-%m-%d %H:%M:%S")

    content = [llm_client.text_block("FACTS:\n" + json.dumps(facts, default=str))]
    content += _run_photo_blocks(executed_steps(records), out, shown, max_run_photos,
                                 record_paths or [])
    for cam, frame in overview:
        keep(frame, f"overview_cam{cam}.jpg", f"camera {cam}, overview")
    budget["images"] -= len(overview)

    if overview:
        content.append(llm_client.text_block(
            f"PHOTOS: {len(overview)} overview photos taken at {facts['photos_taken']}, one "
            f"per fixed camera, in this order: "
            + ", ".join(f"camera {cam}" for cam, _ in overview) + "."))
        for _, frame in overview:
            content.append(llm_client.image_block_from_array(frame, color_order=order))
    else:
        content.append(llm_client.text_block("PHOTOS: none; the cameras are not available."))

    calls: List[dict] = []

    def look_at(object_name):
        if budget["images"] <= 0:
            return "The photo budget is used up; write from what you have.", []
        info, crops = camera.close_up(object_name)
        blocks = []
        for cam, crop in crops[:max(0, budget["images"])]:
            n = sum(1 for _, f in shown if f.startswith("closeup_")) + 1
            keep(crop, f"closeup_{n:02d}_{_slug(object_name)}_cam{cam}.jpg",
                 f"{object_name}, camera {cam}")
            blocks.append(llm_client.image_block_from_array(crop, color_order=order))
            budget["images"] -= 1
        return info, blocks

    def photograph_bench(cameras):
        if budget["images"] <= 0:
            return "The photo budget is used up; write from what you have.", []
        shots = camera.photos(list(cameras) or None)
        blocks = []
        for cam, frame in shots[:max(0, budget["images"])]:
            n = sum(1 for _, f in shown if f.startswith("photo_")) + 1
            keep(frame, f"photo_{n:02d}_cam{cam}.jpg", f"camera {cam}, fresh view")
            blocks.append(llm_client.image_block_from_array(frame, color_order=order))
            budget["images"] -= 1
        return f"Fresh photos from cameras {[cam for cam, _ in shots]}.", blocks

    def record_call(name, args, text):
        print(f"[report] {name}({json.dumps(args)}): {text}")
        calls.append({"tool": name, "args": args, "result": text})

    if camera is not None:
        report = llm_client.tool_completion(
            agent=AGENT_NAME, model=model, system_prompt=_SYSTEM_PROMPT,
            user_content=content, tools=camera_tools(camera),
            handlers={"look_at": look_at, "photograph_bench": photograph_bench},
            schema=REPORT_SCHEMA, schema_name=SCHEMA_NAME,
            max_rounds=max_tool_rounds, temperature=0.2, client=client, on_call=record_call)
    else:
        report = llm_client.structured_completion(
            agent=AGENT_NAME, model=model, system_prompt=_SYSTEM_PROMPT,
            user_content=content, schema=REPORT_SCHEMA, schema_name=SCHEMA_NAME,
            temperature=0.2, client=client)
    facts["camera_calls"] = calls
    (out / "facts.json").write_text(json.dumps(facts, indent=2, default=str), encoding="utf-8")
    path = out / "report.md"
    path.write_text(render_markdown(report, facts, photos=shown,
                                    record_paths=record_paths or [], model=model,
                                    console_logs=[str(c) for c in console_logs]),
                    encoding="utf-8")
    return path


def _run_photo_blocks(steps: Sequence[dict], out: Path, shown: List[Tuple[str, str]],
                      limit: int, record_paths: Sequence[str]) -> List[dict]:
    """
    The photos the run took after its steps, as content blocks in run order,
    each group headed by the step it follows. Copied into the report folder
    and added to ``shown``. Photos that are no longer on disk are skipped.
    """
    import cv2

    groups = []
    for st in steps:
        found = []
        for p in st.get("photos") or []:
            path = Path(p)
            if not path.is_file():
                # The folder may have moved with the records.
                for rp in record_paths:
                    alt = Path(rp).parent / path.parent.name / path.name
                    if alt.is_file():
                        path = alt
                        break
            if path.is_file():
                found.append(path)
        if found:
            groups.append((st, found))
    if not groups or limit <= 0:
        return []
    per_step = max(1, int(limit) // len(groups))
    blocks = [llm_client.text_block(
        "RUN PHOTOS: taken by the robot during the run, right after the step named, "
        "with the arm at home. Each shows the bench as it was at that stage.")]
    for st, paths in groups:
        if len(paths) > per_step:
            step = len(paths) / float(per_step)
            paths = [paths[int(i * step)] for i in range(per_step)]
        caption = (f"after sub-task {st.get('subtask_index')} "
                   f"(session {st.get('session')}): {st.get('subtask')}")
        blocks.append(llm_client.text_block(
            f"{caption}: {len(paths)} photo(s), "
            + ", ".join(p.stem.split("_")[-1] for p in paths) + "."))
        for path in paths:
            image = cv2.imread(str(path))
            if image is None:
                continue
            name = f"run_s{int(st.get('subtask_index') or 0):02d}_{path.name}"
            cv2.imwrite(str(out / name), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            shown.append((f"{caption}, {path.stem.split('_')[-1]}", name))
            blocks.append(llm_client.image_block_from_array(image, color_order="bgr"))
    return blocks


def _cell(text) -> str:
    """Text for a markdown table cell."""
    return " ".join(str(text if text is not None else "").split()).replace("|", "\\|")


def render_markdown(report: dict, facts: dict, *, photos: Sequence[Tuple[str, str]] = (),
                    record_paths: Sequence[str] = (), model: str = "",
                    console_logs: Sequence[str] = ()) -> str:
    steps = facts.get("steps") or []
    done = sum(1 for s in steps if s.get("success"))
    robot_seconds = sum(float(s.get("seconds") or 0) for s in steps)
    sessions = facts.get("sessions") or []
    when = sessions[0].get("started") if sessions else ""
    lines = [f"# {report.get('title', 'Experiment report')}", ""]
    lines += [f"**Run** {when} · **outcome** {facts.get('final_outcome')} · "
              f"**steps** {done} of {len(steps)} succeeded · **sessions** {len(sessions)} · "
              f"**robot time** {robot_seconds:.0f} s", ""]

    lines += ["## Summary", "", report.get("summary", ""), ""]
    lines += ["## Objective", "", report.get("objective", ""), ""]
    lines += ["## Materials", ""] + [f"- {m}" for m in report.get("materials") or []] + [""]

    lines += ["## Procedure", "", report.get("procedure", ""), ""]
    lines += ["| Session | Sub-task | What the robot did | Skill | Result | Time (s) |",
              "|---|---|---|---|---|---|"]
    for s in steps:
        result = "ok" if s.get("success") else f"**FAILED**: {_cell(s.get('failure'))}"
        lines.append(f"| {s.get('session')} | {s.get('subtask')} | {_cell(s.get('description'))} "
                     f"| {s.get('skill')} | {result} | {float(s.get('seconds') or 0):.0f} |")
    lines.append("")

    lines += ["## Observations", ""]
    if photos:
        lines += [f"*Photos taken from {facts.get('photos_taken', '')}, while this report was "
                  f"written: the overview, then what the writer chose to look at.*", ""]
        for caption, name in photos:
            lines += [f"![{caption}]({name})", f"*{caption}*", ""]
    else:
        lines += ["*No photos were taken for this report.*", ""]
    for o in report.get("observations") or []:
        lines.append(f"- **{o.get('subject')}**: {o.get('observed')} *({o.get('source')})*")
    lines.append("")

    expected = (facts.get("instruction") or {}).get("expected_outcome")
    lines += ["## Result versus the expected outcome", ""]
    if expected:
        lines += [f"> Expected (instruction sheet): {expected}", ""]
    lines += [report.get("result_vs_expected", ""), ""]

    lines += ["## Issues", ""]
    issues = report.get("issues") or []
    if issues:
        lines += ["| Sub-task | What happened | Likely cause |", "|---|---|---|"]
        for i in issues:
            sub = i.get("subtask") or "-"
            lines.append(f"| {sub} | {_cell(i.get('what_happened'))} | {_cell(i.get('likely_cause'))} |")
    else:
        lines.append("None recorded.")
    lines.append("")

    if facts.get("operator_notes"):
        lines += ["## Operator notes", ""] + [f"- {n}" for n in facts["operator_notes"]] + [""]

    lines += ["## Conclusion", "", report.get("conclusion", ""), ""]
    if report.get("next_steps"):
        lines += ["## Next steps", ""] + [f"- {n}" for n in report["next_steps"]] + [""]

    lines += ["## Appendix: values measured at each step", ""]
    for s in steps:
        values = s.get("measured") or {}
        if not values:
            continue
        shown = "; ".join(f"{k} = {v}" for k, v in values.items())
        lines.append(f"- **{s.get('session')}.{s.get('subtask')} {s.get('skill')}**: {_cell(shown)}")
    lines.append("")

    if facts.get("camera_calls"):
        lines += ["## Appendix: what the writer looked at", ""]
        for c in facts["camera_calls"]:
            args = ", ".join(f"{k}={v}" for k, v in (c.get("args") or {}).items())
            lines.append(f"- `{c.get('tool')}({args})`: {_cell(c.get('result'))}")
        lines.append("")

    if facts.get("console_excerpts"):
        lines += ["## Appendix: console lines of the failed steps", ""]
        for key, quoted in facts["console_excerpts"].items():
            lines += [f"**Session {key.split('.')[0]}, sub-task {key.split('.')[1]}**", "",
                      "```"] + list(quoted) + ["```", ""]

    lines += ["---", "",
              f"*The header, the procedure table and the appendix are taken from the run "
              f"record; the prose and the observations were written by {model} from that "
              f"record and the photos. Records: "
              + ", ".join(f"`{p}`" for p in record_paths)
              + (". Console logs: " + ", ".join(f"`{c}`" for c in console_logs)
                 if console_logs else "") + "*", ""]
    return "\n".join(lines)


__all__ = ["REPORT_SCHEMA", "BenchCamera", "camera_tools", "latest_record",
           "load_run_chain", "find_console_logs",
           "console_excerpts", "executed_steps", "build_facts", "write_report",
           "render_markdown", "measurements"]
