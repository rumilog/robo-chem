"""
Run a skill in simulation and film it, so the *motion* can be reviewed.

`smoke_test_sim.py` answers "did it report success"; this answers "does it look
right". It picks a tool up, runs the skill under test, and writes a filmstrip of
the stroke from a camera placed to show its profile -- plus, when the source cup
has granules in it, a count of how many ended up in the bowl, which is the one
objective number available for a scoop.

    sim_env/bin/python scripts/film_sim_skill.py --skill arc_scoop
    sim_env/bin/python scripts/film_sim_skill.py --skill scoop --label baseline
    sim_env/bin/python scripts/film_sim_skill.py --skill arc_scoop \
        --params "{'exit_angle_deg': 70, 'cup_tilt_deg': 25}"
    sim_env/bin/python scripts/film_sim_skill.py --skill scoop --then dump \
        --target "white paper cup"

Frames are captured off the physics loop, so the filmstrip is the motion as it
actually ran, not a re-enactment of the commanded poses.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataclasses import replace

from robochem.sim import build_cell
from robochem.sim.bench import Bench
from robochem.sim.powder import ScoopTally, bed_from_prop

ROOT = Path(__file__).resolve().parents[1]

# Camera placements, as (azimuth, elevation, distance) around the source cup.
# "side" looks across the stroke, which runs along +X, so the arc is seen edge
# on -- that is the view that shows whether the path curves.
VIEWS = {
    "side":  (-90.0, -6.0, 0.30),
    "raised": (-80.0, -20.0, 0.32),
    "front": (0.0, -10.0, 0.32),
    "iso":   (-55.0, -18.0, 0.38),
}


def parse_params(raw):
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return ast.literal_eval(raw)


class Film:
    """Grabs frames off the physics loop at a fixed interval of sim time."""

    def __init__(self, cell, view, size=(640, 480), every=0.12, cap=400):
        self.cell = cell
        self.model = cell.scene.model
        self.data = cell.scene.data
        self.renderer = mujoco.Renderer(self.model, size[1], size[0])
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, self.camera)
        azimuth, elevation, distance = VIEWS[view]
        self.camera.azimuth, self.camera.elevation = azimuth, elevation
        self.camera.distance = distance
        self.every = every
        self.cap = cap
        self.frames = []
        self.track = []                  # (t, bowl centre in world) per frame
        self._last = -1e9
        self._original = None
        self._bowl = None

    def look_at(self, xyz):
        self.camera.lookat[:] = np.asarray(xyz, float)

    def track_bowl(self, prop):
        """Follow this prop's bowl centre -- the part that does the scooping."""
        self._bowl = (self.cell.scene.prop_bodies[prop.name],
                      np.asarray(prop.bowl_offset, float))

    def see_through(self, prop, alpha=0.25):
        """Make a prop translucent, so the camera can watch inside it."""
        bid = self.cell.scene.prop_bodies[prop.name]
        for gid in range(self.model.ngeom):
            if self.model.geom_bodyid[gid] == bid:
                self.model.geom_rgba[gid, 3] = alpha

    def grab(self):
        if len(self.frames) >= self.cap:
            return
        self.renderer.update_scene(self.data, camera=self.camera)
        self.frames.append((float(self.data.time),
                            cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR)))
        if self._bowl is not None:
            bid, offset = self._bowl
            mat = self.data.xmat[bid].reshape(3, 3)
            # Position plus the bowl's own forward axis, so the plot can show
            # which way the bowl is facing at each point of the stroke. Roll is
            # half of what makes a scoop a scoop; a path alone cannot show it.
            self.track.append((float(self.data.time),
                               self.data.xpos[bid] + mat @ offset,
                               mat[:, 0].copy()))

    def start(self):
        """Wrap the arm's stepper so every motion is filmed, whoever drives it."""
        arm = self.cell.arm
        self._original = arm._step

        def stepped(n=1):
            self._original(n)
            if self.data.time - self._last >= self.every:
                self._last = self.data.time
                self.grab()

        arm._step = stepped
        self.grab()

    def stop(self):
        if self._original is not None:
            self.cell.arm._step = self._original
            self._original = None

    def close(self):
        self.stop()
        self.renderer.close()

    def video(self, path, fps=15):
        """Write the captured frames out as a video you can actually watch."""
        if not self.frames:
            return None
        h, w = self.frames[0][1].shape[:2]
        for fourcc in ("mp4v", "MJPG"):
            target = path if fourcc == "mp4v" else path.with_suffix(".avi")
            writer = cv2.VideoWriter(str(target),
                                     cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            if not writer.isOpened():
                writer.release()
                continue
            t0 = self.frames[0][0]
            for t, frame in self.frames:
                stamped = frame.copy()
                cv2.rectangle(stamped, (0, 0), (190, 26), (20, 20, 20), -1)
                cv2.putText(stamped, f"t+{t - t0:5.2f}s", (8, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                            cv2.LINE_AA)
                writer.write(stamped)
            writer.release()
            return target
        return None

    def strip(self, count, cols=6, scale=0.5):
        """Evenly spaced frames tiled into one sheet, each stamped with its time."""
        if not self.frames:
            return None
        picks = np.linspace(0, len(self.frames) - 1, min(count, len(self.frames)))
        tiles = []
        t0 = self.frames[0][0]
        for i in picks.astype(int):
            t, frame = self.frames[i]
            tile = cv2.resize(frame, None, fx=scale, fy=scale)
            cv2.rectangle(tile, (0, 0), (tile.shape[1], 22), (20, 20, 20), -1)
            cv2.putText(tile, f"t+{t - t0:5.2f}s", (8, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(tile)
        rows = []
        for r in range(0, len(tiles), cols):
            row = tiles[r:r + cols]
            while len(row) < cols:
                row.append(np.zeros_like(tiles[0]))
            rows.append(np.hstack(row))
        return np.vstack(rows)


class ContactWatch:
    """
    Record every touch between the held tool and the source cup.

    A stroke can look right in the viewer and still be clipping the rim -- and
    which wall it clips decides which way to move it. This reports the contact
    positions relative to the cup centre, so "it hits the front" becomes a
    number and a sign.
    """

    def __init__(self, cell, tool_prop, cup_prop, shot_path=None):
        self.cell = cell
        self.model = cell.scene.model
        self.data = cell.scene.data
        self.tool = cell.scene.prop_bodies[tool_prop.name]
        self.cup = cell.scene.prop_bodies[cup_prop.name]
        # The arm and hand too: "no collision" means the robot as well as the
        # tool, and in a shallow dish it is the fingers that come close.
        hand = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        root = self.model.body_rootid[hand]
        self.robot = {b for b in range(self.model.nbody)
                      if self.model.body_rootid[b] == root}
        self.hits = []                   # (t, contact point relative to the cup)
        self.robot_hits = []             # (t, body name) for arm/hand/finger touches
        # Everything else the arm or the tool touches -- a forearm swinging
        # into a neighbouring beaker is as much a collision as a bowl in the
        # rim, and watching only the cup would never see it.
        self.mine = self.robot | {self.tool}
        self.other_hits = {}             # (ours, theirs) -> [steps, t0, t1]
        self._original = None
        # A picture of the first touch. Counts say a stroke clips; only the
        # picture says whether it is the rim, the wall or the floor.
        self.shot_path = shot_path
        self._shot = None
        self._renderer = None
        if shot_path is not None:
            self.model.vis.global_.offwidth = max(int(self.model.vis.global_.offwidth), 848)
            self.model.vis.global_.offheight = max(int(self.model.vis.global_.offheight), 480)
            self._renderer = mujoco.Renderer(self.model, 480, 848)
            self._camera = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(self.model, self._camera)
            self._camera.azimuth, self._camera.elevation = -90.0, -5.0
            self._camera.distance = 0.16
            self._camera.lookat[:] = self.data.xpos[self.cup] + np.array([0, 0, 0.015])
            for gid in range(self.model.ngeom):
                if self.model.geom_bodyid[gid] == self.cup:
                    self.model.geom_rgba[gid, 3] = 0.3

    def _capture(self):
        if self._renderer is None or self._shot is not None:
            return
        self._renderer.update_scene(self.data, camera=self._camera)
        self._shot = cv2.cvtColor(self._renderer.render(), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(self.shot_path), self._shot)

    def _scan(self):
        data, model = self.data, self.model
        cup_xyz = data.xpos[self.cup]
        for i in range(data.ncon):
            con = data.contact[i]
            b1 = model.geom_bodyid[con.geom1]
            b2 = model.geom_bodyid[con.geom2]
            if {b1, b2} == {self.tool, self.cup}:
                gid = con.geom1 if b1 == self.tool else con.geom2
                self.hits.append((float(data.time),
                                  np.asarray(con.pos, float) - cup_xyz,
                                  mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(gid))
                                  or f"geom{int(gid)}"))
                self._capture()
            elif self.cup in (b1, b2) and ({b1, b2} - {self.cup}) & self.robot:
                other = b1 if b2 == self.cup else b2
                self.robot_hits.append((float(data.time), mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_BODY, int(other))))
                self._capture()
            elif (b1 in self.mine) != (b2 in self.mine):
                ours, theirs = (b1, b2) if b1 in self.mine else (b2, b1)
                # Body 0 is the world, which is what the table is part of.
                key = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(ours)),
                       "table" if theirs == 0 else
                       mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(theirs)))
                t = float(data.time)
                entry = self.other_hits.setdefault(key, [0, t, t])
                entry[0] += 1
                entry[2] = t

    def start(self):
        arm = self.cell.arm
        self._original = arm._step

        def stepped(n=1):
            self._original(n)
            # The arm refreshes only poses after it moves a held prop, so
            # data.contact is from the start of the step. Recompute contacts
            # for the post-step state (collision only: it writes nothing the
            # integrator reads, and costs ~2 ms against the ~20-50 ms full
            # forward pass the arm used to do here) so hits keep their timing.
            mujoco.mj_fwdPosition(self.model, self.data)
            self._scan()

        arm._step = stepped

    def stop(self):
        if self._original is not None:
            self.cell.arm._step = self._original
            self._original = None

    def other_report(self):
        if not self.other_hits:
            return "no contact between the arm/scoop and anything else"
        return "CONTACT WITH OTHER THINGS: " + ", ".join(
            f"{a} x {b} ({n} steps, t={t0:.2f}..{t1:.2f}s)"
            for (a, b), (n, t0, t1) in sorted(self.other_hits.items(),
                                              key=lambda kv: -kv[1][0]))

    def report(self):
        from collections import Counter
        robot = ("no contact between the arm/hand and the cup" if not self.robot_hits
                 else f"{len(self.robot_hits)} arm/hand contacts with the cup: "
                      + ", ".join(f"{k} x{v}" for k, v in
                                  Counter(h[1] for h in self.robot_hits).most_common()))
        robot += "\n  " + self.other_report()
        if not self.hits:
            return "no contact between the scoop and the cup\n  " + robot
        pts = np.array([h[1] for h in self.hits])
        from collections import Counter
        parts = Counter(h[2] for h in self.hits)
        near = pts[pts[:, 0] < 0]
        far = pts[pts[:, 0] >= 0]
        lines = [f"{len(self.hits)} contacts with the cup "
                 f"between t={self.hits[0][0]:.2f}s and t={self.hits[-1][0]:.2f}s",
                 "    parts touching: " + ", ".join(f"{k} x{v}" for k, v in parts.most_common())]
        for name, side in (("toward the base (-X)", near), ("away from the base (+X)", far)):
            if len(side):
                lines.append(f"    {len(side):4d} on the side {name}: "
                             f"x {side[:, 0].min() * 1000:+.0f}..{side[:, 0].max() * 1000:+.0f}mm, "
                             f"z {side[:, 2].min() * 1000:.0f}..{side[:, 2].max() * 1000:.0f}mm")
        lines.append("  " + robot)
        return "\n".join(lines)


def path_plot(track, surface_z, rim_z, base_z, rim_x, size=(900, 620), pad=70):
    """
    The bowl's path drawn in the X-Z plane -- the profile of the stroke.

    This is the picture that answers "is it an arc": a straight push is a flat
    line, a human scoop is a J. The cup is drawn in for scale, so a stroke that
    never reaches the powder is obvious rather than having to be inferred.
    """
    if len(track) < 2:
        return None
    pts = np.array([row[1] for row in track])
    axes = np.array([row[2] for row in track]) if len(track[0]) > 2 else None
    xs, zs = pts[:, 0], pts[:, 2]
    x_lo, x_hi = min(xs.min(), rim_x[0]) - 0.01, max(xs.max(), rim_x[1]) + 0.01
    z_lo, z_hi = min(zs.min(), base_z) - 0.01, max(zs.max(), rim_z) + 0.02
    w, h = size

    def to_px(x, z):
        u = pad + (x - x_lo) / max(x_hi - x_lo, 1e-6) * (w - 2 * pad)
        v = h - pad - (z - z_lo) / max(z_hi - z_lo, 1e-6) * (h - 2 * pad)
        return int(round(u)), int(round(v))

    img = np.full((h, w, 3), 250, np.uint8)
    # The cup: walls, base, powder surface.
    for x in rim_x:
        cv2.line(img, to_px(x, base_z), to_px(x, rim_z), (170, 170, 170), 3)
    cv2.line(img, to_px(rim_x[0], base_z), to_px(rim_x[1], base_z), (170, 170, 170), 3)
    cv2.line(img, to_px(rim_x[0], surface_z), to_px(rim_x[1], surface_z),
             (120, 190, 240), 3)
    cv2.putText(img, "powder surface", (to_px(rim_x[1], surface_z)[0] + 8,
                                        to_px(rim_x[1], surface_z)[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 160, 200), 1, cv2.LINE_AA)

    # The path, coloured from start (dark) to end (light), with direction dots.
    for i in range(1, len(pts)):
        shade = int(40 + 160 * i / len(pts))
        cv2.line(img, to_px(xs[i - 1], zs[i - 1]), to_px(xs[i], zs[i]),
                 (shade, 60, 200 - shade // 2), 2)
    step = max(1, len(pts) // 28)
    for i in range(0, len(pts), step):
        cv2.circle(img, to_px(xs[i], zs[i]), 3, (30, 30, 30), -1)
        if axes is not None:
            # A tick along the bowl's forward axis: pointing down-forward is a
            # bite, level is a carry, tipped back is cupping the load.
            ax = axes[i]
            scale = 0.016
            u0, v0 = to_px(xs[i], zs[i])
            u1, v1 = to_px(xs[i] + ax[0] * scale, zs[i] + ax[2] * scale)
            cv2.arrowedLine(img, (u0, v0), (u1, v1), (170, 110, 30), 1,
                            tipLength=0.35)
    cv2.circle(img, to_px(xs[0], zs[0]), 7, (0, 140, 0), 2)
    cv2.circle(img, to_px(xs[-1], zs[-1]), 7, (0, 0, 220), 2)

    cv2.putText(img, "bowl path in the X-Z plane   (green = start, red = end)",
                (pad, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(img, f"+X, away from the base ->", (pad, h - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.putText(img, f"depth below surface: {(surface_z - zs.min()) * 1000:.0f} mm",
                (pad, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1, cv2.LINE_AA)
    return img


def grains_in_bowl(cell, source_name, spoon_name="larger spoon", margin=0.004):
    """How many of the source's granules are sitting inside the scoop's bowl."""
    scene = cell.scene
    prop = scene.bench.find(spoon_name)
    if prop is None or prop.bowl_size is None:
        return None
    grains = None
    for key, ids in scene.grain_bodies.items():
        if key.lower() == source_name.lower() or source_name.lower() in key.lower():
            grains = ids
            break
    if grains is None:
        prop_src = scene.bench.find(source_name)
        grains = scene.grain_bodies.get(prop_src.name) if prop_src else None
    if not grains:
        return None

    bid = scene.prop_bodies[prop.name]
    mat = scene.data.xmat[bid].reshape(3, 3)
    origin = scene.data.xpos[bid]
    half = np.asarray(prop.bowl_size, float) + margin
    centre = np.asarray(prop.bowl_offset, float)

    inside = 0
    for gid in grains:
        local = mat.T @ (scene.data.xpos[gid] - origin) - centre
        if np.all(np.abs(local) <= half):
            inside += 1
    return inside


def run_then(args, cell, tool_prop, tool_offset, tally, out):
    """
    Run the follow-on skill -- a dump -- with the tool still in the jaws and
    the load the first skill left in it, and measure it the same way: contacts
    with the target, whether the target moved, where the powder went.
    """
    target_prop = cell.scene.bench.find(args.target)
    if target_prop is None:
        print(f"\n--target {args.target!r} matches nothing on the bench")
        return {"skill": args.then, "success": False,
                "error": f"no target {args.target!r}"}
    params = {"target_container": args.target}
    if tool_offset is not None:
        # The ground-truth offset the scoop was aimed with: it is what keeps
        # the bowl, not the wrist, over the target through the whole tip.
        params["tool_offset"] = [float(v) for v in tool_offset]
    if tool_prop is not None and tool_prop.bowl_size is not None:
        # The bowl's outside, from that offset (its bottom) up to the mouth:
        # what the dump keeps above the rim at every tilt.
        params["bowl_length"] = float(2 * tool_prop.bowl_size[0])
        params["bowl_width"] = float(2 * tool_prop.bowl_size[1])
        params["bowl_depth"] = float(2 * tool_prop.bowl_size[2])
    params.update(parse_params(args.then_params))

    body = cell.scene.prop_bodies[target_prop.name]
    start = cell.scene.data.xpos[body].copy()
    watch = (ContactWatch(cell, tool_prop, target_prop,
                          shot_path=out / "then_first_contact.png")
             if tool_prop else None)
    film = None if args.no_film else Film(cell, args.view, every=args.every)
    if film:
        # Frame the rim and the space above it, where the bowl tips: the dump
        # happens several centimetres over the target, not inside it.
        film.look_at(np.array([start[0], start[1],
                               cell.scene.bench.table_z + target_prop.height + 0.05]))
        film.camera.distance *= 1.5
        film.see_through(target_prop)
        if tool_prop is not None and tool_prop.bowl_size is not None:
            film.track_bowl(tool_prop)
    if watch:
        watch.start()
    if tally:
        tally.start()             # same tally: it carries the load over
    if film:
        film.start()
    ok, result = cell.skills.execute(args.then, params)
    if film:
        film.stop()
    if tally:
        tally.stop()
    if watch:
        watch.stop()

    shift = cell.scene.data.xpos[body] - start
    moved = float(np.linalg.norm(shift))
    print(f"\n{args.then}: {'SUCCESS' if ok else 'FAILED'}")
    if not ok:
        print(f"  {result.get('error', result)}")
    if tally:
        print("  " + tally.report_dump(target_prop.name))
    print(f"  {target_prop.name} moved: {moved * 1000:.1f} mm "
          f"(dx {shift[0] * 1000:+.1f}, dy {shift[1] * 1000:+.1f}, "
          f"dz {shift[2] * 1000:+.1f})")
    if watch:
        print("  " + watch.report())
    if film:
        clip = film.video(out / "then_motion.mp4", fps=args.fps)
        if clip is not None:
            print(f"  video -> {clip}")
        strip = film.strip(args.frames)
        if strip is not None:
            cv2.imwrite(str(out / "then_filmstrip.png"), strip)
            print(f"  {len(film.frames)} frames -> {out / 'then_filmstrip.png'}")
        film.close()
    return {"skill": args.then, "success": bool(ok), "target": target_prop.name,
            "powder_estimate": tally.result() if tally else None,
            "robot_target_contacts": len(watch.robot_hits) if watch else None,
            "tool_target_contacts": len(watch.hits) if watch else None,
            "other_contacts": ({f"{a} x {b}": v[0] for (a, b), v in watch.other_hits.items()}
                               if watch else None),
            "target_moved_mm": round(moved * 1000, 2),
            "params": params, "result": result}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skill", default="arc_scoop", help="Skill under test")
    parser.add_argument("--params", default=None,
                        help="Extra params for the skill, JSON or a dict literal")
    parser.add_argument("--source", default="citric acid", help="Powder source")
    parser.add_argument("--pick", default="larger spoon",
                        help="Tool to pick up first; empty to skip")
    parser.add_argument("--view", default="side", choices=sorted(VIEWS))
    parser.add_argument("--frames", type=int, default=18, help="Tiles in the strip")
    parser.add_argument("--fps", type=int, default=15,
                        help="Playback rate of the written video")
    parser.add_argument("--every", type=float, default=0.12,
                        help="Seconds of sim time between captured frames")
    parser.add_argument("--label", default=None, help="Output folder name")
    parser.add_argument("--out", default="diag_out/film")
    parser.add_argument("--viewer", action="store_true", help="Open the live window")
    parser.add_argument("--no-film", action="store_true",
                        help="Do not capture frames at all. Filming renders a "
                             "frame every --every seconds of sim time, which is "
                             "most of the cost of a run; skip it to just watch")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback multiplier. 0.25 runs the stroke at a "
                             "quarter speed, which is how you actually see a "
                             "1.6s sweep; 2 or more genuinely changes the "
                             "dynamics, so keep it at 1 when it matters")
    parser.add_argument("--hold", type=float, default=0.0,
                        help="With --viewer: seconds to keep the window open at "
                             "the end. 0 waits until you close it")
    parser.add_argument("--save-frames", action="store_true",
                        help="Write every captured frame, not just the strip")
    parser.add_argument("--granules", action="store_true",
                        help="Simulate the powder as physical granules (slow, and "
                             "coarser than any powder). Default: a drawn powder "
                             "bed with no contacts, and a geometric estimate of "
                             "what the scoop collects (robochem.sim.powder)")
    parser.add_argument("--powder-level", type=float, default=None,
                        help="Powder depth above the dish's inside floor, metres "
                             "(default: the bench's, 0.018)")
    parser.add_argument("--density", type=float, default=0.9,
                        help="Powder bulk density in g/ml, to express the estimate "
                             "as a mass")
    parser.add_argument("--fill", type=int, default=90,
                        help="With --granules: granules in the source cup")
    parser.add_argument("--grain-radius", type=float, default=0.006,
                        help="With --granules: grain radius; coarse grains cost "
                             "fewer bodies for a given bed depth")
    parser.add_argument("--surface", type=float, default=None,
                        help="Powder surface in world z. Default: measured from "
                             "the granules. With --fill 0, set it by hand to test "
                             "the stroke shape without paying for the physics")
    parser.add_argument("--opaque", action="store_true",
                        help="Do not make the source cup translucent")
    parser.add_argument("--then", default=None, choices=["dump"],
                        help="A second skill to run after --skill with the "
                             "same tool and its load, e.g. --skill scoop --then dump")
    parser.add_argument("--then-params", default=None,
                        help="Params for the --then skill, same format as --params")
    parser.add_argument("--target", default="white paper cup",
                        help="Container --then dump empties the scoop into")
    args = parser.parse_args()

    out = ROOT / args.out / (args.label or args.skill)
    out.mkdir(parents=True, exist_ok=True)

    # A deeper bed than the default bench: with a sparse scatter the measured
    # surface is the rim and the stroke never meets any powder, which tells you
    # nothing about the stroke.
    bench = Bench()
    is_source = lambda p: p.label and args.source.lower() in (p.label or "").lower()
    if args.granules:
        bench.props = [replace(p, fill=args.fill, grain_radius=args.grain_radius)
                       if is_source(p) else p for p in bench.props]
    elif args.powder_level is not None:
        bench.props = [replace(p, powder_level=args.powder_level)
                       if is_source(p) else p for p in bench.props]

    cell = build_cell(bench=bench, viewer=args.viewer, realtime=args.viewer,
                      speed=args.speed, granules=args.granules, verbose=False,
                      workspace_min=[0.25, -0.40, -0.13])
    film = None if args.no_film else Film(cell, args.view, every=args.every)
    try:
        measured_offset = None
        if args.pick:
            picked, pick_result = cell.skills.execute(
                "pick_up", {"object_name": args.pick, "z_offset": 0.0,
                            "grasp_force": 1.0})
            print(f"pick_up {args.pick!r}: {'ok' if picked else 'FAILED'}")
            if not picked:
                print(f"  {pick_result.get('error', pick_result)}")
                return 1
            # Use the offset pick_up measured off the cloud rather than a
            # hardcoded one: a scoop aimed with the wrong tool offset digs
            # somewhere other than where the skill thinks it is digging.
            measured_offset = pick_result.get("suggested_tool_offset")
            cell.vision.clear_cache()

        source_prop = cell.scene.bench.find(args.source)
        cup_xyz = cell.scene.data.xpos[cell.scene.prop_bodies[source_prop.name]]
        tool_prop = cell.scene.bench.find(args.pick or "larger spoon")
        if film:
            film.look_at(cup_xyz + np.array([0.01, 0, 0.05]))
            if not args.opaque:
                film.see_through(source_prop)
            if tool_prop is not None and tool_prop.bowl_size is not None:
                film.track_bowl(tool_prop)

        # Where the powder actually is, for the path plot -- ground truth,
        # independent of what the skill's own perception decided.
        grains = cell.scene.grain_bodies.get(source_prop.name, [])
        bed = None if args.granules else bed_from_prop(cell.scene, source_prop)
        if grains:
            bed_z = (max(float(cell.scene.data.xpos[g][2]) for g in grains)
                     + source_prop.grain_radius)
        elif bed is not None and source_prop.powder_level > 0:
            bed_z = bed.surface_z
        else:
            bed_z = args.surface if args.surface is not None else cup_xyz[2]
        rim_z = cell.scene.bench.table_z + source_prop.height
        print(f"bed surface z={bed_z:.4f}, cup rim z={rim_z:.4f}, "
              + (f"{len(grains)} granules" if args.granules else
                 f"drawn powder bed {source_prop.powder_level * 1000:.0f}mm deep, "
                 f"dish inside radius {(source_prop.radius - source_prop.wall) * 1000:.1f}mm"))

        before = grains_in_bowl(cell, args.source, args.pick or "larger spoon")
        # Where the cup starts. A stroke that shoves the cup has not just made
        # a mess -- it has invalidated the scan every following skill relies on,
        # so "did the cup stay put" is a pass/fail number, not a nicety.
        cup_start = cell.scene.data.xpos[
            cell.scene.prop_bodies[source_prop.name]].copy()
        # The true offset from the TCP to the bowl, straight out of the sim.
        # Printed next to what perception measured, because a stroke aimed with
        # the wrong one cannot be judged by how it looks.
        if tool_prop is not None and tool_prop.bowl_size is not None:
            # Ground truth: TCP -> the FLOOR of the bowl, in the tool frame.
            # The floor, not the centre, because that is the surface that digs.
            tcp, tcp_mat = cell.arm._tcp()
            bid = cell.scene.prop_bodies[tool_prop.name]
            floor_local = (np.asarray(tool_prop.bowl_offset, float)
                           - np.array([0, 0, tool_prop.bowl_size[2]]))
            floor_world = (cell.scene.data.xpos[bid]
                           + cell.scene.data.xmat[bid].reshape(3, 3) @ floor_local)
            truth = tcp_mat.T @ (floor_world - tcp)
            print(f"tool offset: perception {np.round(measured_offset, 4) if measured_offset else None}"
                  f"   CAD/ground truth {np.round(truth, 4)}")
            # Use ground truth for the test. measure_tool_offset's z is a lower
            # bound by its own docstring, and its x is only right when the jaws
            # closed where the grasp planner intended -- here they close ~11mm
            # further along the handle, and at a 50 degree bite that alone digs
            # 8mm shallow. Feeding the truth in isolates the question being
            # asked, which is whether the STROKE is right. On hardware this is
            # the CAD offset plus wherever the grasp actually landed.
            if measured_offset is not None:
                error = np.asarray(measured_offset, float) - truth
                print(f"   perception is off by {np.round(error * 1000, 1)} mm "
                      f"-> using ground truth for this test")
            measured_offset = [float(truth[0]), float(truth[1]), float(abs(truth[2]))]

        params = {"powder_source": args.source}
        if tool_prop is not None and tool_prop.bowl_size is not None:
            # The bowl's CIRCUMSCRIBED width: in a round dish its corners meet
            # the angled walls before its leading face does, so the clamp has
            # to reserve the half-diagonal, not the half-length.
            params["tool_span"] = float(2 * np.hypot(tool_prop.bowl_size[0],
                                                     tool_prop.bowl_size[1]))
            # How far the tool reaches back from the jaws. The handle length is
            # a deliberate over-estimate of jaws-to-handle-end: erring long
            # only makes the clearance check stricter, and the handle end is
            # what swings into the near wall when the wrist rolls nose-up.
            params["tool_back_reach"] = float(tool_prop.height)
            # The bowl's own length and inside depth, as distinct from the
            # circumscribed tool_span above: length sets how far the leading
            # edge drops when the bowl tilts, depth sets how deep the stroke
            # must go before the MOUTH is under the powder rather than the
            # edge alone.
            params["bowl_length"] = float(2 * tool_prop.bowl_size[0])
            params["bowl_depth"] = float(2 * tool_prop.bowl_size[2])
        if measured_offset is not None:
            params["tool_offset"] = [float(v) for v in measured_offset]
        # Hand the skill the real powder surface. The measured container top is
        # the rim, so without this the stroke is aimed tens of millimetres above
        # the bed and nothing about the motion can be judged.
        surface = args.surface if args.surface is not None else (
            bed_z if (grains or (bed is not None and source_prop.powder_level > 0))
            else None)
        if surface is not None and args.skill in ("arc_scoop", "scoop"):
            params["powder_surface_z"] = float(surface)
        elif surface is not None:
            # A skill with no surface parameter: convert to a depth below the
            # rim so the skills are compared digging to the same place.
            params["scoop_depth"] = float(rim_z - surface) + float(
                parse_params(args.params).get("scoop_depth", 0.015))
        if args.skill == "scoop":
            # The dish's INSIDE floor and inside radius, from the model:
            # scene._add_container stands the cup on a base disc 2*wall thick
            # and rings it with a wall `wall` thick. Perception can see neither
            # under a bed (its lowest points are the outer bottom, ~1 mm low;
            # its "opening radius" is the outside of the rim plus the centre
            # error), so hand the truth in, as with the surface: the test is of
            # the stroke, not of the estimates.
            params["container_floor_z"] = float(cell.scene.bench.table_z
                                                + 2 * source_prop.wall)
            params["container_radius"] = float(source_prop.radius - source_prop.wall)
            params["container_center"] = [float(v) for v in cup_xyz[:2]]
            if tool_prop is not None and tool_prop.bowl_size is not None:
                params["bowl_width"] = float(2 * tool_prop.bowl_size[1])
        params.update(parse_params(args.params))

        watch = (ContactWatch(cell, tool_prop, source_prop,
                              shot_path=out / "first_contact.png")
                 if tool_prop else None)
        if watch:
            watch.start()          # wraps _step; Film wraps whatever it finds
        # What the stroke collects from the drawn bed, estimated geometrically
        # (robochem.sim.powder), since the bed has no particles to count.
        tally = (ScoopTally(cell, tool_prop, bed, bulk_density=args.density)
                 if bed is not None and source_prop.powder_level > 0
                 and tool_prop is not None and tool_prop.bowl_size is not None
                 else None)
        if tally:
            tally.start()
        if film:
            film.start()
        ok, result = cell.skills.execute(args.skill, params)
        if film:
            film.stop()
        if tally:
            tally.stop()
        if watch:
            watch.stop()

        after = grains_in_bowl(cell, args.source, args.pick or "larger spoon")
        cup_shift = cell.scene.data.xpos[
            cell.scene.prop_bodies[source_prop.name]] - cup_start
        cup_moved = float(np.linalg.norm(cup_shift))
        print(f"\n{args.skill}: {'SUCCESS' if ok else 'FAILED'}")
        if not ok:
            print(f"  {result.get('error', result)}")
        if after is not None and args.granules:
            print(f"  granules in the bowl: {after} (was {before})")
        if tally:
            print("  " + tally.report())
        print(f"  cup moved: {cup_moved * 1000:.1f} mm "
              f"(dx {cup_shift[0] * 1000:+.1f}, dy {cup_shift[1] * 1000:+.1f}, "
              f"dz {cup_shift[2] * 1000:+.1f})")
        if watch is not None:
            print("  " + watch.report())

        if film:
            clip = film.video(out / "motion.mp4", fps=args.fps)
            if clip is not None:
                print(f"  video -> {clip}")

            strip = film.strip(args.frames)
            if strip is not None:
                cv2.imwrite(str(out / "filmstrip.png"), strip)
                print(f"  {len(film.frames)} frames -> {out / 'filmstrip.png'}")

            plot = path_plot(film.track, bed_z, rim_z,
                             cell.scene.bench.table_z,
                             (float(cup_xyz[0]) - source_prop.radius,
                              float(cup_xyz[0]) + source_prop.radius))
        else:
            plot = None
        if plot is not None:
            cv2.imwrite(str(out / "path.png"), plot)
            zs = np.array([row[1][2] for row in film.track])
            print(f"  bowl dipped to z={zs.min():.4f} "
                  f"({(bed_z - zs.min()) * 1000:+.0f}mm vs the bed surface) "
                  f"-> {out / 'path.png'}")
        if args.save_frames and film:
            for i, (t, frame) in enumerate(film.frames):
                cv2.imwrite(str(out / f"f{i:04d}.png"), frame)

        then = None
        if args.then and ok:
            then = run_then(args, cell, tool_prop, measured_offset, tally, out)
        elif args.then:
            print(f"\n{args.then}: skipped, {args.skill} failed")

        summary = {"skill": args.skill, "success": bool(ok),
                   "powder_estimate": tally.result() if tally else None,
                   "robot_cup_contacts": len(watch.robot_hits) if watch else None,
                   "tool_cup_contacts": len(watch.hits) if watch else None,
                   "other_contacts": ({f"{a} x {b}": v[0] for (a, b), v in watch.other_hits.items()}
                                      if watch else None),
                   "granules_in_bowl": after,
                   "cup_moved_mm": round(cup_moved * 1000, 2),
                   "params": params, "result": result, "then": then}
        (out / "result.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8")
        return 0 if ok and (then is None or then["success"]) else 1
    finally:
        if args.viewer:
            print("\nClose the window when you have seen enough."
                  if args.hold <= 0 else
                  f"\nHolding the window for {args.hold:.0f}s.")
            cell.arm.hold(args.hold if args.hold > 0 else 1e9)
        if film:
            film.close()
        cell.close()


if __name__ == "__main__":
    raise SystemExit(main())
