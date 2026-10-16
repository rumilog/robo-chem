"""
Offline check of the cube-model calibration against a known answer.

Renders what each cage camera would see of the 39.8 mm cube held in the
gripper: true extrinsics are the installed ones, the finger pads hide the
middle of the cube's +-y faces and the fingers rise above it, depth is z-buffered
one point per pixel with 1.5 mm noise along the ray. Then solves twice:

  - the old way, median of the visible points + Kabsch, which should show
    the per-camera bias seen on the bench (cameras disagreeing by centimetres);
  - the cube-model fit started from that, which should recover the truth.

Errors are measured where it matters, on the table: how far each solve moves
a grid of table points (z = 0) from where the true extrinsics put them.

Usage:
    source scripts/env.sh
    python scripts/test_cube_calibration.py
"""

import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robochem.vision.cube_calibration import fit_cube_model, implied_centres, kabsch, subsample

CALIB = "/home/rumi/Desktop/camera examples/robomail/robomail/vision/calib"
CAMS = [2, 3, 4, 5]
CUBE = 0.0398
FX = FY = 615.0
CX, CY, W, H = 320.0, 240.0, 640, 480


def box_surface(lo, hi, step):
    """Points on the six faces of an axis-aligned box, with face normals."""
    pts, nrm = [], []
    for ax in range(3):
        o = [a for a in range(3) if a != ax]
        g0 = np.arange(lo[o[0]], hi[o[0]] + 1e-9, step)
        g1 = np.arange(lo[o[1]], hi[o[1]] + 1e-9, step)
        a, b = np.meshgrid(g0, g1)
        for side, val in ((-1, lo[ax]), (1, hi[ax])):
            p = np.zeros((a.size, 3))
            p[:, o[0]], p[:, o[1]], p[:, ax] = a.ravel(), b.ravel(), val
            n = np.zeros((a.size, 3))
            n[:, ax] = side
            pts.append(p)
            nrm.append(n)
    return np.vstack(pts), np.vstack(nrm)


def build_scene():
    """Cube and finger points in the cube frame, labelled 1 (cube) / 0 (finger)."""
    h = CUBE / 2
    cube, cube_n = box_surface(np.array([-h] * 3), np.array([h] * 3), 0.0005)
    parts, normals, labels = [cube], [cube_n], [np.ones(len(cube), bool)]
    # Finger pads 18 mm wide, 8 mm thick, on the +-y faces, rising 45 mm above
    # the cube centre toward the palm (cube frame z is down: hand z points down).
    for side in (-1, 1):
        y0, y1 = (h, h + 0.008) if side > 0 else (-h - 0.008, -h)
        f, fn = box_surface(np.array([-0.009, y0, -h - 0.045]),
                            np.array([0.009, y1, h - 0.002]), 0.001)
        parts.append(f)
        normals.append(fn)
        labels.append(np.zeros(len(f), bool))
    return np.vstack(parts), np.vstack(normals), np.concatenate(labels)


def render(T_cam, centre, Rc, scene, rng):
    """Cube points one camera sees (camera frame), z-buffered per pixel."""
    pts, nrm, lab = scene
    world = pts @ Rc.T + centre
    wn = nrm @ Rc.T
    cam_pos = T_cam[:3, 3]
    facing = np.einsum("ij,ij->i", wn, cam_pos - world) > 0
    Ti = np.linalg.inv(T_cam)
    pc = world[facing] @ Ti[:3, :3].T + Ti[:3, 3]
    lab = lab[facing]
    z = pc[:, 2]
    ok = z > 0.28                                     # D435 minimum range
    pc, lab, z = pc[ok], lab[ok], z[ok]
    u = np.round(pc[:, 0] / z * FX + CX).astype(int)
    v = np.round(pc[:, 1] / z * FY + CY).astype(int)
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    pc, lab, z, u, v = pc[inside], lab[inside], z[inside], u[inside], v[inside]
    order = np.argsort(z)
    pix = v[order] * W + u[order]
    _, first = np.unique(pix, return_index=True)
    keep = order[first]
    seen = pc[keep][lab[keep]]
    noise = 1.0 + rng.normal(0, 0.0015, len(seen)) / np.linalg.norm(seen, axis=1)
    # 5 % edge pixels whose depth mixes in the table behind: pushed back
    # along the ray by up to 15 cm.
    stray = rng.random(len(seen)) < 0.05
    noise[stray] += rng.uniform(0, 0.15, stray.sum()) / np.linalg.norm(seen[stray], axis=1)
    return seen * noise[:, None]


def table_error(T_est, T_true):
    """mm: how far T_est puts table points from where T_true does."""
    xs, ys = np.meshgrid(np.linspace(0.30, 0.80, 6), np.linspace(-0.35, 0.35, 6))
    p = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(xs.size), np.ones(xs.size)])
    est = (T_est @ np.linalg.inv(T_true) @ p.T).T[:, :3]
    return (est - p[:, :3]) * 1000


def main() -> int:
    rng = np.random.default_rng(1)
    T_true = {c: np.load(f"{CALIB}/realsense_camera{c}w.npy") for c in CAMS}
    hand = np.diag([1.0, -1.0, -1.0])                 # top-down, fingers close along y
    tilt = Rotation.from_euler("y", 4.0, degrees=True).as_matrix()
    Rc_true = hand @ tilt                              # cube pitched 4 deg in the grip
    scene = build_scene()

    poses = [np.array([x, y, z]) for z in np.linspace(0.07, 0.25, 3)
             for y in np.linspace(-0.25, 0.25, 4) for x in np.linspace(0.38, 0.66, 4)]
    obs = {c: [] for c in CAMS}
    for tcp in poses:
        for c in CAMS:
            pts = render(T_true[c], tcp, Rc_true, scene, rng)
            if len(pts) >= 30:
                obs[c].append({"points": subsample(pts, 300), "tcp": tcp,
                               "median": np.median(pts, axis=0)})
    print("observations per camera:", {c: len(v) for c, v in obs.items()})

    T_med = {}
    for c in CAMS:
        A = np.array([o["median"] for o in obs[c]]).T
        B = np.array([o["tcp"] for o in obs[c]]).T
        R, t = kabsch(A, B)
        T_med[c] = np.eye(4)
        T_med[c][:3, :3], T_med[c][:3, 3] = R, t

    fit = fit_cube_model(obs, hand, CUBE, T_med)
    print(f"cube tilt recovered {fit['cube_tilt_deg']:.1f} deg (true 4.0)")

    failures = []
    print("\ntable points moved vs truth, mm: mean |xy| / mean z / max |xyz|")
    for c in CAMS:
        em, ef = table_error(T_med[c], T_true[c]), table_error(fit["transforms"][c], T_true[c])
        print(f"  cam {c}: median+Kabsch {np.linalg.norm(em[:, :2], axis=1).mean():5.1f} / "
              f"{em[:, 2].mean():+5.1f} / {np.linalg.norm(em, axis=1).max():5.1f}    "
              f"cube model {np.linalg.norm(ef[:, :2], axis=1).mean():4.1f} / "
              f"{ef[:, 2].mean():+4.1f} / {np.linalg.norm(ef, axis=1).max():4.1f}   "
              f"rms {fit['rms_mm'][c]:.1f} mm")
        if np.linalg.norm(ef, axis=1).max() > 3.0:
            failures.append(f"cam {c}: cube model off by more than 3 mm on the table")

    o = obs[2][len(obs[2]) // 2]
    c2 = implied_centres(o["points"], fit["transforms"][2], fit["cube_rotation"], CUBE, o["tcp"])
    err = np.linalg.norm(c2 - o["tcp"]) * 1000
    print(f"\nimplied centre, cam 2, one pose: {err:.1f} mm from the TCP")
    if err > 3.0:
        failures.append("implied_centres more than 3 mm off")

    print("\n" + ("FAIL:\n  " + "\n  ".join(failures) if failures else "PASS"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
