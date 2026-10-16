"""
Camera extrinsics from a gripped cube, fitted as a cube rather than a point.

The first calibration paired each camera's median of the cube's *visible*
surface with the TCP. A camera only sees the cube's top and the faces toward
it, so that median sits toward the camera and up. Kabsch folds the bias into
the extrinsics, and every camera then pushes its whole cloud away from itself
and down: on 2026-10-10 cameras on opposite sides put the same flat paper
~36 mm apart, and the table ~14 mm low.

Here each camera's raw cube points are instead required to lie on the surface
of a cube of known size, centred on the TCP at every pose. Which faces a
camera happens to see no longer matters. The median/Kabsch solve is kept as
the starting point.

The cube's offset from the TCP is not solvable from these data (shifting every
camera and the cube together fits equally well), so the cube centre is taken
as the TCP, as before. That shift is common to all cameras: it cannot make
them disagree, only move the table. `table_height` measures that afterwards.

Pure numpy/scipy; no robot or camera imports, so it can be tested offline.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def kabsch(A: np.ndarray, B: np.ndarray):
    """
    Least-squares rigid transform taking A onto B.

    Args:
        A: 3xN points in the camera frame
        B: 3xN corresponding points in the robot frame

    Returns:
        (R, t) with R @ A + t ~= B
    """
    ca = A.mean(axis=1, keepdims=True)
    cb = B.mean(axis=1, keepdims=True)

    H = (A - ca) @ (B - cb).T
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Guard against the reflection solution SVD can return.
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = cb - R @ ca
    return R, t.ravel()


def box_distance(u: np.ndarray, half: float) -> np.ndarray:
    """Signed distance from points u (cube frame, Nx3) to a cube of half-side `half`."""
    q = np.abs(u) - half
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(q.max(axis=1), 0.0)
    return outside + inside


def _transform(params: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(params[:3]).as_matrix()
    T[:3, 3] = params[3:6]
    return T


def _params(T: np.ndarray) -> np.ndarray:
    return np.concatenate([Rotation.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]])


def fit_cube_model(observations: Dict[int, List[dict]],
                   cube_rotation: np.ndarray,
                   cube_size: float,
                   initial: Dict[int, np.ndarray],
                   loss_scale: float = 0.003,
                   table: Optional[Dict[int, np.ndarray]] = None,
                   table_z: Optional[float] = None) -> dict:
    """
    Refine every camera's camera-to-world transform, and the cube's
    orientation in the grip, so each camera's cube points lie on the cube.

    Args:
        observations: {cam_id: [{"points": Nx3 camera frame, "tcp": (3,)}, ...]}
        cube_rotation: 3x3 world-from-cube guess; the frozen hand orientation,
            since the finger pads square the cube to the hand
        cube_size: edge length, metres
        initial: {cam_id: 4x4} starting transforms (the median/Kabsch solve)
        loss_scale: soft-L1 scale, metres; points further off than this
            (edge pixels, stray green) count less
        table: optional {cam_id: Nx3 camera-frame points} of the bare
            tabletop, used with table_z
        table_z: the table's height in the base frame. Each camera's table
            points are then also required to lie at it, and the cube centre's
            height above the TCP becomes a fitted parameter instead of being
            folded into every camera. The cube sweep stops 8 cm above the
            table, so on its own it leaves each camera's height and tilt at
            the table to extrapolation (cameras disagreed by 5.6 mm there on
            2026-10-10).

    Returns:
        {"transforms": {cam: 4x4}, "cube_rotation": 3x3,
         "cube_tilt_deg": float, "rms_mm": {cam: float},
         "cube_above_tcp_mm": float or None}
    """
    cams = sorted(c for c in observations if c in initial and observations[c])
    half = cube_size / 2.0

    # Stack each camera's points once, with the TCP of the pose they came from.
    stacked = {}
    for c in cams:
        pts = np.vstack([o["points"] for o in observations[c]])
        tcp = np.vstack([np.repeat(np.asarray(o["tcp"], float)[None], len(o["points"]), 0)
                         for o in observations[c]])
        stacked[c] = (pts, tcp)

    use_table = table_z is not None and bool(table)
    flat = {c: table[c] for c in cams if use_table and c in table and len(table[c])}
    n_cam = 3 + 6 * len(cams)
    x0 = np.concatenate([np.zeros(3)] + [_params(initial[c]) for c in cams]
                        + ([np.zeros(1)] if use_table else []))

    def residuals(x):
        Rc = Rotation.from_rotvec(x[:3]).as_matrix() @ cube_rotation
        lift = np.array([0.0, 0.0, x[n_cam]]) if use_table else 0.0
        out = []
        for i, c in enumerate(cams):
            T = _transform(x[3 + 6 * i: 9 + 6 * i])
            pts, tcp = stacked[c]
            world = pts @ T[:3, :3].T + T[:3, 3]
            out.append(box_distance((world - tcp - lift) @ Rc, half))
        for i, c in enumerate(cams):
            if c in flat:
                T = _transform(x[3 + 6 * i: 9 + 6 * i])
                out.append(flat[c] @ T[2, :3] + T[2, 3] - table_z)
        return np.concatenate(out)

    sol = least_squares(residuals, x0, loss="soft_l1", f_scale=loss_scale,
                        x_scale="jac", max_nfev=200)

    Rc = Rotation.from_rotvec(sol.x[:3]).as_matrix() @ cube_rotation
    transforms, rms, start = {}, {}, 0
    r = residuals(sol.x)
    for i, c in enumerate(cams):
        transforms[c] = _transform(sol.x[3 + 6 * i: 9 + 6 * i])
        n = len(stacked[c][0])
        rc = r[start:start + n]
        start += n
        # RMS over the points within 10 mm of the cube: the outliers the
        # robust loss down-weighted would otherwise dominate the number.
        rc = rc[np.abs(rc) < 0.010]
        rms[c] = float(np.sqrt(np.mean(rc ** 2)) * 1000) if len(rc) else float("nan")

    return {
        "transforms": transforms,
        "cube_rotation": Rc,
        "cube_tilt_deg": float(np.degrees(np.linalg.norm(sol.x[:3]))),
        "rms_mm": rms,
        "cube_above_tcp_mm": float(sol.x[n_cam] * 1000) if use_table else None,
        "success": bool(sol.success),
    }


def table_inliers(points_cam: np.ndarray, T: np.ndarray, table_z: float,
                  xy_box=((0.30, 0.75), (-0.35, 0.35)), n: int = 3000) -> np.ndarray:
    """
    Camera-frame points of the bare tabletop, picked with a rough transform T:
    inside the xy box, within 25 mm of table_z, on the dominant plane there
    (so papers count, objects and the arm do not). Subsampled to n.
    """
    world = points_cam @ T[:3, :3].T + T[:3, 3]
    (x0, x1), (y0, y1) = xy_box
    keep = ((world[:, 0] > x0) & (world[:, 0] < x1) & (world[:, 1] > y0) &
            (world[:, 1] < y1) & (np.abs(world[:, 2] - table_z) < 0.025))
    if keep.sum() < 200:
        return np.empty((0, 3))
    w = world[keep]
    A = np.column_stack([w[:, 0], w[:, 1], np.ones(len(w))])
    g = np.linalg.lstsq(A[np.abs(w[:, 2] - np.median(w[:, 2])) < 0.006],
                        w[np.abs(w[:, 2] - np.median(w[:, 2])) < 0.006, 2], rcond=None)[0]
    on_plane = np.abs(w[:, 2] - A @ g) < 0.004
    return subsample(points_cam[keep][on_plane], n)


def implied_centres(points_cam: np.ndarray, T: np.ndarray, cube_rotation: np.ndarray,
                    cube_size: float, guess: np.ndarray) -> Optional[np.ndarray]:
    """
    Where one camera's points put the cube centre at one pose, in world.

    Compared with the TCP this is the per-observation error, the same quantity
    the median solve's residual measured, but free of which faces were seen.
    A camera that sees one face only fixes the centre across that face, not
    along it; a weak pull toward `guess` keeps the free directions there
    instead of letting the fit slide.
    """
    if len(points_cam) < 10:
        return None
    world = points_cam @ T[:3, :3].T + T[:3, 3]
    half = cube_size / 2.0
    guess = np.asarray(guess, float)
    # Edge pixels sit well off the cube; start from the points near it.
    near = np.abs(box_distance((world - guess) @ cube_rotation, half)) < 0.010
    if near.sum() >= 10:
        world = world[near]

    def res(c):
        return np.concatenate([box_distance((world - c) @ cube_rotation, half),
                               0.1 * (c - guess)])

    sol = least_squares(res, guess, loss="soft_l1", f_scale=0.003)
    # Few points from a corner view let edge pixels pull the centre; drop
    # what is clearly off the cube at that fit and fit again.
    near = np.abs(box_distance((world - sol.x) @ cube_rotation, half)) < 0.006
    if near.sum() >= 10 and not near.all():
        world = world[near]
        sol = least_squares(res, sol.x, loss="soft_l1", f_scale=0.003)
    return sol.x


def table_height(points_world: np.ndarray, xy_box=((0.30, 0.75), (-0.35, 0.35)),
                 band=(-0.05, 0.02), iters=200, thresh=0.004):
    """
    Plane through the tabletop: (z at the box centre, tilt deg, inliers), or None.

    Only points in the xy box and the z band are tried, so the arm, the cube
    and the cage cannot win.
    """
    (x0, x1), (y0, y1) = xy_box
    p = points_world
    keep = ((p[:, 0] > x0) & (p[:, 0] < x1) & (p[:, 1] > y0) & (p[:, 1] < y1) &
            (p[:, 2] > band[0]) & (p[:, 2] < band[1]))
    p = p[keep]
    if len(p) < 200:
        return None
    rng = np.random.default_rng(0)
    best = None
    for _ in range(iters):
        a, b, c = p[rng.choice(len(p), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs((p - a) @ n) < thresh
        if best is None or inl.sum() > best.sum():
            best = inl
    q = p[best]
    A = np.column_stack([q[:, 0], q[:, 1], np.ones(len(q))])
    gx, gy, g0 = np.linalg.lstsq(A, q[:, 2], rcond=None)[0]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    tilt = np.degrees(np.arctan(np.hypot(gx, gy)))
    return float(gx * cx + gy * cy + g0), float(tilt), int(best.sum())


def subsample(points: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    if len(points) <= n:
        return points
    idx = np.random.default_rng(seed).choice(len(points), n, replace=False)
    return points[idx]


__all__: Sequence[str] = ("kabsch", "box_distance", "fit_cube_model", "implied_centres",
                          "table_height", "table_inliers", "subsample")
