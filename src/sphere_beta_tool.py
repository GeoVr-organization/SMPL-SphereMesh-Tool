"""
Sphere Beta Tool – interactive shape-space explorer built on Polyscope.

Loads per-beta CSV offsets and lets you drive sphere positions/radii
with real-time sliders.
"""

from __future__ import annotations

import csv
import glob
import os
import sys
from pathlib import Path

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------

TEMPLATE_POS = np.array([
    [0.0, 0.0, 0.0],
    [2.0, 0.0, 0.0],
    [4.0, 0.0, 0.0],
], dtype=np.float64)

TEMPLATE_R = np.array([1.0, 1.0, 1.0], dtype=np.float64)

NUM_SPHERES = len(TEMPLATE_R)

MIN_RADIUS = 1e-4

# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = {"sphere_id", "dx", "dy", "dz", "dr"}


def load_offsets_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (dp, dr) arrays from a single beta CSV file.

    dp has shape (NUM_SPHERES, 3), dr has shape (NUM_SPHERES,).
    Rows may appear in any order; sphere_id determines placement.
    """
    dp = np.zeros((NUM_SPHERES, 3), dtype=np.float64)
    dr = np.zeros(NUM_SPHERES, dtype=np.float64)

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
            missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
            raise ValueError(f"{path}: missing columns {missing}")

        for row in reader:
            sid = int(row["sphere_id"])
            if sid < 0 or sid >= NUM_SPHERES:
                raise ValueError(f"{path}: sphere_id {sid} out of range [0, {NUM_SPHERES})")
            dp[sid] = [float(row["dx"]), float(row["dy"]), float(row["dz"])]
            dr[sid] = float(row["dr"])

    return dp, dr


def load_all_betas(data_dir: str | Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Scan *data_dir* for beta*.csv files sorted by name and return lists."""
    pattern = os.path.join(data_dir, "beta*.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"[warn] no beta*.csv files found in {data_dir}, using hardcoded defaults")
        return _default_offsets()

    all_dp: list[np.ndarray] = []
    all_dr: list[np.ndarray] = []
    for p in paths:
        dp, dr = load_offsets_csv(p)
        all_dp.append(dp)
        all_dr.append(dr)
        print(f"  loaded {p}")
    return all_dp, all_dr


def _default_offsets() -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Hardcoded fallback so the tool always starts."""
    dp0 = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]])
    dp1 = np.array([[0, -0.5, 0], [0, 0, -0.5], [0.5, 0, 0]])
    dr0 = np.array([0.2, 0.1, 0.3])
    dr1 = np.array([-0.1, 0.2, -0.2])
    return [dp0, dp1], [dr0, dr1]

# ---------------------------------------------------------------------------
# Runtime computation
# ---------------------------------------------------------------------------


def compute(
    betas: list[float],
    offsets_pos: list[np.ndarray],
    offsets_r: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    pos = TEMPLATE_POS.copy()
    r = TEMPLATE_R.copy()
    for k, b in enumerate(betas):
        pos += b * offsets_pos[k]
        r += b * offsets_r[k]
    r = np.clip(r, MIN_RADIUS, None)
    return pos, r

# ---------------------------------------------------------------------------
# Sphere mesh generation (icosphere-style for Polyscope)
# ---------------------------------------------------------------------------


def _unit_sphere_mesh(subdivisions: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Generate a unit icosphere (vertices, faces)."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ], dtype=np.float64)
    verts /= np.linalg.norm(verts[0])

    faces = [
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
    ]

    for _ in range(subdivisions):
        mid_cache: dict[tuple[int, int], int] = {}
        new_faces = []
        v_list = list(verts)

        def midpoint(a: int, b: int) -> int:
            key = (min(a, b), max(a, b))
            if key in mid_cache:
                return mid_cache[key]
            m = (v_list[a] + v_list[b]) / 2.0
            m /= np.linalg.norm(m)
            idx = len(v_list)
            v_list.append(m)
            mid_cache[key] = idx
            return idx

        for tri in faces:
            a, b, c = tri
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]

        verts = np.array(v_list)
        faces = new_faces

    return verts, np.array(faces, dtype=np.int32)


UNIT_V, UNIT_F = _unit_sphere_mesh(subdivisions=2)

# ---------------------------------------------------------------------------
# Polyscope UI
# ---------------------------------------------------------------------------


def main() -> None:
    data_dir = Path(__file__).parent / "data"
    offsets_pos, offsets_r = load_all_betas(data_dir)
    num_betas = len(offsets_pos)

    betas = [0.0] * num_betas

    ps.init()
    ps.set_ground_plane_mode("shadow_only")
    ps.set_up_dir("y_up")

    handles: list[ps.SurfaceMesh | None] = [None] * NUM_SPHERES
    colors = [
        (0.26, 0.52, 0.96),
        (0.96, 0.52, 0.26),
        (0.26, 0.96, 0.52),
    ]

    def refresh() -> None:
        pos, radii = compute(betas, offsets_pos, offsets_r)
        for i in range(NUM_SPHERES):
            v = UNIT_V * radii[i] + pos[i]
            name = f"sphere_{i}"
            if handles[i] is None:
                handles[i] = ps.register_surface_mesh(name, v, UNIT_F, smooth_shade=True)
                handles[i].set_color(colors[i % len(colors)])
                handles[i].set_transparency(0.85)
            else:
                handles[i].update_vertex_positions(v)

    refresh()

    def ui_callback() -> None:
        changed = False
        psim.SetNextItemWidth(300)
        psim.TextUnformatted("Beta sliders")
        psim.Separator()
        for k in range(num_betas):
            c, val = psim.SliderFloat(f"beta{k}", betas[k], v_min=-2.0, v_max=2.0)
            if c:
                betas[k] = val
                changed = True

        psim.Separator()
        if psim.Button("Reset all"):
            for k in range(num_betas):
                betas[k] = 0.0
            changed = True

        if changed:
            refresh()

    ps.set_user_callback(ui_callback)
    ps.show()


if __name__ == "__main__":
    main()
