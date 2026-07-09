#!/usr/bin/env python3
"""Append mirrored sphere/connection copies to a SphereMeshTool JSON save.

The original spheres and connections are kept.
For each original sphere with |center_x| > tolerance, a mirrored sphere is
added (x -> -x for center/rest_offset x). Then each connection that references
only mirrored-added spheres gets a mirrored copy with remapped indices.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def maybe_negate_x(x: float, tol: float) -> float:
    if abs(x) <= tol:
        return x
    return -x


def mirror_sphere_mesh_payload(data: dict, tol: float) -> dict:
    out = copy.deepcopy(data)
    if not isinstance(out.get("spheres"), list):
        raise ValueError("JSON must contain a 'spheres' list")
    if not isinstance(out.get("connections"), list):
        raise ValueError("JSON must contain a 'connections' list")

    orig_spheres = out["spheres"]
    orig_count = len(orig_spheres)
    mirror_index_of: dict[int, int] = {}

    for i in range(orig_count):
        sp = copy.deepcopy(orig_spheres[i])
        center = sp.get("center")
        if not (isinstance(center, list) and len(center) >= 3):
            continue

        x = float(center[0])
        if abs(x) <= tol:
            # Skip creating mirrored duplicate for near-midline spheres.
            continue

        center[0] = -x
        ro = sp.get("rest_offset")
        if ro is not None and isinstance(ro, list) and len(ro) >= 3:
            ro[0] = maybe_negate_x(float(ro[0]), tol)

        new_idx = len(out["spheres"])
        out["spheres"].append(sp)
        mirror_index_of[i] = new_idx

    for conn in list(out["connections"]):
        indices = conn.get("indices")
        if not isinstance(indices, list) or len(indices) < 2:
            continue
        if any((not isinstance(idx, int)) or (idx not in mirror_index_of) for idx in indices):
            continue

        mirrored_conn = {"indices": [mirror_index_of[idx] for idx in indices]}
        out["connections"].append(mirrored_conn)

    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parent / "save1.json",
        help="Input JSON (default: save1.json next to this script)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path (default: <input_stem>_mirrored.json)",
    )
    p.add_argument(
        "--tolerance",
        type=float,
        default=0.02,
        help="Do not negate x when |x| is at or below this (default: 0.02)",
    )
    args = p.parse_args()

    inp = args.input
    if not inp.is_file():
        raise SystemExit(f"Input not found: {inp}")

    out_path = args.output
    if out_path is None:
        out_path = inp.with_name(f"{inp.stem}_mirrored{inp.suffix}")

    with open(inp, "r", encoding="utf-8") as f:
        data = json.load(f)

    mirrored = mirror_sphere_mesh_payload(data, args.tolerance)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(mirrored, f, indent=2)
        f.write("\n")

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
