"""Score several sphere-mesh templates side by side with the exact metric.

    .venv/Scripts/python.exe scripts/compare_topologies.py data/template.json data/generated/auto_male.json
    .venv/Scripts/python.exe scripts/compare_topologies.py --poses data/template.json data/generated/auto_male.json

Every file is loaded the way the tool loads it (tool JSON; ``quad_groups`` are folded
back into engine quads when present) and run through ``src.topology.validate`` on the
same 3725 modelled vertices of the neutral body, so the columns are comparable across
hand-made and generated templates. ``--poses`` adds the 54-pose sweep (about 30 s per file).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.topology.anatomy import load_body, mocap_poses, random_poses  # noqa: E402
from src.topology.export import merge_quads  # noqa: E402
from src.topology.validate import Thresholds, validate  # noqa: E402


def load_template(path: Path) -> tuple[np.ndarray, np.ndarray, list[tuple[int, ...]], list[str] | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    centers = np.array([sphere["center"] for sphere in payload["spheres"]], dtype=np.float64)
    radii = np.array([sphere["radius"] for sphere in payload["spheres"]], dtype=np.float64)
    connections = [tuple(int(index) for index in entry["indices"]) for entry in payload["connections"]]
    if payload.get("quad_groups"):
        connections = merge_quads(connections, payload["quad_groups"])
    names = payload.get("generated_by", {}).get("sphere_names")
    return centers, radii, connections, names


COLUMNS = (
    ("spheres", 7, "{:d}"), ("tri", 3, "{:d}"), ("quad", 4, "{:d}"),
    ("rms", 5, "{:.2f}"), ("maxgap", 6, "{:.1f}"), (">15mm%", 6, "{:.2f}"),
    ("protr", 5, "{:.1f}"), ("<-5mm%", 6, "{:.2f}"), ("walk%", 5, "{:.2f}"),
    ("unreach", 7, "{:d}"), ("overlap%", 8, "{:.2f}"), ("xjoint", 6, "{:d}"),
    ("posed", 6, "{}"), ("passed", 6, "{}"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--gender", default="male")
    parser.add_argument("--poses", action="store_true", help="include the 54-pose sweep")
    args = parser.parse_args()

    body = load_body(args.gender)
    poses = None
    if args.poses:
        poses = np.concatenate([random_poses(), mocap_poses(stride=5)], axis=0)

    header = f"{'file':<28}" + "".join(f" {name:>{width}}" for name, width, _ in COLUMNS)
    print(header)
    for path in args.files:
        centers, radii, connections, names = load_template(path)
        report = validate(
            centers, radii, connections, body=body, thresholds=Thresholds(), poses=poses,
            names=names, check_symmetry=True,
        )
        rest = report.stats["rest"]
        walk = report.stats["walk"]
        crossings = [entry for entry in report.worst.get("joint_crossing_cones", []) if not entry["seam"]]
        overlap = next(
            (check.value for check in report.checks if check.name == "watchdog/non-adjacent-overlap"),
            float("nan"),
        )
        posed = report.stats.get("posed")
        values = [
            len(centers), sum(1 for c in connections if len(c) == 3),
            sum(1 for c in connections if len(c) == 4),
            rest["rms"] * 1000, rest["max_gap"] * 1000, rest["gap_fraction"] * 100,
            rest["max_protrusion"] * 1000, rest["protrusion_fraction"] * 100,
            walk["fraction"] * 100, walk["unreachable"], (overlap or 0.0) * 100, len(crossings),
            f"{posed['worst']['max_protrusion'] * 1000:.1f}" if posed else "-",
            f"{sum(1 for check in report.checks if check.passed)}/{len(report.checks)}",
        ]
        row = f"{path.stem:<28}" + "".join(
            f" {fmt.format(value):>{width}}" for (name, width, fmt), value in zip(COLUMNS, values)
        )
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
