
"""Convert SphereMeshTool JSON to SMPL20-style text format.
Example usage: (.venv) PS C:> python .\SphereMeshTool\scripts\json2txt.py .\SphereMeshTool\data\template.json
Output format:
1) title line
2) "<num_spheres> <num_triples> <num_doubles> <num_quadruples>"
3) one line per sphere: "x y z radius"
4) one line per connection: indices separated by spaces (2/3/4 indices)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _validate_sphere(sphere: dict, idx: int) -> tuple[float, float, float, float]:
    center = sphere.get("center")
    radius = sphere.get("radius")
    if not isinstance(center, list) or len(center) < 3:
        raise ValueError(f"sphere[{idx}] has invalid 'center'")
    if radius is None:
        raise ValueError(f"sphere[{idx}] is missing 'radius'")
    return float(center[0]), float(center[1]), float(center[2]), float(radius)


def _validate_connection(
    conn: dict, idx: int, sphere_count: int
) -> tuple[int, ...]:
    indices = conn.get("indices")
    if not isinstance(indices, list):
        raise ValueError(f"connection[{idx}] has invalid 'indices'")
    if len(indices) not in (2, 3, 4):
        raise ValueError(
            f"connection[{idx}] has {len(indices)} indices, expected 2/3/4"
        )
    out: list[int] = []
    for j, v in enumerate(indices):
        if not isinstance(v, int):
            raise ValueError(f"connection[{idx}].indices[{j}] is not an integer")
        if v < 0 or v >= sphere_count:
            raise ValueError(
                f"connection[{idx}].indices[{j}]={v} out of range [0, {sphere_count - 1}]"
            )
        out.append(v)
    return tuple(out)


def _truncate(value: float, decimals: int = 4) -> float:
    factor = 10**decimals
    return math.trunc(value * factor) / factor


def convert_json_to_text(input_path: Path, title: str) -> str:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object")
    spheres = data.get("spheres")
    connections = data.get("connections")
    if not isinstance(spheres, list):
        raise ValueError("JSON must contain a 'spheres' list")
    if not isinstance(connections, list):
        raise ValueError("JSON must contain a 'connections' list")

    sphere_rows = [_validate_sphere(sp, i) for i, sp in enumerate(spheres)]
    conn_rows = [
        _validate_connection(conn, i, len(sphere_rows))
        for i, conn in enumerate(connections)
    ]

    n_triples = sum(1 for c in conn_rows if len(c) == 3)
    n_doubles = sum(1 for c in conn_rows if len(c) == 2)
    n_quads = sum(1 for c in conn_rows if len(c) == 4)

    conn_rows = sorted(
        conn_rows,
        key=lambda c: {3: 0, 2: 1, 4: 2}[len(c)],
    )

    lines: list[str] = [
        title,
        f"{len(sphere_rows)} {n_triples} {n_doubles} {n_quads}",
    ]
    for x, y, z, r in sphere_rows:
        # Truncate coordinates to 4 decimals (no rounding), keep radius unchanged.
        lines.append(
            f"{_truncate(x):.4f} {_truncate(y):.4f} {_truncate(z):.4f} {r}"
        )
    for conn in conn_rows:
        lines.append(" ".join(str(i) for i in conn))

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parent / "save1.json",
        help="Input JSON path (default: save1.json next to this script)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output text path (default: <input_stem>.txt)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Sphere Mesh 2.0",
        help="Title line for the output text",
    )
    args = parser.parse_args()

    input_path = args.input
    if not input_path.is_file():
        raise SystemExit(f"Input file not found: {input_path}")

    output_path = args.output
    if output_path is None:
        output_path = input_path.with_suffix(".txt")

    text = convert_json_to_text(input_path=input_path, title=args.title)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Wrote {output_path}")
    print(
        f"Spheres: {text.splitlines()[1].split()[0]}, "
        f"triples: {text.splitlines()[1].split()[1]}, "
        f"doubles: {text.splitlines()[1].split()[2]}, "
        f"quads: {text.splitlines()[1].split()[3]}"
    )


if __name__ == "__main__":
    main()
