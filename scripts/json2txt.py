"""Convert SphereMeshTool JSON to SMPL20-style text format.

Example usage (from the repo root):
    .venv/Scripts/python.exe scripts/json2txt.py data/template.json

Output format:
1) title line
2) "<num_spheres> <num_triples> <num_doubles> <num_quadruples>"
3) one line per sphere: "x y z radius" (x, y, z truncated to 4 decimals, radius verbatim)
4) one line per connection: indices separated by spaces, grouped triples, doubles, quads

Quads
-----
The tool itself only accepts 2- and 3-index connections, so a generated template stores
each engine quad ``(a, b, c, d)`` as its two split triangles ``(a, b, c)`` and ``(a, c, d)``
in ``connections`` and records the original grouping under ``quad_groups``. When that key is
present every listed quad replaces its two split triangles in the emitted text, so the
counts line reports the quads and the triangle count drops by two per quad, exactly as
``src.topology.export.write_engine_text`` would have written it. A quad whose split
triangles are not both present is reported with a warning and its triangles are kept.

JSON without ``quad_groups`` converts exactly as before; 4-index entries in
``connections`` (legacy files) are still written out as quads.
"""

from __future__ import annotations

import argparse
import json
import warnings
from decimal import ROUND_DOWN, Decimal
from pathlib import Path


def _validate_sphere(sphere: dict, idx: int) -> tuple[float, float, float, float]:
    center = sphere.get("center")
    radius = sphere.get("radius")
    if not isinstance(center, list) or len(center) < 3:
        raise ValueError(f"sphere[{idx}] has invalid 'center'")
    if radius is None:
        raise ValueError(f"sphere[{idx}] is missing 'radius'")
    return float(center[0]), float(center[1]), float(center[2]), float(radius)


def _validate_indices(
    indices, label: str, sphere_count: int, allowed_lengths: tuple[int, ...]
) -> tuple[int, ...]:
    if not isinstance(indices, list):
        raise ValueError(f"{label} has invalid 'indices'")
    if len(indices) not in allowed_lengths:
        raise ValueError(
            f"{label} has {len(indices)} indices, expected "
            + "/".join(str(length) for length in allowed_lengths)
        )
    validated: list[int] = []
    for position, sphere_index in enumerate(indices):
        if not isinstance(sphere_index, int):
            raise ValueError(f"{label}.indices[{position}] is not an integer")
        if sphere_index < 0 or sphere_index >= sphere_count:
            raise ValueError(
                f"{label}.indices[{position}]={sphere_index} out of range "
                f"[0, {sphere_count - 1}]"
            )
        validated.append(sphere_index)
    return tuple(validated)


def _validate_connection(
    conn: dict, idx: int, sphere_count: int
) -> tuple[int, ...]:
    return _validate_indices(
        conn.get("indices"), f"connection[{idx}]", sphere_count, (2, 3, 4)
    )


def _validate_quad_group(quad, idx: int, sphere_count: int) -> tuple[int, ...]:
    quad_indices = _validate_indices(quad, f"quad_groups[{idx}]", sphere_count, (4,))
    if len(set(quad_indices)) != 4:
        raise ValueError(f"quad_groups[{idx}] repeats a sphere: {list(quad_indices)}")
    return quad_indices


def merge_quad_groups(
    connection_rows: list[tuple[int, ...]], quad_groups: list[tuple[int, ...]]
) -> list[tuple[int, ...]]:
    """Undo the 0-2 diagonal split: replace ``(a,b,c),(a,c,d)`` by ``(a,b,c,d)``.

    The quad takes the position of the earlier of its two triangles, so the relative order
    of connections -- and therefore the order of the emitted lines -- is the order the
    quads had before ``src.topology.export.split_quads`` split them. Each triangle can be
    consumed by one quad only; a quad missing either triangle is left out with a warning.
    """
    consumed = [False] * len(connection_rows)
    quad_at_position: dict[int, tuple[int, ...]] = {}

    def find_unconsumed(triangle: tuple[int, int, int]) -> int | None:
        for position, row in enumerate(connection_rows):
            if not consumed[position] and row == triangle:
                return position
        return None

    for quad_index, quad in enumerate(quad_groups):
        first, second, third, fourth = quad
        first_position = find_unconsumed((first, second, third))
        second_position = find_unconsumed((first, third, fourth))
        if first_position is None or second_position is None:
            warnings.warn(
                f"quad_groups[{quad_index}]={list(quad)}: split triangles "
                f"{[first, second, third]} and {[first, third, fourth]} not both present "
                "in 'connections'; keeping the triangles as they are.",
                UserWarning,
                stacklevel=2,
            )
            continue
        consumed[first_position] = True
        consumed[second_position] = True
        quad_at_position[min(first_position, second_position)] = quad

    merged: list[tuple[int, ...]] = []
    for position, row in enumerate(connection_rows):
        if position in quad_at_position:
            merged.append(quad_at_position[position])
        if not consumed[position]:
            merged.append(row)
    return merged


def _truncate(value: float, decimals: int = 4) -> float:
    """Truncate (toward zero, no rounding) to ``decimals`` places, exactly.

    Truncating on the shortest round-trip decimal representation instead of
    ``math.trunc(value * 10**decimals)`` avoids the binary-float artefact where a value
    such as ``0.0192`` becomes ``191.99999999999997`` after the multiplication and is
    truncated one unit low. It matters for round trips: a text file written with 4
    decimals must read back and write out to the very same digits. Kept identical to
    ``src.topology.export._truncate``.
    """
    quantum = Decimal(1).scaleb(-decimals)
    truncated = Decimal(repr(float(value))).quantize(quantum, rounding=ROUND_DOWN)
    # ``+ 0.0`` turns a truncated ``-0.0000`` into ``0.0`` so the sign never prints.
    return float(truncated) + 0.0


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

    quad_groups = data.get("quad_groups")
    if quad_groups is not None:
        if not isinstance(quad_groups, list):
            raise ValueError("'quad_groups' must be a list of 4-index lists")
        quad_rows = [
            _validate_quad_group(quad, i, len(sphere_rows))
            for i, quad in enumerate(quad_groups)
        ]
        conn_rows = merge_quad_groups(conn_rows, quad_rows)

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
