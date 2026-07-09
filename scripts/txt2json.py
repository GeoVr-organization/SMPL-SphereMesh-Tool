"""Convert SMPL20-style SphereMeshTool text back to JSON (inverse of json2txt.py).

Input format (same as json2txt output):
1) title line (ignored for JSON; not stored)
2) "<num_spheres> <num_triples> <num_doubles> <num_quadruples>"
3) one line per sphere: "x y z radius"
4) triple connections (3 indices), then doubles (2), then quads (4)

Sphere fields not present in the text file use the same defaults as
``ManagedSphere`` / ``SphereManager.load_from_json`` when keys are missing:
``parent_joint``, ``rest_offset``, ``rest_radius``, ``nearest_vertex_index`` → null;
``locked`` → false.

Note: SphereMeshTool's in-app loader currently accepts only 2- or 3-index
connections; quadruple lines are still written to JSON with a warning.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

# Keep in sync with src/sphere_manager.py JSON_VERSION
JSON_VERSION = 1

SPHERE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TXT = SPHERE_ROOT / "data" / "save1.txt"


def _parse_header_line(line: str) -> tuple[int, int, int, int]:
    parts = line.split()
    if len(parts) != 4:
        raise ValueError(
            f"Expected 4 integers on counts line (spheres triples doubles quads), got: {line!r}"
        )
    a, b, c, d = (int(x) for x in parts)
    return a, b, c, d


def _parse_sphere_line(line: str, idx: int) -> tuple[float, float, float, float]:
    parts = line.split()
    if len(parts) != 4:
        raise ValueError(f"sphere line {idx}: expected 4 numbers, got {line!r}")
    x, y, z, r = (float(t) for t in parts)
    return x, y, z, r


def _parse_connection_line(line: str, expected_len: int, idx: int) -> list[int]:
    parts = line.split()
    if len(parts) != expected_len:
        raise ValueError(
            f"connection line {idx}: expected {expected_len} indices, got {len(parts)} in {line!r}"
        )
    out: list[int] = []
    for p in parts:
        if "." in p:
            raise ValueError(f"connection line {idx}: index must be integer, got {p!r}")
        out.append(int(p))
    return out


def _validate_connection(indices: list[int], sphere_count: int, label: str) -> None:
    if len(set(indices)) != len(indices):
        raise ValueError(f"{label}: duplicate sphere index in {indices}")
    for j, v in enumerate(indices):
        if v < 0 or v >= sphere_count:
            raise ValueError(
                f"{label}: indices[{j}]={v} out of range [0, {sphere_count - 1}]"
            )


def convert_text_to_json_dict(lines: list[str]) -> dict:
    """Parse non-empty stripped lines into the save payload dict."""
    if len(lines) < 2:
        raise ValueError("Need at least a title line and a counts line")

    n_spheres, n_triples, n_doubles, n_quads = _parse_header_line(lines[1])
    need = 2 + n_spheres + n_triples + n_doubles + n_quads
    if len(lines) < need:
        raise ValueError(
            f"File too short: need {need} lines after stripping blanks, got {len(lines)}"
        )

    spheres_out: list[dict] = []
    offset = 2
    for i in range(n_spheres):
        x, y, z, r = _parse_sphere_line(lines[offset + i], i)
        spheres_out.append(
            {
                "center": [x, y, z],
                "radius": r,
                "parent_joint": None,
                "rest_offset": None,
                "rest_radius": None,
                "nearest_vertex_index": None,
                "locked": False,
            }
        )
    offset += n_spheres

    connections_out: list[dict] = []
    for i in range(n_triples):
        idxs = _parse_connection_line(lines[offset + i], 3, i)
        _validate_connection(idxs, n_spheres, f"triple[{i}]")
        connections_out.append({"indices": idxs})
    offset += n_triples

    for i in range(n_doubles):
        idxs = _parse_connection_line(lines[offset + i], 2, i)
        _validate_connection(idxs, n_spheres, f"double[{i}]")
        connections_out.append({"indices": idxs})
    offset += n_doubles

    for i in range(n_quads):
        idxs = _parse_connection_line(lines[offset + i], 4, i)
        _validate_connection(idxs, n_spheres, f"quad[{i}]")
        connections_out.append({"indices": idxs})

    if n_quads > 0:
        warnings.warn(
            "Text contains quadruple connections; JSON will include them, but "
            "SphereMeshTool.load_from_json currently only loads 2- or 3-index connections.",
            UserWarning,
            stacklevel=2,
        )

    trailing = lines[offset + n_quads :]
    if any(line.strip() for line in trailing):
        warnings.warn(
            f"Ignoring {len(trailing)} non-empty line(s) after parsed connections.",
            UserWarning,
            stacklevel=2,
        )

    return {
        "version": JSON_VERSION,
        "spheres": spheres_out,
        "connections": connections_out,
    }


def convert_text_to_json_payload(text: str) -> dict:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() != ""]
    return convert_text_to_json_dict(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=DEFAULT_TXT,
        help=f"Input .txt path (default: {DEFAULT_TXT})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: <input_stem>.json next to the .txt)",
    )
    args = parser.parse_args()

    input_path = args.input
    if not input_path.is_file():
        raise SystemExit(f"Input file not found: {input_path}")

    text = input_path.read_text(encoding="utf-8")
    payload = convert_text_to_json_payload(text)

    output_path = args.output
    if output_path is None:
        output_path = input_path.with_suffix(".json")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    nsp = len(payload["spheres"])
    nconn = len(payload["connections"])
    print(f"Wrote {output_path}")
    print(f"Spheres: {nsp}, connections: {nconn}")


if __name__ == "__main__":
    main()
