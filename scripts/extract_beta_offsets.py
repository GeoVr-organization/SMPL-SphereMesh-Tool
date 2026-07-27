"""
extract_beta_offsets.py
-----------------------
Compute per-sphere position/radius offsets between a template topology JSON
and a modified (hand-edited) topology JSON, then write the result as a
beta*.csv file ready for SphereMeshTool's "Load Beta Offsets" feature.

Workflow (repeat for each SMPL beta you want to capture):
  1. Open SphereMeshTool with the rest-pose sphere layout.
  2. Save it as  data/template.json  (leave ALL beta sliders at 0).
  3. Move ONLY beta[N] slider to 1.0 in the tool.
  4. Manually adjust sphere positions / radii as desired.
  5. Save as  data/save1.json.
  6. Run this script:
       python extract_beta_offsets.py --beta-index N
     → writes  data/offsets/betaN.csv

The produced CSV has one row per sphere:
    sphere_id, dx, dy, dz, dr

where dx/dy/dz = modified_center - template_center  and
       dr      = modified_radius  - template_radius.

Usage
-----
  python extract_beta_offsets.py [--template PATH] [--modified PATH]
                                 [--beta-index N] [--output-dir PATH]

All arguments have defaults that match the standard folder layout.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Default paths (relative to this script's location)
# ---------------------------------------------------------------------------
_SCRIPT_DIR   = Path(__file__).resolve().parent
_REPO_ROOT    = _SCRIPT_DIR.parent          # python/
_DATA_DIR     = _REPO_ROOT / "SphereMeshTool" / "data"
_OFFSETS_DIR  = _DATA_DIR / "offsets"

DEFAULT_TEMPLATE  = _DATA_DIR / "template.json"
DEFAULT_MODIFIED  = _DATA_DIR / "save1.json"
DEFAULT_OUTPUT_DIR = _OFFSETS_DIR


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _parse_spheres(data: dict, label: str) -> list[dict]:
    """Validate and return the sphere list from a SphereMeshTool JSON."""
    if "spheres" not in data:
        raise ValueError(f"{label}: missing 'spheres' key")
    spheres = data["spheres"]
    if not isinstance(spheres, list) or len(spheres) == 0:
        raise ValueError(f"{label}: 'spheres' must be a non-empty list")
    return spheres


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_offsets(
    template_spheres: list[dict],
    modified_spheres: list[dict],
) -> list[dict]:
    """Return a list of per-sphere offset dicts.

    Each entry: {"sphere_id": i, "dx": ..., "dy": ..., "dz": ..., "dr": ...}

    Sphere order is assumed to be consistent between template and modified.
    Sphere count must match.
    """
    n_t = len(template_spheres)
    n_m = len(modified_spheres)
    if n_t != n_m:
        raise ValueError(
            f"Sphere count mismatch: template has {n_t}, modified has {n_m}. "
            "Both JSONs must have the same number of spheres in the same order."
        )

    rows = []
    for i, (t, m) in enumerate(zip(template_spheres, modified_spheres)):
        tc = t["center"]
        mc = m["center"]
        rows.append({
            "sphere_id": i,
            "dx": (mc[0] - tc[0])*0.5,
            "dy": (mc[1] - tc[1])*0.5,
            "dz": (mc[2] - tc[2])*0.5,
            "dr": (float(m["radius"]) - float(t["radius"]))*0.5,
        })
    return rows


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sphere_id", "dx", "dy", "dz", "dr"])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "sphere_id": row["sphere_id"],
                "dx": f"{row['dx']:.8f}",
                "dy": f"{row['dy']:.8f}",
                "dz": f"{row['dz']:.8f}",
                "dr": f"{row['dr']:.8f}",
            })
    print(f"Written: {output_path}  ({len(rows)} spheres)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract sphere offsets between a template and a modified JSON, "
                    "and write betaN.csv for use with SphereMeshTool.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=DEFAULT_TEMPLATE,
        help=f"Path to template JSON (rest pose). Default: {DEFAULT_TEMPLATE}",
    )
    parser.add_argument(
        "--modified",
        type=Path,
        default=DEFAULT_MODIFIED,
        help=f"Path to modified/hand-edited JSON. Default: {DEFAULT_MODIFIED}",
    )
    parser.add_argument(
        "--beta-index",
        type=int,
        default=0,
        metavar="N",
        help="SMPL beta slider index this offset corresponds to (0-based). "
             "Produces betaN.csv. Default: 0",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write betaN.csv into. Default: {DEFAULT_OUTPUT_DIR}",
    )

    args = parser.parse_args()

    # --- validate inputs ---
    if not args.template.is_file():
        print(f"ERROR: template file not found: {args.template}", file=sys.stderr)
        sys.exit(1)
    if not args.modified.is_file():
        print(f"ERROR: modified file not found: {args.modified}", file=sys.stderr)
        sys.exit(1)
    if args.beta_index < 0:
        print("ERROR: --beta-index must be >= 0", file=sys.stderr)
        sys.exit(1)

    # --- load ---
    print(f"Template : {args.template}")
    print(f"Modified : {args.modified}")
    template_data = _load_json(args.template)
    modified_data = _load_json(args.modified)

    template_spheres = _parse_spheres(template_data, "template")
    modified_spheres = _parse_spheres(modified_data, "modified")

    # --- compute ---
    rows = compute_offsets(template_spheres, modified_spheres)

    non_zero = sum(
        1 for r in rows
        if abs(r["dx"]) > 1e-12 or abs(r["dy"]) > 1e-12
        or abs(r["dz"]) > 1e-12 or abs(r["dr"]) > 1e-12
    )
    print(f"Spheres : {len(rows)} total, {non_zero} with non-zero offset")

    # --- write ---
    output_path = args.output_dir / f"beta{args.beta_index}.csv"
    write_csv(rows, output_path)

    print()
    print("Done.  Now in SphereMeshTool:")
    print("  1. Load template.json")
    print("  2. Click 'Load Beta Offsets'")
    print(f"  3. Move the Beta {args.beta_index + 1} slider to 1.0 → "
          "the spheres should match your hand-edited save.")


if __name__ == "__main__":
    main()
