"""
extract_beta_offsets.py
-----------------------
Compute per-sphere position/radius offsets between a template sphere mesh and a
modified (hand-edited or re-fitted) sphere mesh, then write the result as a
beta*.csv file ready for SphereMeshTool's "Load Beta Offsets" feature.

Inputs are engine text files (``.txt``, as written by SMFitting) or tool JSON files;
the format is picked from the file extension. Only each sphere's centre and radius
are read.

Default layout:
    data/template_male_1.3_starpose.txt   the template (star pose, all betas 0), which
                                          the GUI also starts on
    SMFitting/output/beta0.txt, beta1.txt ...
                                          the sphere mesh fitted for SMPL beta N

The template must come from the same fitting run as the betaN.txt files:
SMFitting/mainbeta.py seeded with data/template_male_1.3_seed_starpose.txt, with the base
mesh (examples/SMPL_starpose.obj) as the "beta" mesh. The procedure moves some spheres by
tens of millimetres even when nothing changes (up to 31 mm from the 1.3 seed); taken against
that control fit, the move cancels out of every offset instead of being added sum(beta)
times by the GUI.

Usage
-----
  python scripts/extract_beta_offsets.py --beta-index N   -> data/offsets/betaN.csv
  python scripts/extract_beta_offsets.py --all            -> every betaN.txt found

The produced CSV has one row per sphere:
    sphere_id, dx, dy, dz, dr

where dx/dy/dz = (modified_center - template_center)  and
       dr      = (modified_radius - template_radius) .

Other options: --template, --modified, --input-dir, --output-dir (see --help).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

from txt2json import convert_text_to_json_payload

# ---------------------------------------------------------------------------
# Default paths (relative to this script's location)
# ---------------------------------------------------------------------------
_SCRIPT_DIR   = Path(__file__).resolve().parent
_REPO_ROOT    = _SCRIPT_DIR.parent
_OFFSETS_DIR  = _REPO_ROOT / "data" / "offsets"
_FIT_OUTPUT_DIR = _REPO_ROOT / "SMFitting" / "output"

DEFAULT_INPUT_DIR  = _FIT_OUTPUT_DIR
DEFAULT_TEMPLATE   = _REPO_ROOT / "data" / "template_male_1.3_starpose.txt"
DEFAULT_OUTPUT_DIR = _OFFSETS_DIR

# betaN.txt -> N, for --all
_BETA_TXT_PATTERN = re.compile(r"^beta(\d+)\.txt$")


# ---------------------------------------------------------------------------
# Loading helpers
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


def load_spheres(path: Path, label: str) -> list[dict]:
    """Return the sphere list of an engine ``.txt`` file or a tool ``.json`` file."""
    if path.suffix.lower() == ".txt":
        data = convert_text_to_json_payload(path.read_text(encoding="utf-8"))
    else:
        data = _load_json(path)
    return _parse_spheres(data, label)


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
            "Both files must have the same number of spheres in the same order."
        )

    rows = []
    for i, (t, m) in enumerate(zip(template_spheres, modified_spheres)):
        tc = t["center"]
        mc = m["center"]
        rows.append({
            "sphere_id": i,
            "dx": (mc[0] - tc[0]),
            "dy": (mc[1] - tc[1]),
            "dz": (mc[2] - tc[2]),
            "dr": (float(m["radius"]) - float(t["radius"])),
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

def extract_one(
    template_spheres: list[dict],
    modified_path: Path,
    beta_index: int,
    output_dir: Path,
) -> None:
    """Compare one modified file with the template and write ``beta<N>.csv``."""
    print(f"Modified : {modified_path}")
    modified_spheres = load_spheres(modified_path, "modified")
    rows = compute_offsets(template_spheres, modified_spheres)

    non_zero = sum(
        1 for r in rows
        if abs(r["dx"]) > 1e-12 or abs(r["dy"]) > 1e-12
        or abs(r["dz"]) > 1e-12 or abs(r["dr"]) > 1e-12
    )
    print(f"Spheres : {len(rows)} total, {non_zero} with non-zero offset")

    write_csv(rows, output_dir / f"beta{beta_index}.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract sphere offsets between a template and a modified sphere mesh "
                    "(.txt or .json), and write betaN.csv for use with SphereMeshTool.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=DEFAULT_TEMPLATE,
        help=f"Path to the template (rest shape). Default: {DEFAULT_TEMPLATE}",
    )
    parser.add_argument(
        "--modified",
        type=Path,
        default=None,
        help="Path to the modified file. Default: <input-dir>/beta<N>.txt",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory holding betaN.txt. Default: {DEFAULT_INPUT_DIR}",
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
        "--all",
        action="store_true",
        help="Process every betaN.txt in --input-dir (ignores --beta-index and --modified).",
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
    if args.beta_index < 0:
        print("ERROR: --beta-index must be >= 0", file=sys.stderr)
        sys.exit(1)

    # --- decide which (file, beta index) pairs to process ---
    if args.all:
        modified_files = {}
        for candidate in args.input_dir.glob("beta*.txt"):
            match = _BETA_TXT_PATTERN.match(candidate.name)
            if match:
                modified_files[int(match.group(1))] = candidate
        if not modified_files:
            print(f"ERROR: no betaN.txt found in {args.input_dir}", file=sys.stderr)
            sys.exit(1)
    else:
        modified_path = args.modified or args.input_dir / f"beta{args.beta_index}.txt"
        if not modified_path.is_file():
            print(f"ERROR: modified file not found: {modified_path}", file=sys.stderr)
            sys.exit(1)
        modified_files = {args.beta_index: modified_path}

    # --- load template once, then process each modified file ---
    print(f"Template : {args.template}")
    template_spheres = load_spheres(args.template, "template")
    for beta_index in sorted(modified_files):
        extract_one(template_spheres, modified_files[beta_index], beta_index, args.output_dir)
        print()

    print("Done.  Now in SphereMeshTool:")
    print("  1. Load the template (same sphere layout as the template file)")
    print("  2. Click 'Load Beta Offsets'")
    print("  3. Move a beta slider to 1.0 -> the spheres should match the modified file.")


if __name__ == "__main__":
    main()
