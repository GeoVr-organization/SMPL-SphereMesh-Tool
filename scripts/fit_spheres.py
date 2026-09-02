"""CLI for the geometric least-squares sphere fit in ``src/sphere_fit.py``.

Loads a sphere-mesh template and a neutral SMPL body, optimises the sphere centres and
radii against the body vertices at one or more centre-displacement caps, and prints the
two-column RMS report. Writing the result back to JSON refuses to overwrite an existing
file without ``--force``.

Examples
--------
    python scripts/fit_spheres.py --dry-run
    python scripts/fit_spheres.py --sweep 0,1,2,5,10,20
    python scripts/fit_spheres.py --max-offset 0.005 --out data/template_male_fit.json --force
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch as tc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sphere_fit import optimize_spheres, select_band, spheremesh_distance  # noqa: E402
from src.smpl_forward_compat import my_function_forward  # noqa: E402
from src.smpl_models import GENDERS, MALE, load_smpl_model, spheremesh_template_path  # noqa: E402


def load_template(path: Path) -> tuple[dict, np.ndarray, np.ndarray, list[tuple[int, ...]]]:
    """Return the raw payload plus centres, radii and connection index tuples."""
    payload = json.loads(path.read_text())
    centers = np.array([s["center"] for s in payload["spheres"]], dtype=np.float64)
    radii = np.array([s["radius"] for s in payload["spheres"]], dtype=np.float64)
    connections = [tuple(int(i) for i in c["indices"]) for c in payload["connections"]]
    return payload, centers, radii, connections


def neutral_body_vertices(gender: str) -> np.ndarray:
    """Vertices of the shape- and pose-neutral body, as float64."""
    model = load_smpl_model(gender)
    beta = tc.zeros(10, dtype=tc.float64)
    theta = tc.zeros(72, dtype=tc.float64)
    verts, _, _, _ = my_function_forward(model, beta, theta)
    return verts.detach().cpu().numpy().astype(np.float64)


def write_template(payload: dict, centers: np.ndarray, radii: np.ndarray, path: Path, force: bool) -> None:
    """Write the optimised spheres back into the template payload.

    Only ``center`` and ``radius`` are touched; sphere count, order, connections,
    parent_joint, rest_offset and lbs_weight are carried through untouched.
    """
    if path.exists() and not force:
        raise SystemExit(f"Refusing to overwrite {path} without --force")
    for sphere, center, radius in zip(payload["spheres"], centers, radii):
        sphere["center"] = [float(x) for x in center]
        sphere["radius"] = float(radius)
    path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {path}")


def report_run(label: str, info: dict) -> None:
    print(f"\n=== max_offset = {label} ===")
    print(
        "  iter  E              RMS spheres   RMS full      neg-frac  median f   at-delta  at-r_min"
    )
    for index, iteration in enumerate(info["iterations"]):
        print(
            f"  {index:<5} {iteration['E']:<14.8f} {iteration['rms_spheres']:<13.6f} "
            f"{iteration['rms_full']:<13.6f} {iteration['negative_fraction']:<9.3f} "
            f"{iteration['mean_median_residual'] * 1000:<10.2f} "
            f"{iteration['at_max_offset']:<9} {iteration['at_min_radius']}"
        )
    print(
        f"  negative-residual fraction  {info['negative_fraction_before']:.3f} (template) -> "
        f"{info['negative_fraction_after']:.3f} (fitted)"
    )
    print(
        "     the primary instrument, lower is better; the symmetric loss reaches 0.516 here"
    )
    print(
        f"  spheres-only RMS  {info['rms_spheres_before']:.6f} -> {info['rms_spheres_after']:.6f} m"
        + ("   (higher by design under an asymmetric loss)"
           if info["rms_spheres_after"] > info["rms_spheres_before"] else "")
    )
    print(
        f"  full metric  RMS  {info['rms_full_before']:.6f} -> {info['rms_full_after']:.6f} m"
        + ("   (WORSE -- the segment terms are doing work the spheres now undo)"
           if info["rms_full_after"] > info["rms_full_before"] else "")
    )
    print(
        f"  centre displacement  max {info['max_displacement'] * 1000:.3f} mm, "
        f"mean {info['mean_displacement'] * 1000:.3f} mm"
    )
    print(
        f"  radius change        max |dr| {info['max_radius_change'] * 1000:.3f} mm, "
        f"mean dr {info['mean_radius_change'] * 1000:.3f} mm"
    )
    frozen_empty = info["frozen_empty"]
    frozen_sparse = info["frozen_sparse"]
    print(f"  frozen, no points assigned ({len(frozen_empty)}): {frozen_empty}")
    print(f"  frozen, fewer than 4 points ({len(frozen_sparse)}): {frozen_sparse}")
    finite_conditions = info["condition_numbers"][np.isfinite(info["condition_numbers"])]
    if finite_conditions.size:
        print(
            f"  condition number of H  median {np.median(finite_conditions):.3e}, "
            f"max {finite_conditions.max():.3e}"
        )
    print(
        f"  unassigned by the {info['assign_radius_scale']:g}*r cap: {info['unassigned']} "
        f"of {info['num_points']} points"
    )
    print(
        f"  end-of-run partition drift {info['would_reassign_owner_fraction'] * 100:.2f}%, "
        f"cap-membership churn {info['would_recap_fraction'] * 100:.2f}%"
    )
    spread = info["per_sphere"]["axial_spread"]
    print(
        f"  assigned-set diameter  median {np.median(spread) * 1000:.1f} mm, "
        f"max {spread.max() * 1000:.1f} mm"
    )
    if info["degenerate_points"]:
        print(f"  degenerate |d_j| guarded: {info['degenerate_points']} point(s)")
    if not info["converged"] and not info["rolled_back"]:
        print(
            f"  NOTE: stopped at max_outer ({len(info['iterations'])}) before E settled; "
            "raise --max-outer for a tighter fit"
        )
    if info["rolled_back"]:
        print(
            "  NOTE: stopped when the next reassignment would have raised E; that round was "
            "rolled back, so the figures above are the best found"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gender", choices=GENDERS, default=MALE)
    parser.add_argument("--template", type=Path, default=None, help="Sphere-mesh JSON; defaults to the gender template")
    parser.add_argument(
        "--max-offset", type=float, default=0.010,
        help="Centre displacement cap in metres. See the max_offset section of "
             "src/sphere_fit.py for how the 10 mm default was chosen",
    )
    parser.add_argument("--sweep", type=str, default=None, help="Comma-separated caps in mm, e.g. 0,1,2,5,10,20")
    parser.add_argument("--min-radius", type=float, default=1e-3)
    parser.add_argument("--max-radius", type=float, default=np.inf)
    parser.add_argument(
        "--protrusion-weight", type=float, default=10.0,
        help="kappa: how much harder a protrusion is penalised than a gap. 1 is the plain "
             "squared loss, which inflates radii; larger values approach the inscribed sphere",
    )
    parser.add_argument(
        "--assignment", choices=("euclidean", "signed"), default="euclidean",
        help="Fit partition rule. 'signed' restores the pre-fix additively weighted argmin",
    )
    parser.add_argument(
        "--assign-radius-scale", type=float, default=1.75,
        help="beta: discard a point from a sphere when |v - c| > beta*r",
    )
    parser.add_argument(
        "--reassign-passes", type=int, default=0,
        help="Outer iterations after the first in which the partition is recomputed; "
             "0 freezes it at the input configuration",
    )
    parser.add_argument("--band-max", type=float, default=0.05, help="Band half-width in metres")
    parser.add_argument("--max-outer", type=int, default=20)
    parser.add_argument("--max-inner", type=int, default=10)
    parser.add_argument("--out", type=Path, default=None, help="Write the optimised template here")
    parser.add_argument("--force", action="store_true", help="Allow overwriting --out")
    parser.add_argument("--dry-run", action="store_true", help="Report the baseline only, do not optimise")
    args = parser.parse_args()

    template_path = args.template or spheremesh_template_path(args.gender)
    payload, centers, radii, connections = load_template(template_path)
    vertices = neutral_body_vertices(args.gender)

    band = select_band(vertices, centers, radii, connections, band_max=args.band_max)
    points = vertices[band]

    print(f"Template   {template_path}")
    print(f"           {len(centers)} spheres, {len(connections)} connections")
    print(f"Body       {args.gender}, neutral shape and pose, {len(vertices)} vertices")
    print(
        f"Band       |full metric| <= {args.band_max} m -> {len(points)} vertices "
        f"({100.0 * len(points) / len(vertices):.1f}% of the mesh)"
    )
    baseline_full = spheremesh_distance(points, centers, radii, connections)
    print(f"Baseline   full-metric RMS in band {np.sqrt(np.mean(baseline_full ** 2)):.6f} m")

    if args.dry_run:
        return 0

    caps = (
        [float(v) / 1000.0 for v in args.sweep.split(",")]
        if args.sweep
        else [args.max_offset]
    )

    last_centers, last_radii = centers, radii
    for cap in caps:
        last_centers, last_radii, info = optimize_spheres(
            centers,
            radii,
            points,
            max_offset=cap,
            min_radius=args.min_radius,
            max_radius=args.max_radius,
            max_outer=args.max_outer,
            max_inner=args.max_inner,
            protrusion_weight=args.protrusion_weight,
            assignment=args.assignment,
            assign_radius_scale=args.assign_radius_scale,
            reassign_passes=args.reassign_passes,
            connections=connections,
        )
        report_run(f"{cap * 1000:g} mm", info)

    if args.out is not None:
        if len(caps) > 1:
            print(f"\nWriting the last cap in the sweep ({caps[-1] * 1000:g} mm).")
        write_template(payload, last_centers, last_radii, args.out, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
