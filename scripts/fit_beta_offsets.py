"""
fit_beta_offsets.py
-------------------
Compute, for each SMPL shape coefficient, the per-sphere position/radius offsets
that keep a sphere-mesh template flush with the body surface, and write them as
the beta*.csv files SphereMeshTool blends in "Load Beta Offsets".

Full write-up -- derivations, provenance, measured numbers:  docs/fit_beta_offsets.md

This is the automatic counterpart of extract_beta_offsets.py: instead of diffing
a hand-edited save against the template, the offsets come from the SMPL shape
blendshapes.

What is preserved
-----------------
Skeleton and topology are untouched: same sphere count and order, same
connections, same parent_joint / rest_offset / lbs_weight.  Only two things move
-- the radius, and the centre, nudged slightly.

What a CSV means
----------------
Each betaN.csv is computed with beta[N] = 1 and every other coefficient at 0, so
the ten files are independent linear modes.  Each holds the FULL delta at
beta = 1, matching how SphereManager.apply_beta_offsets blends them:

    c_i(beta) = c_i^tpl + sum_k beta[k] * offsets_pos[k][i]
    r_i(beta) = r_i^tpl + sum_k beta[k] * offsets_r[k][i]

(extract_beta_offsets.py halves its delta before writing; this script does not.)

Method, per sphere i and coefficient k
--------------------------------------
At zero pose the SMPL surface is exactly affine in beta, so the vertex
displacement field

    u_j = v_j(e_k) - v_j(0)

is exactly the k-th shape blendshape column -- verified to 3e-16 m, i.e. float64
epsilon, not an approximation.

1. CENTRE -- first-order moving least squares.  Fit an affine model of u over the
   mesh vertices around the sphere, in coordinates local to the centre:

       min_{A,t}  sum_j w_ij |A (v_j(0) - c_i) + t - u_j|^2 + lam_i (|A|^2 + |t|^2)

   Gaussian weights w_ij with sigma = support/2, over a ball of radius
   max(support_scale * r_i, support_min); the 40 nearest vertices are used
   instead when the ball holds fewer than 16.  Stacking X = [p_j^T, 1] and
   U = [u_j^T] this is a single 4x4 normal-equation solve shared by the three
   output axes:

       (X^T W X + lam_i I_4) M = X^T W U,      M = [A^T ; t^T]

   with lam_i = lam * trace(X^T W X) / 4, which makes lam dimensionless and the
   same value valid for spheres of any size.  Because the fit is centred on c_i,
   the constant term IS the model evaluated at the centre:

       delta_c_i = t = M[3]

   Why affine and not a weighted average of the displacements: a torso sphere is
   MEDIAL, and its nearest vertices all sit on one side.  Under a local dilation
   u(x) = alpha * x the weighted average returns t = alpha * (weighted mean of x)
   != 0 and drags the centre toward the near surface, while the first-order model
   contains the truth (A = alpha, t = 0) and recovers t = 0.  A never moves the
   sphere -- it only removes that bias from t.

   Consequence: delta_c is a fixed linear operator (X, W, lam_i depend on the
   rest mesh only) applied to a field that is itself linear in beta, so the
   centre modes compose EXACTLY -- verified to 3e-16 m at beta all-ones,
   all-twos and mixed.  They are the exact Jacobian d c_i / d beta_k.

2. RADIUS -- preserve the tool's own flushness measure: the ratio between a
   sphere's radius and its distance to the closest SMPL vertex, which is what
   SphereManager.update_sphere_radius maintains through the rest_radius field.

       kappa_i = r_i(0) / dist(c_i(0), V(0))        kappa = 1 <=> tangent
       r_i(k)  = kappa_i * dist(c_i(k), V(k))

   kappa_i is recomputed from the geometry, NOT read from the template's
   rest_radius field: that field is stale in the shipped templates, because the
   routine that refreshes it is no longer called on beta changes.  In
   template_male_1.0.json sphere 32 stores 0.16 where the geometry says 1.00.

   The rule contains a min over vertices, so unlike the centre it is NOT linear
   in beta: composing all ten modes costs 0.95 mm on average (6.68 mm worst)
   with every slider at 1.  That is the sole source of the degradation seen when
   several sliders move together.

3. CSV row = (c_i(k) - c_i(0),  r_i(k) - r_i(0)), eight decimals, one per sphere.

Reported quality
----------------
RMS signed distance from the SMPL vertices to the sphere-mesh, using the same
distance function as src/spheremesh.py, restricted to the vertices the template
actually models (3837 of 6890 for the male template -- hands, feet and skull cap
are covered by no sphere and would swamp the number).  The figure to match is the
template on the neutral body: 6.01 mm for template_male_1.0.json.

Usage
-----
  python scripts/fit_beta_offsets.py --force              # all 10 betas, male
  python scripts/fit_beta_offsets.py --dry-run            # measure, write nothing
  python scripts/fit_beta_offsets.py --betas 1 3 --force
  python scripts/fit_beta_offsets.py --gender female \
      --template data/template_female_1.0.json --output-dir data/offsets_female
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))

_DATA_DIR = _REPO_ROOT / "data"
_MODELS_DIR = _REPO_ROOT / "SMPL_models"

DEFAULT_TEMPLATE = _DATA_DIR / "template_male_1.0.json"
DEFAULT_OUTPUT_DIR = _DATA_DIR / "offsets"
MODEL_FILES = {
    "male": "basicmodel_m_lbs_10_207_0_v1.0.0.npz",
    "female": "basicModel_f_lbs_10_207_0_v1.0.0.npz",
}

NUM_BETAS = 10
MIN_NEIGHBOURS = 16          # below this the ball query falls back to k-NN
FALLBACK_NEIGHBOURS = 40     # how many the k-NN fallback takes
MIRROR_TOLERANCE = 0.02      # same tolerance SphereManager uses for twins
REPORT_BAND = 0.015          # |sd| <= 15 mm marks the modelled body region


# ---------------------------------------------------------------------------
# SMPL
# ---------------------------------------------------------------------------

def load_smpl(model_path: Path):
    """Instantiate the SMPL model from a .npz. Imported late so --help stays fast."""
    from smpl_pytorch.smpl import SMPL, load_numpy_params
    return SMPL(load_numpy_params(model_path))


def smpl_vertices(model, beta_index: int | None, beta_value: float) -> np.ndarray:
    """Zero-pose SMPL vertices with a single beta set, every other one at zero.

    theta = 0 is essential: there the linear blend skinning collapses to the
    identity and the surface is exactly affine in beta,

        v_j(beta) = v_j(0) + sum_k beta[k] * S_jk

    so the difference of two calls reads one blendshape column S_.k exactly.
    Goes through the same forward wrapper the GUI uses, so these vertices are
    identical to the ones the tool renders.

    Pass beta_index=None for the neutral body.
    """
    import torch as tc
    from src.smpl_forward_compat import my_function_forward

    beta = tc.zeros(NUM_BETAS, dtype=tc.float64)
    if beta_index is not None:
        beta[beta_index] = float(beta_value)
    theta = tc.zeros(72, dtype=tc.float64)
    verts, _, _, _ = my_function_forward(model, beta, theta)
    return verts.detach().cpu().numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------

def load_template(path: Path):
    """Read (centers, radii, connections) from a SphereMeshTool topology JSON.

    Only the geometry is needed: parent joints, rest offsets and LBS weights are
    carried by the file the tool loads and are never touched by the offsets.
    Note the tool rounds centers to three decimals on load
    (SphereManager._trim_center_xyz_3decimals); the shipped templates already
    store them that way, so the snapshot the tool takes on "Load Beta Offsets"
    matches the base used here.
    """
    data = json.loads(path.read_text())
    if "spheres" not in data:
        raise ValueError(f"{path}: missing 'spheres' key")
    spheres = data["spheres"]
    if not isinstance(spheres, list) or not spheres:
        raise ValueError(f"{path}: 'spheres' must be a non-empty list")
    centers = np.array([s["center"] for s in spheres], dtype=np.float64)
    radii = np.array([float(s["radius"]) for s in spheres], dtype=np.float64)
    connections = [tuple(int(i) for i in c["indices"])
                   for c in data.get("connections", [])]
    return centers, radii, connections


def mirror_pairs(centers: np.ndarray, tol: float = MIRROR_TOLERANCE) -> dict[int, int]:
    """Map each sphere to its mirror twin across the X axis, where one exists.

    Same rule and same 0.02 m tolerance as SphereManager._find_mirror_sphere_index,
    so --symmetrize pairs spheres exactly the way the tool's mirrored editing does.
    On template_male_1.0.json 54 of 60 spheres get a twin.
    """
    pairs: dict[int, int] = {}
    for i, c in enumerate(centers):
        target = np.array([-c[0], c[1], c[2]])
        d = np.linalg.norm(centers - target, axis=1)
        d[i] = np.inf
        j = int(np.argmin(d))
        if d[j] <= tol:
            pairs[i] = j
    return pairs


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def local_affine_displacement(
    center: np.ndarray,
    radius: float,
    verts_rest: np.ndarray,
    displacement: np.ndarray,
    tree_rest: cKDTree,
    scale: float,
    min_radius: float,
    lam: float,
) -> np.ndarray:
    """How far one sphere centre moves, from a weighted local affine fit of the
    surface displacement field.  This is the core of the method.

    Fits, over the rest-mesh vertices around the centre and in coordinates local
    to it (p_j = v_j(0) - c_i),

        min_{A,t}  sum_j w_ij |A p_j + t - u_j|^2  +  lam_i (|A|^2 + |t|^2)

    and returns t, the model evaluated at the centre itself (p = 0).

    Support and weights
        rho_i = max(scale * radius, min_radius)
        N_i   = { j : |p_j| <= rho_i },  or the 40 nearest vertices if |N_i| < 16
        w_ij  = exp(-0.5 * (|p_j| / (rho_i/2))^2)
    The support edge therefore sits at two standard deviations, where the weight
    is already down to exp(-2), so truncating the Gaussian costs little.

    Solution
        X = [p_j^T, 1] (n,4),  U = [u_j^T] (n,3),  W = diag(w),  M = [A^T; t^T]
        (X^T W X + lam_i I_4) M = X^T W U
        lam_i = lam * trace(X^T W X) / 4
    The normal matrix is 4x4 whatever n is, and the three output axes share it:
    one solve per sphere, constant cost.  Its top-left block is the weighted
    scatter matrix of the neighbourhood, so it goes singular exactly when the
    patch is degenerate (vertices coplanar with c_i, or n < 4) -- which is what
    the ridge is for.  Scaling the ridge by the mean diagonal keeps lam
    dimensionless; measured on the male template it cuts the worst condition
    number from 7.2e4 to 3.8e3 and barely touches the typical case.

    Returning M[3] rather than re-evaluating the model is not a shortcut: because
    p is measured from c_i, the constant row *is* the value at the centre.

    Why the first-order model and not a plain weighted average of u -- see the
    module docstring; in short, an average drags a medial sphere toward whichever
    surface its neighbourhood happens to face, and the linear term removes that
    bias without ever being used to move the sphere.
    """
    # --- support and neighbourhood:  rho_i = max(scale * r_i, min_radius) -----
    # A sphere that is flush has its nearest vertices *at* distance r_i rather
    # than inside it, so for the big interior spheres the ball comes back nearly
    # empty and the k-NN fallback is the normal path, not an exception: on the
    # male template 24 of 60 spheres take it (mean radius 91 mm vs 57 mm).
    support = max(scale * radius, min_radius)
    idx = tree_rest.query_ball_point(center, support)
    if len(idx) < MIN_NEIGHBOURS:
        idx = tree_rest.query(center, FALLBACK_NEIGHBOURS)[1]
    idx = np.asarray(idx, dtype=int)

    # --- Gaussian weights, sigma = rho_i / 2 ---------------------------------
    # p_j = v_j(0) - c_i, expressed local to the centre so that the fit's
    # constant term ends up being the model evaluated at the centre.
    local = verts_rest[idx] - center
    dist = np.linalg.norm(local, axis=1)
    weight = np.exp(-0.5 * (dist / (0.5 * support)) ** 2)[:, None]

    # --- weighted least squares:  (X^T W X + lam_i I) M = X^T W U ------------
    # X = [p_j^T, 1], U = [u_j^T], M = [A^T ; t^T].  4x4 whatever n is, and the
    # three output axes share it, so one solve does all of them.
    design = np.hstack([local, np.ones((len(idx), 1))])          # X   (n, 4)
    normal = design.T @ (weight * design)                        # X^T W X (4, 4)
    rhs = design.T @ (weight * displacement[idx])                # X^T W U (4, 3)
    # Ridge scaled by the mean diagonal, which makes `lam` dimensionless and
    # pulls M toward "no displacement" when the patch is near-planar and so
    # leaves A undetermined along the normal.
    ridge = lam * np.trace(normal) / 4.0
    model = np.linalg.solve(normal + ridge * np.eye(4), rhs)
    return model[3]      # t -- the constant row IS the fit at p = 0, i.e. at c_i


def compute_offsets(
    centers: np.ndarray,
    radii: np.ndarray,
    verts_rest: np.ndarray,
    verts_shaped: np.ndarray,
    scale: float,
    min_radius: float,
    lam: float,
):
    """Offsets that take the template from the neutral body to a shaped one.

    Returns (delta_centers (N,3), delta_radii (N,)), i.e. the two halves of a
    betaN.csv row, for the shaped mesh handed in.

    Centre: the local affine transfer of the displacement field u = shaped - rest
    (see local_affine_displacement).  Exactly linear in the field, hence exactly
    linear in beta.

    Radius: preserve each sphere's flushness ratio against the deformed surface,

        kappa_i = r_i / dist(c_i, rest vertices)          measured once, on rest
        r_i^new = kappa_i * dist(c_i^new, shaped vertices)

    kappa is recomputed here rather than read from the template's rest_radius
    field, which is stale in the shipped files.  Note the radius rule is NOT
    linear in beta -- it contains a min over vertices -- so summing modes
    approximates it rather than reproducing it.

    Raises ValueError if a sphere centre coincides with a mesh vertex, which
    would make kappa undefined.
    """
    tree_rest = cKDTree(verts_rest)
    tree_shaped = cKDTree(verts_shaped)

    # The template's own "flushness": radius over distance to the closest vertex.
    clearance = tree_rest.query(centers)[0]
    if np.any(clearance <= 1e-12):
        raise ValueError("a sphere center coincides with a mesh vertex")
    ratio = radii / clearance

    displacement = verts_shaped - verts_rest
    new_centers = np.array([
        centers[i] + local_affine_displacement(
            centers[i], radii[i], verts_rest, displacement, tree_rest,
            scale, min_radius, lam,
        )
        for i in range(len(centers))
    ])
    new_radii = ratio * tree_shaped.query(new_centers)[0]
    return new_centers - centers, new_radii - radii


def symmetrize(delta_pos: np.ndarray, delta_r: np.ndarray, pairs: dict[int, int]):
    """Average each offset with its X-mirrored twin: dx antisymmetric, the rest even.

        delta_p_i <- (delta_p_i + R delta_p_j) / 2,   R = diag(-1, 1, 1)
        delta_r_i <- (delta_r_i + delta_r_j) / 2

    Off by default.  The SMPL shape blendshapes are genuinely asymmetric about
    the sagittal plane -- 0.88 mm on beta[0] rising to 2.13 mm on beta[8], where
    the asymmetry exceeds the mode's own magnitude -- so symmetrising discards
    real shape variation.  It costs little if wanted anyway: beta[0] goes from
    5.91 to 5.94 mm.
    """
    out_pos = delta_pos.copy()
    out_r = delta_r.copy()
    for i, j in pairs.items():
        twin = delta_pos[j] * np.array([-1.0, 1.0, 1.0])
        out_pos[i] = 0.5 * (delta_pos[i] + twin)
        out_r[i] = 0.5 * (delta_r[i] + delta_r[j])
    return out_pos, out_r


# ---------------------------------------------------------------------------
# Quality metric: distance from the SMPL surface to the sphere-mesh
# ---------------------------------------------------------------------------

def segment_pairs(connections) -> list[tuple[int, int]]:
    """Unordered sphere pairs spanned by the connections, deduplicated.

    A 2-sphere connection contributes one segment; a 3-sphere one contributes the
    three edges of its triangle.  template_male_1.0.json: 58 connections (40
    binary, 18 ternary) -> 88 segments.
    """
    pairs = set()
    for c in connections:
        edges = [(c[0], c[1])] if len(c) == 2 else [
            (c[0], c[1]), (c[1], c[2]), (c[0], c[2])
        ]
        for a, b in edges:
            pairs.add((min(a, b), max(a, b)))
    return sorted(pairs)


def spheremesh_distance(centers, radii, points, pairs) -> np.ndarray:
    """Signed distance from points to the sphere-mesh.  Negative = inside.

    Vectorised mirror of Spheremesh.get_distance / Trispheremesh.get_distance in
    src/spheremesh.py: the minimum over the spheres and over the tapered segments,

        sd(p) = min(  min_i ( |p - c_i| - r_i ),
                      min_(a,b) ( |p - q_ab(t*)| - r_ab(t*) )  )

        t*        = clamp( <p - c_a, c_b - c_a> / |c_b - c_a|^2 , 0, 1 )
        q_ab(t)   = c_a + t (c_b - c_a)
        r_ab(t)   = r_a + t (r_b - r_a)

    Like the tool's own function this measures against the sphere interpolated at
    the clamped ORTHOGONAL projection, which is not the exact distance to the
    tangent-cone envelope (that would need projecting along the slant).  The
    approximation is reproduced deliberately, so the reported error matches what
    the tool draws and what its heatmap shows.
    """
    sd = (np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
          - radii[None, :]).min(axis=1)
    for a, b in pairs:
        axis = centers[b] - centers[a]
        denom = float(axis @ axis)
        if denom < 1e-12:
            continue
        t = np.clip((points - centers[a]) @ axis / denom, 0.0, 1.0)
        closest = centers[a][None, :] + t[:, None] * axis[None, :]
        taper = radii[a] + t * (radii[b] - radii[a])
        sd = np.minimum(sd, np.linalg.norm(points - closest, axis=1) - taper)
    return sd


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def write_csv(delta_pos: np.ndarray, delta_r: np.ndarray, path: Path) -> None:
    """Write one betaN.csv: header plus a row per sphere, eight decimals.

    Same columns and precision as extract_beta_offsets.py, so hand-made and
    fitted files are interchangeable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sphere_id", "dx", "dy", "dz", "dr"])
        for i in range(len(delta_pos)):
            writer.writerow([
                i,
                f"{delta_pos[i, 0]:.8f}",
                f"{delta_pos[i, 1]:.8f}",
                f"{delta_pos[i, 2]:.8f}",
                f"{delta_r[i]:.8f}",
            ])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI: validate inputs, fit the requested betas, report quality, write CSVs.

    The report measures each beta twice -- once with the template untouched and
    once with the fitted offsets applied -- against the same band of vertices,
    so the two columns are directly comparable, and both against the template on
    the neutral body, which is the figure to match rather than zero.

    Refuses to overwrite existing files unless --force, since the offsets
    directory may hold hand-made modes.
    """
    parser = argparse.ArgumentParser(
        description="Fit sphere-mesh beta offsets against the SMPL shape space.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE,
                        help=f"Template topology JSON at beta = 0. Default: {DEFAULT_TEMPLATE}")
    parser.add_argument("--gender", choices=sorted(MODEL_FILES), default="male",
                        help="Which SMPL model to shape. Default: male")
    parser.add_argument("--model", type=Path, default=None,
                        help="Explicit SMPL .npz path (overrides --gender)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"Where to write betaN.csv. Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--betas", type=int, nargs="+", default=list(range(NUM_BETAS)),
                        metavar="N", help="Beta indices to fit (0-based). Default: 0..9")
    parser.add_argument("--beta-value", type=float, default=1.0,
                        help="Slider value each beta is driven to. Default: 1.0")
    parser.add_argument("--support-scale", type=float, default=1.0,
                        help="Affine-fit support radius as a multiple of the sphere radius. Default: 1.0")
    parser.add_argument("--support-min", type=float, default=0.06,
                        help="Floor for the support radius, in metres. Default: 0.06")
    parser.add_argument("--regularization", type=float, default=1e-3,
                        help="Ridge term on the affine fit. Default: 1e-3")
    parser.add_argument("--symmetrize", action="store_true",
                        help="Average each offset with its X-mirrored twin. Off by default: "
                             "the SMPL blendshapes are genuinely asymmetric.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing betaN.csv files.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report only, write nothing.")
    args = parser.parse_args()

    model_path = args.model or (_MODELS_DIR / MODEL_FILES[args.gender])
    if not args.template.is_file():
        sys.exit(f"ERROR: template not found: {args.template}")
    if not model_path.is_file():
        sys.exit(f"ERROR: SMPL model not found: {model_path}")
    bad = [b for b in args.betas if not 0 <= b < NUM_BETAS]
    if bad:
        sys.exit(f"ERROR: beta indices out of range [0, {NUM_BETAS}): {bad}")

    targets = {b: args.output_dir / f"beta{b}.csv" for b in sorted(set(args.betas))}
    if not args.dry_run and not args.force:
        clash = [str(p) for p in targets.values() if p.exists()]
        if clash:
            sys.exit("ERROR: refusing to overwrite existing files (use --force):\n  "
                     + "\n  ".join(clash))

    centers, radii, connections = load_template(args.template)
    pairs = segment_pairs(connections)
    twins = mirror_pairs(centers)

    print(f"Template : {args.template}  ({len(centers)} spheres, {len(connections)} connections)")
    print(f"Model    : {model_path}")
    print(f"Output   : {args.output_dir}")
    print(f"Betas    : {sorted(targets)}  driven to {args.beta_value}"
          f"{'  (symmetrized)' if args.symmetrize else ''}")
    if args.symmetrize:
        print(f"           mirrored twins found for {len(twins)}/{len(centers)} spheres")
    print()

    model = load_smpl(model_path)
    verts_rest = smpl_vertices(model, None, 0.0)

    sd_rest = spheremesh_distance(centers, radii, verts_rest, pairs)
    band = sd_rest <= REPORT_BAND
    base_rms = float(np.sqrt((sd_rest[band] ** 2).mean()) * 1000.0)
    print(f"Fit quality is the RMS distance from the SMPL surface to the sphere-mesh,")
    print(f"over the {int(band.sum())} of {len(verts_rest)} vertices the template models.")
    print(f"Template on the neutral body: {base_rms:.2f} mm  <- the target to match")
    print()
    print(f"{'beta':>5}  {'no offsets':>11}  {'fitted':>8}  {'max |dc|':>9}  {'max |dr|':>9}  "
          f"{'mean |dc|':>10}")

    for beta_index, out_path in targets.items():
        verts_shaped = smpl_vertices(model, beta_index, args.beta_value)
        delta_pos, delta_r = compute_offsets(
            centers, radii, verts_rest, verts_shaped,
            args.support_scale, args.support_min, args.regularization,
        )
        if args.symmetrize:
            delta_pos, delta_r = symmetrize(delta_pos, delta_r, twins)

        sd_off = spheremesh_distance(centers, radii, verts_shaped, pairs)[band]
        sd_fit = spheremesh_distance(centers + delta_pos, radii + delta_r,
                                     verts_shaped, pairs)[band]
        print(f"{beta_index:>5}  {np.sqrt((sd_off**2).mean())*1000:8.2f} mm  "
              f"{np.sqrt((sd_fit**2).mean())*1000:5.2f} mm  "
              f"{np.abs(delta_pos).max()*1000:6.2f} mm  "
              f"{np.abs(delta_r).max()*1000:6.2f} mm  "
              f"{np.linalg.norm(delta_pos, axis=1).mean()*1000:7.2f} mm")

        if not args.dry_run:
            write_csv(delta_pos, delta_r, out_path)

    print()
    if args.dry_run:
        print("Dry run: no files written.")
    else:
        print(f"Wrote {len(targets)} file(s) into {args.output_dir}")
    print()
    print("In SphereMeshTool:")
    print(f"  1. Load Topology -> {args.template.name}   (all beta sliders at 0)")
    print("  2. Click 'Load Beta Offsets'   (snapshots the template)")
    print("  3. Move any Beta slider: index N in the CSV name is the 'Beta N+1' slider.")


if __name__ == "__main__":
    main()
