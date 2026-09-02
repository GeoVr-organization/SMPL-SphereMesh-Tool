"""Acceptance tests for the radius-inflation fixes: asymmetric loss and Euclidean partition.

Covers Phase 1 and Phase 2 of the second implementation brief. The Phase 1 bit-identity
regression compares against a snapshot of the symmetric-loss implementation, regenerated
with --write-baseline when the pre-change behaviour is deliberately superseded.

Run with: python scripts/test_sphere_fit_inflation.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch as tc
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sphere_fit import (  # noqa: E402
    MAX_PROTRUSION_WEIGHT,
    assign_partition,
    optimize_spheres,
    select_band,
)
from src.smpl_forward_compat import my_function_forward  # noqa: E402
from src.smpl_models import MALE, load_smpl_model, spheremesh_template_path  # noqa: E402

FAILURES: list[str] = []

CYLINDER_RADIUS_MM = 30.0
CYLINDER_HALF_EXTENT_MM = 30.0
CYLINDER_SAMPLES = 4001

BRIEF_TABLE = {1.0: 34.44, 2.0: 32.44, 5.0: 30.90, 10.0: 30.38, 20.0: 30.16}


def check(condition: bool, label: str, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f"  ({detail})" if detail else ""))
    if not condition:
        FAILURES.append(label)


def cylinder_points(radius: float, half_extent: float, count: int) -> np.ndarray:
    """Points on a cylinder of the given radius, axis along z, centred on the origin.

    Azimuth advances by the golden angle so the samples wrap the full circumference and the
    normal matrix is non-singular in the two radial directions.
    """
    axial = np.linspace(-half_extent, half_extent, count)
    angle = np.arange(count) * np.pi * (3.0 - np.sqrt(5.0))
    return np.column_stack(
        (radius * np.cos(angle), radius * np.sin(angle), axial)
    ).astype(np.float64)


def weighted_mean_fixed_point(distances: np.ndarray, protrusion_weight: float) -> float:
    """Fixed point of r <- sum w(r) d / sum w(r), the exact minimiser at a held centre."""
    radius = float(distances.mean())
    for _ in range(500):
        loss_weights = np.where(distances - radius < 0.0, protrusion_weight ** 2, 1.0)
        updated = float(np.sum(loss_weights * distances) / np.sum(loss_weights))
        if abs(updated - radius) < 1e-15:
            return updated
        radius = updated
    return radius


def fit_cylinder(protrusion_weight: float) -> float:
    """Fitted radius in mm for the brief's cylinder, centre held at the axis midpoint."""
    points = cylinder_points(CYLINDER_RADIUS_MM, CYLINDER_HALF_EXTENT_MM, CYLINDER_SAMPLES)
    _, radii, _ = optimize_spheres(
        np.zeros((1, 3)),
        np.array([CYLINDER_RADIUS_MM]),
        points,
        max_offset=0.0,
        min_radius=1e-6,
        max_inner=200,
        max_outer=60,
        tol=1e-16,
        protrusion_weight=protrusion_weight,
        assign_radius_scale=np.inf,
    )
    return float(radii[0])


def load_template() -> tuple[np.ndarray, np.ndarray, list[tuple[int, ...]]]:
    payload = json.loads(spheremesh_template_path(MALE).read_text())
    centers = np.array([s["center"] for s in payload["spheres"]], dtype=np.float64)
    radii = np.array([s["radius"] for s in payload["spheres"]], dtype=np.float64)
    connections = [tuple(int(i) for i in c["indices"]) for c in payload["connections"]]
    return centers, radii, connections


def neutral_band_points(centers, radii, connections) -> np.ndarray:
    model = load_smpl_model(MALE)
    verts, _, _, _ = my_function_forward(
        model, tc.zeros(10, dtype=tc.float64), tc.zeros(72, dtype=tc.float64)
    )
    vertices = verts.detach().cpu().numpy().astype(np.float64)
    return vertices[select_band(vertices, centers, radii, connections, band_max=0.05)]


def phase1_bit_identity(baseline_path: Path, centers, radii, points, connections) -> None:
    print("\nPhase 1 -- kappa = 1 reproduces the symmetric-loss implementation")
    if not baseline_path.is_file():
        check(False, "baseline snapshot present", f"missing {baseline_path}")
        return
    baseline = np.load(baseline_path)
    new_centers, new_radii, info = optimize_spheres(
        centers,
        radii,
        points,
        max_offset=0.005,
        connections=connections,
        protrusion_weight=1.0,
        assignment="signed",
        assign_radius_scale=np.inf,
        reassign_passes=10 ** 6,
    )
    check(
        np.array_equal(new_centers.view(np.uint8), baseline["centers"].view(np.uint8)),
        "centres bit-identical to the symmetric-loss baseline",
    )
    check(
        np.array_equal(new_radii.view(np.uint8), baseline["radii"].view(np.uint8)),
        "radii bit-identical to the symmetric-loss baseline",
    )
    check(
        np.array_equal(
            np.array([it["E"] for it in info["iterations"]]), baseline["energies"]
        ),
        "per-iteration E bit-identical to the symmetric-loss baseline",
    )


def phase1_cylinder_table() -> None:
    print("\nPhase 1 -- synthetic cylinder reproduces the brief's table")
    print("    kappa   fitted r    protrusion   brief    error")
    previous_radius = None
    monotone = True
    for protrusion_weight, expected in BRIEF_TABLE.items():
        fitted = fit_cylinder(protrusion_weight)
        error = abs(fitted - expected)
        print(
            f"    {protrusion_weight:<7g} {fitted:<11.4f} "
            f"{fitted - CYLINDER_RADIUS_MM:<12.4f} {expected:<8.2f} {error:.4f}"
        )
        check(error < 0.02, f"kappa = {protrusion_weight:g} within 0.02 mm of the brief")
        if previous_radius is not None and fitted > previous_radius:
            monotone = False
        previous_radius = fitted
    check(monotone, "raising kappa shrinks the sphere (residual sign not inverted)")


def phase1_extreme_kappa() -> None:
    print("\nPhase 1 -- kappa far above the clamp")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fitted = fit_cylinder(1000.0)
    clamped = any(issubclass(w.category, RuntimeWarning) for w in caught)
    check(clamped, f"kappa = 1000 clamped to {MAX_PROTRUSION_WEIGHT:g} with a warning")
    check(
        abs(fitted - CYLINDER_RADIUS_MM) < 0.05,
        "clamped kappa drives r to the inscribed radius",
        f"r = {fitted:.4f} mm, protrusion {fitted - CYLINDER_RADIUS_MM:.4f} mm",
    )


def phase1_fixed_point_cross_check() -> None:
    print("\nPhase 1 -- fitted radius equals the iterated weighted mean")
    points = cylinder_points(CYLINDER_RADIUS_MM, CYLINDER_HALF_EXTENT_MM, CYLINDER_SAMPLES)
    distances = np.linalg.norm(points, axis=1)
    for protrusion_weight in (1.0, 5.0, 20.0):
        fitted = fit_cylinder(protrusion_weight)
        expected = weighted_mean_fixed_point(distances, protrusion_weight)
        check(
            abs(fitted - expected) < 1e-6,
            f"kappa = {protrusion_weight:g} matches the fixed point",
            f"{abs(fitted - expected):.3e} mm, solver stopping threshold ~1e-7 mm",
        )


def phase2_kdtree_equivalence(centers, radii, points) -> None:
    print("\nPhase 2 -- Euclidean partition agrees with a KD-tree and a naive loop")
    subset = points[:200]
    owner, _ = assign_partition(subset, centers, radii, assign_radius_scale=np.inf)
    _, tree_owner = cKDTree(centers).query(subset)
    naive = np.array(
        [int(np.argmin([np.linalg.norm(p - c) for c in centers])) for p in subset]
    )
    check(np.array_equal(owner, tree_owner), "dense partition equals cKDTree")
    check(np.array_equal(owner, naive), "dense partition equals the naive loop")


def phase2_partition_effects(centers, radii, points, connections) -> tuple[dict, dict]:
    print("\nPhase 2 -- partition, cap and freezing on the male template")
    _, _, legacy = optimize_spheres(
        centers, radii, points, max_offset=0.005, connections=connections,
        protrusion_weight=1.0, assignment="signed", assign_radius_scale=np.inf,
        reassign_passes=10 ** 6,
    )
    _, _, current = optimize_spheres(
        centers, radii, points, max_offset=0.005, connections=connections,
    )

    legacy_spread = legacy["per_sphere"]["axial_spread"] * 1000.0
    current_spread = current["per_sphere"]["axial_spread"] * 1000.0
    print(
        f"    assigned-set diameter  median {np.median(legacy_spread):.1f} -> "
        f"{np.median(current_spread):.1f} mm, mean {legacy_spread.mean():.1f} -> "
        f"{current_spread.mean():.1f} mm"
    )
    print(
        "    (this measure is the set's full diameter, so on a limb it is dominated by the"
    )
    print(
        "     circumferential component, which the partition cannot reduce; see the notes)"
    )
    print(
        f"    negative fraction  {legacy['negative_fraction_after']:.3f} -> "
        f"{current['negative_fraction_after']:.3f}"
    )
    check(
        current_spread.mean() <= legacy_spread.mean(),
        "assigned-set diameter does not grow versus the symmetric-loss baseline",
    )
    check(
        current["negative_fraction_after"] < legacy["negative_fraction_after"],
        "negative-residual fraction improves",
    )
    energies = [it["E"] for it in current["iterations"]]
    check(
        all(energies[i] <= energies[i - 1] for i in range(1, len(energies))),
        "E monotone over the fixed point set",
        f"{len(energies)} iterations",
    )
    check(not current["rolled_back"], "no rollback needed with a frozen partition")
    print(
        f"    end-of-run reassignment diagnostic: "
        f"{current['would_reassign_fraction'] * 100:.2f}% total, of which "
        f"{current['would_reassign_owner_fraction'] * 100:.2f}% is partition drift and "
        f"{current['would_recap_fraction'] * 100:.2f}% is cap-membership churn"
    )
    check(
        current["would_reassign_owner_fraction"] < 0.05,
        "partition drift small enough for freezing to be defensible",
        f"{current['would_reassign_owner_fraction'] * 100:.2f}% of commonly-assigned points",
    )
    print(
        f"    unassigned by the {current['assign_radius_scale']:g}*r cap: "
        f"{current['unassigned']} of {current['num_points']} points"
    )
    return current, legacy


def phase2_reproduces_phase1(centers, radii, points, connections) -> None:
    print("\nPhase 2 -- inactive cap with reassignment reproduces the Phase 1 setting")
    _, _, capped = optimize_spheres(
        centers, radii, points, max_offset=0.005, connections=connections,
        assign_radius_scale=np.inf, reassign_passes=10 ** 6,
    )
    check(
        capped["unassigned"] == 0,
        "an infinite cap discards nothing",
        f"{capped['unassigned']} unassigned",
    )
    energies = [it["E"] for it in capped["iterations"]]
    check(
        all(energies[i] <= energies[i - 1] for i in range(1, len(energies))),
        "E monotone with reassignment active",
    )


def phase2_unassigned_not_zero(centers, radii, points) -> None:
    print("\nPhase 2 -- discarded points are excluded, not counted as zero residual")
    owner, uncapped = assign_partition(points, centers, radii, assign_radius_scale=1.75)
    check(np.any(owner < 0), "the default cap discards something", f"{int(np.count_nonzero(owner < 0))} points")
    check(np.all(uncapped >= 0), "the uncapped partition assigns every point")
    _, _, info = optimize_spheres(centers, radii, points, max_offset=0.0)
    counts = info["per_sphere"]["count"]
    lost = info["per_sphere"]["lost_to_cap"]
    check(
        int(counts.sum()) == points.shape[0] - info["unassigned"],
        "assigned counts and unassigned count reconcile with the point total",
    )
    check(int(lost.sum()) == info["unassigned"], "lost-to-cap reconciles with unassigned")


def worst_spheres_table(info: dict, label: str, limit: int = 10) -> None:
    print(f"\n{label}: ten spheres with the highest negative-residual fraction")
    per_sphere = info["per_sphere"]
    negative = np.nan_to_num(per_sphere["negative_fraction"], nan=-1.0)
    order = np.argsort(negative)[::-1][:limit]
    print("    idx  count  neg-frac  median f   min f     max f     spread")
    for index in order:
        print(
            f"    {index:<4} {per_sphere['count'][index]:<6} "
            f"{per_sphere['negative_fraction'][index]:<9.3f} "
            f"{per_sphere['median_residual'][index] * 1000:<10.2f} "
            f"{per_sphere['min_residual'][index] * 1000:<9.2f} "
            f"{per_sphere['max_residual'][index] * 1000:<9.2f} "
            f"{per_sphere['axial_spread'][index] * 1000:.1f}"
        )


def main() -> int:
    baseline_path = Path(
        "C:/Users/Mathias/AppData/Local/Temp/claude/"
        "c--Users-Mathias-Documents-GitHub-SMPL-SphereMesh-Tool/"
        "e902d0e8-cc6d-4bee-9f10-46ea3d9a9961/scratchpad/phase0_baseline.npz"
    )
    centers, radii, connections = load_template()
    points = neutral_band_points(centers, radii, connections)
    print(f"Template: {len(centers)} spheres, band: {len(points)} vertices")

    phase1_bit_identity(baseline_path, centers, radii, points, connections)
    phase1_cylinder_table()
    phase1_extreme_kappa()
    phase1_fixed_point_cross_check()
    phase2_kdtree_equivalence(centers, radii, points)
    current, legacy = phase2_partition_effects(centers, radii, points, connections)
    phase2_reproduces_phase1(centers, radii, points, connections)
    phase2_unassigned_not_zero(centers, radii, points)
    worst_spheres_table(legacy, "Phase 0 baseline (symmetric loss, signed partition)")
    worst_spheres_table(current, "After the fix (kappa=10, Euclidean partition, capped)")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): " + ", ".join(FAILURES))
        return 1
    print("All inflation-fix acceptance checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
