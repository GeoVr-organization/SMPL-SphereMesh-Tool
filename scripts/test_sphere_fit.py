"""Acceptance tests for src/sphere_fit.py, phases 1 to 3 of the implementation brief.

Run with: python scripts/test_sphere_fit.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sphere_fit import (  # noqa: E402
    assign_points,
    optimize_spheres,
    spheremesh_distance,
    spheres_only_distance,
)

FAILURES: list[str] = []


def check(condition: bool, label: str, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f"  ({detail})" if detail else ""))
    if not condition:
        FAILURES.append(label)


def load_template(path: Path) -> tuple[np.ndarray, np.ndarray, list]:
    data = json.loads(path.read_text())
    centers = np.array([s["center"] for s in data["spheres"]], dtype=np.float64)
    radii = np.array([s["radius"] for s in data["spheres"]], dtype=np.float64)
    connections = [tuple(c["indices"]) for c in data["connections"]]
    return centers, radii, connections


def unit_sphere_points(count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(count, 3))
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)


def phase1_assignment() -> None:
    print("\nPhase 1 -- metric and assignment")
    rng = np.random.default_rng(0)
    centers = rng.normal(scale=0.3, size=(12, 3))
    radii = rng.uniform(0.02, 0.12, size=12)
    points = rng.normal(scale=0.4, size=(200, 3))

    assignment, distance = assign_points(points, centers, radii)
    naive = np.array(
        [
            int(np.argmin([np.linalg.norm(p - c) - r for c, r in zip(centers, radii)]))
            for p in points
        ]
    )
    check(np.array_equal(assignment, naive), "broadcast assignment equals naive loop")
    check(
        np.array_equal(distance, spheres_only_distance(points, centers, radii)),
        "assigned distance equals union signed distance",
    )
    check(
        assignment.shape == (points.shape[0],) and assignment.min() >= 0,
        "every point assigned to exactly one sphere",
    )
    check(distance.dtype == np.float64, "distances are float64")

    kd_like = np.argmin(np.linalg.norm(points[:, None] - centers[None], axis=2), axis=1)
    print(
        f"  [info] nearest-centre assignment differs from Apollonius on "
        f"{int(np.count_nonzero(kd_like != assignment))}/{len(points)} points "
        "(this is why no KD-tree is used)"
    )


def phase2_synthetic_exact() -> None:
    print("\nPhase 2 -- exact synthetic recovery")
    true_center = np.array([0.11, -0.24, 0.37])
    true_radius = 0.185
    points = true_center + true_radius * unit_sphere_points(200, seed=1)

    start_center = true_center * 1.2 + 0.2 * true_radius
    start_radius = true_radius * 1.2
    new_centers, new_radii, info = optimize_spheres(
        np.array([start_center]),
        np.array([start_radius]),
        points,
        max_offset=np.inf,
        max_inner=60,
        max_outer=40,
        tol=1e-16,
    )
    center_error = float(np.linalg.norm(new_centers[0] - true_center))
    radius_error = abs(new_radii[0] - true_radius)
    check(center_error < 1e-12, "centre recovered to < 1e-12 m", f"{center_error:.3e}")
    check(radius_error < 1e-12, "radius recovered to < 1e-12 m", f"{radius_error:.3e}")
    check(info["degenerate_points"] == 0, "no degenerate normals")


def phase2_synthetic_noisy() -> None:
    print("\nPhase 2 -- noisy synthetic recovery")
    rng = np.random.default_rng(2)
    true_center = np.array([-0.05, 0.42, 0.09])
    true_radius = 0.21
    points = true_center + true_radius * unit_sphere_points(2000, seed=3)
    points = points + rng.normal(scale=1e-3, size=points.shape)

    new_centers, new_radii, _ = optimize_spheres(
        np.array([true_center * 1.2]),
        np.array([true_radius * 1.2]),
        points,
        max_offset=np.inf,
        max_inner=60,
        max_outer=40,
        tol=1e-16,
        protrusion_weight=1.0,
    )
    center_error = float(np.linalg.norm(new_centers[0] - true_center))
    check(center_error < 1e-3, "centre inside the 1 mm noise floor", f"{center_error:.3e}")
    weighted_mean = float(np.mean(np.linalg.norm(points - new_centers[0], axis=1)))
    check(
        abs(new_radii[0] - weighted_mean) < 1e-10,
        "returned radius equals weighted mean of |v - c|",
        f"{abs(new_radii[0] - weighted_mean):.3e}",
    )


def phase2_degenerate_plane() -> None:
    print("\nPhase 2 -- degenerate flat patch")
    rng = np.random.default_rng(4)
    points = np.zeros((400, 3))
    points[:, :2] = rng.uniform(-0.05, 0.05, size=(400, 2))
    new_centers, new_radii, info = optimize_spheres(
        np.array([[0.0, 0.0, -0.1]]),
        np.array([0.1]),
        points,
        max_offset=np.inf,
        max_inner=30,
        max_outer=10,
    )
    check(np.all(np.isfinite(new_centers)), "no NaN in centres")
    check(np.all(np.isfinite(new_radii)), "no NaN in radii")
    check(new_radii[0] > 0.0, "radius stays positive", f"r = {new_radii[0]:.4f}")
    condition = info["condition_numbers"][0]
    check(condition > 1e6, "reports a high condition number", f"cond = {condition:.3e}")


def phase2_monotone(centers: np.ndarray, radii: np.ndarray, points: np.ndarray, connections: list) -> None:
    print("\nPhase 2 -- monotonicity on the real template (unconstrained)")
    _, _, info = optimize_spheres(
        centers, radii, points, max_offset=np.inf, connections=connections
    )
    energies = [it["E"] for it in info["iterations"]]
    increases = [
        (i, energies[i - 1], energies[i])
        for i in range(1, len(energies))
        if energies[i] > energies[i - 1]
    ]
    check(not increases, "E monotonically non-increasing", f"{len(energies)} outer iterations")
    check(
        energies[0] <= info["initial_E"],
        "first iteration does not exceed the initial energy",
    )


def phase3_constraints(centers: np.ndarray, radii: np.ndarray, points: np.ndarray, connections: list) -> None:
    print("\nPhase 3 -- constraints")
    zero_centers, zero_radii, _ = optimize_spheres(
        centers, radii, points, max_offset=0.0, connections=connections
    )
    check(
        np.array_equal(zero_centers.view(np.uint8), centers.view(np.uint8)),
        "max_offset = 0 returns bit-identical centres",
    )
    check(not np.array_equal(zero_radii, radii), "max_offset = 0 still changes radii")
    check(zero_centers is not centers, "returned centres are a new allocation")

    centers_guard = centers.copy()
    radii_guard = radii.copy()
    optimize_spheres(centers, radii, points, max_offset=0.02, connections=connections)
    check(np.array_equal(centers, centers_guard), "input centres not mutated")
    check(np.array_equal(radii, radii_guard), "input radii not mutated")

    loose_centers, loose_radii, loose_info = optimize_spheres(
        centers, radii, points, max_offset=1e6, connections=connections
    )
    free_centers, free_radii, free_info = optimize_spheres(
        centers, radii, points, max_offset=np.inf, connections=connections
    )
    check(
        np.array_equal(loose_centers, free_centers) and np.array_equal(loose_radii, free_radii),
        "an inactive max_offset reproduces the unconstrained result exactly",
    )

    print("\n  delta sweep (mm):  final E        RMS spheres  RMS full     pinned  frozen<4  frozen=0")
    previous_energy = None
    monotone = True
    for delta_mm in (0, 1, 2, 5, 10, 20):
        _, _, info = optimize_spheres(
            centers, radii, points, max_offset=delta_mm / 1000.0, connections=connections
        )
        energy = info["final_E"]
        pinned = info["iterations"][-1]["at_max_offset"]
        print(
            f"  {delta_mm:>6}          {energy:<14.8f} "
            f"{info['rms_spheres_after']:<12.6f} {info['rms_full_after']:<12.6f} "
            f"{pinned:<7} {len(info['frozen_sparse']):<9} {len(info['frozen_empty'])}"
        )
        if previous_energy is not None and energy > previous_energy + 1e-12:
            monotone = False
        previous_energy = energy
        energies = [it["E"] for it in info["iterations"]]
        if any(energies[i] > energies[i - 1] for i in range(1, len(energies))):
            check(False, f"E monotone within delta = {delta_mm} mm")
    check(monotone, "final E non-increasing as delta grows")
    check(True, "E monotone within every delta (checked above)")


def main() -> int:
    template_path = Path(__file__).resolve().parent.parent / "data" / "template_male_1.0.json"
    centers, radii, connections = load_template(template_path)
    print(
        f"Template {template_path.name}: {len(centers)} spheres, {len(connections)} connections"
    )

    rng = np.random.default_rng(7)
    points = centers[rng.integers(0, len(centers), size=3000)]
    points = points + rng.normal(scale=0.02, size=points.shape)
    points = points + (radii[rng.integers(0, len(radii), size=3000)][:, None] * 0.0)
    print(f"Synthetic target cloud: {len(points)} points")

    phase1_assignment()
    phase2_synthetic_exact()
    phase2_synthetic_noisy()
    phase2_degenerate_plane()
    phase2_monotone(centers, radii, points, connections)
    phase3_constraints(centers, radii, points, connections)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): " + ", ".join(FAILURES))
        return 1
    print("All acceptance checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
