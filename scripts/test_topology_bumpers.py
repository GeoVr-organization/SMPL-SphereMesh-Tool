"""Checks for ``src/topology/bumpers.py``.

The distance functions there are closed forms for
``min over the simplex of |p - c(lambda)| - r(lambda)``. Every closed form is checked
against a brute-force minimisation of that same expression on a dense lambda grid, which is
slow but obviously correct, so a sign slip or a mis-set branch cannot hide behind a
plausible-looking number. The graph checks are structural.

Run:  .venv/Scripts/python.exe scripts/test_topology_bumpers.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.topology.bumpers import (  # noqa: E402
    CONE,
    PRYSMOID,
    QUAD,
    SPHERE,
    Bumper,
    BumperGraph,
    prysmoid_distances,
    round_cone_distances,
    sphere_distances,
    tangent_plane,
)

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  ok    {message}")
    else:
        print(f"  FAIL  {message}")
        FAILURES.append(message)


def brute_force_segment(
    points: np.ndarray, start_center, start_radius, end_center, end_radius, samples=200_001
) -> np.ndarray:
    """min over t of |p - lerp(c)| - lerp(r), by exhaustive sampling."""
    parameter = np.linspace(0.0, 1.0, samples)[:, None]
    center = start_center[None, :] + parameter * (end_center - start_center)[None, :]
    radius = start_radius + parameter[:, 0] * (end_radius - start_radius)
    best = np.empty(len(points))
    for index, point in enumerate(points):
        best[index] = np.min(np.linalg.norm(center - point, axis=1) - radius)
    return best


def brute_force_triangle(points: np.ndarray, centers, radii, steps=900) -> np.ndarray:
    """min over the barycentric triangle, by exhaustive sampling of lambda."""
    weights = []
    for first in range(steps + 1):
        for second in range(steps + 1 - first):
            weights.append((first / steps, second / steps, 1.0 - (first + second) / steps))
    weight = np.array(weights)
    center = weight @ centers
    radius = weight @ radii
    best = np.empty(len(points))
    for index, point in enumerate(points):
        best[index] = np.min(np.linalg.norm(center - point, axis=1) - radius)
    return best


def random_points(count: int, scale: float, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return generator.uniform(-scale, scale, size=(count, 3))


# ------------------------------------------------------------------ sphere and cone

print("sphere")
centers = np.array([[0.0, 0.0, 0.0], [0.3, -0.1, 0.05]])
radii = np.array([0.1, 0.05])
points = random_points(200, 0.5, seed=1)
expected = np.stack(
    [np.linalg.norm(points - centers[i], axis=1) - radii[i] for i in range(2)], axis=1
)
check(np.allclose(sphere_distances(points, centers, radii), expected), "matches |p-c|-r")

print("round cone")
CONE_CASES = [
    ("equal radii (capsule)", np.array([0.0, 0.0, 0.0]), 0.08, np.array([0.25, 0.0, 0.0]), 0.08),
    ("mild taper", np.array([0.0, 0.0, 0.0]), 0.09, np.array([0.20, 0.03, 0.0]), 0.05),
    ("strong taper", np.array([0.0, 0.0, 0.0]), 0.12, np.array([0.22, 0.0, 0.0]), 0.02),
    ("reversed taper", np.array([0.0, 0.0, 0.0]), 0.02, np.array([0.22, 0.0, 0.0]), 0.12),
    ("near-swallowed", np.array([0.0, 0.0, 0.0]), 0.10, np.array([0.11, 0.0, 0.0]), 0.02),
]
points = random_points(300, 0.35, seed=2)
for label, start_center, start_radius, end_center, end_radius in CONE_CASES:
    closed_form = round_cone_distances(
        points,
        start_center[None, :], np.array([start_radius]),
        end_center[None, :], np.array([end_radius]),
    )[:, 0]
    reference = brute_force_segment(points, start_center, start_radius, end_center, end_radius)
    worst = np.abs(closed_form - reference).max()
    check(worst < 2e-6, f"{label}: max |closed form - brute force| = {worst:.2e} m")

interior = np.array([[0.125, 0.0, 0.0], [0.125, 0.04, 0.0], [0.02, 0.0, 0.0]])
inside_capsule = round_cone_distances(
    interior, np.array([[0.0, 0.0, 0.0]]), np.array([0.08]),
    np.array([[0.25, 0.0, 0.0]]), np.array([0.08]),
)[:, 0]
check(
    np.allclose(inside_capsule, [-0.08, -0.04, -0.08], atol=1e-9),
    "interior points get clearance to the nearest surface, not to the farthest",
)

swallowed = round_cone_distances(
    points, np.array([[0.0, 0.0, 0.0]]), np.array([0.10]),
    np.array([[0.01, 0.0, 0.0]]), np.array([0.02]),
)[:, 0]
check(
    np.allclose(swallowed, np.linalg.norm(points, axis=1) - 0.10, atol=1e-12),
    "a swallowed sphere falls back to the enclosing sphere",
)

# ------------------------------------------------------------------------- prysmoid

print("prysmoid")
TRIANGLE_CASES = [
    (
        "equal radii, near-equilateral",
        np.array([[0.0, 0.0, 0.0], [0.20, 0.0, 0.0], [0.10, 0.18, 0.0]]),
        np.array([0.06, 0.06, 0.06]),
    ),
    (
        "graded radii",
        np.array([[0.0, 0.0, 0.0], [0.22, 0.01, 0.0], [0.09, 0.19, 0.03]]),
        np.array([0.09, 0.05, 0.07]),
    ),
    (
        "steep radius gradient, tangent plane still exists",
        np.array([[0.0, 0.0, 0.0], [0.30, 0.0, 0.0], [0.12, 0.26, 0.0]]),
        np.array([0.12, 0.02, 0.07]),
    ),
    (
        "sliver triangle",
        np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.13, 0.012, 0.0]]),
        np.array([0.05, 0.045, 0.05]),
    ),
]
points = random_points(150, 0.32, seed=3)
for label, triangle_centers, triangle_radii in TRIANGLE_CASES:
    closed_form = prysmoid_distances(points, triangle_centers[None], triangle_radii[None])[:, 0]
    reference = brute_force_triangle(points, triangle_centers, triangle_radii)
    worst = np.abs(closed_form - reference).max()
    check(worst < 5e-6, f"{label}: max |closed form - brute force| = {worst:.2e} m")

flat_centers = np.array([[0.0, 0.0, 0.0], [0.30, 0.0, 0.0], [0.15, 0.26, 0.0]])
flat_radii = np.array([0.05, 0.05, 0.05])
on_axis = np.array([[0.15, 0.09, 0.0], [0.15, 0.09, 0.025], [0.15, 0.09, 0.05],
                    [0.15, 0.09, 0.09]])
slab = prysmoid_distances(on_axis, flat_centers[None], flat_radii[None])[:, 0]
check(
    np.allclose(slab, [-0.05, -0.025, 0.0, 0.04], atol=1e-9),
    "the slab covers the triangle's interior: centre reads -r, face reads 0",
)

edges_only = np.full(len(on_axis), np.inf)
for first, second in ((0, 1), (1, 2), (2, 0)):
    edges_only = np.minimum(
        edges_only,
        round_cone_distances(
            on_axis,
            flat_centers[None, first], flat_radii[None, first],
            flat_centers[None, second], flat_radii[None, second],
        )[:, 0],
    )
check(
    edges_only[0] > 0.03 and slab[0] < 0.0,
    f"edges alone would call the triangle's centre {edges_only[0] * 1000:.0f} mm outside "
    f"where the slab has it {slab[0] * 1000:.0f} mm inside",
)

_, _, _, exists = tangent_plane(
    np.array([[[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.02, 0.04, 0.0]]]),
    np.array([[0.10, 0.01, 0.05]]),
)
check(not bool(exists[0]), "no tangent plane when radii vary faster than the spacing")

_, _, _, collinear = tangent_plane(
    np.array([[[0.0, 0.0, 0.0], [0.10, 0.0, 0.0], [0.20, 0.0, 0.0]]]),
    np.array([[0.05, 0.05, 0.05]]),
)
check(not bool(collinear[0]), "no tangent plane for collinear centres")

# ---------------------------------------------------------------------------- graph

print("bumper graph")
graph = BumperGraph(5, [(0, 1), (1, 2, 3), (2, 3, 4)])
counts = graph.counts()
check(
    counts == {SPHERE: 5, CONE: 6, PRYSMOID: 2, QUAD: 0},
    f"primitive tally {counts}: 6 distinct cones from one double and two triples "
    "sharing edge 2-3",
)
check(len(graph.components()) == 1, "the example is one connected component")

first_prysmoid = graph.index_of(PRYSMOID, (1, 2, 3))
second_prysmoid = graph.index_of(PRYSMOID, (2, 3, 4))
check(
    second_prysmoid in graph.adjacency[first_prysmoid],
    "slabs sharing rim edge 2-3 are adjacent (neibSide)",
)
check(
    graph.index_of(CONE, (2, 3)) in graph.adjacency[first_prysmoid],
    "a slab is adjacent to its own rim cones",
)
check(
    graph.index_of(SPHERE, (1,)) not in graph.adjacency[first_prysmoid],
    "a sphere is NOT directly adjacent to a slab: the walk goes through the cone",
)
check(
    graph.index_of(CONE, (0, 1)) in graph.adjacency[graph.index_of(SPHERE, (0,))],
    "a sphere is adjacent to every cone incident to it",
)

split = BumperGraph(6, [(0, 1), (2, 3), (4, 5)])
check(len(split.components()) == 3, "three disjoint cones give three components")

quad_graph = BumperGraph(4, [(0, 1, 2, 3)])
quad = Bumper(QUAD, (0, 1, 2, 3))
check(
    set(quad.edges()) == {(0, 1), (1, 2), (2, 3), (0, 3)},
    "a quad's rim is its perimeter; the 0-2 diagonal is not a rim edge",
)
check(
    quad_graph.counts()[CONE] == 4,
    "a quad generates 4 rim cones, not the 6 pairs of its spheres",
)
check(
    quad.triangles() == ((0, 1, 2), (0, 2, 3)),
    "a quad splits into two triangles across the 0-2 diagonal for distance",
)

# A quad and two triangles covering the same four spheres must agree geometrically.
quad_centers = np.array([[0.0, 0.0, 0.0], [0.20, 0.0, 0.0], [0.20, 0.18, 0.0], [0.0, 0.18, 0.0]])
quad_radii = np.array([0.05, 0.045, 0.055, 0.05])
probe = random_points(120, 0.3, seed=4)
as_quad = quad_graph.distances(probe, quad_centers, quad_radii)[
    :, quad_graph.index_of(QUAD, (0, 1, 2, 3))
]
as_pair = np.minimum(
    prysmoid_distances(probe, quad_centers[None, [0, 1, 2]], quad_radii[None, [0, 1, 2]])[:, 0],
    prysmoid_distances(probe, quad_centers[None, [0, 2, 3]], quad_radii[None, [0, 2, 3]])[:, 0],
)
check(
    np.allclose(as_quad, as_pair, atol=1e-12),
    "a quad and its two-triangle split give identical distances",
)

far_apart = (graph.index_of(SPHERE, (0,)), graph.index_of(SPHERE, (4,)))
check(
    np.array_equal(graph.reachable_within(1), graph.adjacency_matrix()),
    "reachable_within(1) is the adjacency matrix itself",
)
hops = next(n for n in range(1, 20) if graph.reachable_within(n)[far_apart])
check(hops == 6, f"spheres 0 and 4 are {hops} walk steps apart (S-C-S-C-S-C-S)")
check(graph.reachable_within(hops).all(), "at that radius the whole graph is reachable")

# ---------------------------------------------------------------------------- owners

print("owners")
owner_graph = BumperGraph(3, [(0, 1), (1, 2)])
owner_centers = np.array([[0.0, 0.0, 0.0], [0.30, 0.0, 0.0], [0.60, 0.0, 0.0]])
owner_radii = np.array([0.05, 0.05, 0.05])
probe = np.array([[0.0, 0.0, 0.20], [0.15, 0.0, 0.06], [0.45, 0.0, 0.06], [-0.20, 0.0, 0.0]])
owner, distance = owner_graph.owners(probe, owner_centers, owner_radii)
named = [str(owner_graph.bumpers[i]) for i in owner]
check(
    named[0] == "sphere[0]" and named[3] == "sphere[0]",
    f"a point over a sphere's cap is owned by the sphere, not the cone that ties there "
    f"(got {named[0]}, {named[3]})",
)
check(
    named[1] == "cone[0, 1]" and named[2] == "cone[1, 2]",
    f"points over the lateral band of each cone are owned by that cone (got {named[1]}, {named[2]})",
)
cap_distances = owner_graph.distances(probe[:1], owner_centers, owner_radii)[0]
check(
    abs(cap_distances[owner_graph.index_of(SPHERE, (0,))]
        - cap_distances[owner_graph.index_of(CONE, (0, 1))]) < 1e-12,
    "the cap is an exact tie between sphere and cone, which is why the tie-break matters",
)

slab_graph = BumperGraph(3, [(0, 1, 2)])
slab_centers = np.array([[0.0, 0.0, 0.0], [0.30, 0.0, 0.0], [0.15, 0.26, 0.0]])
slab_radii = np.array([0.05, 0.05, 0.05])
slab_probe = np.array([[0.15, 0.09, 0.09], [0.15, -0.09, 0.0], [-0.12, -0.05, 0.0]])
slab_owner, _ = slab_graph.owners(slab_probe, slab_centers, slab_radii)
slab_named = [str(slab_graph.bumpers[i]) for i in slab_owner]
check(
    slab_named == ["prysmoid[0, 1, 2]", "cone[0, 1]", "sphere[0]"],
    f"over the face the slab owns, off an edge the cone owns, past a corner the sphere "
    f"owns (got {slab_named})",
)

check(np.all(distance > 0), "those probe points are outside the sphere-mesh")

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED:")
    for message in FAILURES:
        print(f"  - {message}")
    raise SystemExit(1)
print("all checks passed")
