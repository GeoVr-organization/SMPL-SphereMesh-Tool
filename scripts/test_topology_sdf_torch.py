"""Checks for ``src/topology/sdf_torch.py``.

The torch port is a re-implementation of numerics that are already checked against brute
force in ``scripts/test_topology_bumpers.py``, so the question here is a different one: do
the two implementations agree to floating-point noise, column by column, on every branch
the numpy code has -- and are the gradients the port exists to provide actually right.
Agreement is checked on synthetic configurations that each exercise one branch and on the
real template; the gradients are checked with ``torch.autograd.gradcheck`` on a small case
(the ownership-tie filter there hides the cap, swallowed-cone and no-plane branches, so
those get their own single-column gradchecks) and with central differences on the template.
The memory figure a fitter sizes batches by is *measured* -- bytes autograd keeps for
backward, and the process working set -- and checked against the analytic single-tensor
size so the two can never be confused again.

Run:  .venv/Scripts/python.exe scripts/test_topology_sdf_torch.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.topology import anatomy  # noqa: E402
from src.topology.bumpers import (  # noqa: E402
    CONE,
    PRYSMOID,
    QUAD,
    SPHERE,
    BumperGraph,
    round_cone_distances,
    tangent_plane,
)
from src.topology.sdf_torch import TorchSdf, connections_from_payload  # noqa: E402
from src.topology.validate import modelled_vertex_mask  # noqa: E402

FAILURES: list[str] = []
REPO = Path(__file__).resolve().parent.parent


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  ok    {message}")
    else:
        print(f"  FAIL  {message}")
        FAILURES.append(message)


def random_points(count: int, scale: float, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return generator.uniform(-scale, scale, size=(count, 3))


def as_tensor(array: np.ndarray, *, requires_grad: bool = False) -> torch.Tensor:
    tensor = torch.as_tensor(np.ascontiguousarray(array, dtype=np.float64))
    return tensor.clone().requires_grad_(True) if requires_grad else tensor


def max_abs_difference(
    graph: BumperGraph, points: np.ndarray, centers: np.ndarray, radii: np.ndarray
) -> float:
    reference = graph.distances(points, centers, radii)
    sdf = TorchSdf(graph)
    candidate = sdf.distances(
        as_tensor(points), as_tensor(centers, requires_grad=True), as_tensor(radii, requires_grad=True)
    )
    check(candidate.requires_grad, "torch distances carry a gradient")
    check(tuple(candidate.shape) == reference.shape, f"shape {tuple(candidate.shape)} == {reference.shape}")
    return float(np.max(np.abs(candidate.detach().numpy() - reference)))


# ------------------------------------------------------------- 1. synthetic agreement

print("synthetic configurations against numpy")

SYNTHETIC = {
    "capsule (equal radii)": (
        np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]]),
        np.array([0.1, 0.1]),
        [(0, 1)],
    ),
    "tapered cone": (
        np.array([[0.0, 0.0, 0.0], [0.3, 0.05, -0.02]]),
        np.array([0.15, 0.06]),
        [(0, 1)],
    ),
    "reversed taper": (
        np.array([[0.0, 0.0, 0.0], [0.3, 0.05, -0.02]]),
        np.array([0.06, 0.15]),
        [(0, 1)],
    ),
    "swallowed cone": (
        np.array([[0.0, 0.0, 0.0], [0.03, 0.01, 0.0]]),
        np.array([0.2, 0.05]),
        [(0, 1)],
    ),
    "flat equilateral prysmoid": (
        np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.15, 0.3 * np.sqrt(0.75), 0.0]]),
        np.array([0.08, 0.08, 0.08]),
        [(0, 1, 2)],
    ),
    "prysmoid without tangent plane": (
        np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.05, 0.1, 0.0]]),
        np.array([0.05, 0.14, 0.15]),
        [(0, 1, 2)],
    ),
    "quad": (
        np.array([[0.0, 0.0, 0.0], [0.3, 0.02, 0.0], [0.32, 0.3, 0.05], [-0.02, 0.28, -0.03]]),
        np.array([0.08, 0.1, 0.07, 0.09]),
        [(0, 1, 2, 3)],
    ),
}

for configuration_index, (name, (centers, radii, connections)) in enumerate(SYNTHETIC.items()):
    graph = BumperGraph(len(centers), connections)
    points = random_points(400, 0.6, seed=100 + configuration_index)
    difference = max_abs_difference(graph, points, centers, radii)
    check(difference < 1e-10, f"{name}: max |torch - numpy| = {difference:.2e} < 1e-10")

no_plane_centers, no_plane_radii, _ = SYNTHETIC["prysmoid without tangent plane"]
_, _, _, exists = tangent_plane(no_plane_centers[None], no_plane_radii[None])
check(not bool(exists[0]), "the 'without tangent plane' prysmoid really has exists=False")
swallowed_centers, swallowed_radii, _ = SYNTHETIC["swallowed cone"]
swallowed_gap = np.linalg.norm(swallowed_centers[1] - swallowed_centers[0])
check(
    swallowed_gap < abs(swallowed_radii[0] - swallowed_radii[1]),
    "the swallowed cone really is swallowed",
)

# ------------------------------------------------------------ 2. template agreement

print("real template against numpy")

payload = json.loads((REPO / "data/generated/auto_male.json").read_text(encoding="utf-8"))
template_centers = np.array([sphere["center"] for sphere in payload["spheres"]], dtype=np.float64)
template_radii = np.array([sphere["radius"] for sphere in payload["spheres"]], dtype=np.float64)
template_connections = connections_from_payload(payload)
quad_count = sum(1 for connection in template_connections if len(connection) == 4)
check(quad_count == 12, f"connections_from_payload rebuilt {quad_count} quads (expect 12)")
check(
    len(template_connections) == len(payload["connections"]) - quad_count,
    "each quad replaced exactly two split triangles",
)

template_graph = BumperGraph(len(template_centers), template_connections)
counts = template_graph.counts()
print(
    f"  columns: {counts[SPHERE]} sphere, {counts[CONE]} cone, "
    f"{counts[PRYSMOID]} prysmoid, {counts[QUAD]} quad ({len(template_graph.bumpers)} total)"
)
check(counts[QUAD] == 12, "graph has 12 quad columns")

body = anatomy.load_body("male")
sample_generator = np.random.default_rng(7)
sample_vertices = body.vertices[sample_generator.choice(len(body.vertices), 2000, replace=False)]
reference_template = template_graph.distances(sample_vertices, template_centers, template_radii)
template_sdf = TorchSdf(template_graph)
candidate_template = template_sdf.distances(
    as_tensor(sample_vertices),
    as_tensor(template_centers, requires_grad=True),
    as_tensor(template_radii, requires_grad=True),
).detach().numpy()
column_difference = np.abs(candidate_template - reference_template).max(axis=0)
for kind in (SPHERE, CONE, PRYSMOID, QUAD):
    columns = [index for index, bumper in enumerate(template_graph.bumpers) if bumper.kind == kind]
    worst = float(column_difference[columns].max())
    check(worst < 1e-9, f"{kind} columns: max |torch - numpy| = {worst:.2e} < 1e-9")

# ------------------------------------------------------------------- 3. gradcheck

print("gradcheck on a small configuration")

small_centers = np.array([
    [0.0, 0.0, 0.0],
    [0.25, 0.02, 0.0],
    [0.5, 0.0, 0.05],
    [0.7, 0.2, 0.0],
    [0.65, -0.15, 0.1],
])
small_radii = np.array([0.08, 0.06, 0.09, 0.05, 0.07])
small_graph = BumperGraph(5, [(0, 1), (1, 2), (2, 3, 4)])
small_sdf = TorchSdf(small_graph)

candidate_points = random_points(600, 0.9, seed=11) + np.array([0.3, 0.0, 0.0])
column_distances = small_graph.distances(candidate_points, small_centers, small_radii)
two_smallest = np.sort(column_distances, axis=1)[:, :2]
untied = (two_smallest[:, 1] - two_smallest[:, 0]) > 1e-4
gradcheck_points = as_tensor(candidate_points[untied][:30])
check(gradcheck_points.shape[0] == 30, f"found 30 untied points ({int(untied.sum())} candidates)")


def small_loss(centers: torch.Tensor, radii: torch.Tensor) -> torch.Tensor:
    return small_sdf.signed_distance(gradcheck_points, centers, radii).sum()


try:
    passed = torch.autograd.gradcheck(
        small_loss,
        (as_tensor(small_centers, requires_grad=True), as_tensor(small_radii, requires_grad=True)),
        eps=1e-6,
        atol=1e-5,
    )
except Exception as error:  # gradcheck raises with a description of the mismatch
    passed = False
    print(f"  gradcheck: {error}")
check(passed, "torch.autograd.gradcheck of sum of signed distance w.r.t. centres and radii")

# The tie filter above keeps only points whose owner is clear, and on this configuration
# those are lateral or sphere-owned points: the cap regions, the swallowed cone and the
# no-plane prysmoid never survive it. Gradcheck them one column at a time instead.


def column_gradcheck(
    graph: BumperGraph, centers: np.ndarray, radii: np.ndarray, points: np.ndarray, name: str
) -> None:
    sdf = TorchSdf(graph)
    column_points = as_tensor(points)

    def column_loss(grad_centers: torch.Tensor, grad_radii: torch.Tensor) -> torch.Tensor:
        return sdf.distances(column_points, grad_centers, grad_radii)[:, 0].sum()

    try:
        column_passed = torch.autograd.gradcheck(
            column_loss,
            (as_tensor(centers, requires_grad=True), as_tensor(radii, requires_grad=True)),
            eps=1e-6,
            atol=1e-5,
        )
    except Exception as error:
        column_passed = False
        print(f"  gradcheck: {error}")
    check(column_passed, f"column gradcheck: {name} ({len(points)} points)")


branch_generator = np.random.default_rng(23)
capsule_centers, capsule_radii, capsule_connections = SYNTHETIC["capsule (equal radii)"]
capsule_graph = BumperGraph(2, capsule_connections)
# Equal radii: the end cap is exactly the half-space beyond the end plane, the start cap
# exactly the half-space behind the start plane.
end_cap_points = np.column_stack([
    branch_generator.uniform(0.32, 0.6, 20),
    branch_generator.uniform(-0.2, 0.2, 20),
    branch_generator.uniform(-0.2, 0.2, 20),
])
start_cap_points = np.column_stack([
    branch_generator.uniform(-0.3, -0.02, 20),
    branch_generator.uniform(-0.2, 0.2, 20),
    branch_generator.uniform(-0.2, 0.2, 20),
])
column_gradcheck(capsule_graph, capsule_centers, capsule_radii, end_cap_points, "capsule end cap")
column_gradcheck(capsule_graph, capsule_centers, capsule_radii, start_cap_points, "capsule start cap")

taper_centers, taper_radii, taper_connections = SYNTHETIC["tapered cone"]
taper_graph = BumperGraph(2, taper_connections)
taper_axis = taper_centers[1] - taper_centers[0]
taper_axis /= np.linalg.norm(taper_axis)
# Well beyond the small end along the axis, inside a narrow tube: on the end cap.
taper_end_points = (
    taper_centers[1]
    + taper_axis * branch_generator.uniform(0.12, 0.3, (20, 1))
    + branch_generator.uniform(-0.03, 0.03, (20, 3))
)
# Well behind the big end: on the start cap.
taper_start_points = (
    taper_centers[0]
    - taper_axis * branch_generator.uniform(0.2, 0.4, (20, 1))
    + branch_generator.uniform(-0.05, 0.05, (20, 3))
)
column_gradcheck(taper_graph, taper_centers, taper_radii, taper_end_points, "tapered cone end cap")
column_gradcheck(taper_graph, taper_centers, taper_radii, taper_start_points, "tapered cone start cap")
reversed_centers, reversed_radii, reversed_connections = SYNTHETIC["reversed taper"]
column_gradcheck(
    BumperGraph(2, reversed_connections), reversed_centers, reversed_radii,
    taper_end_points, "reversed taper, points past the big end",
)

swallowed_points = random_points(20, 0.5, seed=29)
column_gradcheck(
    BumperGraph(2, SYNTHETIC["swallowed cone"][2]), swallowed_centers, swallowed_radii,
    swallowed_points, "swallowed cone (enclosing-sphere fallback)",
)
swallowed_reversed = swallowed_radii[::-1].copy()
column_gradcheck(
    BumperGraph(2, SYNTHETIC["swallowed cone"][2]), swallowed_centers, swallowed_reversed,
    swallowed_points, "swallowed cone with the end sphere enclosing",
)

# The no-plane prysmoid is the minimum of its three edge cones; drop points where two of
# those tie, which is the only place that minimum is legitimately non-differentiable.
no_plane_candidates = random_points(200, 0.4, seed=31)
edge_distances = np.stack([
    round_cone_distances(
        no_plane_candidates,
        no_plane_centers[[first]], no_plane_radii[[first]],
        no_plane_centers[[second]], no_plane_radii[[second]],
    )[:, 0]
    for first, second in ((0, 1), (1, 2), (2, 0))
], axis=1)
two_smallest_edges = np.sort(edge_distances, axis=1)[:, :2]
no_plane_untied = (two_smallest_edges[:, 1] - two_smallest_edges[:, 0]) > 1e-4
column_gradcheck(
    BumperGraph(3, SYNTHETIC["prysmoid without tangent plane"][2]),
    no_plane_centers, no_plane_radii, no_plane_candidates[no_plane_untied][:20],
    "prysmoid without tangent plane (exists=False)",
)

# ------------------------------------------------ 4. finite differences on the template

print("finite differences on the template")

difference_generator = np.random.default_rng(3)
fd_vertices = as_tensor(
    body.vertices[difference_generator.choice(len(body.vertices), 500, replace=False)]
)


def template_loss(centers: torch.Tensor, radii: torch.Tensor) -> torch.Tensor:
    return (template_sdf.signed_distance(fd_vertices, centers, radii) ** 2).sum()


fd_centers = as_tensor(template_centers, requires_grad=True)
fd_radii = as_tensor(template_radii, requires_grad=True)
template_loss(fd_centers, fd_radii).backward()
autograd_gradient = np.concatenate(
    [fd_centers.grad.numpy().reshape(-1), fd_radii.grad.numpy().reshape(-1)]
)
parameter_count = autograd_gradient.shape[0]



def loss_at(flat_parameters: np.ndarray) -> float:
    centre_count = len(template_radii)
    with torch.no_grad():
        return float(
            template_loss(
                as_tensor(flat_parameters[: 3 * centre_count].reshape(-1, 3)),
                as_tensor(flat_parameters[3 * centre_count:]),
            )
        )


flat_template = np.concatenate([template_centers.reshape(-1), template_radii])
step = 1e-6
worst_relative = 0.0
compared = 0
skipped = 0
for parameter in difference_generator.choice(parameter_count, 20, replace=False):
    plus = flat_template.copy()
    minus = flat_template.copy()
    plus[parameter] += step
    minus[parameter] -= step
    central = (loss_at(plus) - loss_at(minus)) / (2.0 * step)
    analytic = float(autograd_gradient[parameter])
    if abs(analytic) <= 1e-8:
        skipped += 1
        continue
    relative = abs(central - analytic) / abs(analytic)
    worst_relative = max(worst_relative, relative)
    compared += 1
    if relative >= 1e-5:
        print(f"    parameter {parameter}: autograd {analytic:.9e}  central {central:.9e}  rel {relative:.2e}")
check(compared > 0, f"compared {compared} parameters, skipped {skipped} with |gradient| <= 1e-8")
check(worst_relative < 1e-5, f"worst relative error autograd vs central differences = {worst_relative:.2e} < 1e-5")

# ------------------------------------------------------------- 5. timing and memory

print("timing and memory on all modelled vertices")

modelled = body.vertices[modelled_vertex_mask(body)]
timing_points = as_tensor(modelled)

# Memory first, while the process has not yet held a full-size autograd graph, so the
# working-set delta is not hidden by memory the allocator kept from an earlier run.
memory = template_sdf.memory_report(timing_points, as_tensor(template_centers), as_tensor(template_radii))
for line in memory.describe().splitlines():
    print(f"  {line}")
largest_shape, largest_elements = template_sdf.largest_intermediate(len(modelled))
print(
    f"  (the largest single tensor is shape {largest_shape}, {largest_elements} float64 "
    f"elements; it is NOT the memory a forward + backward needs)"
)
check(memory.largest_tensor_bytes == largest_elements * 8, "report's largest-tensor figure is the analytic one")
retained_ratio = memory.saved_for_backward_bytes / memory.largest_tensor_bytes
check(
    retained_ratio > 5.0,
    f"measured retained-for-backward memory is {retained_ratio:.1f}x the largest single tensor (> 5x)",
)
check(memory.saved_storage_count > 20, f"{memory.saved_storage_count} distinct storages saved for backward")
if memory.forward_delta is not None:
    check(
        # The allocator may hand back pages it cached from earlier tests, so the
        # working-set growth is a loose lower bound on the saved bytes, not an equality.
        memory.forward_delta > 0.3 * memory.saved_for_backward_bytes,
        f"process working set grew by {memory.forward_delta / 1e6:.0f} MB over the forward, "
        f"consistent with {memory.saved_for_backward_bytes / 1e6:.0f} MB saved for backward",
    )
else:
    print("  (process working set counter not available; skipping the working-set check)")

# The measured figure scales linearly in the number of points, which is what lets a fitter
# extrapolate it to a different batch.
half_memory = template_sdf.memory_report(
    timing_points[: len(modelled) // 2], as_tensor(template_centers), as_tensor(template_radii)
)
scaling = memory.saved_for_backward_bytes / half_memory.saved_for_backward_bytes
check(1.85 < scaling < 2.15, f"retained memory scales linearly in points: full / half = {scaling:.3f}")
print(
    f"  extrapolated retained memory for all {len(body.vertices)} vertices: "
    f"{memory.retained_bytes_for(len(body.vertices)) / 1e6:.0f} MB"
)

best_seconds = float("inf")
for _ in range(3):
    timing_centers = as_tensor(template_centers, requires_grad=True)
    timing_radii = as_tensor(template_radii, requires_grad=True)
    started = time.perf_counter()
    (template_sdf.signed_distance(timing_points, timing_centers, timing_radii) ** 2).sum().backward()
    best_seconds = min(best_seconds, time.perf_counter() - started)
print(
    f"  {len(modelled)} vertices x {len(template_graph.bumpers)} bumpers: forward + backward "
    f"best of 3 = {best_seconds:.3f} s"
)
check(best_seconds < 30.0, "forward + backward on all modelled vertices completes in reasonable time")
check(np.isfinite(timing_centers.grad.numpy()).all() and np.isfinite(timing_radii.grad.numpy()).all(), "gradients on the template are finite")

# ------------------------------------------------------------------- 6. no_grad parity

print("no_grad parity")

grad_centers = as_tensor(template_centers, requires_grad=True)
grad_radii = as_tensor(template_radii, requires_grad=True)
with_grad = template_sdf.distances(timing_points, grad_centers, grad_radii)
check(with_grad.requires_grad, "distances with requires_grad inputs carry a gradient")
with torch.no_grad():
    without_grad = template_sdf.distances(timing_points, grad_centers, grad_radii)
    without_grad_signed = template_sdf.signed_distance(timing_points, grad_centers, grad_radii)
check(not without_grad.requires_grad, "distances under torch.no_grad carry no gradient")
check(torch.equal(with_grad.detach(), without_grad), "distances under torch.no_grad are bit-identical")
check(
    torch.equal(with_grad.detach().min(dim=1).values, without_grad_signed),
    "signed_distance under torch.no_grad is bit-identical",
)

# ------------------------------------------------------------------ 7. edge cases

print("edge cases")

empty_sdf = TorchSdf(BumperGraph(0, []))
empty_distances = empty_sdf.distances(
    timing_points[:3], torch.zeros((0, 3), dtype=torch.float64), torch.zeros(0, dtype=torch.float64)
)
check(tuple(empty_distances.shape) == (3, 0), "a graph without spheres yields an (M, 0) distance matrix")

# ------------------------------------------------------------------------ summary

print()
if FAILURES:
    print(f"{len(FAILURES)} check(s) failed:")
    for message in FAILURES:
        print(f"  - {message}")
    sys.exit(1)
print("all checks passed")
