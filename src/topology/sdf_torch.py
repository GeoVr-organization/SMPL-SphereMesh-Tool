"""Differentiable exact signed distance of a sphere-mesh, in torch.

``src.topology.bumpers`` evaluates the exact signed distance of every collision primitive a
connection list generates. It is written in numpy and is the reference the validator
measures with; it cannot be differentiated. A fitter that wants to move sphere centres and
radii so the body's vertices sit on the sphere-mesh needs the same function with gradients
with respect to those centres and radii, and it needs them for the *exact* distance -- the
approximate cone that ``src.sphere_fit.spheremesh_distance`` used until 2026-10 slid the
true minimiser toward the larger ball and reported up to 24 mm of phantom gap on points
that are exactly on the surface (``TOPOLOGY_HANDOFF.md`` 4.3), which is a gradient
pointing the wrong way.

So this module is a line-by-line port of the numerics in ``bumpers.py`` to torch, and the
numerics are deliberately *not* rewritten: every branch is the same ``where`` on the same
predicate, and ``scripts/test_topology_sdf_torch.py`` holds the two implementations within
1e-10 of each other. Two things do differ, both for autograd only and both invisible in the
forward values:

* Wherever the numpy code takes ``sqrt(max(x, 0))`` the port takes
  ``sqrt(clamp_min(x, 1e-30))``. The gradient of ``sqrt`` at 0 is infinite, and an infinite
  gradient in the branch a ``where`` did *not* select still poisons the backward pass with
  ``0 * inf = NaN``; a floor of 1e-30 makes that branch finite without changing any value
  above 1e-30. The same floor is put under every norm for the same reason.
* Nothing is ever divided by a quantity that can be exactly zero; every such divisor is
  replaced by 1 under a mask, exactly as the numpy code already does for the swallowed cone
  and the degenerate tangent plane.

``TorchSdf`` precomputes, once per graph, the index tensors that map the bumper list onto
the four batched kernels, so a call is four gathers, four kernels and one permutation --
no Python loop over points or bumpers -- and column ``b`` of the result is
``graph.bumpers[b]`` exactly as in ``BumperGraph.distances``.

Memory is the thing a fitter has to size batches by, and the honest number is not the
biggest tensor a forward pass allocates: autograd keeps every intermediate the backward
needs alive until ``backward`` runs, and for these kernels that is a few dozen tensors of
(M, K) and (M, K, 3) shape, tens of times the largest single one. ``TorchSdf.memory_report``
therefore *measures* it -- the bytes of every distinct storage autograd saves for backward
(via ``torch.autograd.graph.saved_tensors_hooks``) and, where the platform exposes it, the
process working set before, after the forward and at its peak. ``largest_intermediate`` is
kept as the analytic single-tensor figure and says so in its name.
"""

from __future__ import annotations

import ctypes
import gc
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from src.topology.bumpers import (
    CONE,
    PRYSMOID,
    QUAD,
    SPHERE,
    BumperGraph,
)

__all__ = [
    "TorchSdf",
    "MemoryReport",
    "SavedTensorMeter",
    "process_working_set",
    "connections_from_payload",
    "sphere_distances",
    "round_cone_distances",
    "tangent_plane",
    "prysmoid_distances",
]

# Same constants as bumpers.py; they are part of the numerics being reproduced.
_MIN_CONE_SLACK = 1e-9
_MAX_TANGENT_IN_PLANE = 1.0 - 1e-12

# Floor under every square root and norm. sqrt'(0) is infinite, and torch.where does not
# stop an infinite gradient in the unselected branch from becoming NaN (0 * inf); values
# above the floor are untouched.
_SQRT_FLOOR = 1e-30


# --------------------------------------------------------------------------- distances


def sphere_distances(
    points: torch.Tensor, centers: torch.Tensor, radii: torch.Tensor
) -> torch.Tensor:
    """(M, N) signed distance from every point to every sphere."""
    difference = points[:, None, :] - centers[None, :, :]
    squared = torch.einsum("mnk,mnk->mn", difference, difference)
    # sqrt'(0) is infinite: a point sitting exactly on a centre must not poison the backward.
    return torch.sqrt(torch.clamp_min(squared, _SQRT_FLOOR)) - radii[None, :]


def round_cone_distances(
    points: torch.Tensor,
    start_center: torch.Tensor,
    start_radius: torch.Tensor,
    end_center: torch.Tensor,
    end_radius: torch.Tensor,
) -> torch.Tensor:
    """(M, K) exact signed distance to K tapered cones, spherical caps included.

    Port of ``bumpers.round_cone_distances``; see there for the derivation. The swallowed
    cone (centres closer than the radii differ) is answered with the enclosing sphere, and
    that fallback is always evaluated here rather than only when needed, so the graph has
    the same shape for every input.
    """
    axis = end_center - start_center                                     # (K, 3)
    axis_squared = torch.einsum("kj,kj->k", axis, axis)                  # (K,)
    radius_difference = start_radius - end_radius                        # (K,)
    lateral_squared = axis_squared - radius_difference ** 2              # (K,)

    degenerate = (axis_squared <= _MIN_CONE_SLACK) | (lateral_squared <= _MIN_CONE_SLACK)
    safe_axis_squared = torch.where(degenerate, torch.ones_like(axis_squared), axis_squared)
    inverse_axis_squared = 1.0 / safe_axis_squared                       # never divides by 0

    offset = points[:, None, :] - start_center[None, :, :]               # (M, K, 3)
    along = torch.einsum("mkj,kj->mk", offset, axis)                     # (M, K)
    beyond_end = along - safe_axis_squared[None, :]

    perpendicular = (
        offset * safe_axis_squared[None, :, None] - axis[None, :, :] * along[..., None]
    )
    perpendicular_squared = torch.einsum("mkj,mkj->mk", perpendicular, perpendicular)
    along_squared = along * along * safe_axis_squared[None, :]
    beyond_squared = beyond_end * beyond_end * safe_axis_squared[None, :]

    threshold = (
        torch.sign(radius_difference) * radius_difference ** 2
    )[None, :] * perpendicular_squared
    on_end_cap = torch.sign(beyond_end) * lateral_squared[None, :] * beyond_squared > threshold
    on_start_cap = torch.sign(along) * lateral_squared[None, :] * along_squared < threshold

    # numpy: sqrt(max(x, 0)). The floor keeps sqrt' finite where x <= 0 (swallowed cones,
    # points exactly on the axis) so the unselected branch of the where below stays clean.
    lateral_root = torch.sqrt(
        torch.clamp_min(
            perpendicular_squared * lateral_squared[None, :] * inverse_axis_squared[None, :],
            _SQRT_FLOOR,
        )
    )
    distance = (
        lateral_root + along * radius_difference[None, :]
    ) * inverse_axis_squared[None, :] - start_radius[None, :]
    end_cap_distance = (
        torch.sqrt(torch.clamp_min(perpendicular_squared + beyond_squared, _SQRT_FLOOR))
        * inverse_axis_squared[None, :]
        - end_radius[None, :]
    )
    start_cap_distance = (
        torch.sqrt(torch.clamp_min(perpendicular_squared + along_squared, _SQRT_FLOOR))
        * inverse_axis_squared[None, :]
        - start_radius[None, :]
    )
    distance = torch.where(on_end_cap, end_cap_distance, distance)
    distance = torch.where(on_start_cap & ~on_end_cap, start_cap_distance, distance)

    start_encloses = (start_radius >= end_radius)[:, None]
    enclosing_center = torch.where(start_encloses, start_center, end_center)
    enclosing_radius = torch.maximum(start_radius, end_radius)
    fallback = sphere_distances(points, enclosing_center, enclosing_radius)
    return torch.where(degenerate[None, :], fallback, distance)


def tangent_plane(
    centers: torch.Tensor, radii: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The plane tangent to three spheres from one side, for T triangles at once.

    Port of ``bumpers.tangent_plane``: ``centers`` is (T, 3, 3), ``radii`` is (T, 3), and
    the result is ``(in_plane_normal, out_of_plane_scale, plane_axis, exists)``. Where
    ``exists`` is False the other three are finite placeholders, never NaN, so a ``where``
    that discards them also discards a finite gradient.
    """
    first_edge = centers[:, 1] - centers[:, 0]
    second_edge = centers[:, 2] - centers[:, 0]

    first_dot_first = torch.einsum("tj,tj->t", first_edge, first_edge)
    second_dot_second = torch.einsum("tj,tj->t", second_edge, second_edge)
    first_dot_second = torch.einsum("tj,tj->t", first_edge, second_edge)
    determinant = first_dot_first * second_dot_second - first_dot_second ** 2
    determinant_ok = torch.abs(determinant) >= 1e-18
    safe_determinant = torch.where(determinant_ok, determinant, torch.ones_like(determinant))

    right_first = radii[:, 0] - radii[:, 1]
    right_second = radii[:, 0] - radii[:, 2]
    first_weight = (
        second_dot_second * right_first - first_dot_second * right_second
    ) / safe_determinant
    second_weight = (
        first_dot_first * right_second - first_dot_second * right_first
    ) / safe_determinant

    in_plane_normal = (
        first_weight[:, None] * first_edge + second_weight[:, None] * second_edge
    )
    in_plane_squared = torch.einsum("tj,tj->t", in_plane_normal, in_plane_normal)

    plane_axis = torch.linalg.cross(first_edge, second_edge, dim=1)
    # norm'(0) is infinite (it is a sqrt): floor the squared length before the root.
    axis_length = torch.sqrt(
        torch.clamp_min(torch.einsum("tj,tj->t", plane_axis, plane_axis), _SQRT_FLOOR)
    )
    exists = (
        determinant_ok
        & (axis_length > 1e-12)
        & (in_plane_squared < _MAX_TANGENT_IN_PLANE)
    )
    plane_axis = plane_axis / torch.clamp_min(axis_length, 1e-300)[:, None]
    # numpy: sqrt(max(1 - |w|^2, 0)); floored so sqrt' stays finite where no plane exists.
    out_of_plane_scale = torch.sqrt(torch.clamp_min(1.0 - in_plane_squared, _SQRT_FLOOR))
    return in_plane_normal, out_of_plane_scale, plane_axis, exists


def prysmoid_distances(
    points: torch.Tensor, centers: torch.Tensor, radii: torch.Tensor
) -> torch.Tensor:
    """(M, T) exact signed distance to T triangular slabs.

    Port of ``bumpers.prysmoid_distances``: ``centers`` is (T, 3, 3), ``radii`` is (T, 3).
    The three edge cones are always evaluated; the plane term is taken only where the
    tangent plane exists and the point's projection lands inside the tangency triangle.
    """
    best = round_cone_distances(
        points, centers[:, 0], radii[:, 0], centers[:, 1], radii[:, 1]
    )
    for first, second in ((1, 2), (2, 0)):
        best = torch.minimum(
            best,
            round_cone_distances(
                points,
                centers[:, first], radii[:, first],
                centers[:, second], radii[:, second],
            ),
        )

    in_plane_normal, out_of_plane_scale, plane_axis, exists = tangent_plane(centers, radii)

    # The stationary normal points from the interpolated centre towards the point, so its
    # out-of-plane sign is simply the side of the centre triangle the point lies on.
    offset_from_first = points[:, None, :] - centers[None, :, 0, :]
    side = torch.sign(torch.einsum("mtj,tj->mt", offset_from_first, plane_axis))
    side = torch.where(side == 0.0, torch.ones_like(side), side)

    normal = (
        in_plane_normal[None, :, :]
        + (side * out_of_plane_scale[None, :])[..., None] * plane_axis[None, :, :]
    )                                                                    # (M, T, 3)
    plane_offset = torch.einsum("mtj,tj->mt", normal, centers[:, 0]) + radii[None, :, 0]
    plane_distance = torch.einsum("mtj,mj->mt", normal, points) - plane_offset

    # The tangency points are the corners of the flat face. The point's projection has to
    # land inside their triangle for the interior stationary point to be the real minimum.
    tangency = centers[None, :, :, :] + radii[None, :, :, None] * normal[:, :, None, :]
    projected = points[:, None, :] - plane_distance[..., None] * normal

    usable = _inside_triangle(projected, tangency, normal) & exists[None, :]
    return torch.where(usable, torch.minimum(best, plane_distance), best)


def _inside_triangle(
    projected: torch.Tensor, corners: torch.Tensor, normal: torch.Tensor
) -> torch.Tensor:
    """(M, T) test that each projected point lies inside its triangle of corners.

    ``projected`` is (M, T, 3), ``corners`` is (M, T, 3, 3), ``normal`` is (M, T, 3). As in
    the numpy version the three edge turns only have to agree in sign, because the
    tangency triangle is wound one way against one face normal and the other way against
    the opposite face's.
    """
    turns = []
    for corner in range(3):
        edge = corners[:, :, (corner + 1) % 3, :] - corners[:, :, corner, :]
        to_point = projected - corners[:, :, corner, :]
        turns.append(
            torch.einsum("mtj,mtj->mt", torch.linalg.cross(edge, to_point, dim=2), normal)
        )
    turn = torch.stack(turns, dim=2)
    return torch.all(turn >= -1e-12, dim=2) | torch.all(turn <= 1e-12, dim=2)


# --------------------------------------------------------------------------- the graph


class TorchSdf:
    """Batched, differentiable ``BumperGraph.distances``.

    The constructor reads ``graph.bumpers`` once and stores index tensors for each kind:
    which sphere a sphere column reads, which (start, end) pair a cone column reads, and
    the triangles of every slab column -- one for a prysmoid, two for a quad split across
    its 0-2 diagonal exactly as ``Bumper.triangles`` does. All triangles of all slabs are
    evaluated in one batched call and the quad pairs reduced with a minimum, then the four
    blocks are permuted back into bumper order.
    """

    def __init__(self, graph: BumperGraph, *, dtype: torch.dtype = torch.float64):
        self.graph = graph
        self.dtype = dtype
        self.num_spheres = graph.num_spheres
        self.num_bumpers = len(graph.bumpers)

        sphere_columns: list[int] = []
        sphere_owned: list[int] = []
        cone_columns: list[int] = []
        cone_pairs: list[tuple[int, int]] = []
        prysmoid_columns: list[int] = []
        prysmoid_triangles: list[tuple[int, int, int]] = []
        quad_columns: list[int] = []
        quad_triangles: list[tuple[int, int, int]] = []
        for bumper_index, bumper in enumerate(graph.bumpers):
            if bumper.kind == SPHERE:
                sphere_columns.append(bumper_index)
                sphere_owned.append(bumper.spheres[0])
            elif bumper.kind == CONE:
                cone_columns.append(bumper_index)
                cone_pairs.append((bumper.spheres[0], bumper.spheres[1]))
            elif bumper.kind == PRYSMOID:
                prysmoid_columns.append(bumper_index)
                prysmoid_triangles.extend(bumper.triangles())
            elif bumper.kind == QUAD:
                quad_columns.append(bumper_index)
                quad_triangles.extend(bumper.triangles())
            else:
                raise ValueError(f"unknown bumper kind {bumper.kind!r}")

        long = torch.int64
        self.sphere_owned = torch.tensor(sphere_owned, dtype=long)
        self.cone_pairs = torch.tensor(cone_pairs, dtype=long).reshape(-1, 2)
        # Prysmoid triangles first, then the quad triangles two by two, so the batched slab
        # result splits as [:P] and [P:].reshape(-1, Q, 2).
        self.slab_triangles = torch.tensor(
            prysmoid_triangles + quad_triangles, dtype=long
        ).reshape(-1, 3)
        self.num_sphere_columns = len(sphere_columns)
        self.num_cone_columns = len(cone_columns)
        self.num_prysmoid_columns = len(prysmoid_columns)
        self.num_quad_columns = len(quad_columns)

        block_order = sphere_columns + cone_columns + prysmoid_columns + quad_columns
        self.block_to_column = torch.tensor(block_order, dtype=long)
        self.column_from_block = torch.argsort(self.block_to_column)

    # -- shapes --------------------------------------------------------------------

    def largest_intermediate(self, num_points: int) -> tuple[tuple[int, ...], int]:
        """Shape and element count of the biggest *single* tensor a forward call allocates.

        This is an analytic figure for one tensor, not the memory a forward + backward
        needs: autograd retains dozens of intermediates of comparable size until backward
        runs, so the retained total is tens of times larger (about 26x on the auto_male
        template). Size batches with ``memory_report``, which measures that total.

        The candidates are the (M, K, 3) offset of the cone kernel (which the slab kernel
        runs three times, once per edge, with K = its triangle count) and the (M, T, 3, 3)
        tangency corners of the slab kernel.
        """
        num_triangles = int(self.slab_triangles.shape[0])
        candidates = [
            ((num_points, self.num_cone_columns, 3), num_points * self.num_cone_columns * 3),
            ((num_points, num_triangles, 3), num_points * num_triangles * 3),
            ((num_points, num_triangles, 3, 3), num_points * num_triangles * 9),
        ]
        return max(candidates, key=lambda candidate: candidate[1])

    def memory_report(
        self,
        points: torch.Tensor,
        centers: torch.Tensor,
        radii: torch.Tensor,
        loss: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> "MemoryReport":
        """Measure what one forward + backward of ``loss(signed_distance)`` really costs.

        ``centers`` and ``radii`` are cloned with ``requires_grad`` so the caller's tensors
        are untouched. ``loss`` defaults to the sum of squared signed distance, the fitter's
        objective. The report holds the bytes autograd saved for backward (every distinct
        storage, counted once, measured with ``saved_tensors_hooks``), the analytic size of
        the largest single tensor for comparison, and the process working set before, after
        the forward and at its peak when the platform exposes it (``None`` otherwise).
        """
        if loss is None:
            loss = _sum_of_squares
        grad_centers = centers.detach().clone().requires_grad_(True)
        grad_radii = radii.detach().clone().requires_grad_(True)

        gc.collect()
        working_set_before = process_working_set()
        meter = SavedTensorMeter()
        started = time.perf_counter()
        with meter:
            objective = loss(self.signed_distance(points, grad_centers, grad_radii))
        working_set_after_forward = process_working_set()
        objective.backward()
        seconds = time.perf_counter() - started
        working_set_after_backward = process_working_set()

        _, largest_elements = self.largest_intermediate(int(points.shape[0]))
        element_bytes = torch.empty((), dtype=self.dtype).element_size()
        return MemoryReport(
            num_points=int(points.shape[0]),
            num_bumpers=self.num_bumpers,
            largest_tensor_bytes=largest_elements * element_bytes,
            saved_for_backward_bytes=meter.unique_bytes,
            saved_tensor_count=meter.saved_count,
            saved_storage_count=meter.unique_count,
            working_set_before=None if working_set_before is None else working_set_before[0],
            working_set_after_forward=(
                None if working_set_after_forward is None else working_set_after_forward[0]
            ),
            peak_working_set=(
                None if working_set_after_backward is None else working_set_after_backward[1]
            ),
            seconds=seconds,
        )

    # -- evaluation ----------------------------------------------------------------

    def distances(
        self, points: torch.Tensor, centers: torch.Tensor, radii: torch.Tensor
    ) -> torch.Tensor:
        """(M, B) exact signed distance from every point to every bumper.

        ``points`` is (M, 3) and needs no gradient; ``centers`` (N, 3) and ``radii`` (N,)
        may require one. Column ``b`` is ``graph.bumpers[b]``.
        """
        points = points.to(self.dtype)
        centers = centers.to(self.dtype)
        radii = radii.to(self.dtype)
        device = centers.device

        blocks: list[torch.Tensor] = []
        if self.num_sphere_columns:
            owned = self.sphere_owned.to(device)
            blocks.append(sphere_distances(points, centers[owned], radii[owned]))
        if self.num_cone_columns:
            pairs = self.cone_pairs.to(device)
            blocks.append(
                round_cone_distances(
                    points,
                    centers[pairs[:, 0]], radii[pairs[:, 0]],
                    centers[pairs[:, 1]], radii[pairs[:, 1]],
                )
            )
        if self.num_prysmoid_columns or self.num_quad_columns:
            triangles = self.slab_triangles.to(device)
            slab = prysmoid_distances(points, centers[triangles], radii[triangles])
            split = self.num_prysmoid_columns
            if self.num_prysmoid_columns:
                blocks.append(slab[:, :split])
            if self.num_quad_columns:
                quad_pairs = slab[:, split:].reshape(points.shape[0], self.num_quad_columns, 2)
                blocks.append(torch.minimum(quad_pairs[:, :, 0], quad_pairs[:, :, 1]))

        if not blocks:  # a graph without spheres has no columns; torch.cat rejects []
            return torch.zeros((points.shape[0], 0), dtype=self.dtype, device=device)
        combined = torch.cat(blocks, dim=1)
        return combined[:, self.column_from_block.to(device)]

    def signed_distance(
        self, points: torch.Tensor, centers: torch.Tensor, radii: torch.Tensor
    ) -> torch.Tensor:
        """(M,) signed distance to the sphere-mesh: the column-wise minimum."""
        return self.distances(points, centers, radii).min(dim=1).values


# --------------------------------------------------------------------------- memory


def _sum_of_squares(signed_distance: torch.Tensor) -> torch.Tensor:
    return (signed_distance ** 2).sum()


@dataclass(frozen=True)
class MemoryReport:
    """What one forward + backward of ``TorchSdf`` cost; see ``TorchSdf.memory_report``.

    ``saved_for_backward_bytes`` is the measured figure to size batches with: it is
    linear in the number of points, so ``bytes_per_point`` extrapolates to any batch.
    ``largest_tensor_bytes`` is the analytic size of the single biggest tensor and is
    only there to show how much smaller than the retained total it is. The working-set
    numbers are process-wide (``None`` when the platform offers no counter) and the peak
    is the process-lifetime peak, so ``peak_delta`` is only meaningful when this call was
    the largest thing the process has done so far.
    """

    num_points: int
    num_bumpers: int
    largest_tensor_bytes: int
    saved_for_backward_bytes: int
    saved_tensor_count: int
    saved_storage_count: int
    working_set_before: int | None
    working_set_after_forward: int | None
    peak_working_set: int | None
    seconds: float

    @property
    def bytes_per_point(self) -> float:
        return self.saved_for_backward_bytes / max(self.num_points, 1)

    def retained_bytes_for(self, num_points: int) -> float:
        """Extrapolated retained-for-backward bytes for a batch of ``num_points``."""
        return self.bytes_per_point * num_points

    @property
    def forward_delta(self) -> int | None:
        if self.working_set_before is None or self.working_set_after_forward is None:
            return None
        return self.working_set_after_forward - self.working_set_before

    @property
    def peak_delta(self) -> int | None:
        if self.working_set_before is None or self.peak_working_set is None:
            return None
        return max(self.peak_working_set - self.working_set_before, 0)

    def describe(self) -> str:
        megabytes = 1e6
        lines = [
            f"{self.num_points} points x {self.num_bumpers} bumpers, forward + backward "
            f"{self.seconds:.3f} s",
            f"largest single tensor (analytic): {self.largest_tensor_bytes / megabytes:.1f} MB",
            f"saved for backward (measured): {self.saved_for_backward_bytes / megabytes:.1f} MB "
            f"in {self.saved_storage_count} storages ({self.saved_tensor_count} saved tensors), "
            f"{self.bytes_per_point / 1e3:.1f} kB per point",
        ]
        if self.forward_delta is not None:
            lines.append(
                f"process working set: {self.working_set_before / megabytes:.0f} MB before, "
                f"+{self.forward_delta / megabytes:.0f} MB after forward, "
                f"peak +{self.peak_delta / megabytes:.0f} MB"
            )
        else:
            lines.append("process working set: not available on this platform")
        return "\n".join(lines)


class SavedTensorMeter:
    """Context manager that counts the memory autograd saves for backward inside it.

    ``torch.autograd.graph.saved_tensors_hooks`` calls ``pack`` for every tensor an
    operation keeps for its backward. Many of those are views of the same storage, and a
    storage is only paid for once, so the bytes are summed over distinct storages (keyed by
    data pointer and size) while ``saved_count`` still counts every saved tensor.
    """

    def __init__(self) -> None:
        self.saved_count = 0
        self._storages: dict[tuple[int, int], int] = {}
        self._hooks: torch.autograd.graph.saved_tensors_hooks | None = None

    @property
    def unique_bytes(self) -> int:
        return sum(self._storages.values())

    @property
    def unique_count(self) -> int:
        return len(self._storages)

    def _pack(self, tensor: torch.Tensor) -> torch.Tensor:
        self.saved_count += 1
        storage = tensor.untyped_storage()
        self._storages[(storage.data_ptr(), storage.nbytes())] = storage.nbytes()
        return tensor

    @staticmethod
    def _unpack(tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def __enter__(self) -> "SavedTensorMeter":
        self._hooks = torch.autograd.graph.saved_tensors_hooks(self._pack, self._unpack)
        self._hooks.__enter__()
        return self

    def __exit__(self, *exception_info: object) -> None:
        assert self._hooks is not None
        self._hooks.__exit__(*exception_info)
        self._hooks = None


def process_working_set() -> tuple[int, int] | None:
    """(current, peak) resident memory of this process in bytes, or None if unknown.

    Windows reads ``K32GetProcessMemoryInfo``; elsewhere ``getrusage`` gives the peak (in
    kilobytes on Linux, bytes on macOS) and ``/proc/self/statm`` the current size when
    it exists, else the peak stands in for it.
    """
    if sys.platform == "win32":
        return _windows_working_set()
    try:
        import resource
    except ImportError:
        return None
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        peak *= 1024
    try:
        with open("/proc/self/statm", encoding="utf-8") as statm:
            resident_pages = int(statm.read().split()[1])
        current = resident_pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        current = peak
    return int(current), peak


def _windows_working_set() -> tuple[int, int] | None:
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    try:
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(ProcessMemoryCounters)
        query = ctypes.windll.kernel32.K32GetProcessMemoryInfo
        query.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD]
        query.restype = wintypes.BOOL
        if not query(wintypes.HANDLE(-1), ctypes.byref(counters), counters.cb):
            return None
    except (AttributeError, OSError):
        return None
    return int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)


# --------------------------------------------------------------------------- payloads


def connections_from_payload(payload: dict) -> list[tuple[int, ...]]:
    """Rebuild the engine connection list from a tool JSON payload.

    The tool only stores 2- and 3-index connections, so ``export.split_quads`` writes each
    quad ``(a, b, c, d)`` as the triangles ``(a, b, c)`` and ``(a, c, d)`` and records the
    quad under ``quad_groups``. This undoes that: every quad takes the place of its first
    triangle and its second triangle is dropped. The result is deterministic and has the
    same *set* of connections as the engine ``.txt``, but not the same order -- the text
    file groups triples, doubles and quads into separate sections -- so bumper indices of
    a ``BumperGraph`` built from it are not the engine's slab numbers.

    A split triangle is matched exactly first and, failing that, as a vertex set, so a
    triangle rotated by hand still matches; a hand-edited payload that carries a genuine
    prysmoid on the same three vertices as a split triangle would be indistinguishable.
    """
    connections: list[tuple[int, ...] | None] = []
    for connection in payload.get("connections", []):
        indices = connection["indices"] if isinstance(connection, dict) else connection
        connections.append(tuple(int(index) for index in indices))

    for quad in payload.get("quad_groups", []) or []:
        first, second, third, fourth = (int(index) for index in quad)
        first_triangle = (first, second, third)
        second_triangle = (first, third, fourth)
        first_position = _find_triangle(connections, first_triangle)
        second_position = _find_triangle(connections, second_triangle)
        if first_position is None or second_position is None:
            raise ValueError(
                f"quad {quad} has no matching split triangles "
                f"{first_triangle} and {second_triangle} in the connection list"
            )
        connections[first_position] = (first, second, third, fourth)
        connections[second_position] = None
    return [connection for connection in connections if connection is not None]


def _find_triangle(
    connections: Sequence[tuple[int, ...] | None], triangle: tuple[int, int, int]
) -> int | None:
    """Position of ``triangle`` in ``connections``: exact order first, then as a set."""
    for position, connection in enumerate(connections):
        if connection == triangle:
            return position
    wanted = set(triangle)
    for position, connection in enumerate(connections):
        if connection is not None and len(connection) == 3 and set(connection) == wanted:
            return position
    return None
