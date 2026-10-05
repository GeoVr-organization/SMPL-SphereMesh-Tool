"""Fitting sphere centres and radii against the *exact* sphere-mesh distance.

Why this exists
---------------
The previous pipeline handed the placed spheres to ``src.sphere_fit.optimize_spheres``,
which fits every sphere to the vertices nearest its own centre and knows nothing about the
cones and slabs between spheres. Measured, it *raised* sphere-mesh protrusion from 28 mm to
40 mm on one configuration while lowering its own objective, and two further passes -- a
radius floor and a shrink loop -- existed only to undo that. Every one of those numbers was
produced by a metric that could not see the primitives it was scoring.

This module optimises the thing that is actually measured: the signed distance from every
modelled body vertex to the union of spheres, cones and slabs, evaluated exactly by
``sdf_torch.TorchSdf`` and differentiated by autograd. The loss is one-sided --

    rho(d) = d^2                 d >= 0   (a gap: the skin is outside the collider)
           = (kappa * d)^2       d <  0   (a protrusion: the collider is outside the skin)

-- with ``kappa`` well above one, because a protrusion pushes cloth *off* the body and is
worse than a gap of the same size. Optionally the same protrusion term is evaluated under a
set of real poses, with the spheres moved by the tool's own rigid single-joint rule, so
that a cone which pops out of the arm when the elbow bends is penalised where the user
would see it.

What is held fixed
------------------
* **Bilateral symmetry, by construction.** Only the left half and the midline are free
  parameters; every ``R_`` sphere is the exact reflection of its ``L_`` twin and midline
  spheres keep ``x = 0``. Symmetry is therefore never "restored" afterwards, it simply
  cannot be lost.
* **Bindings.** Centres may move at most ``max_offset`` per axis from where the placement
  rules put them, and after the fit every sphere's tool-assigned joint is re-derived. A
  sphere whose binding changed is pinned back to its placed centre and the fit is repeated
  with that sphere's centre frozen, because a centre that crossed a bone boundary would be
  bound to a different joint on load and undo the animation reasoning that placed it.
* **Topology.** Connections are never changed here; that is ``repair``'s job.

The parameterisation is ``centre = placed + delta`` with ``|delta| <= max_offset`` per
axis and ``radius = placed_radius * exp(log_scale)`` with the scale bounded, both handled as
box constraints by L-BFGS-B, so no penalty term has to be tuned for them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .anatomy import (
    TRUNK_JOINTS,
    Body,
    assign_parent_joint_like_tool,
    joints_compatible,
    load_body,
)
from .bumpers import BumperGraph

__all__ = [
    "binding_margin",
    "SymmetricParameters",
    "FitOptions",
    "FitResult",
    "vertex_areas",
    "fit_topology",
]


# ------------------------------------------------------------------ parameterisation


def _mirror_name(name: str) -> str:
    if name.startswith("L_"):
        return "R_" + name[2:]
    if name.startswith("R_"):
        return "L_" + name[2:]
    return name


@dataclass
class SymmetricParameters:
    """Maps a flat vector of free parameters to full ``(centers, radii)`` arrays.

    Free spheres are the ``L_`` spheres and everything not prefixed ``L_``/``R_`` (the
    midline). A ``R_`` sphere whose ``L_`` twin exists is derived; a ``R_`` sphere with no
    twin is treated as free (it is asymmetric by the author's choice, not ours).
    """

    names: list[str]
    placed_centers: np.ndarray                 # (N, 3)
    placed_radii: np.ndarray                   # (N,)
    free_index: np.ndarray = field(init=False)  # (N,) index into the free list per sphere
    mirror_sign: np.ndarray = field(init=False)  # (N,) +1 or -1 on x
    midline: np.ndarray = field(init=False)    # (F,) bool per free sphere
    free_names: list[str] = field(init=False)

    def __post_init__(self) -> None:
        index_of = {name: index for index, name in enumerate(self.names)}
        free_names: list[str] = []
        free_slot: dict[str, int] = {}
        free_index = np.zeros(len(self.names), dtype=np.int64)
        mirror_sign = np.ones(len(self.names), dtype=np.float64)
        for index, name in enumerate(self.names):
            twin = _mirror_name(name)
            if name.startswith("R_") and twin in index_of:
                continue
            free_slot[name] = len(free_names)
            free_names.append(name)
        for index, name in enumerate(self.names):
            twin = _mirror_name(name)
            if name.startswith("R_") and twin in index_of:
                free_index[index] = free_slot[twin]
                mirror_sign[index] = -1.0
            else:
                free_index[index] = free_slot[name]
        self.free_names = free_names
        self.free_index = free_index
        self.mirror_sign = mirror_sign
        self.midline = np.array(
            [not (name.startswith("L_") or name.startswith("R_")) for name in free_names]
        )

    @property
    def free_count(self) -> int:
        return len(self.free_names)

    @property
    def size(self) -> int:
        return 4 * self.free_count

    def bounds(
        self, max_offset: float, radius_scale_bounds: tuple[float, float],
        frozen_centers: Sequence[str] = (),
        offset_overrides: dict[str, float] | None = None,
        anchor_centers: np.ndarray | None = None,
        anchor_radii: np.ndarray | None = None,
    ) -> list[tuple[float, float]]:
        """Box bounds for L-BFGS-B over the flat vector ``[dx, dy, dz, log_scale] * F``.

        ``offset_overrides`` gives individual spheres a smaller centre box than
        ``max_offset``; ``frozen_centers`` pins them entirely. ``anchor_centers`` (N, 3)
        are the centres the offset is measured *from*: when a fit starts from the result
        of an earlier fit, the box is shifted so the total displacement from the original
        placement stays within ``max_offset`` instead of compounding call after call.
        """
        frozen = set(frozen_centers)
        overrides = offset_overrides or {}
        shift = np.zeros((len(self.names), 3)) if anchor_centers is None else (
            self.placed_centers - np.asarray(anchor_centers, dtype=np.float64)
        )
        # Radius bounds are per sphere and measured from the anchor radius too: a start
        # already at 0.9 of its anchor may only shrink to floor/0.9 of itself. (A global
        # translation of the floor let spheres shrink to 0.59 of placement over repeated
        # refits, which is the shoulder's bad basin.)
        ratio = np.ones(len(self.names)) if anchor_radii is None else (
            self.placed_radii / np.asarray(anchor_radii, dtype=np.float64)
        )
        # The sphere index whose own (un-mirrored) coordinates each free slot carries.
        representative = {}
        for index, name in enumerate(self.names):
            if self.mirror_sign[index] > 0:
                representative.setdefault(int(self.free_index[index]), index)
        bounds: list[tuple[float, float]] = []
        for slot, name in enumerate(self.free_names):
            offset = 0.0 if name in frozen else float(overrides.get(name, max_offset))
            x_offset = 0.0 if self.midline[slot] else offset
            already = shift[representative[slot]]
            for limit, moved in zip((x_offset, offset, offset), already):
                bounds.append((min(-limit - moved, 0.0), max(limit - moved, 0.0)))
            own_ratio = ratio[representative[slot]]
            bounds.append((
                min(np.log(radius_scale_bounds[0] / own_ratio), 0.0),
                max(np.log(radius_scale_bounds[1] / own_ratio), 0.0),
            ))
        return bounds

    def initial(self) -> np.ndarray:
        return np.zeros(self.size)

    def expand_numpy(self, flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        block = flat.reshape(self.free_count, 4)
        delta = block[self.free_index, :3].copy()
        delta[:, 0] *= self.mirror_sign
        centers = self.placed_centers + delta
        radii = self.placed_radii * np.exp(block[self.free_index, 3])
        return centers, radii

    def expand_torch(self, flat):
        """Torch version of ``expand_numpy`` for autograd; ``flat`` is a (4F,) tensor."""
        import torch

        block = flat.view(self.free_count, 4)
        gather = torch.as_tensor(self.free_index)
        delta = block[gather, :3]
        sign = torch.as_tensor(
            np.stack([self.mirror_sign, np.ones_like(self.mirror_sign),
                      np.ones_like(self.mirror_sign)], axis=1)
        )
        centers = torch.as_tensor(self.placed_centers) + delta * sign
        radii = torch.as_tensor(self.placed_radii) * torch.exp(block[gather, 3])
        return centers, radii


# ---------------------------------------------------------------------- options


@dataclass
class FitOptions:
    """Everything the fit can be told. Defaults are the ones the notes argue for."""

    max_offset: float = 0.012
    #: Radii may shrink to this fraction of the placed (skin-touching) radius and grow to
    #: the other. Growth is allowed because the placement radius is the *inscribed* one
    #: and a cone between two inscribed spheres is usually slightly inside the skin. The
    #: floor is deliberately high: with 0.55 the one-sided loss has a second basin at the
    #: shoulder where the deltoid, clavicle and first upper-arm spheres shrink to the floor
    #: to silence a cone's protrusion and leave a 35 mm gap behind (RMS 9-10 mm against
    #: 5.3 mm in the good basin, which never needed a radius below 0.86 of placement).
    radius_scale_bounds: tuple[float, float] = (0.70, 1.30)
    protrusion_weight: float = 6.0
    #: Poses: (P, 72) thetas. The protrusion term under pose only penalises what pokes
    #: out further than ``pose_tolerance``; gaps under pose are not penalised because the
    #: skin stretches away from a rigid collider at every joint and fighting that would
    #: wreck the rest fit.
    poses: np.ndarray | None = None
    pose_weight: float = 0.5
    pose_tolerance: float = 0.006
    pose_vertex_stride: int = 3
    #: Vertex weights: area-weighted so dense mesh regions (face, hands) do not dominate.
    area_weighted: bool = True
    #: Points used during the fit: every ``point_stride``-th modelled vertex. 1 uses all.
    point_stride: int = 1
    max_iterations: int = 150
    tolerance: float = 1e-10
    #: Small pull towards the placed centres, relative to the loss scale, to break the
    #: flat-patch degeneracy (translate along the normal and grow the radius) that the
    #: bounds alone leave undecided. Zero disables it.
    center_regularization: float = 1e-4
    #: Rounds of "check bindings, restrain violators, refit". A sphere whose tool binding
    #: changed has its centre box halved and is re-solved from its placed centre; only in
    #: the last round is it pinned outright. Freezing at the first offence left whole
    #: regions under-fitted (measured: the shoulder top at 30 mm gap with three spheres
    #: frozen, against 24 mm with none).
    binding_rounds: int = 4
    #: A sphere may never end up inside another: it would be a bumper that can never be
    #: nearest to anything, and its cones would have no lateral surface. Penalised over
    #: every sphere pair once the surfaces come within ``containment_margin`` of nesting.
    containment_weight: float = 1.0
    containment_margin: float = 0.006
    verbose: bool = True


@dataclass
class FitResult:
    centers: np.ndarray
    radii: np.ndarray
    info: dict


# ------------------------------------------------------------------------ helpers


def vertex_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """One third of the area of every incident face, per vertex (the mixed-area lump)."""
    triangle = vertices[faces]
    cross = np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])
    face_area = 0.5 * np.linalg.norm(cross, axis=1)
    area = np.zeros(len(vertices))
    for corner in range(3):
        np.add.at(area, faces[:, corner], face_area / 3.0)
    return area


def binding_margin(centers: np.ndarray, rest_joints: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """How far each sphere is from changing its tool-assigned bone, in metres.

    The tool binds a sphere to the closest kintree bone. The margin is the distance to the
    second-closest bone minus the distance to the closest: a sphere with a margin of a
    millimetre binds by floating-point luck and its mirror twin may bind differently.
    """
    centers = np.atleast_2d(np.asarray(centers, dtype=np.float64))
    distances = []
    for child in range(1, len(parents)):
        parent = int(parents[child])
        if parent < 0:
            continue
        start, end = rest_joints[parent], rest_joints[child]
        along = end - start
        length_squared = float(along @ along)
        if length_squared < 1e-12:
            distances.append(np.linalg.norm(centers - start, axis=1))
            continue
        parameter = np.clip((centers - start) @ along / length_squared, 0.0, 1.0)
        distances.append(np.linalg.norm(centers - (start + parameter[:, None] * along), axis=1))
    ranked = np.sort(np.stack(distances, axis=1), axis=1)
    return ranked[:, 1] - ranked[:, 0]


def _acceptable_binding_change(old_joint: int, new_joint: int, parents: np.ndarray) -> bool:
    return (
        old_joint in TRUNK_JOINTS and new_joint in TRUNK_JOINTS
        and joints_compatible(old_joint, new_joint, parents)
    )


def _one_sided(distance, kappa: float):
    import torch

    return torch.where(distance >= 0.0, distance * distance, (kappa * kappa) * distance * distance)


# -------------------------------------------------------------------------- the fit


def fit_topology(
    body: Body,
    centers: np.ndarray,
    radii: np.ndarray,
    connections: Sequence[Sequence[int]],
    names: Sequence[str],
    *,
    vertex_mask: np.ndarray,
    options: FitOptions | None = None,
    frozen_centers: Sequence[str] = (),
    anchor_centers: np.ndarray | None = None,
    anchor_radii: np.ndarray | None = None,
) -> FitResult:
    """Optimise centres and radii against the exact sphere-mesh distance.

    ``centers``/``radii`` are where this fit starts; ``vertex_mask`` selects the modelled
    body vertices. ``anchor_centers``/``anchor_radii`` are the original placement when
    the start is itself the result of an earlier fit: the centre box and the radius-scale
    bounds are then measured from the anchor, so repeated fits cannot drift further than
    one fit could. Returns new arrays plus a dict with the loss history, binding audit
    and timing.
    """
    import torch
    from scipy.optimize import minimize

    from .sdf_torch import TorchSdf

    torch.set_default_dtype(torch.float64)
    options = options or FitOptions()
    names = list(names)
    centers = np.ascontiguousarray(centers, dtype=np.float64)
    radii = np.ascontiguousarray(radii, dtype=np.float64)
    connections = [tuple(int(index) for index in connection) for connection in connections]
    started = time.perf_counter()

    graph = BumperGraph(len(centers), connections)
    sdf = TorchSdf(graph)
    parameters = SymmetricParameters(names, centers, radii)
    if anchor_centers is not None:
        anchor_centers = np.asarray(anchor_centers, dtype=np.float64)
    if anchor_radii is not None:
        anchor_radii = np.asarray(anchor_radii, dtype=np.float64)

    # ---------------------------------------------------------------- rest-pose points
    modelled_index = np.flatnonzero(vertex_mask)
    fit_index = modelled_index[:: max(1, int(options.point_stride))]
    points = torch.as_tensor(body.vertices[fit_index])
    if options.area_weighted:
        area = vertex_areas(body.vertices, body.faces)[fit_index]
        weights = torch.as_tensor(area / area.mean())
    else:
        weights = torch.ones(len(fit_index))
    weight_total = float(weights.sum())

    # ------------------------------------------------------------------ pose set
    placed_parent, _, _ = assign_parent_joint_like_tool(centers, body.joints, body.parents)
    pose_terms: list[tuple] = []
    if options.poses is not None and len(options.poses) and options.pose_weight > 0.0:
        rest_joints = torch.as_tensor(body.joints)
        parent_index = torch.as_tensor(placed_parent)
        pose_index = modelled_index[:: max(1, int(options.pose_vertex_stride))]
        for theta in np.asarray(options.poses, dtype=np.float64):
            posed = load_body(body.gender, theta=theta)
            pose_terms.append((
                torch.as_tensor(posed.joints[placed_parent]),
                torch.as_tensor(posed.rotations[placed_parent]),
                torch.as_tensor(posed.vertices[pose_index]),
            ))
        pose_rest_offset_base = rest_joints[parent_index]
    kappa = float(options.protrusion_weight)
    kappa_pose_squared = kappa * kappa
    off_diagonal = 1.0 - torch.eye(len(centers))
    tolerance = float(options.pose_tolerance)

    history: list[dict] = []

    def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
        flat_tensor = torch.as_tensor(flat).clone().requires_grad_(True)
        current_centers, current_radii = parameters.expand_torch(flat_tensor)

        rest_distance = sdf.signed_distance(points, current_centers, current_radii)
        rest_loss = (weights * _one_sided(rest_distance, kappa)).sum() / weight_total
        loss = rest_loss

        pose_loss = torch.zeros(())
        if pose_terms:
            offsets = current_centers - pose_rest_offset_base
            for posed_joint, posed_rotation, posed_points in pose_terms:
                posed_centers = posed_joint + torch.einsum("nij,nj->ni", posed_rotation, offsets)
                posed_distance = sdf.signed_distance(posed_points, posed_centers, current_radii)
                excess = torch.clamp_min(-posed_distance - tolerance, 0.0)
                pose_loss = pose_loss + kappa_pose_squared * (excess * excess).mean()
            pose_loss = pose_loss / len(pose_terms)
            loss = loss + float(options.pose_weight) * pose_loss

        if options.containment_weight > 0.0:
            separation = torch.cdist(current_centers, current_centers)
            nesting = (
                torch.abs(current_radii[:, None] - current_radii[None, :])
                + float(options.containment_margin) - separation
            )
            nesting = torch.clamp_min(nesting, 0.0) * off_diagonal
            loss = loss + float(options.containment_weight) * (nesting * nesting).sum()

        if options.center_regularization > 0.0:
            block = flat_tensor.view(parameters.free_count, 4)
            pull = (block[:, :3] ** 2).sum() / max(float(options.max_offset) ** 2, 1e-12)
            loss = loss + float(options.center_regularization) * pull / parameters.free_count

        loss.backward()
        gradient = flat_tensor.grad.detach().numpy().astype(np.float64)
        history.append({
            "loss": loss.item(), "rest": rest_loss.item(), "pose": pose_loss.item(),
            "max_gap": rest_distance.max().item(),
            "max_protrusion": (-rest_distance.min()).item(),
        })
        return loss.item(), gradient

    # ------------------------------------------------------------ binding-safe rounds
    frozen = set(frozen_centers)
    offset_overrides: dict[str, float] = {}
    flat = parameters.initial()
    audit: list[dict] = []
    rounds = max(1, int(options.binding_rounds))
    for binding_round in range(rounds):
        bounds = parameters.bounds(
            options.max_offset, options.radius_scale_bounds, frozen, offset_overrides,
            anchor_centers, anchor_radii,
        )
        solution = minimize(
            objective, flat, jac=True, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": int(options.max_iterations), "ftol": options.tolerance,
                     "gtol": 1e-12, "maxcor": 30},
        )
        flat = solution.x
        fitted_centers, fitted_radii = parameters.expand_numpy(flat)
        fitted_parent, _, _ = assign_parent_joint_like_tool(
            fitted_centers, body.joints, body.parents
        )
        # A sphere sliding across the boundary between two compatible *trunk* joints
        # (spine3 to the collar, spine1 to spine2) changes its animation as little as the
        # trunk rule allows for a cone; restarting it costs far more than it protects.
        # Anything else -- a limb joint on either side -- is a real crossing.
        changed = np.array([
            index for index in np.flatnonzero(fitted_parent != placed_parent)
            if not _acceptable_binding_change(
                int(placed_parent[index]), int(fitted_parent[index]), body.parents
            )
        ], dtype=np.int64)
        audit.append({
            "round": binding_round, "iterations": int(solution.nit),
            "evaluations": int(solution.nfev), "loss": float(solution.fun),
            "binding_changed": [names[index] for index in changed],
            "message": str(solution.message),
        })
        if options.verbose:
            print(
                f"fit       round {binding_round}: {solution.nit} iterations, "
                f"loss {solution.fun:.4e}, worst gap {history[-1]['max_gap'] * 1000:.1f} mm, "
                f"worst protrusion {history[-1]['max_protrusion'] * 1000:.1f} mm"
                + ("" if len(changed) == 0
                   else f"; {len(changed)} sphere(s) changed binding, pinning: "
                        + ", ".join(names[index] for index in changed[:6]))
            )
        if len(changed) == 0:
            break
        # Restrain every violator (and its twin, which the parameterisation shares):
        # halve its centre box and restart it from its placed centre; pin it only if it
        # still crosses in the final round.
        last_round = binding_round >= rounds - 2
        for index in changed:
            slot = parameters.free_index[index]
            name = parameters.free_names[slot]
            if last_round:
                frozen.add(name)
            else:
                offset_overrides[name] = 0.5 * offset_overrides.get(name, float(options.max_offset))
            flat.reshape(parameters.free_count, 4)[slot, :3] = 0.0

    fitted_centers, fitted_radii = parameters.expand_numpy(flat)
    fitted_parent, _, _ = assign_parent_joint_like_tool(fitted_centers, body.joints, body.parents)
    accepted = [
        names[index] for index in np.flatnonzero(fitted_parent != placed_parent)
        if _acceptable_binding_change(int(placed_parent[index]), int(fitted_parent[index]), body.parents)
    ]
    displacement = np.linalg.norm(fitted_centers - centers, axis=1)
    margin = binding_margin(fitted_centers, body.joints, body.parents)
    info = {
        "binding_margin_min_mm": float(margin.min() * 1000),
        "binding_fragile": [
            names[index] for index in np.flatnonzero(margin < 0.003)
        ],
        "history": history,
        "rounds": audit,
        "frozen_centers": sorted(frozen),
        "restrained_centers": {name: round(value * 1000, 2) for name, value in offset_overrides.items()},
        "binding_changed": [
            names[index] for index in np.flatnonzero(fitted_parent != placed_parent)
            if names[index] not in accepted
        ],
        "binding_accepted": accepted,
        "mean_displacement": float(displacement.mean()),
        "max_displacement": float(displacement.max()),
        "radius_scale_min": float((fitted_radii / radii).min()),
        "radius_scale_max": float((fitted_radii / radii).max()),
        "free_parameters": int(parameters.size),
        "fit_points": int(len(fit_index)),
        "poses": int(len(pose_terms)),
        "seconds": time.perf_counter() - started,
    }
    return FitResult(fitted_centers, fitted_radii, info)
