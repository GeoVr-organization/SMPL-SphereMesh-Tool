"""Measurements of an SMPL body that a sphere placement rule can be written in terms of.

A topology generator should not contain coordinates. It should contain rules like "put a
sphere at the widest inscribed point of the cross-section one third of the way down the
upper arm, with the radius that inscribed circle allows" -- so that the same rules produce
a template for a different body, a different gender or a different shape without anyone
retyping numbers. This module supplies the quantities those rules quantify over.

Two ideas carry most of the weight.

**Stations.** A limb is measured by walking its bone and taking a cross-section at each
step. ``bone_stations`` returns, per step, the section's *medial* point -- the centre of
the largest circle inscribed in the section, not the bone axis, which in a thigh or a
forearm is noticeably off-centre -- and the radius that circle allows. A chain of spheres
placed on those points with those radii is the best tapered cone approximation of the limb
available at that sphere count.

**Sheets.** A trunk is not a tube and its medial axis is not a curve; it is a sheet, wide
in x and thin in z, because the trunk cross-section is much wider than it is deep. For an
ellipse with semi-axes ``a > b`` the locus of inscribed-circle centres is a segment of
half-length ``(a^2 - b^2) / a`` along the long direction, and the inscribed radius falls
from ``b`` at the centre to zero at the ends. ``trunk_slices`` measures that locus
directly, so a generator can lay spheres along it and let the triangle slabs between them
do the front-to-back covering.

Sections are treated as convex. A horizontal cut through a human trunk, thigh or upper arm
is convex to well inside the accuracy anything here is used at, and convexity buys an exact
inscribed-circle radius as a plain minimum over the hull's half-planes rather than a
rasterised distance transform.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np

__all__ = [
    "Body",
    "Station",
    "Section",
    "JOINT_NAMES",
    "JOINT_INDEX",
    "load_body",
    "pose_body",
    "vertex_normals",
    "dominant_joint",
    "joint_region_mask",
<<<<<<< Updated upstream
    "plane_frame",
=======
>>>>>>> Stashed changes
    "section_at",
    "bone_stations",
    "trunk_slices",
    "inscribed_radius",
    "assign_parent_joint_like_tool",
    "pose_spheres_like_tool",
<<<<<<< Updated upstream
    "TRUNK_JOINTS",
    "kintree_distance",
    "joints_compatible",
=======
>>>>>>> Stashed changes
    "random_poses",
    "mocap_poses",
]

# SMPL's 24 joints. "left" is +X in this model: joint 16, L_shoulder, sits at x = +0.199.
JOINT_NAMES: tuple[str, ...] = (
    "pelvis", "L_hip", "R_hip", "spine1", "L_knee", "R_knee", "spine2", "L_ankle",
    "R_ankle", "spine3", "L_foot", "R_foot", "neck", "L_collar", "R_collar", "head",
    "L_shoulder", "R_shoulder", "L_elbow", "R_elbow", "L_wrist", "R_wrist", "L_hand",
    "R_hand",
)
JOINT_INDEX: dict[str, int] = {name: index for index, name in enumerate(JOINT_NAMES)}


@dataclass
class Body:
    """One SMPL body evaluated at a shape and a pose, with everything placement needs."""

    gender: str
    vertices: np.ndarray        # (6890, 3)
    joints: np.ndarray          # (24, 3) world positions
    rotations: np.ndarray       # (24, 3, 3) global joint rotations
    faces: np.ndarray           # (13776, 3)
    skin_weights: np.ndarray    # (6890, 24)
    parents: np.ndarray         # (24,) kintree parent index, -1 at the root

    @property
    def normals(self) -> np.ndarray:
        cached = getattr(self, "_normals", None)
        if cached is None:
            cached = vertex_normals(self.vertices, self.faces)
            object.__setattr__(self, "_normals", cached)
        return cached

<<<<<<< Updated upstream
    @property
    def vertex_tree(self):
        """A ``scipy.spatial.cKDTree`` over the vertices, built once per body."""
        cached = getattr(self, "_vertex_tree", None)
        if cached is None:
            from scipy.spatial import cKDTree

            cached = cKDTree(self.vertices)
            object.__setattr__(self, "_vertex_tree", cached)
        return cached

=======
>>>>>>> Stashed changes
    def bone(self, parent: str | int, child: str | int) -> tuple[np.ndarray, np.ndarray, float]:
        """(origin, unit axis, length) of the bone from ``parent`` to ``child``."""
        start = self.joints[_joint(parent)]
        end = self.joints[_joint(child)]
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length < 1e-9:
            raise ValueError(f"bone {parent}->{child} has no length")
        return start, direction / length, length


@dataclass
class Section:
    """A planar cut through the body, expressed in the cut's own 2D frame."""

    origin: np.ndarray          # (3,) a point on the cutting plane
    axis: np.ndarray            # (3,) unit normal of the cutting plane
    first_axis: np.ndarray      # (3,) unit, in-plane
    second_axis: np.ndarray     # (3,) unit, in-plane, completes a right-handed frame
    planar: np.ndarray          # (K, 2) the section's vertices in (first, second)
    vertex_indices: np.ndarray  # (K,) which body vertices these are

    def to_world(self, planar_points: np.ndarray) -> np.ndarray:
        planar_points = np.atleast_2d(planar_points)
        return (
            self.origin
            + planar_points[:, 0:1] * self.first_axis
            + planar_points[:, 1:2] * self.second_axis
        )

    def extent(self) -> tuple[float, float]:
        """Full width along each in-plane axis."""
        span = self.planar.max(axis=0) - self.planar.min(axis=0)
        return float(span[0]), float(span[1])

    def half_planes(self) -> tuple[np.ndarray, np.ndarray]:
        """Outward unit normals and offsets of the convex hull, as ``n . x <= offset``."""
        from scipy.spatial import ConvexHull

        hull = ConvexHull(self.planar)
        normals = hull.equations[:, :2]
        offsets = -hull.equations[:, 2]
        lengths = np.linalg.norm(normals, axis=1)
        return normals / lengths[:, None], offsets / lengths

    def inscribed_radius_at(self, planar_points: np.ndarray) -> np.ndarray:
        """Largest circle centred at each point that stays inside the section outline.

        Negative where the point is outside the outline.
        """
        normals, offsets = self.half_planes()
        planar_points = np.atleast_2d(planar_points)
        return (offsets[None, :] - planar_points @ normals.T).min(axis=1)

    def best_transverse(self, along: np.ndarray, samples: int = 601) -> tuple[np.ndarray, np.ndarray]:
        """For each position along the long axis, the transverse position of deepest inset.

        Returns ``(planar_points, radii)``. This is the deterministic sibling of
        ``medial_axis``: the caller says *where* across the body it wants a sphere and this
        answers how deep to put it and how big it can be, instead of the caller reading a
        position off a ridge that wanders.

        Deterministic matters here. A sheet whose column positions are chosen by the caller
        and whose depths are solved for is exactly mirror-symmetric when the section is;
        one that follows a sampled ridge inherits the ridge's sampling noise, and a trunk
        built that way comes out a centimetre off-centre.
        """
        normals, offsets = self.half_planes()
        long_axis = 0 if np.ptp(self.planar[:, 0]) >= np.ptp(self.planar[:, 1]) else 1
        short_axis = 1 - long_axis
        across = np.linspace(
            self.planar[:, short_axis].min(), self.planar[:, short_axis].max(), samples
        )
        along = np.atleast_1d(np.asarray(along, dtype=np.float64))

        grid = np.empty((len(along), samples, 2))
        grid[..., long_axis] = along[:, None]
        grid[..., short_axis] = across[None, :]
        clearance = (
            offsets[None, :] - grid.reshape(-1, 2) @ normals.T
        ).min(axis=1).reshape(len(along), samples)

        best = clearance.argmax(axis=1)
        points = np.empty((len(along), 2))
        points[:, long_axis] = along
        points[:, short_axis] = across[best]
        return points, clearance[np.arange(len(along)), best]

    def medial_axis(
        self, samples: int = 401, *, maximal_only: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """The section's medial locus and the inscribed radius along it.

        Returns ``(planar_points, radii)`` sampled along the section's long in-plane
        direction: for each position along that direction, the transverse position that
        maximising the inscribed radius.

        ``maximal_only`` keeps only circles that are *maximal* -- not contained in another
        circle of the family -- and it is on by default because without it this returns the
        symmetry segment rather than the medial axis. The difference is not academic. For an
        ellipse with semi-axes ``a > b`` the ridge of the clearance function runs the full
        width ``+-a``, but the medial axis stops at ``+-(a^2 - b^2)/a``, where the inscribed
        circle has radius ``b^2/a`` and osculates the end of the ellipse. Every circle past
        that point is swallowed by the one at the end. On the male trunk that is the
        difference between a medial segment of +-70 mm and a spurious one of +-170 mm -- and
        spheres placed on the spurious part are contained in their neighbours, which the
        validator correctly reports as dead bumpers.
        """
        normals, offsets = self.half_planes()
        long_axis = 0 if np.ptp(self.planar[:, 0]) >= np.ptp(self.planar[:, 1]) else 1
        short_axis = 1 - long_axis

        along = np.linspace(
            self.planar[:, long_axis].min(), self.planar[:, long_axis].max(), samples
        )
        across = np.linspace(
            self.planar[:, short_axis].min(), self.planar[:, short_axis].max(), samples
        )
        grid = np.empty((samples, samples, 2))
        grid[..., long_axis] = along[:, None]
        grid[..., short_axis] = across[None, :]
        clearance = (
            offsets[None, :] - grid.reshape(-1, 2) @ normals.T
        ).min(axis=1).reshape(samples, samples)

        best = clearance.argmax(axis=1)
        radii = clearance[np.arange(samples), best]
        ridge = np.empty((samples, 2))
        ridge[:, long_axis] = along
        ridge[:, short_axis] = across[best]
        keep = radii > 0.0
        ridge, radii = ridge[keep], radii[keep]
        if not maximal_only or len(ridge) == 0:
            return ridge, radii

        # A circle is medial when it touches the outline at two places that are genuinely
        # on opposite sides -- that is the definition, and it is what a containment test
        # only approximates. Near the end of an elongated section the containment is
        # tangent to within rounding, so the test decides by noise; the contact-angle test
        # decides by geometry. Contacts are the half-planes whose clearance ties with the
        # minimum, and "opposite sides" is their normals differing by more than 50 degrees.
        clearance = offsets[None, :] - ridge @ normals.T                 # (N, H)
        tolerance = 0.03 * radii[:, None] + 5e-4
        touching = clearance <= radii[:, None] + tolerance
        alignment = normals @ normals.T                                  # (H, H)
        pairwise = np.where(
            touching[:, :, None] & touching[:, None, :], alignment[None, :, :], 1.0
        )
        opposed = pairwise.min(axis=(1, 2)) < np.cos(np.deg2rad(50.0))
        if not opposed.any():
            return ridge, radii
        return ridge[opposed], radii[opposed]


@dataclass
class Station:
    """One cross-section of a bone, reduced to what a sphere needs."""

    parameter: float            # 0 at the parent joint, 1 at the child joint
    axis_point: np.ndarray      # (3,) where the bone axis crosses the section
    medial_point: np.ndarray    # (3,) centre of the section's largest inscribed circle
    inscribed_radius: float     # that circle's radius, in the section plane
    touch_radius: float         # distance from medial_point to the nearest body vertex
    section_extent: tuple[float, float]
    vertex_count: int

    @property
    def aspect(self) -> float:
        """Long over short extent of the section: 1 is circular."""
        wide, deep = self.section_extent
        return float(max(wide, deep) / max(min(wide, deep), 1e-12))


def _joint(name: str | int) -> int:
    return JOINT_INDEX[name] if isinstance(name, str) else int(name)


# ------------------------------------------------------------------------ loading


def load_body(gender: str = "male", beta=None, theta=None) -> Body:
    """Evaluate an SMPL body. Defaults are the neutral shape in T-pose."""
    import torch as tc

    from src.smpl_forward_compat import my_function_forward
    from src.smpl_models import load_smpl_model, smpl_model_path

    model = load_smpl_model(gender)
    beta_tensor = (
        tc.zeros(10, dtype=tc.float64) if beta is None
        else tc.as_tensor(np.asarray(beta, dtype=np.float64).reshape(-1))
    )
    theta_tensor = (
        tc.zeros(72, dtype=tc.float64) if theta is None
        else tc.as_tensor(np.asarray(theta, dtype=np.float64).reshape(-1))
    )
    vertices, joints, rotations, _ = my_function_forward(model, beta_tensor, theta_tensor)
    return Body(
        gender=gender,
        vertices=vertices.detach().cpu().numpy().astype(np.float64),
        joints=joints.detach().cpu().numpy().astype(np.float64),
        rotations=rotations.detach().cpu().numpy().astype(np.float64),
        faces=np.asarray(model.faces, dtype=np.int64),
        skin_weights=_skin_weights(gender),
        parents=np.asarray(model.parents, dtype=np.int64),
    )


def pose_body(gender: str, theta, beta=None) -> Body:
    """The same body under a pose. Thin alias that reads better at call sites."""
    return load_body(gender=gender, beta=beta, theta=theta)


@lru_cache(maxsize=4)
def _skin_weights(gender: str) -> np.ndarray:
    from src.smpl_models import smpl_model_path

    parameters = np.load(smpl_model_path(gender), allow_pickle=True)
    return np.asarray(parameters["weights"], dtype=np.float64)


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals, outward for SMPL's face winding."""
    normals = np.zeros_like(vertices)
    triangle = vertices[faces]
    face_normal = np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normal)
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)


# ------------------------------------------------------------------- segmentation


def dominant_joint(body: Body) -> tuple[np.ndarray, np.ndarray]:
    """Per-vertex (joint with the largest skinning weight, that weight)."""
    return np.argmax(body.skin_weights, axis=1), np.max(body.skin_weights, axis=1)


def joint_region_mask(
    body: Body, joints: Sequence[str | int], *, weight_min: float = 0.5
) -> np.ndarray:
    """Vertices whose skinning weight on the given joints sums to at least ``weight_min``.

    Summing over a set rather than testing the argmax matters at the seams: a vertex on the
    lower ribs may split its weight between spine2 and spine3 and be dominated by neither,
    yet it plainly belongs to the trunk.
    """
    columns = [_joint(name) for name in joints]
    return body.skin_weights[:, columns].sum(axis=1) >= float(weight_min)


def inscribed_radius(body: Body, centers: np.ndarray) -> np.ndarray:
    """Distance from each centre to the nearest body vertex.

    This is the radius that makes a sphere touch the surface without crossing it, and it is
    deliberately the same quantity ``SphereManager._compute_touch_radius_and_nearest_vertex``
    uses, so a template generated here and one grown in the tool agree.
    """
    centers = np.atleast_2d(np.asarray(centers, dtype=np.float64))
<<<<<<< Updated upstream
    distances, _ = body.vertex_tree.query(centers)
    return np.asarray(distances, dtype=np.float64).reshape(len(centers))
=======
    radius = np.empty(len(centers))
    for index, center in enumerate(centers):
        radius[index] = np.min(np.linalg.norm(body.vertices - center, axis=1))
    return radius
>>>>>>> Stashed changes


# ---------------------------------------------------------------------- sections


<<<<<<< Updated upstream
def plane_frame(
    axis: np.ndarray, reference_up: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(axis, first_axis, second_axis)``: the deterministic in-plane frame of a section.

    The first in-plane axis is ``reference_up`` (world +X by default, +Y when that is
    nearly parallel to the normal) projected into the plane; the second completes a
    right-handed frame. Shared by the vertex-band and the mesh-cut sections so that the
    two are comparable coordinate for coordinate.
    """
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    reference = np.array([1.0, 0.0, 0.0]) if reference_up is None else np.asarray(
        reference_up, dtype=np.float64
    )
    if abs(float(reference @ axis)) > 0.95:
        reference = np.array([0.0, 1.0, 0.0])
    first_axis = reference - float(reference @ axis) * axis
    first_axis /= np.linalg.norm(first_axis)
    second_axis = np.cross(axis, first_axis)
    return axis, first_axis, second_axis


=======
>>>>>>> Stashed changes
def section_at(
    body: Body,
    origin: np.ndarray,
    axis: np.ndarray,
    *,
    half_thickness: float = 0.012,
    vertex_mask: np.ndarray | None = None,
    reference_up: np.ndarray | None = None,
    symmetrize: bool = False,
<<<<<<< Updated upstream
    method: str = "band",
    min_points: int = 12,
) -> Section:
    """A cross-section of the body through ``origin`` normal to ``axis``.

    ``method="band"`` (the default) takes the vertices within ``half_thickness`` of the
    plane, which is cheap but sparse and, through both thighs, returns both thighs.
    ``method="cut"`` intersects the mesh with the plane (``sections.plane_cut``) and keeps
    the loop that ``vertex_mask`` prefers and that contains ``origin``; ``half_thickness``
    is then unused.
=======
) -> Section:
    """Vertices within ``half_thickness`` of the plane through ``origin`` normal to ``axis``.
>>>>>>> Stashed changes

    The in-plane frame is chosen deterministically so that repeated calls along a bone give
    comparable coordinates: the first in-plane axis is ``reference_up`` (world +X by
    default) projected into the plane.
    """
<<<<<<< Updated upstream
    if method == "cut":
        section = _cut_section(
            body, origin, axis, vertex_mask=vertex_mask, reference_up=reference_up,
            symmetrize=symmetrize,
        )
        # Where the trimmed cut is too sparse to hull -- a station right at a joint on a
        # slimmer body, where the mask keeps only a sliver of the loop -- the vertex band
        # is the better measurement, so degrade to it rather than fail the station.
        if len(section.planar) >= int(min_points):
            return section
        method = "band"
    if method != "band":
        raise ValueError(f"section_at: method must be 'band' or 'cut', not {method!r}")
    axis, first_axis, second_axis = plane_frame(axis, reference_up)
=======
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    reference = np.array([1.0, 0.0, 0.0]) if reference_up is None else np.asarray(
        reference_up, dtype=np.float64
    )
    if abs(float(reference @ axis)) > 0.95:
        reference = np.array([0.0, 1.0, 0.0])
    first_axis = reference - float(reference @ axis) * axis
    first_axis /= np.linalg.norm(first_axis)
    second_axis = np.cross(axis, first_axis)
>>>>>>> Stashed changes

    candidates = (
        np.arange(len(body.vertices)) if vertex_mask is None
        else np.flatnonzero(vertex_mask)
    )
    offset = body.vertices[candidates] - origin
    within = np.abs(offset @ axis) <= float(half_thickness)
    selected = candidates[within]
    local = body.vertices[selected] - origin
    planar = np.stack([local @ first_axis, local @ second_axis], axis=1)
    if symmetrize:
        # The body is bilaterally symmetric but its vertex sampling is not exactly, and a
        # section built from raw vertices comes out a few millimetres lopsided. Unioning
        # the section with its own reflection makes the outline exactly symmetric, so a
        # sheet placed at +-x lands on true mirror positions with equal radii and the
        # generated template passes a strict mirror check instead of nearly passing it.
        flip = np.array([-1.0, 1.0, 1.0])
        reflected_local = (body.vertices[selected] * flip) - origin
        reflected = np.stack(
            [reflected_local @ first_axis, reflected_local @ second_axis], axis=1
        )
        planar = np.concatenate([planar, reflected], axis=0)
        selected = np.concatenate([selected, selected])
    return Section(
        origin=np.asarray(origin, dtype=np.float64),
        axis=axis,
        first_axis=first_axis,
        second_axis=second_axis,
        planar=planar,
        vertex_indices=selected,
    )


<<<<<<< Updated upstream
def _cut_section(
    body: Body,
    origin: np.ndarray,
    axis: np.ndarray,
    *,
    vertex_mask: np.ndarray | None,
    reference_up: np.ndarray | None,
    symmetrize: bool,
) -> Section:
    """The mesh-plane cut, reduced to the part of the loop the caller's mask selects.

    A horizontal plane at shoulder height passes through the trunk *and* both arms of a
    T-posed body as one connected loop 1.3 m wide, and a plane across the top of the thigh
    passes through the pelvis as well. Choosing a loop cannot separate those; the loop has
    to be *trimmed* to the cut points whose nearer mesh vertex the mask accepts, which is
    exactly the population the vertex band used. Measured on the generator's own rows
    (review of the first version): trimming recovers the band's trunk width (377 mm) and
    hip-sphere position (x = +102 mm, r = 70 mm) where the untrimmed loop gave 1316 mm and
    a 136 mm sphere on the midline.

    Where the band degrades to an empty section -- a plane that misses the body, or grazes
    it at a vertex -- this does too, so ``bone_stations`` skips the station instead of
    raising.
    """
    from .sections import CutLoop, loop_section, plane_cut, select_loop

    origin = np.asarray(origin, dtype=np.float64)
    unit_axis, first_axis, second_axis = plane_frame(axis, reference_up)
    empty = Section(
        origin=origin, axis=unit_axis, first_axis=first_axis, second_axis=second_axis,
        planar=np.zeros((0, 2)), vertex_indices=np.zeros(0, dtype=np.int64),
    )
    loops = [loop for loop in plane_cut(body.vertices, body.faces, origin, axis) if len(loop) >= 3]
    if not loops:
        return empty
    try:
        loop = select_loop(loops, near=origin, prefer_mask=vertex_mask)
    except ValueError:
        return empty
    if vertex_mask is not None:
        keep = np.asarray(vertex_mask, dtype=bool)[loop.nearer_vertices()]
        if keep.sum() < 3:
            return empty
        loop = CutLoop(
            points=loop.points[keep], edge_vertices=loop.edge_vertices[keep],
            edge_parameters=loop.edge_parameters[keep], closed=bool(keep.all() and loop.closed),
        )
    if np.ptp(loop.points, axis=0).max() < 1e-6:
        return empty
    return loop_section(loop, origin, axis, reference_up=reference_up, symmetrize=symmetrize)


=======
>>>>>>> Stashed changes
def bone_stations(
    body: Body,
    parent: str | int,
    child: str | int,
    *,
    parameters: Sequence[float] | None = None,
    vertex_mask: np.ndarray | None = None,
    half_thickness: float | None = None,
    min_vertices: int = 12,
<<<<<<< Updated upstream
    method: str = "band",
=======
>>>>>>> Stashed changes
) -> list[Station]:
    """Cross-sections along a bone, reduced to a medial point and a radius each.

    ``parameters`` are positions along the bone, 0 at the parent joint and 1 at the child.
    Values outside [0, 1] are allowed and useful: a shoulder chain often wants to start
    slightly *before* the shoulder joint, inside the deltoid.

    The returned ``medial_point`` is the centre of the section's largest inscribed circle,
    not the point where the bone axis crosses it. On a thigh those differ by well over a
    centimetre, and a chain of spheres on the bone axis inherits that error as a systematic
    protrusion on one side and a gap on the other.
<<<<<<< Updated upstream

    ``method`` is passed to ``section_at``; with ``"cut"`` the ``min_vertices`` floor
    applies to the cut loop's point count.
=======
>>>>>>> Stashed changes
    """
    origin, axis, length = body.bone(parent, child)
    if parameters is None:
        parameters = np.linspace(0.0, 1.0, 6)
    if half_thickness is None:
        half_thickness = max(0.008, 0.06 * length)

    stations: list[Station] = []
    for parameter in parameters:
        axis_point = origin + float(parameter) * length * axis
        section = section_at(
            body, axis_point, axis,
<<<<<<< Updated upstream
            half_thickness=half_thickness, vertex_mask=vertex_mask, method=method,
=======
            half_thickness=half_thickness, vertex_mask=vertex_mask,
>>>>>>> Stashed changes
        )
        if len(section.planar) < min_vertices:
            continue
        ridge, radii = section.medial_axis()
        if len(ridge) == 0:
            continue
        best = int(np.argmax(radii))
        medial_point = section.to_world(ridge[best])[0]
        stations.append(
            Station(
                parameter=float(parameter),
                axis_point=axis_point,
                medial_point=medial_point,
                inscribed_radius=float(radii[best]),
                touch_radius=float(inscribed_radius(body, medial_point[None])[0]),
                section_extent=section.extent(),
                vertex_count=len(section.planar),
            )
        )
    return stations


def trunk_slices(
    body: Body,
    heights: Sequence[float],
    *,
    vertex_mask: np.ndarray | None = None,
    half_thickness: float = 0.018,
<<<<<<< Updated upstream
    method: str = "band",
=======
>>>>>>> Stashed changes
) -> list[tuple[float, Section, np.ndarray, np.ndarray]]:
    """Horizontal cuts through the trunk with their medial locus.

    Returns ``(height, section, medial_points_world, medial_radii)`` per cut. The medial
    locus is the segment a sheet of spheres should be laid along; its radii are the
    half-depths those spheres should carry, which is what makes the slab between two of them
    thicken the trunk front to back without any sphere ever poking through the skin.
    """
    up = np.array([0.0, 1.0, 0.0])
    result = []
    for height in heights:
        origin = np.array([0.0, float(height), 0.0])
        section = section_at(
            body, origin, up,
            half_thickness=half_thickness, vertex_mask=vertex_mask,
<<<<<<< Updated upstream
            reference_up=np.array([1.0, 0.0, 0.0]), method=method,
=======
            reference_up=np.array([1.0, 0.0, 0.0]),
>>>>>>> Stashed changes
        )
        if len(section.planar) < 12:
            continue
        ridge, radii = section.medial_axis()
        if len(ridge) == 0:
            continue
        result.append((float(height), section, section.to_world(ridge), radii))
    return result


# --------------------------------------------------------------- tool-compatible LBS


<<<<<<< Updated upstream
#: Joints of the trunk. The spine joints are nearly dead (no vertex weights spine1 above
#: 0.73), the collars barely rotate, and the skin between them blends smoothly, so a cone
#: spanning two of them within two kintree steps behaves like one inside a bone. A limb
#: joint is a hinge and gets no such allowance.
TRUNK_JOINTS: frozenset[int] = frozenset(
    JOINT_INDEX[name]
    for name in ("pelvis", "spine1", "spine2", "spine3", "neck", "L_collar", "R_collar")
)


def kintree_distance(first_joint: int, second_joint: int, parents: np.ndarray) -> int:
    """Number of bones between two joints along the kinematic tree."""
    def ancestors(joint: int) -> list[int]:
        chain = [int(joint)]
        while parents[chain[-1]] >= 0:
            chain.append(int(parents[chain[-1]]))
        return chain

    first_chain, second_chain = ancestors(first_joint), ancestors(second_joint)
    common = next(joint for joint in first_chain if joint in second_chain)
    return first_chain.index(common) + second_chain.index(common)


def joints_compatible(first_joint: int, second_joint: int, parents: np.ndarray) -> bool:
    """A cone may join two spheres only when rigid binding keeps it inside one bone.

    Same joint: both ends move together. Parent and child: the cone is the seam between
    two chains and spans one joint, which is unavoidable and cheap while it is short. On
    the trunk, two steps apart is allowed as well (``TRUNK_JOINTS``). Anything else spans
    joints that rotate independently and will pop out.
    """
    first_joint, second_joint = int(first_joint), int(second_joint)
    if first_joint == second_joint:
        return True
    distance = kintree_distance(first_joint, second_joint, parents)
    if distance == 1:
        return True
    return distance == 2 and first_joint in TRUNK_JOINTS and second_joint in TRUNK_JOINTS


=======
>>>>>>> Stashed changes
def assign_parent_joint_like_tool(
    centers: np.ndarray, rest_joints: np.ndarray, parents: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce ``SphereManager.assign_parent_joint``, vectorised over spheres.

    The tool finds the kintree bone whose segment is closest to the sphere centre and binds
    the sphere to that bone's **parent** joint. Any generator that wants to predict how its
    spheres will move under animation has to use this rule and not a skinning-weight
    argmax, because this is the rule the tool will actually apply on load.

    Returns ``(parent_joint, child_joint, distance_to_that_bone)``. The child end is what
    the tool records as the *secondary* joint, and the position of the sphere along that
    bone is what it turns into the LBS blend weight -- so a caller writing a template has to
    know it too, even though the blend itself is currently disabled in the tool.
    """
    centers = np.atleast_2d(np.asarray(centers, dtype=np.float64))
    best_parent = np.zeros(len(centers), dtype=np.int64)
    best_child = np.zeros(len(centers), dtype=np.int64)
    best_distance = np.full(len(centers), np.inf)
    for child in range(1, len(parents)):
        parent = int(parents[child])
        if parent < 0:
            continue
        start = rest_joints[parent]
        end = rest_joints[child]
        along = end - start
        length_squared = float(along @ along)
        if length_squared < 1e-12:
            distance = np.linalg.norm(centers - start, axis=1)
        else:
            parameter = np.clip((centers - start) @ along / length_squared, 0.0, 1.0)
            distance = np.linalg.norm(centers - (start + parameter[:, None] * along), axis=1)
        closer = distance < best_distance
        best_distance[closer] = distance[closer]
        best_parent[closer] = parent
        best_child[closer] = child
    return best_parent, best_child, best_distance


def pose_spheres_like_tool(
    rest_centers: np.ndarray,
    parent_joints: np.ndarray,
    rest_joints: np.ndarray,
    posed_joints: np.ndarray,
    posed_rotations: np.ndarray,
) -> np.ndarray:
    """Move sphere centres exactly as ``SphereManager.update_sphere_positions`` does.

    ``centre' = joints'[p] + R'[p] @ (centre - joints_rest[p])`` -- rigid and single-joint.
    The secondary-joint blend is commented out in the tool, so it is absent here too: a
    validator that blended would report artifacts the user will never see and miss the ones
    they will.
    """
    rest_centers = np.atleast_2d(np.asarray(rest_centers, dtype=np.float64))
    parent_joints = np.asarray(parent_joints, dtype=np.int64)
    offsets = rest_centers - rest_joints[parent_joints]
    return posed_joints[parent_joints] + np.einsum(
        "nij,nj->ni", posed_rotations[parent_joints], offsets
    )


# ------------------------------------------------------------------------- pose sets


def random_poses() -> np.ndarray:
    """(24, 72) thetas from ``data/random_pose.csv``, in SMPL axis order.

    The CSV's header names every axis, and its column order is already the SMPL one, so the
    rows are usable as-is. These are independent draws rather than a motion, which makes
    them a wide but incoherent sample -- good for finding a bumper that breaks under some
    joint combination, poor evidence about what happens during an actual movement.
    """
    import csv

    from src.app_paths import data_file

    with open(data_file("random_pose.csv"), newline="") as handle:
        rows = list(csv.reader(handle))
    return np.array([[float(value) for value in row] for row in rows[1:]], dtype=np.float64)


def mocap_poses(path=None, stride: int = 5) -> np.ndarray:
    """(F, 72) thetas from a captured take, every ``stride``-th frame.

    Complements ``random_poses``: a real take visits poses that actually occur together,
    which is where a topology that survives every joint moved alone still fails.
    """
    import pickle

    source = Path(path) if path is not None else (
        Path(__file__).resolve().parent.parent.parent / "00163_take2_fps30.pkl"
    )
    with open(source, "rb") as handle:
        data = pickle.load(handle, encoding="latin1")
    poses = np.asarray(data["poses"], dtype=np.float64)
    return poses[::int(stride)].reshape(-1, poses.shape[1] * poses.shape[2])
