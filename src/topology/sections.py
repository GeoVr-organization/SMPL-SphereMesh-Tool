"""True mesh-plane cross-sections, as a replacement for the vertex band in ``anatomy``.

``anatomy.section_at`` collects the vertices that lie within a slab around the cutting plane.
That is cheap and it was enough to get a first generator running, but it is not a
cross-section: a 12 mm slab through the SMPL trunk catches only 25-42 of the vertices along
a perimeter of about a metre, and at y = -0.05 it under-reports the trunk's depth by
45 mm because no vertex happens to sit at the front or the back of the belly inside the
slab. Worse, a slab through both thighs returns both thighs as one point cloud, and the
convex hull that every measurement here is built on then bridges the crotch and puts an
inscribed circle in mid-air.

Cutting the watertight mesh by the plane fixes both. Every face with vertices on both sides
of the plane contributes one segment between its two edge crossings, and chaining those
segments by the edge they share gives one closed polygon per connected piece of the cut --
a few hundred points around the trunk instead of a few dozen, and the two thighs as two
loops that a caller can tell apart. Skinning weights are interpolated along the cut edges,
so the loop can still be attributed to joints the way a vertex band was.

The output is deliberately the same ``anatomy.Section`` the rest of the code consumes, in
the same in-plane frame, so a generator can switch between the two methods with a keyword
and compare.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .anatomy import Section, plane_frame

__all__ = [
    "CutLoop",
    "plane_cut",
    "interpolate_weights",
    "loop_section",
    "select_loop",
    "loop_perimeter",
]


@dataclass
class CutLoop:
    """One closed polygon of a mesh-plane cut.

    Every point lies on a mesh edge; the edge and the position along it are kept so that
    per-vertex attributes (skinning weights above all) can be interpolated to the point.
    """

    points: np.ndarray            # (K, 3) world coordinates, ordered around the loop
    edge_vertices: np.ndarray     # (K, 2) the mesh edge each point lies on, as vertex indices
    edge_parameters: np.ndarray   # (K,) position along that edge, 0 at edge_vertices[:, 0]
    closed: bool                  # False only for a cut through a mesh with a boundary

    def __len__(self) -> int:
        return len(self.points)

    def nearer_vertices(self) -> np.ndarray:
        """(K,) the endpoint of each cut edge that the cut point lies closer to."""
        return np.where(
            self.edge_parameters < 0.5, self.edge_vertices[:, 0], self.edge_vertices[:, 1]
        )

    def centroid(self) -> np.ndarray:
        return self.points.mean(axis=0)


def loop_perimeter(loop: CutLoop) -> float:
    """Length of the polygon, closing it if it is closed."""
    if len(loop.points) < 2:
        return 0.0
    steps = np.diff(loop.points, axis=0)
    length = float(np.linalg.norm(steps, axis=1).sum())
    if loop.closed:
        length += float(np.linalg.norm(loop.points[-1] - loop.points[0]))
    return length


# ------------------------------------------------------------------------ the cut


def plane_cut(
    vertices: np.ndarray,
    faces: np.ndarray,
    origin: np.ndarray,
    axis: np.ndarray,
    *,
    require_closed: bool = True,
) -> list[CutLoop]:
    """Cut a triangle mesh by the plane through ``origin`` with normal ``axis``.

    Returns one ``CutLoop`` per connected component of the intersection curve. On a
    watertight mesh every loop is closed; with ``require_closed`` (the default) an open
    chain raises with a count of what went wrong, since it means the mesh or the cut is
    not what the caller believes it is.

    Exact zeros of the signed distance are nudged by +1e-12 so that every crossing is an
    edge crossing and no face has a vertex *on* the plane. That is the whole reason the
    chaining below can be a plain degree-two walk: with no vertices on the plane, each
    cut edge is shared by exactly the two faces adjacent to that edge, so each cut point
    has exactly two segment neighbours.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    origin = np.asarray(origin, dtype=np.float64)

    signedDistance = (vertices - origin) @ axis
    signedDistance = np.where(signedDistance == 0.0, 1e-12, signedDistance)
    positive = signedDistance > 0.0

    # Edges of every face, in the face's own order: (0,1), (1,2), (2,0).
    cornerA = faces
    cornerB = np.roll(faces, -1, axis=1)
    crossing = positive[cornerA] != positive[cornerB]                  # (F, 3)
    mixedFaces = np.flatnonzero(crossing.sum(axis=1) == 2)
    if len(mixedFaces) == 0:
        return []

    # The two crossing edges of each mixed face, as sorted vertex pairs so that the same
    # mesh edge seen from its two faces gets the same key.
    crossingEdgesA = cornerA[mixedFaces][crossing[mixedFaces]].reshape(-1, 2)   # (M, 2)
    crossingEdgesB = cornerB[mixedFaces][crossing[mixedFaces]].reshape(-1, 2)
    edgeLow = np.minimum(crossingEdgesA, crossingEdgesB)
    edgeHigh = np.maximum(crossingEdgesA, crossingEdgesB)
    edgeKeys = np.stack([edgeLow.ravel(), edgeHigh.ravel()], axis=1)           # (2M, 2)
    uniqueEdges, segmentEnds = np.unique(edgeKeys, axis=0, return_inverse=True)
    segmentEnds = segmentEnds.reshape(-1, 2)                                   # (M, 2)

    # Intersection of each unique edge with the plane, parametrised from the low vertex.
    distanceLow = signedDistance[uniqueEdges[:, 0]]
    distanceHigh = signedDistance[uniqueEdges[:, 1]]
    parameters = distanceLow / (distanceLow - distanceHigh)
    cutPoints = (
        vertices[uniqueEdges[:, 0]]
        + parameters[:, None] * (vertices[uniqueEdges[:, 1]] - vertices[uniqueEdges[:, 0]])
    )

    # Chain the segments. Each cut point should have exactly two neighbours.
    pointCount = len(uniqueEdges)
    neighbours: list[list[int]] = [[] for _ in range(pointCount)]
    for first, second in segmentEnds:
        neighbours[int(first)].append(int(second))
        neighbours[int(second)].append(int(first))

    visited = np.zeros(pointCount, dtype=bool)
    loops: list[CutLoop] = []
    openLoops = 0
    for startPoint in range(pointCount):
        if visited[startPoint]:
            continue
        visited[startPoint] = True
        forward, closed = _walk(neighbours, visited, startPoint)
        order = [startPoint] + forward
        if not closed:
            # An open chain (a mesh with a boundary): walk the other way from the start
            # too, so the chain is still reported whole.
            backward, _ = _walk(neighbours, visited, startPoint)
            order = list(reversed(backward)) + order
            openLoops += 1
        orderArray = np.asarray(order, dtype=np.int64)
        loops.append(
            CutLoop(
                points=cutPoints[orderArray],
                edge_vertices=uniqueEdges[orderArray],
                edge_parameters=parameters[orderArray],
                closed=closed,
            )
        )

    if require_closed and openLoops:
        raise AssertionError(
            f"plane_cut: {openLoops} of {len(loops)} loops did not close "
            f"({len(mixedFaces)} cut faces, {pointCount} cut points); the mesh is not "
            "watertight along the cut"
        )
    return loops


def _walk(neighbours: list[list[int]], visited: np.ndarray, startPoint: int) -> tuple[list[int], bool]:
    """Follow unvisited neighbours from ``startPoint``; True if the walk returned to it."""
    order: list[int] = []
    previous, current = -1, startPoint
    while True:
        following = -1
        for candidate in neighbours[current]:
            if candidate == startPoint and previous != -1 and candidate != previous:
                return order, True
            if not visited[candidate]:
                following = candidate
                break
        if following < 0:
            return order, False
        order.append(following)
        visited[following] = True
        previous, current = current, following


def interpolate_weights(loop: CutLoop, skin_weights: np.ndarray) -> np.ndarray:
    """(K, J) per-vertex attributes interpolated linearly along each cut edge.

    With skinning weights this keeps every row summing to one, so the loop can be filtered
    or attributed by joint exactly as a vertex band was.
    """
    skin_weights = np.asarray(skin_weights, dtype=np.float64)
    parameter = loop.edge_parameters[:, None]
    return (
        (1.0 - parameter) * skin_weights[loop.edge_vertices[:, 0]]
        + parameter * skin_weights[loop.edge_vertices[:, 1]]
    )


# ------------------------------------------------------------------- to a Section


def loop_section(
    loop: CutLoop,
    origin: np.ndarray,
    axis: np.ndarray,
    *,
    reference_up: np.ndarray | None = None,
    symmetrize: bool = False,
) -> Section:
    """Express a cut loop as an ``anatomy.Section`` in the frame ``section_at`` would use.

    ``vertex_indices`` are the nearer endpoints of the cut edges, so callers that look a
    section's points up in the body (for masks, normals, joint weights) keep working.
    ``symmetrize`` unions the loop with its own x-reflection, for the same reason
    ``section_at`` offers it: an exactly symmetric outline gives exactly mirrored sheets.
    """
    origin = np.asarray(origin, dtype=np.float64)
    axis, first_axis, second_axis = plane_frame(axis, reference_up)

    local = loop.points - origin
    planar = np.stack([local @ first_axis, local @ second_axis], axis=1)
    vertex_indices = loop.nearer_vertices()
    if symmetrize:
        flip = np.array([-1.0, 1.0, 1.0])
        reflected_local = (loop.points * flip) - origin
        reflected = np.stack(
            [reflected_local @ first_axis, reflected_local @ second_axis], axis=1
        )
        planar = np.concatenate([planar, reflected], axis=0)
        vertex_indices = np.concatenate([vertex_indices, vertex_indices])
    return Section(
        origin=origin,
        axis=axis,
        first_axis=first_axis,
        second_axis=second_axis,
        planar=planar,
        vertex_indices=vertex_indices,
    )


# ----------------------------------------------------------------- loop selection


def _loop_frame(loop: CutLoop) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """An orthonormal in-plane frame of a planar loop, from the loop's own geometry."""
    centroid = loop.centroid()
    relative = loop.points - centroid
    # Newell's normal: robust for any polygon, exact for a planar one.
    following = np.roll(relative, -1, axis=0)
    normal = np.cross(relative, following).sum(axis=0)
    normalLength = np.linalg.norm(normal)
    if normalLength < 1e-18:
        raise ValueError("select_loop: a loop with no area cannot be tested for containment")
    normal /= normalLength
    firstAxis = relative[np.argmax(np.linalg.norm(relative, axis=1))]
    firstAxis = firstAxis - float(firstAxis @ normal) * normal
    firstAxis /= np.linalg.norm(firstAxis)
    secondAxis = np.cross(normal, firstAxis)
    return centroid, firstAxis, secondAxis


def _contains(loop: CutLoop, point: np.ndarray) -> bool:
    """Even-odd point-in-polygon test in the loop's plane."""
    centroid, firstAxis, secondAxis = _loop_frame(loop)
    relative = loop.points - centroid
    polygon = np.stack([relative @ firstAxis, relative @ secondAxis], axis=1)
    queryX, queryY = float((point - centroid) @ firstAxis), float((point - centroid) @ secondAxis)
    startX, startY = polygon[:, 0], polygon[:, 1]
    endX, endY = np.roll(startX, -1), np.roll(startY, -1)
    straddles = (startY > queryY) != (endY > queryY)
    with np.errstate(divide="ignore", invalid="ignore"):
        crossingX = startX + (queryY - startY) * (endX - startX) / (endY - startY)
    crossings = np.count_nonzero(straddles & (queryX < crossingX))
    return bool(crossings % 2 == 1)


def select_loop(
    loops: list[CutLoop],
    *,
    near: np.ndarray | None = None,
    side: str | None = None,
    prefer_mask: np.ndarray | None = None,
    vertices: np.ndarray | None = None,
) -> CutLoop:
    """Pick the loop a caller means, out of everything a plane cut through the body hit.

    Filters apply in order. ``side`` keeps loops by the sign of their mean x ("left" is
    +X in SMPL); ``prefer_mask`` (a boolean per body vertex) keeps the loop(s) whose
    nearer-endpoint vertices have the highest mask fraction; ``near`` then prefers the loop
    whose polygon contains the in-plane projection of the point, falling back to the loop
    whose centroid is closest. Ties are broken deterministically in favour of the largest
    loop (by perimeter). ``vertices`` is accepted for callers that have the body at hand
    but is not needed: a loop carries its own points and edge indices.
    """
    if not loops:
        raise ValueError("select_loop: the cut produced no loops")
    ordered = sorted(loops, key=loop_perimeter, reverse=True)

    if side is not None:
        if side == "left":
            ordered = [loop for loop in ordered if loop.points[:, 0].mean() > 0.0]
        elif side == "right":
            ordered = [loop for loop in ordered if loop.points[:, 0].mean() < 0.0]
        else:
            raise ValueError(f"select_loop: side must be 'left' or 'right', not {side!r}")
        if not ordered:
            raise ValueError(f"select_loop: no loop on the {side} side")

    if prefer_mask is not None:
        prefer_mask = np.asarray(prefer_mask, dtype=bool)
        fractions = np.array(
            [float(prefer_mask[loop.nearer_vertices()].mean()) for loop in ordered]
        )
        best = fractions.max()
        ordered = [
            loop for loop, fraction in zip(ordered, fractions) if fraction >= best - 1e-9
        ]

    if near is not None and len(ordered) > 1:
        near = np.asarray(near, dtype=np.float64)
        containing = [loop for loop in ordered if _contains(loop, near)]
        if containing:
            return containing[0]
        distances = np.array([np.linalg.norm(loop.centroid() - near) for loop in ordered])
        return ordered[int(np.argmin(distances))]

    return ordered[0]
