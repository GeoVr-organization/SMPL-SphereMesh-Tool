"""Spending the spare sphere budget where the validator says the error is.

The placement rules in ``generate`` produce the *structure* of a topology -- chains along
bones, a ladder and a sheet on the trunk, inscribed spheres in the bulges. They cannot see
what they missed. The previous result left 20 of the 99 spheres unspent and its residual
error sat in exactly the places no rule addressed: the armpit, the groin, the lower back,
and a handful of surface steps where two bumpers touch in space but are far apart in the
graph. Both of those are things a loop can find and fix from the validator's own output,
which is what this module does.

Two repairs, both on the *named* spec so bilateral symmetry is kept by construction: a
left-side repair is written once and mirrored, a midline repair is pinned to ``x = 0``.

``fill_gaps``: take the largest cluster of vertices further than a threshold from the
collider, search the body interior near it for the largest sphere that reaches the cluster
without crossing the skin, and connect it to the nearest existing spheres whose binding is
compatible (same joint, or parent and child) and whose cone would not protrude. Repeat
until the budget is spent or nothing worth fixing remains.

``link_walk_breaks``: take the pair of bumpers between which the most surface steps need
more hops than the watchdog can afford, and add the one cone between their spheres that
removes the most of those steps, subject to the same binding and protrusion rules. Every
cone added here is a topological link the solver can walk; geometric proximity alone is
never trusted.

Both loops re-measure after every change rather than trusting a prediction, because the
handoff's most expensive lesson was that 2-D reasoning does not predict the 3-D union.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .anatomy import (
    Body,
    JOINT_NAMES,
    assign_parent_joint_like_tool,
    inscribed_radius,
    joints_compatible,
)
from .bumpers import SPHERE, BumperGraph, round_cone_distances
from .generate import SphereSpec, TopologySpec, mirror

__all__ = [
    "Cluster",
    "PoseSet",
    "clusters_of",
    "joints_compatible",
    "cone_protrusion",
    "fill_gaps",
    "link_walk_breaks",
    "replace_degenerate_slabs",
    "signed_distances",
]

#: Anything within this of the midline is treated as a midline feature and pinned to x = 0.
MIDLINE_TOLERANCE = 0.015


@dataclass
class PoseSet:
    """A few real poses, evaluated once, for judging joint-spanning cones.

    A cone whose two spheres bind to different joints is rigidly posed at both ends by the
    tool; when the joint between them rotates, the cone sweeps out of the body. At rest
    such a cone can look perfect, so the repairs check it under these poses as well and
    reject it when it pokes out further than ``tolerance``. Built from the same poses the
    fitter uses, on the same masked vertices.
    """

    posed_joints: list[np.ndarray]      # per pose, (24, 3)
    posed_rotations: list[np.ndarray]   # per pose, (24, 3, 3)
    posed_points: list[np.ndarray]      # per pose, (M, 3) the masked vertices, posed
    tolerance: float = 0.025

    @classmethod
    def from_thetas(cls, body: Body, thetas: np.ndarray, vertex_mask: np.ndarray,
                    *, stride: int = 3, tolerance: float = 0.025) -> "PoseSet":
        from .anatomy import load_body

        index = np.flatnonzero(vertex_mask)[:: max(1, int(stride))]
        joints, rotations, points = [], [], []
        for theta in np.asarray(thetas, dtype=np.float64):
            posed = load_body(body.gender, theta=theta)
            joints.append(posed.joints)
            rotations.append(posed.rotations)
            points.append(posed.vertices[index])
        return cls(joints, rotations, points, float(tolerance))

    def cone_protrusion(
        self, rest_joints: np.ndarray, first_center, first_radius, first_joint: int,
        second_center, second_radius, second_joint: int,
    ) -> float:
        """Worst protrusion of the cone over the poses, both ends moved rigidly."""
        worst = 0.0
        first_offset = np.asarray(first_center) - rest_joints[first_joint]
        second_offset = np.asarray(second_center) - rest_joints[second_joint]
        for joints, rotations, points in zip(self.posed_joints, self.posed_rotations, self.posed_points):
            posed_first = joints[first_joint] + rotations[first_joint] @ first_offset
            posed_second = joints[second_joint] + rotations[second_joint] @ second_offset
            worst = max(worst, cone_protrusion(
                points, posed_first, first_radius, posed_second, second_radius
            ))
        return worst


@dataclass
class Cluster:
    vertices: np.ndarray        # indices into the *body* vertex array
    worst: float                # metres, positive
    centroid: np.ndarray        # (3,)
    facing: np.ndarray          # (3,) mean outward normal
    side: str                   # "L", "R" or "M"


def replace_degenerate_slabs(
    spec: TopologySpec, centers: np.ndarray, radii: np.ndarray, *, verbose: bool = True,
) -> list[dict]:
    """Turn every 3-connection with no tangent plane into its three cones.

    A slab exists only while a plane tangent to all three spheres exists, which fails when
    the radii vary faster than the centres are spaced -- and the fit changes radii, so a
    triangle that was fine at placement can lose its face (on the female body the
    clavicle / chest-corner triangle does). The GUI's ``SphereMesh.Prysmoid`` then draws and
    measures only its three edge capsuloids, and the engine carries a slab that covers
    nothing beyond them. Replacing it by its rim cones
    keeps every sphere linked (the walk passes through the shared spheres) and reports
    the change instead of hiding it.
    """
    from .bumpers import tangent_plane

    index_of = spec.index_of()
    replaced: list[dict] = []
    kept: list[tuple[str, ...]] = []
    existing = {tuple(sorted(connection)) for connection in spec.connections}
    for connection in spec.connections:
        if len(connection) != 3:
            kept.append(connection)
            continue
        used = np.array([index_of[name] for name in connection])
        _, _, _, exists = tangent_plane(centers[used][None], radii[used][None])
        if bool(exists[0]):
            kept.append(connection)
            continue
        replaced.append({"triangle": list(connection),
                         "radii_mm": [round(float(radii[i]) * 1000, 1) for i in used]})
        for first, second in ((0, 1), (1, 2), (2, 0)):
            edge = (connection[first], connection[second])
            if tuple(sorted(edge)) not in existing:
                kept.append(edge)
                existing.add(tuple(sorted(edge)))
    spec.connections = kept
    if verbose:
        for entry in replaced:
            print(f"slab      {entry['triangle']} has no tangent plane (radii {entry['radii_mm']} mm); "
                  "replaced by its three cones")
    return replaced


def signed_distances(
    body: Body, graph: BumperGraph, centers: np.ndarray, radii: np.ndarray,
    vertex_mask: np.ndarray,
) -> np.ndarray:
    """Exact signed distance of every masked body vertex to the sphere-mesh."""
    return graph.distances(body.vertices[vertex_mask], centers, radii).min(axis=1)


def cone_protrusion(
    points: np.ndarray, start_center, start_radius, end_center, end_radius,
) -> float:
    """How far (metres) a single cone would poke through the given surface points."""
    distance = round_cone_distances(
        points,
        np.asarray(start_center, dtype=np.float64)[None, :], np.array([float(start_radius)]),
        np.asarray(end_center, dtype=np.float64)[None, :], np.array([float(end_radius)]),
    )[:, 0]
    return float(max(0.0, -distance.min()))


# ------------------------------------------------------------------------- clusters


def clusters_of(
    body: Body, flagged: np.ndarray, values: np.ndarray, *, min_size: int = 4,
) -> list[Cluster]:
    """Connected components, over the mesh edges, of the flagged body vertices.

    ``flagged`` is a boolean over *all* body vertices, ``values`` the per-vertex error
    (metres, positive) used to report the worst. Largest cluster first.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    flagged_index = np.flatnonzero(flagged)
    if len(flagged_index) == 0:
        return []
    remap = -np.ones(len(body.vertices), dtype=np.int64)
    remap[flagged_index] = np.arange(len(flagged_index))
    faces = body.faces
    pairs = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    pairs = remap[pairs]
    pairs = pairs[(pairs >= 0).all(axis=1)]
    size = len(flagged_index)
    adjacency = coo_matrix(
        (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(size, size)
    )
    count, label = connected_components(adjacency, directed=False)
    normals = body.normals
    clusters: list[Cluster] = []
    for component in range(count):
        members = flagged_index[label == component]
        if len(members) < int(min_size):
            continue
        centroid = body.vertices[members].mean(axis=0)
        side = "M" if abs(centroid[0]) <= MIDLINE_TOLERANCE else ("L" if centroid[0] > 0 else "R")
        facing = normals[members].mean(axis=0)
        facing /= max(np.linalg.norm(facing), 1e-12)
        clusters.append(Cluster(
            vertices=members, worst=float(values[members].max()),
            centroid=centroid, facing=facing, side=side,
        ))
    clusters.sort(key=lambda cluster: -len(cluster.vertices))
    return clusters


# --------------------------------------------------------------------- gap filling


def _interior_candidates(
    body: Body, lower: np.ndarray, upper: np.ndarray, *, resolution: float,
    min_clearance: float, midline: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Grid points inside the body within a box, with their clearance to the skin."""
    from scipy.spatial import cKDTree

    axes = [np.arange(lower[axis], upper[axis] + 1e-9, resolution) for axis in range(3)]
    if midline:
        axes[0] = np.array([0.0])
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    tree = getattr(body, "_repair_tree", None)
    if tree is None:
        tree = cKDTree(body.vertices)
        object.__setattr__(body, "_repair_tree", tree)
    clearance, nearest = tree.query(grid)
    outward = np.einsum("ij,ij->i", grid - body.vertices[nearest], body.normals[nearest])
    keep = (outward < 0.0) & (clearance >= float(min_clearance))
    return grid[keep], clearance[keep]


def _propose_fill(
    body: Body, cluster: Cluster, *, reach_tolerance: float, resolution: float,
    min_clearance: float, box_margin: float,
) -> tuple[np.ndarray, float, int] | None:
    """The interior sphere that reaches the most of the cluster's vertices.

    A sphere "reaches" a vertex when the vertex is within ``reach_tolerance`` of its
    surface. Among candidates the one reaching most vertices wins; ties go to the larger
    sphere, because a larger sphere also covers more of whatever surrounds the cluster.
    """
    cluster_points = body.vertices[cluster.vertices]
    lower = cluster_points.min(axis=0) - box_margin
    upper = cluster_points.max(axis=0) + box_margin
    midline = cluster.side == "M"
    grid, clearance = _interior_candidates(
        body, lower, upper, resolution=resolution, min_clearance=min_clearance,
        midline=midline,
    )
    if len(grid) == 0:
        return None

    # Prefer candidates the tool will bind to the joint that actually drives the
    # cluster's skin. The skin under the shoulder is three-quarters shoulder-weighted yet
    # lies medial to the shoulder joint, so the largest sphere reaching it binds to the
    # collar by the closest-bone rule and pops 45 mm out of the arm when the arm moves;
    # a slightly smaller sphere a centimetre further out binds to the shoulder and moves
    # with the skin. Only when no candidate binds to the dominant joint is the choice
    # left to geometry alone.
    dominant_joint = int(np.argmax(body.skin_weights[cluster.vertices].sum(axis=0)))
    candidate_joint, _, _ = assign_parent_joint_like_tool(grid, body.joints, body.parents)
    preferred = candidate_joint == dominant_joint
    if preferred.any():
        grid, clearance = grid[preferred], clearance[preferred]

    # (G, K) distance from each candidate's surface to each cluster vertex.
    surface_gap = np.linalg.norm(
        grid[:, None, :] - cluster_points[None, :, :], axis=2
    ) - clearance[:, None]
    reached = (np.abs(surface_gap) <= float(reach_tolerance)).sum(axis=1)
    if reached.max() == 0:
        return None
    score = reached + clearance / clearance.max()   # clearance only breaks ties
    best = int(np.argmax(score))
    return grid[best].copy(), float(clearance[best]), int(reached[best])


def _buried_fraction(
    graph: BumperGraph, centers: np.ndarray, radii: np.ndarray, center: np.ndarray,
    radius: float, *, samples: int = 96, seed: int = 0,
) -> float:
    """Share of a candidate sphere's surface that lies inside the existing collider."""
    generator = np.random.default_rng(seed)
    directions = generator.normal(size=(int(samples), 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    surface = np.asarray(center)[None, :] + float(radius) * directions
    return float((graph.distances(surface, centers, radii).min(axis=1) < -1e-9).mean())


def _partner_links(
    body: Body, new_center: np.ndarray, new_radius: float, new_joint: int,
    centers: np.ndarray, radii: np.ndarray, joints: np.ndarray, names: Sequence[str],
    surface_points: np.ndarray, *, max_links: int, protrusion_tolerance: float,
    forbid_side: str | None, poses: PoseSet | None = None,
) -> list[tuple[str, float]]:
    """Existing spheres a new sphere may be joined to, nearest surface first."""
    gap = np.linalg.norm(centers - new_center, axis=1) - radii - new_radius
    order = np.argsort(gap)
    links: list[tuple[str, float]] = []
    for index in order:
        name = names[index]
        if forbid_side and name.startswith(forbid_side):
            continue
        if not joints_compatible(new_joint, joints[index], body.parents):
            continue
        separation = np.linalg.norm(centers[index] - new_center)
        if separation <= abs(radii[index] - new_radius) + 1e-6:
            continue   # one sphere swallows the other: no cone surface at all
        protrusion = cone_protrusion(
            surface_points, new_center, new_radius, centers[index], radii[index]
        )
        if protrusion > float(protrusion_tolerance):
            continue
        if poses is not None and int(joints[index]) != int(new_joint):
            posed = poses.cone_protrusion(
                body.joints, new_center, new_radius, new_joint,
                centers[index], radii[index], int(joints[index]),
            )
            if posed > poses.tolerance:
                continue
        links.append((name, protrusion))
        if len(links) >= int(max_links):
            break
    return links


def fill_gaps(
    body: Body,
    spec: TopologySpec,
    *,
    vertex_mask: np.ndarray,
    refit: Callable[[TopologySpec], tuple[np.ndarray, np.ndarray]],
    max_spheres: int,
    gap_threshold: float = 0.012,
    min_cluster: int = 8,
    reach_tolerance: float = 0.010,
    resolution: float = 0.006,
    min_clearance: float = 0.015,
    box_margin: float = 0.07,
    max_links: int = 2,
    protrusion_tolerance: float = 0.008,
    max_rounds: int = 12,
    verbose: bool = True,
    poses: PoseSet | None = None,
    max_buried_fraction: float = 1.0,
) -> tuple[TopologySpec, list[dict]]:
    """Add spheres, one cluster at a time, until the budget or the residual runs out.

    ``max_buried_fraction`` can reject a proposal whose surface is mostly inside the
    existing collider already. It is off by default because, measured on the fitted
    result, it does not discriminate: the fills that earned their place are 59-94 %
    buried (a sphere in a gap between big trunk spheres exposes only the face toward the
    gap) and the one harmful near-duplicate was 87 %. What does exclude that duplicate --
    a 91 mm sphere two centimetres from a 113 mm one, placed for a six-vertex cluster --
    is ``min_cluster``: clusters that small are within the mesh's own noise.

    ``refit`` is called with the spec after every addition and must return the current
    ``(centers, radii)`` in spec order -- a cheap fit is enough here; the caller runs the
    full one afterwards. The spec's own centres and radii are updated from it so the next
    round measures the fitted geometry, not the placement.
    """
    log: list[dict] = []
    for round_index in range(int(max_rounds)):
        if len(spec.spheres) >= int(max_spheres):
            log.append({"round": round_index, "stop": "budget"})
            break
        centers, radii = refit(spec)
        for sphere, center, radius in zip(spec.spheres, centers, radii):
            sphere.center = np.asarray(center, dtype=np.float64)
            sphere.radius = float(radius)
        centers, radii, connections, names = spec.assemble()
        graph = BumperGraph(len(centers), connections)
        signed = signed_distances(body, graph, centers, radii, vertex_mask)
        values = np.zeros(len(body.vertices))
        values[np.flatnonzero(vertex_mask)] = signed
        flagged = np.zeros(len(body.vertices), dtype=bool)
        flagged[np.flatnonzero(vertex_mask)] = signed > float(gap_threshold)
        clusters = [
            cluster for cluster in clusters_of(body, flagged, values, min_size=min_cluster)
            if cluster.side != "R"
        ]
        if not clusters:
            log.append({"round": round_index, "stop": "no cluster left"})
            break

        joints, _, _ = assign_parent_joint_like_tool(centers, body.joints, body.parents)
        surface_points = body.vertices[vertex_mask]
        added = None
        for cluster in clusters:
            proposal = _propose_fill(
                body, cluster, reach_tolerance=reach_tolerance, resolution=resolution,
                min_clearance=min_clearance, box_margin=box_margin,
            )
            if proposal is None:
                continue
            new_center, new_radius, reached = proposal
            buried = _buried_fraction(graph, centers, radii, new_center, new_radius)
            if buried > float(max_buried_fraction):
                continue
            side = cluster.side
            if side == "L" and abs(new_center[0]) <= MIDLINE_TOLERANCE:
                # The cluster is off-centre but the sphere that covers it is not: mirroring
                # it would produce two spheres a centimetre apart, one inside the other.
                side = "M"
                new_center[0] = 0.0
                new_radius = float(inscribed_radius(body, new_center[None])[0])
            new_joint = int(assign_parent_joint_like_tool(
                new_center[None], body.joints, body.parents
            )[0][0])
            links = _partner_links(
                body, new_center, new_radius, new_joint, centers, radii, joints, names,
                surface_points, max_links=max_links,
                protrusion_tolerance=protrusion_tolerance,
                forbid_side="R_" if side == "L" else None, poses=poses,
            )
            if not links:
                continue
            prefix = "M" if side == "M" else "L"
            name = f"{prefix}_fill_{sum(1 for s in spec.spheres if '_fill_' in s.name) // (1 if prefix == 'M' else 2)}"
            while any(sphere.name == name for sphere in spec.spheres):
                name += "b"
            spec.add(
                [SphereSpec(name=name, center=new_center, radius=new_radius, group="fill")],
                [(name, partner) for partner, _ in links],
            )
            mirror(spec)
            added = {
                "round": round_index, "name": name, "center": new_center.round(4).tolist(),
                "radius_mm": round(new_radius * 1000, 1), "reached": reached,
                "cluster_size": int(len(cluster.vertices)),
                "cluster_worst_mm": round(cluster.worst * 1000, 1),
                "cluster_joint": JOINT_NAMES[int(np.argmax(
                    body.skin_weights[cluster.vertices].sum(axis=0)))],
                "links": [(partner, round(protrusion * 1000, 1)) for partner, protrusion in links],
                "binding": JOINT_NAMES[new_joint],
                "spheres_now": len(spec.spheres),
            }
            break
        if added is None:
            log.append({"round": round_index, "stop": "no admissible fill for any cluster"})
            break
        log.append(added)
        if verbose:
            print(
                f"fill      {added['name']} r={added['radius_mm']} mm at {added['center']} "
                f"({added['binding']}) reaches {added['reached']}/{added['cluster_size']} of a "
                f"{added['cluster_joint']} cluster, worst {added['cluster_worst_mm']} mm; "
                f"links {added['links']}; {added['spheres_now']} spheres"
            )
    return spec, log


# ------------------------------------------------------------------ walk-break links


def _expensive_pairs(
    graph: BumperGraph, centers: np.ndarray, radii: np.ndarray, points: np.ndarray,
    edges: np.ndarray, *, budget: int,
) -> tuple[dict[tuple[int, int], int], int]:
    owner, _ = graph.owners(points, centers, radii)
    hop = graph.hop_distances()
    first, second = owner[edges[:, 0]], owner[edges[:, 1]]
    hops = hop[first, second]
    expensive = (hops < 0) | (hops > int(budget))
    tally: dict[tuple[int, int], int] = {}
    for left, right in zip(first[expensive], second[expensive]):
        key = (int(min(left, right)), int(max(left, right)))
        tally[key] = tally.get(key, 0) + 1
    return tally, int(expensive.sum())


def link_walk_breaks(
    body: Body,
    spec: TopologySpec,
    *,
    vertex_mask: np.ndarray,
    edges: np.ndarray,
    walk_budget: int = 3,
    max_links: int = 8,
    protrusion_tolerance: float = 0.010,
    min_gain: int = 3,
    verbose: bool = True,
    poses: PoseSet | None = None,
) -> tuple[TopologySpec, list[dict]]:
    """Add cones that shorten the watchdog's walk where it is measurably too long.

    ``edges`` are the surface steps in *masked-point* index space, as ``validate``
    builds them. Each round takes the worst bumper pair, tries every cone between one
    sphere of each that passes the binding and protrusion rules, keeps the one that
    removes the most expensive steps overall, and stops when no candidate removes at
    least ``min_gain`` of them.
    """
    points = body.vertices[vertex_mask]
    log: list[dict] = []
    for round_index in range(int(max_links)):
        centers, radii, connections, names = spec.assemble()
        graph = BumperGraph(len(centers), connections)
        tally, expensive_before = _expensive_pairs(
            graph, centers, radii, points, edges, budget=walk_budget
        )
        if not tally:
            log.append({"round": round_index, "stop": "no expensive steps"})
            break
        joints, _, _ = assign_parent_joint_like_tool(centers, body.joints, body.parents)
        existing = {tuple(sorted(connection)) for connection in connections}
        ranked = sorted(tally.items(), key=lambda item: -item[1])

        best = None
        for (first_bumper, second_bumper), count in ranked[:6]:
            first_spheres = graph.bumpers[first_bumper].spheres
            second_spheres = graph.bumpers[second_bumper].spheres
            for first in first_spheres:
                for second in second_spheres:
                    if first == second or tuple(sorted((first, second))) in existing:
                        continue
                    # Keep the spec mirrorable: a left sphere never links to a right one.
                    if names[first][:2] + names[second][:2] in ("L_R_", "R_L_"):
                        continue
                    if not joints_compatible(joints[first], joints[second], body.parents):
                        continue
                    separation = np.linalg.norm(centers[first] - centers[second])
                    if separation <= abs(radii[first] - radii[second]) + 1e-6:
                        continue
                    protrusion = cone_protrusion(
                        points, centers[first], radii[first], centers[second], radii[second]
                    )
                    if protrusion > float(protrusion_tolerance):
                        continue
                    if poses is not None and int(joints[first]) != int(joints[second]):
                        posed = poses.cone_protrusion(
                            body.joints, centers[first], radii[first], int(joints[first]),
                            centers[second], radii[second], int(joints[second]),
                        )
                        if posed > poses.tolerance:
                            continue
                    trial = TopologySpec(
                        spheres=list(spec.spheres),
                        connections=list(spec.connections) + [(names[first], names[second])],
                    )
                    mirror(trial)
                    trial_centers, trial_radii, trial_connections, _ = trial.assemble()
                    trial_graph = BumperGraph(len(trial_centers), trial_connections)
                    _, expensive_after = _expensive_pairs(
                        trial_graph, trial_centers, trial_radii, points, edges,
                        budget=walk_budget,
                    )
                    gain = expensive_before - expensive_after
                    if best is None or gain > best[0]:
                        best = (gain, names[first], names[second], protrusion, count)
        if best is None or best[0] < int(min_gain):
            log.append({
                "round": round_index,
                "stop": "no admissible cone gains enough"
                        + ("" if best is None else f" (best {best[0]})"),
            })
            break
        gain, first_name, second_name, protrusion, count = best
        spec.connect(first_name, second_name)
        mirror(spec)
        entry = {
            "round": round_index, "cone": (first_name, second_name),
            "steps_removed": int(gain), "pair_steps": int(count),
            "protrusion_mm": round(protrusion * 1000, 1),
            "expensive_before": int(expensive_before),
        }
        log.append(entry)
        if verbose:
            print(
                f"link      cone[{first_name}, {second_name}] removes {gain} of "
                f"{expensive_before} expensive steps (protrusion {protrusion * 1000:.1f} mm)"
            )
    return spec, log
