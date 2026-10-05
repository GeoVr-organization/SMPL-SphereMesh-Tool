"""Everything a candidate topology has to survive before it is worth looking at.

The checks fall into four groups, in rising order of how much they cost to run and how
much they actually tell you.

**Budget and structure.** Sphere and triangle counts against the limits; the sphere graph
and the bumper graph each a single connected component; no degenerate primitive -- a sphere
swallowed by another sphere is a bumper that can never be nearest to anything, and a
triangle whose radii vary faster than its centre spacing has no flat face and is three
cones wearing a slab's name.

**Coverage and protrusion.** Signed distance from every modelled body vertex to the
sphere-mesh, measured with the exact metric in ``bumpers`` rather than
``sphere_fit.spheremesh_distance``, which never evaluates a triangle's slab and so scores a
correct sheet as a hole. Positive is a gap, negative means the collider is outside the skin.
<<<<<<< Updated upstream
The report says *where* as well as how much: badly covered vertices are grouped into
connected clusters over the mesh edges, and each cluster is described by its size, its
worst value, its position, the joint that skins it, the direction it faces and the bumper
that owns it. A missing midline column shows up as "every gap cluster faces front" long
before anyone opens the result in a viewer.
=======
>>>>>>> Stashed changes

**The walk-break rate.** This is the check the whole exercise exists for. A cloth node
sliding across the body moves between neighbouring surface points; the solver's watchdog
follows it by stepping through the bumper adjacency graph. So: take every edge of the SMPL
mesh, find which bumper owns each end, and count how many adjacency hops separate them.

It has to be a hop *count*, not an adjacency test. Two consecutive cones along a limb share
a sphere and are two hops apart through it -- ordinary traversal, and calling it a break
would condemn every well-formed chain. What is genuinely expensive is a step whose two
bumpers are many hops apart, and what is fatal is a step between different components,
where the walk cannot get there at all and the solver falls back to a brute-force rescan.
So the reported figure is the fraction of surface steps needing more hops than the walk can
afford, plus the mean walk length, plus the histogram behind both. Unlike a coverage number
this punishes exactly the failure described in the brief: two bumpers overlapping in space
with no topological link between them.

<<<<<<< Updated upstream
**Harmful overlap.** The same failure seen from the collider's side rather than the
skin's. Each bumper's surface is sampled; a sample that is not buried by the bumper's own
topological neighbourhood is *locally exposed*, and it is *harmful* when a bumper far away
in the graph swallows it anyway -- because a node resolved against that exposed patch can
be captured by the far bumper with no link for the watchdog to follow. Earlier versions
counted any interpenetration between unlinked bumpers and failed every design ever built,
including the user's own; see ``harmful_overlap`` for the metric and ``Thresholds`` for the
numbers it was calibrated against.

**Binding.** The tool poses every sphere rigidly by a single joint chosen by the closest
kintree bone, so a cone whose two spheres bind to unrelated joints tears apart under
animation (260 mm RMS across the elbow, measured). Every rim edge of every bumper is
checked for that in the rest pose, and, when the generator says which joint it intended,
the tool's rule is checked against the intention.

=======
>>>>>>> Stashed changes
**The pose sweep.** All of the above again, under real poses. The body is re-evaluated with
full SMPL skinning while the spheres are moved by the tool's *rigid, single-joint* rule, so
any bumper that spans two joints which rotate relative to each other shows up as protrusion
exactly where the user would see it pop out.
"""

from __future__ import annotations

<<<<<<< Updated upstream
import json
=======
>>>>>>> Stashed changes
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
<<<<<<< Updated upstream
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
=======
>>>>>>> Stashed changes

from .anatomy import (
    JOINT_INDEX,
    JOINT_NAMES,
    Body,
    assign_parent_joint_like_tool,
<<<<<<< Updated upstream
    joints_compatible,
    load_body,
    pose_spheres_like_tool,
)
from .bumpers import CONE, PRYSMOID, QUAD, SPHERE, BumperGraph, tangent_plane
=======
    load_body,
    pose_spheres_like_tool,
)
from .bumpers import CONE, PRYSMOID, QUAD, SPHERE, BumperGraph
>>>>>>> Stashed changes

__all__ = [
    "Thresholds",
    "Report",
    "validate",
<<<<<<< Updated upstream
    "format_report",
    "bumper_label",
=======
>>>>>>> Stashed changes
    "modelled_vertex_mask",
    "surface_edges",
    "walk_breaks",
    "non_adjacent_overlap",
<<<<<<< Updated upstream
    "harmful_overlap",
    "bumper_surface_samples",
    "coverage_clusters",
    "joint_crossing_cones",
=======
>>>>>>> Stashed changes
]


@dataclass
class Thresholds:
    """Pass/fail limits. Every one of these is a judgement call, so each carries its why."""

    max_spheres: int = 99
    # The user's budget. Triangles are the scarce resource because each one is a prysmoid
    # the engine has to carry; quads are cheap engine-side and are counted separately.
    max_triangles: int = 20
    max_quads: int = 40

    # A gap this size means cloth rests visibly off the body. 15 mm is about one SMPL edge
    # length, so anything under it is inside the mesh's own resolution.
    max_gap: float = 0.015
    # Protrusion is worse than a gap: the collider is outside the skin and cloth is pushed
    # off the body. Held to a third of the gap budget.
    max_protrusion: float = 0.005
    # A few vertices may sit outside these limits without the result being unusable; a
    # broad failure may not.
    max_gap_fraction: float = 0.02
    max_protrusion_fraction: float = 0.02

    # The headline number. 1% of surface edges losing the walk is roughly one brute-force
    # reacquisition per hundred node-crossings, which is the point where the watchdog is
    # still paying for itself.
    max_walk_break_fraction: float = 0.01

<<<<<<< Updated upstream
    # Harmful overlap, global: over every bumper's *locally exposed* surface samples (those
    # not buried by a bumper within the walk radius), the fraction that a bumper further
    # away than the walk radius swallows anyway. Calibrated against the user's hand-made
    # template, the thing this validator must not reject. Measured with 96 samples per
    # bumper and a walk radius of 3:
    #   data/template.json            567 of 5295 exposed samples, 10.71% globally; worst
    #                                 single bumper sphere[40], 100% (10 of 10) into
    #                                 sphere[35]; 39 of 166 bumpers affected.
    #   data/generated/auto_male.json 126 of 5996 exposed samples, 2.10% globally; worst
    #                                 cone[R_thigh_0, R_thigh_1], 100% (2 of 2) into
    #                                 sphere[M_lower_r4]; 17 of 225 bumpers affected.
    # The limit is set at just under twice the template's global figure. The per-bumper
    # worst is reported but not gated: with 96 samples a sphere that presents ten exposed
    # samples and pokes into one distant neighbour reads as 100%, which is one local defect
    # for the worst list, not a topology-wide verdict -- both designs above score 100%
    # there while differing five-fold globally, in the same direction as their walk-break
    # rates (3.68% against 1.83%).
    max_non_adjacent_overlap: float = 0.20
=======
    # Two bumpers that are not adjacent should barely interpenetrate. Measured as the
    # fraction of one bumper's surface samples lying inside the other.
    max_non_adjacent_overlap: float = 0.05
>>>>>>> Stashed changes

    # Under pose the rigid binding cannot track skinning exactly and some slack is honest.
    posed_max_protrusion: float = 0.020
    posed_max_protrusion_fraction: float = 0.03


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    value: float | None = None
    limit: float | None = None


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    worst: dict = field(default_factory=dict)
<<<<<<< Updated upstream
    # The graph the report was computed on and the sphere names it was labelled with, so
    # ``format_report`` can label bumper indices after the fact.
    graph: BumperGraph | None = None
    names: list[str] | None = None
=======
>>>>>>> Stashed changes

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def add(self, name, passed, detail, value=None, limit=None) -> None:
        self.checks.append(Check(name, bool(passed), detail, value, limit))

    def text(self) -> str:
        lines = []
        width = max((len(check.name) for check in self.checks), default=0)
        for check in self.checks:
            mark = "PASS" if check.passed else "FAIL"
            lines.append(f"  [{mark}] {check.name:<{width}}  {check.detail}")
        failed = sum(1 for check in self.checks if not check.passed)
        lines.append("")
        lines.append(
            f"  {len(self.checks) - failed}/{len(self.checks)} checks passed"
            + ("" if failed == 0 else f"  ({failed} FAILED)")
        )
        return "\n".join(lines)


<<<<<<< Updated upstream
# ---------------------------------------------------------------------------- labels


def sphere_label(index: int, names: Sequence[str] | None = None) -> str:
    """A sphere's name when one is known, otherwise its index."""
    if names is not None and 0 <= int(index) < len(names):
        return str(names[int(index)])
    return str(int(index))


def bumper_label(graph: BumperGraph, index: int, names: Sequence[str] | None = None) -> str:
    """``cone[L_glute_0, L_thigh_0]`` when names are known, ``cone[3, 7]`` otherwise.

    ``Bumper.__str__`` stays index-based on purpose: it is the stable key used by tests and
    by the graph itself, and a name list is a property of one design, not of the primitive.
    """
    bumper = graph.bumpers[int(index)]
    if names is None:
        return str(bumper)
    return f"{bumper.kind}[{', '.join(sphere_label(sphere, names) for sphere in bumper.spheres)}]"


=======
>>>>>>> Stashed changes
# ------------------------------------------------------------------ body-side helpers


#: Where the model stops. Each entry is ``(bone parent, bone child, offset in metres)``:
#: everything beyond the plane through the child joint, normal to the bone, offset by that
#: much further along it, is out of scope.
DEFAULT_TRUNCATIONS: tuple[tuple[str, str, float], ...] = (
    ("L_elbow", "L_wrist", 0.0),
    ("R_elbow", "R_wrist", 0.0),
    ("L_knee", "L_ankle", 0.0),
    ("R_knee", "R_ankle", 0.0),
    ("spine3", "neck", 0.06),
)


def modelled_vertex_mask(
    body: Body,
    *,
    exclude: Sequence[str] = ("head", "L_hand", "R_hand", "L_foot", "R_foot"),
    exclude_weight: float = 0.5,
    truncations: Sequence[tuple[str, str, float]] = DEFAULT_TRUNCATIONS,
) -> np.ndarray:
    """Vertices the topology is meant to cover.

    Hands, feet and the skull are out of scope, and leaving them in drags every reported
    average toward a region nothing models -- a 120 mm "gap" at the fingertips says nothing
    about the quality of the torso.

    Two filters, because skinning weight alone is not enough. A vertex is dropped when the
    excluded joints together own at least ``exclude_weight`` of its skinning; and a vertex
    is also dropped when it lies beyond one of the ``truncations``, which are the planes
    where the chains actually stop. The second filter is what removes the palm and the
    instep, whose skinning is still dominated by the wrist and the ankle even though no
    sphere reaches them.
    """
    columns = [JOINT_INDEX[name] for name in exclude]
    keep = body.skin_weights[:, columns].sum(axis=1) < float(exclude_weight)
    for parent, child, offset in truncations:
        start, axis, _ = body.bone(parent, child)
        cut = body.joints[JOINT_INDEX[child]] + float(offset) * axis
        keep &= (body.vertices - cut) @ axis <= 0.0
    return keep


def surface_edges(faces: np.ndarray) -> np.ndarray:
    """(E, 2) unique undirected edges of a triangle mesh.

    These are the steps a cloth node takes across the surface, and therefore the steps the
    watchdog's walk has to be able to follow.
    """
    pairs = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    pairs = np.sort(pairs, axis=1)
    return np.unique(pairs, axis=0)


# ------------------------------------------------------------------------- the checks


def walk_breaks(
    graph: BumperGraph,
    centers: np.ndarray,
    radii: np.ndarray,
    points: np.ndarray,
    edges: np.ndarray,
    *,
    budget: int = 3,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """How far the watchdog has to walk for each step a node takes across the surface.

    Returns ``(owner_per_point, hops_per_edge, tally_of_expensive_pairs)``, where a hop
    count of ``-1`` means the two bumpers are in different components of the graph and the
<<<<<<< Updated upstream
    walk cannot get there at all. The tally is keyed by ``(bumper index, bumper index)``,
    lower index first, so the caller can label the pair however it likes.
=======
    walk cannot get there at all.
>>>>>>> Stashed changes

    ``budget`` is how many hops the walk is assumed to afford before the fallback is
    cheaper. It has to be at least 2 to be fair: consecutive cones along a limb share a
    sphere and are two hops apart through it, which is ordinary traversal and not a fault.
    3 additionally allows the step from one slab, across a rim cone, into the next.
    """
    owner, _ = graph.owners(points, centers, radii)
    hop = graph.hop_distances()
    first = owner[edges[:, 0]]
    second = owner[edges[:, 1]]
    hops = hop[first, second]

    expensive = (hops < 0) | (hops > int(budget))
<<<<<<< Updated upstream
    tally: dict[tuple[int, int], int] = {}
    for left, right in zip(first[expensive], second[expensive]):
        key = (int(min(left, right)), int(max(left, right)))
=======
    tally: dict[tuple[str, str], int] = {}
    for left, right in zip(first[expensive], second[expensive]):
        key = tuple(sorted((str(graph.bumpers[left]), str(graph.bumpers[right]))))
>>>>>>> Stashed changes
        tally[key] = tally.get(key, 0) + 1
    return owner, hops, tally


<<<<<<< Updated upstream
def _unit_directions(generator: np.random.Generator, count: int) -> np.ndarray:
    direction = generator.normal(size=(count, 3))
    return direction / np.linalg.norm(direction, axis=1, keepdims=True)


def bumper_surface_samples(
    graph: BumperGraph,
    centers: np.ndarray,
    radii: np.ndarray,
    *,
    samples_per_bumper: int = 96,
    seed: int = 0,
) -> list[np.ndarray]:
    """Points on the *true* surface of each bumper, one (S, 3) array per bumper.

    The naive way -- a point on a random interpolated ball in a random direction -- puts
    almost every sample strictly inside the bumper it is meant to sample, because the
    other balls of the same cone or slab cover it. Those are interior points, and an
    interior point is inside its own bumper by definition, which makes any "is this sample
    inside something" question meaningless. So each kind is sampled on its actual boundary:

    * a sphere: uniform directions on its surface;
    * a cone: its lateral surface, which is where its balls' common tangent lines run. A
      unit normal ``n`` is tangent to every ball of the cone when ``n . axis`` equals
      ``-(r1 - r0) / |c1 - c0|``; pick a random in-plane direction, tilt it by that much,
      and the point ``c(t) + r(t) n`` lies exactly on the cone for every ``t``. The end
      caps belong to the sphere bumpers and are sampled there;
    * a prysmoid or quad: its two flat faces. The plane tangent to all three spheres is
      tangent to every interpolated ball of the triangle, so ``c(lambda) + r(lambda) n``
      is on the slab for every barycentric ``lambda``. Its rim cones are sampled as cones.

    A cone with no lateral surface or a triangle with no tangent plane yields no samples,
    which is the honest answer: there is no surface to expose.
    """
    generator = np.random.default_rng(seed)
    samples: list[np.ndarray] = []
    for bumper in graph.bumpers:
        if bumper.kind == SPHERE:
            sphere = bumper.spheres[0]
            direction = _unit_directions(generator, samples_per_bumper)
            samples.append(centers[sphere] + radii[sphere] * direction)
            continue

        if bumper.kind == CONE:
            start, end = bumper.spheres
            axis = centers[end] - centers[start]
            length = float(np.linalg.norm(axis))
            tilt = -(radii[end] - radii[start]) / length if length > 1e-12 else 2.0
            if abs(tilt) >= 1.0 - 1e-9:
                samples.append(np.zeros((0, 3)))
                continue
            axis = axis / length
            random_direction = _unit_directions(generator, samples_per_bumper)
            in_plane = random_direction - (random_direction @ axis)[:, None] * axis
            in_plane /= np.maximum(np.linalg.norm(in_plane, axis=1, keepdims=True), 1e-12)
            normal = np.sqrt(1.0 - tilt * tilt) * in_plane + tilt * axis
            parameter = generator.uniform(size=samples_per_bumper)[:, None]
            medial_center = centers[start] + parameter * (centers[end] - centers[start])
            medial_radius = radii[start] + parameter[:, 0] * (radii[end] - radii[start])
            samples.append(medial_center + medial_radius[:, None] * normal)
            continue

        triangles = bumper.triangles()
        per_triangle = max(1, samples_per_bumper // (2 * len(triangles)))
        slab_points: list[np.ndarray] = []
        for triangle in triangles:
            corners = np.array(triangle)
            in_plane_normal, out_of_plane_scale, plane_axis, exists = tangent_plane(
                centers[corners][None], radii[corners][None]
            )
            if not bool(exists[0]):
                continue
            weight = generator.dirichlet(np.ones(3), size=per_triangle)
            medial_center = weight @ centers[corners]
            medial_radius = weight @ radii[corners]
            for side in (1.0, -1.0):
                normal = in_plane_normal[0] + side * out_of_plane_scale[0] * plane_axis[0]
                slab_points.append(medial_center + medial_radius[:, None] * normal)
        samples.append(
            np.concatenate(slab_points, axis=0) if slab_points else np.zeros((0, 3))
        )
    return samples


def harmful_overlap(
    graph: BumperGraph,
    centers: np.ndarray,
    radii: np.ndarray,
    *,
    walk_radius: int = 3,
    samples_per_bumper: int = 96,
    seed: int = 0,
    chunk_size: int = 8192,
) -> dict:
    """Surface of each bumper that a topologically distant bumper swallows.

    The failure the watchdog cannot absorb is a node resolved against one bumper being
    captured by another that overlaps it in space but is far away in the graph. Overlap as
    such is not the problem: a sphere lies inside every cone that ends on it and a cone
    inside every slab that carries it, and that is the collider being seamless. Counting
    every unlinked interpenetration therefore fails every design, including good ones.

    So each bumper's surface is sampled (``bumper_surface_samples``) and split three ways.
    A sample strictly inside any bumper within ``walk_radius`` hops of its own -- the own
    bumper included -- is *buried*: nothing resolves against it, and nothing is lost. The
    rest are *locally exposed*: the surface the solver actually presents. An exposed sample
    that is strictly inside some bumper *more* than ``walk_radius`` hops away (or in
    another component) is *harmful*: that far bumper is nearer, will take the node, and the
    walk cannot get there.

    Returns a dict with ``per_bumper`` (worst first: bumper index, exposed and harmful
    sample counts, the harmful fraction and the far bumper that swallows most of it),
    ``global_fraction`` (harmful samples over all exposed samples), and ``buried`` -- the
    non-sphere bumpers ranked by *dead surface*, the fraction of their samples inside any
    other bumper at all, which is informational: a cone entirely inside its slab costs the
    engine a primitive and buys nothing.
    """
    samples = bumper_surface_samples(
        graph, centers, radii, samples_per_bumper=samples_per_bumper, seed=seed
    )
    sample_owner = np.concatenate(
        [np.full(len(points), index, dtype=np.int64) for index, points in enumerate(samples)]
    ) if samples else np.zeros(0, dtype=np.int64)
    all_points = (
        np.concatenate(samples, axis=0) if samples else np.zeros((0, 3))
    )
    bumper_count = len(graph.bumpers)

    inside = np.zeros((len(all_points), bumper_count), dtype=bool)
    for start in range(0, len(all_points), chunk_size):
        stop = min(start + chunk_size, len(all_points))
        distance = graph.distances(all_points[start:stop], centers, radii)
        inside[start:stop] = distance < -1e-9

    hop = graph.hop_distances()
    near_matrix = (hop >= 0) & (hop <= int(walk_radius))
    near = near_matrix[sample_owner]
    buried = (inside & near).any(axis=1)
    exposed = ~buried
    far_inside = inside & ~near
    harmful = exposed & far_inside.any(axis=1)
    other = np.ones((len(all_points), bumper_count), dtype=bool)
    other[np.arange(len(all_points)), sample_owner] = False
    dead = (inside & other).any(axis=1)

    per_bumper: list[dict] = []
    dead_surface: list[dict] = []
    for index in range(bumper_count):
        mine = sample_owner == index
        exposed_count = int(exposed[mine].sum())
        harmful_count = int(harmful[mine].sum())
        into = None
        into_count = 0
        if harmful_count:
            captured = far_inside[mine & harmful].sum(axis=0)
            into = int(np.argmax(captured))
            into_count = int(captured[into])
        per_bumper.append({
            "bumper": index,
            "samples": int(mine.sum()),
            "exposed": exposed_count,
            "harmful": harmful_count,
            "fraction": harmful_count / exposed_count if exposed_count else 0.0,
            "into": into,
            "into_fraction": into_count / exposed_count if exposed_count else 0.0,
        })
        if graph.bumpers[index].kind != SPHERE and mine.any():
            dead_surface.append({
                "bumper": index,
                "dead_fraction": float(dead[mine].mean()),
            })
    per_bumper.sort(key=lambda entry: (-entry["fraction"], -entry["harmful"]))
    dead_surface.sort(key=lambda entry: -entry["dead_fraction"])

    exposed_total = int(exposed.sum())
    harmful_total = int(harmful.sum())
    return {
        "per_bumper": per_bumper,
        "buried": dead_surface,
        "exposed_total": exposed_total,
        "harmful_total": harmful_total,
        "global_fraction": harmful_total / exposed_total if exposed_total else 0.0,
        "walk_radius": int(walk_radius),
        "samples_per_bumper": int(samples_per_bumper),
    }


=======
>>>>>>> Stashed changes
def non_adjacent_overlap(
    graph: BumperGraph,
    centers: np.ndarray,
    radii: np.ndarray,
    *,
    walk_radius: int = 3,
    samples_per_bumper: int = 96,
    seed: int = 0,
) -> list[tuple[str, str, float]]:
    """Bumper pairs that interpenetrate while being topologically far apart, worst first.

<<<<<<< Updated upstream
    The original overlap metric, kept for comparison. It reports the fraction of one
    bumper's (interpolated-ball) samples lying strictly inside another bumper that shares
    no sphere with it and is more than ``walk_radius`` hops away. It fails every design
    ever measured -- 95.8% on the user's template -- because a sphere legitimately sits
    inside a slab two rows over, and because the samples are interior points to begin
    with. ``harmful_overlap`` is what ``validate`` uses.
=======
    The failure this looks for is the one the watchdog cannot absorb: two primitives that
    overlap in space but belong to disjoint parts of the topology, so a node sliding from
    one to the other finds no link to follow and falls back to a brute-force rescan.

    Pairs that merely *neighbour* each other are excluded, and the exclusion has to be
    generous or the check is all false positives. Bumpers built from a shared sphere
    necessarily overlap -- a sphere lies inside every cone that ends on it, and a cone lies
    inside every slab that carries it as a rim -- and that overlap is the collider being
    seamless, not a fault. So a pair is reported only when it shares no sphere at all *and*
    is more than ``walk_radius`` hops apart in the bumper graph.

    The measure is the fraction of one bumper's surface samples lying strictly inside the
    other: scale-free, so it reads the same for two fat torso spheres and two thin forearm
    cones.
>>>>>>> Stashed changes
    """
    generator = np.random.default_rng(seed)
    direction = generator.normal(size=(samples_per_bumper, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)

    surface_samples: list[np.ndarray] = []
    for bumper in graph.bumpers:
        used = np.array(bumper.spheres)
        weight = generator.dirichlet(np.ones(len(used)), size=samples_per_bumper)
        medial_center = weight @ centers[used]
        medial_radius = weight @ radii[used]
        surface_samples.append(medial_center + medial_radius[:, None] * direction)

    nearby = graph.reachable_within(walk_radius)
    spheres_of = [set(bumper.spheres) for bumper in graph.bumpers]
    found: list[tuple[str, str, float]] = []
    for first in range(len(graph.bumpers)):
        candidates = [
            second for second in range(first + 1, len(graph.bumpers))
            if not nearby[first, second] and not (spheres_of[first] & spheres_of[second])
        ]
        if not candidates:
            continue
        distance = graph.distances(surface_samples[first], centers, radii)
        inside_fraction = (distance < -1e-9).mean(axis=0)
        for second in candidates:
            overlap = float(inside_fraction[second])
            if overlap > 0.0:
                found.append(
                    (str(graph.bumpers[first]), str(graph.bumpers[second]), overlap)
                )
    found.sort(key=lambda entry: -entry[2])
    return found


def degenerate_primitives(
<<<<<<< Updated upstream
    graph: BumperGraph,
    centers: np.ndarray,
    radii: np.ndarray,
    names: Sequence[str] | None = None,
) -> list[str]:
    """Bumpers that carry no surface of their own, or no flat face."""
=======
    graph: BumperGraph, centers: np.ndarray, radii: np.ndarray
) -> list[str]:
    """Bumpers that carry no surface of their own, or no flat face."""
    from .bumpers import tangent_plane

>>>>>>> Stashed changes
    problems: list[str] = []

    separation = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    for first in range(len(centers)):
        for second in range(len(centers)):
            if first == second:
                continue
            if separation[first, second] + radii[first] <= radii[second] + 1e-9:
                problems.append(
<<<<<<< Updated upstream
                    f"sphere {sphere_label(first, names)} (r={radii[first] * 1000:.0f} mm) "
                    f"lies entirely inside sphere {sphere_label(second, names)} "
                    f"(r={radii[second] * 1000:.0f} mm): it can never be the nearest bumper"
                )
                break

    for bumper_index, bumper in enumerate(graph.bumpers):
        label = bumper_label(graph, bumper_index, names)
=======
                    f"sphere {first} (r={radii[first] * 1000:.0f} mm) lies entirely inside "
                    f"sphere {second} (r={radii[second] * 1000:.0f} mm): it can never be "
                    "the nearest bumper"
                )
                break

    for bumper in graph.bumpers:
>>>>>>> Stashed changes
        if bumper.kind == CONE:
            first, second = bumper.spheres
            slack = separation[first, second] - abs(radii[first] - radii[second])
            if slack <= 1e-6:
                problems.append(
<<<<<<< Updated upstream
                    f"{label}: centres {separation[first, second] * 1000:.1f} mm apart but "
=======
                    f"{bumper}: centres {separation[first, second] * 1000:.1f} mm apart but "
>>>>>>> Stashed changes
                    f"radii differ by {abs(radii[first] - radii[second]) * 1000:.1f} mm, so "
                    "the cone has no lateral surface"
                )
        elif bumper.kind in (PRYSMOID, QUAD):
            for triangle in bumper.triangles():
                used = np.array(triangle)
                _, _, _, exists = tangent_plane(centers[used][None], radii[used][None])
                if not bool(exists[0]):
<<<<<<< Updated upstream
                    triangle_label = ", ".join(sphere_label(s, names) for s in triangle)
                    problems.append(
                        f"{label}: triangle [{triangle_label}] admits no tangent plane -- "
                        "its centres are near-collinear or its radii vary faster than "
                        "their spacing, so the slab has no flat face"
=======
                    problems.append(
                        f"{bumper}: triangle {triangle} admits no tangent plane -- its "
                        "centres are near-collinear or its radii vary faster than their "
                        "spacing, so the slab has no flat face"
>>>>>>> Stashed changes
                    )
    return problems


<<<<<<< Updated upstream
def mirror_asymmetry(
    centers: np.ndarray,
    radii: np.ndarray,
    tolerance: float = 2e-3,
    names: Sequence[str] | None = None,
) -> list[str]:
=======
def mirror_asymmetry(centers: np.ndarray, radii: np.ndarray, tolerance: float = 2e-3) -> list[str]:
>>>>>>> Stashed changes
    """Spheres with no mirror twin across x = 0."""
    mirrored = centers.copy()
    mirrored[:, 0] *= -1.0
    problems: list[str] = []
    for index, (center, radius) in enumerate(zip(mirrored, radii)):
        distance = np.linalg.norm(centers - center, axis=1)
        twin = int(np.argmin(distance))
        if distance[twin] > tolerance or abs(radii[twin] - radius) > tolerance:
            problems.append(
<<<<<<< Updated upstream
                f"sphere {sphere_label(index, names)} at {np.round(centers[index], 3).tolist()} "
                f"has no twin (nearest mirror candidate {sphere_label(twin, names)}, off by "
=======
                f"sphere {index} at {np.round(centers[index], 3).tolist()} has no twin "
                f"(nearest mirror candidate {twin}, off by "
>>>>>>> Stashed changes
                f"{distance[twin] * 1000:.1f} mm / {abs(radii[twin] - radius) * 1000:.1f} mm r)"
            )
    return problems


<<<<<<< Updated upstream
def joint_crossing_cones(
    graph: BumperGraph,
    parent_joints: np.ndarray,
    kintree_parents: np.ndarray,
) -> list[dict]:
    """Rim edges whose two spheres the tool binds to different joints.

    Every rim edge of every bumper is a cone bumper in the graph (``BumperGraph`` creates
    one for each slab edge as well as each explicit 2-connection), so walking the cones
    covers every edge exactly once.

    Returns one entry per crossing edge: ``cone`` (bumper index), ``joints`` (the two
    joint names) and ``seam`` -- True when the two joints are kintree parent and child.
    A seam is where a chain hands over from one bone to the next and is the one crossing
    that has to exist; the spheres on either side rotate about the shared joint, and the
    error that buys is the 0-2 cm band in the handoff table. Anything else -- two joints
    that are siblings, or grandparent and grandchild, or on different limbs -- rotates
    about two different centres and the cone between them tears. Failing entries are
    listed first.
    """
    parent_joints = np.asarray(parent_joints, dtype=np.int64)
    kintree_parents = np.asarray(kintree_parents, dtype=np.int64)
    crossings: list[dict] = []
    for bumper_index, bumper in enumerate(graph.bumpers):
        if bumper.kind != CONE:
            continue
        first, second = bumper.spheres
        joint_a = int(parent_joints[first])
        joint_b = int(parent_joints[second])
        if joint_a == joint_b:
            continue
        # "seam" is the shared rule the repairs use to admit a cone: parent and child
        # anywhere, and two steps apart on the trunk, where the spine joints are nearly
        # dead and the skin blends smoothly (anatomy.joints_compatible).
        seam = joints_compatible(joint_a, joint_b, kintree_parents)
        crossings.append({
            "cone": bumper_index,
            "spheres": [int(first), int(second)],
            "joints": [JOINT_NAMES[joint_a], JOINT_NAMES[joint_b]],
            "seam": bool(seam),
        })
    crossings.sort(key=lambda entry: (entry["seam"], entry["cone"]))
    return crossings


_FACING_LABELS = {
    (0, 1): "left", (0, -1): "right",
    (1, 1): "up", (1, -1): "down",
    (2, 1): "front", (2, -1): "back",
}


def coverage_clusters(
    signed: np.ndarray,
    edges: np.ndarray,
    selected: np.ndarray,
    points: np.ndarray,
    normals: np.ndarray,
    dominant_joints: np.ndarray,
    owner: np.ndarray,
    *,
    sign: float,
    min_vertices: int = 5,
) -> list[dict]:
    """Connected patches of badly covered vertices, largest first.

    ``selected`` marks the offending vertices; they are grouped by connectivity over
    ``edges`` (both ends selected), and each patch of at least ``min_vertices`` is
    summarised by what a person would want to know to go and look at it: how many
    vertices, the worst value in mm (``sign`` +1 for gaps, -1 for protrusion), where it is
    (centroid and bounding box, metres), which joint skins it, which way it faces (the
    dominant axis of its mean normal, remembering that +X is the body's left) and which
    bumper owns most of it -- the last three being what separates "the belly is uncovered"
    from "the midline column is missing".
    """
    selected = np.asarray(selected, dtype=bool)
    count = len(signed)
    kept = edges[selected[edges[:, 0]] & selected[edges[:, 1]]]
    adjacency = coo_matrix(
        (np.ones(len(kept), dtype=np.int8), (kept[:, 0], kept[:, 1])),
        shape=(count, count),
    )
    _, labels = connected_components(adjacency, directed=False)
    labels = np.where(selected, labels, -1)

    clusters: list[dict] = []
    for label in np.unique(labels[labels >= 0]):
        members = np.flatnonzero(labels == label)
        if len(members) < int(min_vertices):
            continue
        mean_normal = normals[members].mean(axis=0)
        facing_axis = int(np.argmax(np.abs(mean_normal)))
        facing_sign = 1 if mean_normal[facing_axis] >= 0.0 else -1
        joint_counts = np.bincount(dominant_joints[members], minlength=len(JOINT_NAMES))
        owner_counts = np.bincount(owner[members])
        clusters.append({
            "vertices": int(len(members)),
            "worst_mm": round(float(sign * (sign * signed[members]).max()) * 1000, 1),
            "mean_mm": round(float(signed[members].mean()) * 1000, 1),
            "centroid": np.round(points[members].mean(axis=0), 3).tolist(),
            "bbox_min": np.round(points[members].min(axis=0), 3).tolist(),
            "bbox_max": np.round(points[members].max(axis=0), 3).tolist(),
            "joint": JOINT_NAMES[int(np.argmax(joint_counts))],
            "facing": _FACING_LABELS[(facing_axis, facing_sign)],
            "bumper": int(np.argmax(owner_counts)),
            "vertex_indices": members.tolist(),
        })
    clusters.sort(key=lambda entry: (-entry["vertices"], -abs(entry["worst_mm"])))
    return clusters


=======
>>>>>>> Stashed changes
# ------------------------------------------------------------------------------ driver


def validate(
    centers: np.ndarray,
    radii: np.ndarray,
    connections: Iterable[Sequence[int]],
    *,
    body: Body | None = None,
    gender: str = "male",
    thresholds: Thresholds | None = None,
    poses: np.ndarray | None = None,
    pose_vertex_stride: int = 3,
    walk_budget: int = 3,
    check_symmetry: bool = True,
<<<<<<< Updated upstream
    names: Sequence[str] | None = None,
    expected_parent_joints: Sequence[int] | np.ndarray | None = None,
    min_cluster_vertices: int = 5,
) -> Report:
    """Run the whole suite. ``poses`` is (P, 72); pass None to skip the pose sweep.

    ``names`` labels every bumper in the report by sphere name instead of index.
    ``expected_parent_joints`` (N,) is the joint the generator meant each sphere to bind
    to; when given, the tool's own binding rule is checked against it.
    """
=======
) -> Report:
    """Run the whole suite. ``poses`` is (P, 72); pass None to skip the pose sweep."""
>>>>>>> Stashed changes
    centers = np.ascontiguousarray(centers, dtype=np.float64)
    radii = np.ascontiguousarray(radii, dtype=np.float64)
    connections = [tuple(int(index) for index in c) for c in connections]
    limits = thresholds or Thresholds()
    body = body or load_body(gender)
<<<<<<< Updated upstream
    names = None if names is None else [str(name) for name in names]
    if names is not None and len(names) != len(centers):
        raise ValueError(f"{len(names)} names for {len(centers)} spheres")
    report = Report()

    graph = BumperGraph(len(centers), connections)
    report.graph = graph
    report.names = names

    def label(index: int) -> str:
        return bumper_label(graph, index, names)

=======
    report = Report()

    graph = BumperGraph(len(centers), connections)
>>>>>>> Stashed changes
    counts = graph.counts()
    triangles = sum(1 for c in connections if len(c) == 3)
    quads = sum(1 for c in connections if len(c) == 4)
    doubles = sum(1 for c in connections if len(c) == 2)
    report.stats["counts"] = {
        "spheres": len(centers), "doubles": doubles, "triangles": triangles,
        "quads": quads, "bumpers": dict(counts),
    }

    report.add(
        "budget/spheres", len(centers) <= limits.max_spheres,
        f"{len(centers)} spheres (limit {limits.max_spheres})",
        len(centers), limits.max_spheres,
    )
    report.add(
        "budget/triangles", triangles <= limits.max_triangles,
        f"{triangles} three-sphere connections (limit {limits.max_triangles})",
        triangles, limits.max_triangles,
    )
    report.add(
        "budget/quads", quads <= limits.max_quads,
        f"{quads} four-sphere connections (limit {limits.max_quads})",
        quads, limits.max_quads,
    )

    sphere_components = graph.sphere_graph_components()
    report.add(
        "graph/spheres-connected", len(sphere_components) == 1,
        f"{len(sphere_components)} component(s)"
        + ("" if len(sphere_components) == 1
<<<<<<< Updated upstream
           else "; strays " + str([
               [sphere_label(s, names) for s in component]
               for component in sphere_components[1:]
           ][:4])),
=======
           else f"; strays {[c for c in sphere_components[1:]][:4]}"),
>>>>>>> Stashed changes
        len(sphere_components), 1,
    )
    bumper_components = graph.components()
    report.add(
        "graph/bumpers-connected", len(bumper_components) == 1,
        f"{len(bumper_components)} component(s) over "
        f"{len(graph.bumpers)} bumpers "
        f"({counts[SPHERE]} spheres, {counts[CONE]} cones, {counts[PRYSMOID]} prysmoids, "
        f"{counts[QUAD]} quads)",
        len(bumper_components), 1,
    )

<<<<<<< Updated upstream
    degenerate = degenerate_primitives(graph, centers, radii, names)
=======
    degenerate = degenerate_primitives(graph, centers, radii)
>>>>>>> Stashed changes
    report.add(
        "graph/no-degenerate-primitives", not degenerate,
        "none" if not degenerate
        else f"{len(degenerate)}: " + "; ".join(degenerate[:3]),
        len(degenerate), 0,
    )
    report.worst["degenerate"] = degenerate

    if check_symmetry:
<<<<<<< Updated upstream
        asymmetric = mirror_asymmetry(centers, radii, names=names)
=======
        asymmetric = mirror_asymmetry(centers, radii)
>>>>>>> Stashed changes
        report.add(
            "geometry/mirror-symmetric", not asymmetric,
            "every sphere has an x-mirror twin" if not asymmetric
            else f"{len(asymmetric)} unpaired: " + "; ".join(asymmetric[:2]),
            len(asymmetric), 0,
        )

    # ---------------------------------------------------------------- rest-pose fidelity
    modelled = modelled_vertex_mask(body)
    points = body.vertices[modelled]
    signed = graph.distances(points, centers, radii).min(axis=1)

    gap = signed.max()
    protrusion = -signed.min()
    gap_fraction = float((signed > limits.max_gap).mean())
    protrusion_fraction = float((signed < -limits.max_protrusion).mean())
    report.stats["rest"] = {
        "modelled_vertices": int(modelled.sum()),
        "rms": float(np.sqrt(np.mean(signed ** 2))),
        "max_gap": float(gap), "max_protrusion": float(protrusion),
        "gap_fraction": gap_fraction, "protrusion_fraction": protrusion_fraction,
    }
    report.add(
        "fidelity/max-gap", gap <= limits.max_gap,
        f"worst gap {gap * 1000:.1f} mm, {gap_fraction * 100:.2f}% of vertices over "
        f"{limits.max_gap * 1000:.0f} mm (RMS {report.stats['rest']['rms'] * 1000:.1f} mm)",
        float(gap), limits.max_gap,
    )
    report.add(
        "fidelity/gap-fraction", gap_fraction <= limits.max_gap_fraction,
        f"{gap_fraction * 100:.2f}% of modelled vertices sit further than "
        f"{limits.max_gap * 1000:.0f} mm from the collider",
        gap_fraction, limits.max_gap_fraction,
    )
    report.add(
        "fidelity/max-protrusion", protrusion <= limits.max_protrusion,
        f"collider pokes {protrusion * 1000:.1f} mm through the skin at worst, "
        f"{protrusion_fraction * 100:.2f}% of vertices affected",
        float(protrusion), limits.max_protrusion,
    )
    report.add(
        "fidelity/protrusion-fraction", protrusion_fraction <= limits.max_protrusion_fraction,
        f"{protrusion_fraction * 100:.2f}% of modelled vertices are inside the collider "
        f"by more than {limits.max_protrusion * 1000:.0f} mm",
        protrusion_fraction, limits.max_protrusion_fraction,
    )

    # ------------------------------------------------------------------- the walk itself
    modelled_index = np.flatnonzero(modelled)
    remap = -np.ones(len(body.vertices), dtype=np.int64)
    remap[modelled_index] = np.arange(len(modelled_index))
    edges = surface_edges(body.faces)
    edges = edges[(remap[edges[:, 0]] >= 0) & (remap[edges[:, 1]] >= 0)]
    edges = remap[edges]

    owner, hops, tally = walk_breaks(
        graph, centers, radii, points, edges, budget=walk_budget
    )
    unreachable = hops < 0
    expensive = unreachable | (hops > walk_budget)
    break_fraction = float(expensive.mean())
    histogram = {
        str(count): int(((hops == count)).sum())
        for count in range(0, walk_budget + 1)
    }
    histogram[f">{walk_budget}"] = int((hops > walk_budget).sum())
    histogram["unreachable"] = int(unreachable.sum())
    report.stats["walk"] = {
        "edges": int(len(edges)),
        "budget": int(walk_budget),
        "over_budget": int(expensive.sum()),
        "fraction": break_fraction,
        "unreachable": int(unreachable.sum()),
        "mean_hops": float(hops[~unreachable].mean()) if (~unreachable).any() else float("nan"),
        "histogram": histogram,
        "distinct_bumpers_used": int(len(np.unique(owner))),
        "bumpers_total": len(graph.bumpers),
    }
    report.worst["walk_breaks"] = sorted(
<<<<<<< Updated upstream
        (
            {
                "pair": [label(pair[0]), label(pair[1])],
                "pair_indices": list(pair),
                "edges": count,
            }
            for pair, count in tally.items()
        ),
=======
        ({"pair": list(pair), "edges": count} for pair, count in tally.items()),
>>>>>>> Stashed changes
        key=lambda entry: -entry["edges"],
    )[:15]
    report.add(
        "watchdog/walk-break-rate", break_fraction <= limits.max_walk_break_fraction,
        f"{expensive.sum()} of {len(edges)} surface steps ({break_fraction * 100:.2f}%) need "
        f"more than {walk_budget} hops ({unreachable.sum()} unreachable at all); "
        f"mean walk {report.stats['walk']['mean_hops']:.2f} hops"
        + ("" if not report.worst["walk_breaks"]
           else "; worst " + " / ".join(
               f"{entry['pair'][0]}<->{entry['pair'][1]} x{entry['edges']}"
               for entry in report.worst["walk_breaks"][:3])),
        break_fraction, limits.max_walk_break_fraction,
    )
    # A sphere buried inside its own cones is normal and harmless -- along a straight limb
    # every interior sphere of a chain is covered by the two cones that meet on it, and the
    # walk simply passes through. A *cone or slab* that owns nothing is a different matter:
    # it is a connection that buys no surface, so it is pure cost.
    dead = sorted(set(range(len(graph.bumpers))) - set(owner.tolist()))
    dead_connections = [i for i in dead if graph.bumpers[i].kind != SPHERE]

    # A connection owning no surface is not automatically waste. Some exist to shorten the
    # walk: the cone from the pubic node to the bottom of the trunk covers nothing, because
    # the node sits inside its neighbour, but without it a cloth node crossing the groin
    # takes a detour the solver cannot afford. So a dead connection counts against the
    # topology only when it is *also* not load-bearing -- when deleting it would leave its
    # own endpoints within the walk budget anyway.
    # Only explicit 2-connections are judged. A cone that exists solely because it is the
    # rim of a slab was never a choice the author made -- it comes with the quad, and the
    # only way to remove it is to remove the quad, which is a different question.
    explicit = {
        tuple(sorted(connection)) for connection in graph.connections if len(connection) == 2
    }
    wasteful = []
    for bumper_index in dead_connections:
        if tuple(sorted(graph.bumpers[bumper_index].spheres)) not in explicit:
            continue
        spheres = graph.bumpers[bumper_index].spheres
        without = BumperGraph(
            graph.num_spheres,
            [c for c in graph.connections if set(spheres) - set(c) or len(c) != len(spheres)],
        )
        hop = without.hop_distances()
        first = without.index_of(SPHERE, (spheres[0],))
        second = without.index_of(SPHERE, (spheres[-1],))
        detour = hop[first, second]
        if 0 <= detour <= walk_budget:
            wasteful.append(bumper_index)
<<<<<<< Updated upstream
    report.worst["dead_bumpers"] = [label(i) for i in dead]
    report.worst["dead_bumper_indices"] = list(dead)
=======
    report.worst["dead_bumpers"] = [str(graph.bumpers[i]) for i in dead]
>>>>>>> Stashed changes
    report.add(
        "watchdog/no-wasteful-connections", not wasteful,
        ("every connection either owns surface or shortens the walk"
         if not wasteful
         else f"{len(wasteful)} connection(s) own no surface and are not load-bearing: "
<<<<<<< Updated upstream
              + ", ".join(label(i) for i in wasteful[:6]))
=======
              + ", ".join(str(graph.bumpers[i]) for i in wasteful[:6]))
>>>>>>> Stashed changes
        + f" ({len(dead_connections) - len(wasteful)} more own nothing but keep the walk "
          f"short; {len(dead) - len(dead_connections)} spheres are buried inside their own "
          "cones, which is expected along a chain)",
        len(wasteful), 0,
    )

    # Where the failures are matters more than how many there are: 3% of vertices badly
    # covered is a different problem when it is the whole lower back than when it is a ring
    # around each armpit.
    dominant = np.argmax(body.skin_weights, axis=1)[modelled]
    broken_at_vertex = np.zeros(len(points), dtype=bool)
    broken_at_vertex[edges[expensive].ravel()] = True
    by_region = []
    for joint_index in sorted(set(dominant.tolist())):
        selected = dominant == joint_index
        if selected.sum() < 10:
            continue
        region_signed = signed[selected]
        by_region.append({
            "joint": JOINT_NAMES[joint_index],
            "vertices": int(selected.sum()),
            "max_gap_mm": round(float(region_signed.max()) * 1000, 1),
            "max_protrusion_mm": round(float(-region_signed.min()) * 1000, 1),
            "rms_mm": round(float(np.sqrt(np.mean(region_signed ** 2))) * 1000, 1),
            "walk_break_pct": round(float(broken_at_vertex[selected].mean()) * 100, 2),
        })
    report.stats["by_region"] = sorted(by_region, key=lambda entry: -entry["rms_mm"])

<<<<<<< Updated upstream
    # And the same question at the resolution of a patch rather than a joint: which
    # connected pieces of skin are badly covered, where they are, and which way they face.
    normals = body.normals[modelled]
    for key, mask, sign in (
        ("gap_clusters", signed > limits.max_gap, 1.0),
        ("protrusion_clusters", signed < -limits.max_protrusion, -1.0),
    ):
        clusters = coverage_clusters(
            signed, edges, mask, points, normals, dominant, owner,
            sign=sign, min_vertices=min_cluster_vertices,
        )
        for cluster in clusters:
            cluster["bumper_label"] = label(cluster["bumper"])
            cluster["vertex_indices"] = modelled_index[cluster["vertex_indices"]].tolist()
        report.stats[key] = clusters

    # ------------------------------------------------------------------ harmful overlap
    overlap = harmful_overlap(graph, centers, radii, walk_radius=walk_budget)
    worst_entries = [entry for entry in overlap["per_bumper"] if entry["harmful"] > 0]
    global_overlap = float(overlap["global_fraction"])
    report.stats["overlap"] = {
        "global_fraction": global_overlap,
        "exposed_samples": overlap["exposed_total"],
        "harmful_samples": overlap["harmful_total"],
        "walk_radius": overlap["walk_radius"],
        "samples_per_bumper": overlap["samples_per_bumper"],
        "bumpers_affected": len(worst_entries),
        "worst_bumper_fraction": worst_entries[0]["fraction"] if worst_entries else 0.0,
    }
    report.worst["non_adjacent_overlap"] = [
        {
            "a": label(entry["bumper"]),
            "b": label(entry["into"]),
            "a_index": entry["bumper"],
            "b_index": entry["into"],
            "fraction": round(entry["fraction"], 4),
            "into_fraction": round(entry["into_fraction"], 4),
            "exposed": entry["exposed"],
            "harmful": entry["harmful"],
        }
        for entry in worst_entries[:15]
    ]
    report.worst["buried_bumpers"] = [
        {
            "bumper": label(entry["bumper"]),
            "bumper_index": entry["bumper"],
            "dead_fraction": round(entry["dead_fraction"], 4),
        }
        for entry in overlap["buried"][:5]
    ]
    report.add(
        "watchdog/non-adjacent-overlap", global_overlap <= limits.max_non_adjacent_overlap,
        f"{overlap['harmful_total']} of {overlap['exposed_total']} locally exposed surface "
        f"samples ({global_overlap * 100:.2f}%) lie inside a bumper more than "
        f"{walk_budget} hops away; {len(worst_entries)} of {len(graph.bumpers)} bumpers "
        "affected"
        + ("" if not worst_entries
           else f"; worst {report.worst['non_adjacent_overlap'][0]['a']} "
                f"{worst_entries[0]['fraction'] * 100:.1f}% into "
                f"{report.worst['non_adjacent_overlap'][0]['b']}"),
        global_overlap, limits.max_non_adjacent_overlap,
    )

    # ------------------------------------------------------------------------- binding
    parent_joints, _, _ = assign_parent_joint_like_tool(centers, body.joints, body.parents)
    report.stats["binding"] = {
        "joints_used": sorted(set(int(j) for j in parent_joints)),
        "parent_joints": [int(j) for j in parent_joints],
    }
    if expected_parent_joints is not None:
        expected = np.asarray(expected_parent_joints, dtype=np.int64)
        if len(expected) != len(centers):
            raise ValueError(
                f"{len(expected)} expected parent joints for {len(centers)} spheres"
            )
        mismatched = np.flatnonzero(expected != parent_joints)
        mismatches = [
            {
                "sphere": sphere_label(int(index), names),
                "sphere_index": int(index),
                "intended": JOINT_NAMES[int(expected[index])],
                "tool": JOINT_NAMES[int(parent_joints[index])],
            }
            for index in mismatched
        ]
        report.worst["binding_mismatches"] = mismatches
        report.add(
            "animation/binding-as-intended", not mismatches,
            f"every one of {len(centers)} spheres binds to the joint it was placed for"
            if not mismatches
            else f"{len(mismatches)} of {len(centers)} spheres bind elsewhere: "
                 + ", ".join(
                     f"{entry['sphere']} meant {entry['intended']} got {entry['tool']}"
                     for entry in mismatches[:6]
                 )
                 + ("" if len(mismatches) <= 6 else ", ..."),
            len(mismatches), 0,
        )

    crossings = joint_crossing_cones(graph, parent_joints, body.parents)
    for entry in crossings:
        entry["label"] = label(entry["cone"])
    unrelated = [entry for entry in crossings if not entry["seam"]]
    seams = [entry for entry in crossings if entry["seam"]]
    report.worst["joint_crossing_cones"] = crossings
    cone_count = counts[CONE]
    report.add(
        "animation/one-bone-per-cone", not unrelated,
        f"{cone_count - len(crossings)} of {cone_count} rim edges stay on one bone; "
        f"{len(seams)} cross a chain seam (parent-child, or two trunk joints; allowed); "
        f"{len(unrelated)} span unrelated joints"
        + ("" if not unrelated
           else ": " + "; ".join(
               f"{entry['label']} spans {entry['joints'][0]} / {entry['joints'][1]}"
               for entry in unrelated[:5]
           ) + ("" if len(unrelated) <= 5 else "; ...")),
        len(unrelated), 0,
=======
    overlaps = non_adjacent_overlap(graph, centers, radii)
    worst_overlap = overlaps[0][2] if overlaps else 0.0
    report.worst["non_adjacent_overlap"] = [
        {"a": a, "b": b, "fraction": round(f, 4)} for a, b, f in overlaps[:15]
    ]
    report.add(
        "watchdog/non-adjacent-overlap", worst_overlap <= limits.max_non_adjacent_overlap,
        f"worst unlinked interpenetration {worst_overlap * 100:.1f}%"
        + ("" if not overlaps
           else f" ({overlaps[0][0]} into {overlaps[0][1]}); "
                f"{len(overlaps)} unlinked pairs touch at all"),
        worst_overlap, limits.max_non_adjacent_overlap,
>>>>>>> Stashed changes
    )

    # -------------------------------------------------------------------- the pose sweep
    if poses is not None and len(poses):
<<<<<<< Updated upstream
=======
        parent_joints, _, _ = assign_parent_joint_like_tool(centers, body.joints, body.parents)
        report.stats["binding"] = {
            "joints_used": sorted(set(int(j) for j in parent_joints)),
        }
>>>>>>> Stashed changes
        sampled = np.flatnonzero(modelled)[::pose_vertex_stride]
        per_pose = []
        for pose_index, theta in enumerate(poses):
            posed = load_body(body.gender, theta=theta)
            posed_centers = pose_spheres_like_tool(
                centers, parent_joints, body.joints, posed.joints, posed.rotations
            )
            posed_signed = graph.distances(
                posed.vertices[sampled], posed_centers, radii
            ).min(axis=1)
            per_pose.append({
                "pose": pose_index,
                "max_protrusion": float(-posed_signed.min()),
                "protrusion_fraction": float(
                    (posed_signed < -limits.max_protrusion).mean()
                ),
                "max_gap": float(posed_signed.max()),
                "rms": float(np.sqrt(np.mean(posed_signed ** 2))),
            })
        worst_pose = max(per_pose, key=lambda entry: entry["max_protrusion"])
        worst_protrusion = worst_pose["max_protrusion"]
        worst_fraction = max(entry["protrusion_fraction"] for entry in per_pose)
        report.stats["posed"] = {
            "poses": len(per_pose),
            "vertices_per_pose": int(len(sampled)),
            "mean_max_protrusion": float(
                np.mean([entry["max_protrusion"] for entry in per_pose])
            ),
            "worst": worst_pose,
        }
        report.worst["poses"] = sorted(
            per_pose, key=lambda entry: -entry["max_protrusion"]
        )[:8]
        report.add(
            "animation/posed-protrusion", worst_protrusion <= limits.posed_max_protrusion,
            f"over {len(per_pose)} poses the collider pokes at worst "
            f"{worst_protrusion * 1000:.1f} mm through the skin (pose {worst_pose['pose']}); "
            f"mean over poses {report.stats['posed']['mean_max_protrusion'] * 1000:.1f} mm",
            worst_protrusion, limits.posed_max_protrusion,
        )
        report.add(
            "animation/posed-protrusion-fraction",
            worst_fraction <= limits.posed_max_protrusion_fraction,
            f"worst pose has {worst_fraction * 100:.2f}% of sampled vertices inside the "
            "collider",
            worst_fraction, limits.posed_max_protrusion_fraction,
        )
    return report
<<<<<<< Updated upstream


# ------------------------------------------------------------------------ formatting


def _cluster_table(title: str, clusters: list[dict], label, limit: int = 8) -> list[str]:
    if not clusters:
        return [f"{title}: none"]
    lines = [
        f"{title} ({len(clusters)} of >= 5 vertices, largest {limit} shown):",
        f"  {'verts':>5} {'worst':>7} {'joint':<12} {'facing':<6} "
        f"{'centroid (m)':<24} {'bbox (m)':<48} owner",
    ]
    for cluster in clusters[:limit]:
        centroid = "({:+.3f}, {:+.3f}, {:+.3f})".format(*cluster["centroid"])
        bbox = "({:+.3f}, {:+.3f}, {:+.3f}) .. ({:+.3f}, {:+.3f}, {:+.3f})".format(
            *cluster["bbox_min"], *cluster["bbox_max"]
        )
        lines.append(
            f"  {cluster['vertices']:5d} {cluster['worst_mm']:6.1f}mm {cluster['joint']:<12} "
            f"{cluster['facing']:<6} {centroid:<24} {bbox:<48} {label(cluster['bumper'])}"
        )
    return lines


def format_report(report: Report, *, names: Sequence[str] | None = None) -> str:
    """The whole report as one text block: checks, histogram, tables and worst lists.

    ``names`` relabels every bumper index kept in the report's worst lists and cluster
    tables; when omitted the names the report was validated with are used, and when there
    were none, indices. Check details were composed at validation time and keep whatever
    labels they were given then.
    """
    names = report.names if names is None else [str(name) for name in names]
    graph = report.graph

    def label(index) -> str:
        if index is None:
            return "-"
        if graph is None:
            return str(index)
        return bumper_label(graph, int(index), names)

    lines = [report.text(), ""]

    walk = report.stats.get("walk")
    if walk:
        lines += ["walk hop histogram   " + json.dumps(walk["histogram"]), ""]

    if report.stats.get("by_region"):
        lines.append(
            f"  {'region':<12} {'verts':>6} {'rms':>7} {'gap':>8} {'protr':>8} {'walkbrk':>8}"
        )
        for entry in report.stats["by_region"]:
            lines.append(
                f"  {entry['joint']:<12} {entry['vertices']:6d} {entry['rms_mm']:7.1f} "
                f"{entry['max_gap_mm']:8.1f} {entry['max_protrusion_mm']:8.1f} "
                f"{entry['walk_break_pct']:7.2f}%"
            )

    if "gap_clusters" in report.stats:
        lines += [""] + _cluster_table("gap clusters", report.stats["gap_clusters"], label)
    if "protrusion_clusters" in report.stats:
        lines += [""] + _cluster_table(
            "protrusion clusters", report.stats["protrusion_clusters"], label
        )

    if report.worst.get("walk_breaks"):
        lines += ["", "worst walk breaks:"]
        for entry in report.worst["walk_breaks"][:10]:
            pair = entry.get("pair_indices")
            first, second = (
                (label(pair[0]), label(pair[1])) if pair else tuple(entry["pair"])
            )
            lines.append(f"  {first} <-> {second}  x{entry['edges']}")

    if report.worst.get("non_adjacent_overlap"):
        lines += ["", "worst harmful overlaps (share of a bumper's exposed surface swallowed "
                      "by a bumper beyond the walk radius):"]
        for entry in report.worst["non_adjacent_overlap"][:10]:
            first = label(entry["a_index"]) if "a_index" in entry else entry["a"]
            second = label(entry["b_index"]) if "b_index" in entry else entry["b"]
            lines.append(
                f"  {first}: {entry['fraction'] * 100:5.1f}% "
                f"({entry.get('harmful', '?')} of {entry.get('exposed', '?')} samples), "
                f"mostly into {second}"
            )

    if report.worst.get("buried_bumpers"):
        lines += ["", "most buried connections (share of surface inside any other bumper):"]
        for entry in report.worst["buried_bumpers"][:5]:
            name = (
                label(entry["bumper_index"]) if "bumper_index" in entry else entry["bumper"]
            )
            lines.append(f"  {name}: {entry['dead_fraction'] * 100:5.1f}%")

    crossings = report.worst.get("joint_crossing_cones")
    if crossings is not None:
        unrelated = [entry for entry in crossings if not entry["seam"]]
        seams = [entry for entry in crossings if entry["seam"]]
        lines += [
            "",
            f"joint-crossing cones: {len(unrelated)} across unrelated joints, "
            f"{len(seams)} on chain seams",
        ]
        for entry in unrelated:
            lines.append(
                f"  FAIL {label(entry['cone'])} spans "
                f"{entry['joints'][0]} / {entry['joints'][1]}"
            )
        for entry in seams[:12]:
            lines.append(
                f"  seam {label(entry['cone'])} spans "
                f"{entry['joints'][0]} / {entry['joints'][1]}"
            )
        if len(seams) > 12:
            lines.append(f"  ... {len(seams) - 12} more seams")

    if report.worst.get("binding_mismatches"):
        lines += ["", "binding mismatches (intended -> tool):"]
        for entry in report.worst["binding_mismatches"][:12]:
            lines.append(f"  {entry['sphere']}: {entry['intended']} -> {entry['tool']}")

    if report.worst.get("poses"):
        lines += ["", "worst poses:"]
        for entry in report.worst["poses"][:6]:
            lines.append(
                f"  pose {entry['pose']:3d}  protrusion {entry['max_protrusion'] * 1000:6.1f} mm "
                f"({entry['protrusion_fraction'] * 100:5.2f}% of verts), "
                f"gap {entry['max_gap'] * 1000:6.1f} mm"
            )
    return "\n".join(lines)
=======
>>>>>>> Stashed changes
