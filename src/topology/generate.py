"""Building a sphere-mesh from placement rules rather than from typed coordinates.

A topology is described as a ``TopologySpec``: named spheres, and connections between
names. Names rather than indices, for three reasons. They survive insertion -- adding a
sphere mid-chain does not renumber the connection list; they make mirroring mechanical --
``L_upperarm_2`` reflects to ``R_upperarm_2`` and every connection reflects with it; and a
connection list written as ``("L_chest_out", "L_chest_in", "L_waist_in")`` says what it is,
which an index triple never does.

Three builders cover the body:

``chain`` walks a bone and drops spheres on the *medial* point of each cross-section, which
is where the largest inscribed sphere sits -- not on the bone axis, which in a thigh or a
shin is off by two to four centimetres.

``sheet`` lays spheres along the medial locus of a horizontal trunk cut, so that the slabs
between them thicken the trunk front-to-back. This is the part that makes a cheap torso
possible: a triangle of three spheres covers a *sheet* of surface on both sides, so a wide
flat region costs a handful of triangles rather than a forest of cones.

``stub`` terminates a chain where the model stops -- wrist, ankle, neck -- with a single
sphere that closes the surface without pretending to model a hand, foot or head.

Radii come from the skin, not from a fit: a sphere whose radius is the distance to the
nearest body vertex touches the surface and cannot cross it. That gives zero sphere
protrusion by construction, and leaves the optimiser with only the interesting problem --
closing gaps -- rather than the one it is bad at.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Sequence

import numpy as np

from .anatomy import Body, JOINT_INDEX, bone_stations, inscribed_radius, section_at
from .bumpers import BumperGraph

__all__ = [
    "SphereSpec",
    "TopologySpec",
    "chain",
    "sheet",
    "stub",
    "mirror",
    "snap_radii_to_skin",
    "relieve_protrusion",
    "refine",
    "symmetrize",
    "blobs",
    "ring",
]

#: A sphere placed exactly on a joint centre binds to an unpredictable bone: the tool's
#: closest-segment search sees two bones at distance zero and the tie falls to whichever
#: floating-point comparison happens to win. Backing off along the incoming bone by this
#: much makes every case deterministic and parent-bound.
JOINT_CENTRE_BACKOFF = 0.015


@dataclass
class SphereSpec:
    """One sphere, named, with the group it belongs to for reporting."""

    name: str
    center: np.ndarray
    radius: float
    group: str = ""

    def mirrored(self, name: str) -> "SphereSpec":
        center = self.center.copy()
        center[0] = -center[0]
        return replace(self, name=name, center=center, group=_mirror_name(self.group))


@dataclass
class TopologySpec:
    """Named spheres plus connections between names, resolvable to the wire format."""

    spheres: list[SphereSpec] = field(default_factory=list)
    connections: list[tuple[str, ...]] = field(default_factory=list)

    def add(self, spheres: Iterable[SphereSpec], connections: Iterable[Sequence[str]] = ()) -> "TopologySpec":
        known = {sphere.name for sphere in self.spheres}
        for sphere in spheres:
            if sphere.name in known:
                raise ValueError(f"duplicate sphere name {sphere.name!r}")
            known.add(sphere.name)
            self.spheres.append(sphere)
        for connection in connections:
            self.connections.append(tuple(connection))
        return self

    def connect(self, *names: str) -> "TopologySpec":
        self.connections.append(tuple(names))
        return self

    def index_of(self) -> dict[str, int]:
        return {sphere.name: index for index, sphere in enumerate(self.spheres)}

    def assemble(self) -> tuple[np.ndarray, np.ndarray, list[tuple[int, ...]], list[str]]:
        """Resolve to ``(centers, radii, connections, names)``.

        Connections are deduplicated and validated here rather than at use: a repeated
        triple would create a duplicate prysmoid bumper, which is invisible in the file and
        doubles that slab's cost in the solver.
        """
        lookup = self.index_of()
        missing = {
            name for connection in self.connections for name in connection
            if name not in lookup
        }
        if missing:
            raise ValueError(f"connections reference unknown spheres: {sorted(missing)}")

        seen: set[tuple[int, ...]] = set()
        resolved: list[tuple[int, ...]] = []
        for connection in self.connections:
            indices = tuple(lookup[name] for name in connection)
            if len(set(indices)) != len(indices):
                raise ValueError(f"connection {connection} repeats a sphere")
            key = tuple(sorted(indices))
            if key in seen:
                continue
            seen.add(key)
            resolved.append(indices)

        centers = np.array([sphere.center for sphere in self.spheres], dtype=np.float64)
        radii = np.array([sphere.radius for sphere in self.spheres], dtype=np.float64)
        names = [sphere.name for sphere in self.spheres]
        return centers, radii, resolved, names

    def summary(self) -> str:
        by_group: dict[str, int] = {}
        for sphere in self.spheres:
            by_group[sphere.group] = by_group.get(sphere.group, 0) + 1
        arity: dict[int, int] = {}
        for connection in self.connections:
            arity[len(connection)] = arity.get(len(connection), 0) + 1
        groups = ", ".join(f"{name} {count}" for name, count in sorted(by_group.items()))
        return (
            f"{len(self.spheres)} spheres ({groups}); "
            f"{arity.get(2, 0)} doubles, {arity.get(3, 0)} triples, {arity.get(4, 0)} quads"
        )


def _mirror_name(name: str) -> str:
    if name.startswith("L_"):
        return "R_" + name[2:]
    if name.startswith("R_"):
        return "L_" + name[2:]
    return name


# ------------------------------------------------------------------------- builders


def chain(
    body: Body,
    parent: str,
    child: str,
    parameters: Sequence[float],
    *,
    name: str,
    group: str = "",
    vertex_mask: np.ndarray | None = None,
    radius_scale: float = 1.0,
    on_axis: bool = False,
    connect: bool = True,
    half_thickness: float | None = None,
    min_vertices: int = 12,
<<<<<<< Updated upstream
    method: str | None = None,
=======
>>>>>>> Stashed changes
) -> tuple[list[SphereSpec], list[tuple[str, str]]]:
    """Spheres along a bone at the given parameters, with the cones between them.

    ``parameters`` are positions along the bone, 0 at ``parent`` and 1 at ``child``; values
    outside that range are allowed and often wanted, since a deltoid starts before the
    shoulder joint. Set ``on_axis`` to place on the bone axis instead of the section's
    medial point, which is only right for a limb whose cross-section is centred on its bone.
    """
<<<<<<< Updated upstream
    section_options = {} if method is None else {"method": method}
    stations = bone_stations(
        body, parent, child, parameters=parameters, vertex_mask=vertex_mask,
        half_thickness=half_thickness, min_vertices=min_vertices, **section_options,
=======
    stations = bone_stations(
        body, parent, child, parameters=parameters, vertex_mask=vertex_mask,
        half_thickness=half_thickness, min_vertices=min_vertices,
>>>>>>> Stashed changes
    )
    if len(stations) != len(parameters):
        found = {round(station.parameter, 4) for station in stations}
        raise ValueError(
            f"chain {name}: no usable cross-section at parameter(s) "
            f"{[p for p in parameters if round(p, 4) not in found]} of bone "
            f"{parent}->{child}; the vertex mask is probably too tight there"
        )

    spheres = [
        SphereSpec(
            name=f"{name}_{index}",
            center=station.axis_point if on_axis else station.medial_point,
            radius=float(station.touch_radius * radius_scale),
            group=group or name,
        )
        for index, station in enumerate(stations)
    ]
    links = (
        [(spheres[index].name, spheres[index + 1].name) for index in range(len(spheres) - 1)]
        if connect else []
    )
    return spheres, links


def sheet(
    body: Body,
    heights: Sequence[float],
    columns: Sequence[float],
    *,
    name: str,
    column_names: Sequence[str] | None = None,
    group: str = "",
    vertex_mask: np.ndarray | None = None,
    radius_scale: float = 1.0,
    half_thickness: float = 0.018,
<<<<<<< Updated upstream
    method: str | None = None,
=======
>>>>>>> Stashed changes
) -> tuple[list[SphereSpec], dict[tuple[int, int], str]]:
    """A grid of spheres on the trunk's medial locus.

    ``columns`` are *reach fractions* in [0, 1]: 0 places a column at the row's widest
    inscribed circle, 1 at the furthest its spheres can reach, and the values in between
    interpolate that. Naming columns by how far they reach rather than by a position means
    the sheet follows the body as it narrows from ribcage to waist with no per-row tuning,
    and it makes a swallowed sphere impossible -- see the derivation at the call site.

    ``column_names`` labels the columns, and it exists so that a sheet can be built as a
    half plus a centre line and mirrored. Naming a column ``L`` gives every sphere in it a
    name beginning ``L_``, which is what ``mirror`` keys on -- so the right half, and every
    connection touching it, is generated rather than measured, and the result is exactly
    symmetric. Building both halves by measurement instead produces near-mirrors, and every
    connection written to a bare column index silently attaches *both* limbs to the same
    side of the body.

    Returns the spheres and a ``(row, column) -> name`` map, so the caller can write the
    connection pattern over the grid rather than over generated names.
    """
    up = np.array([0.0, 1.0, 0.0])
    spheres: list[SphereSpec] = []
    grid: dict[tuple[int, int], str] = {}
    diagnostics: list[dict] = []

<<<<<<< Updated upstream
    section_options = {} if method is None else {"method": method}
=======
>>>>>>> Stashed changes
    for row, height in enumerate(heights):
        section = section_at(
            body, np.array([0.0, float(height), 0.0]), up,
            half_thickness=half_thickness, vertex_mask=vertex_mask,
<<<<<<< Updated upstream
            reference_up=np.array([1.0, 0.0, 0.0]), symmetrize=True, **section_options,
=======
            reference_up=np.array([1.0, 0.0, 0.0]), symmetrize=True,
>>>>>>> Stashed changes
        )
        if len(section.planar) < 12:
            raise ValueError(f"sheet {name}: too few vertices in the cut at y={height}")
        ridge, _ = section.medial_axis()
        if len(ridge) == 0:
            raise ValueError(f"sheet {name}: no medial locus in the cut at y={height}")

        # Column positions are chosen by the *radius* they should carry, not by a fraction
        # of the medial span, and the depth at each is solved for. Two reasons.
        #
        # Fractions of the span do not survive a row whose span is short. At the top of the
        # trunk the medial segment is only about 33 mm long while the spheres are 87 mm in
        # radius, so columns at 0 and 0.55 of the span land 18 mm apart with radii differing
        # by 18 mm -- one sphere entirely inside the other, a cone with no lateral surface,
        # and a bumper that can never be nearest to anything. Asking instead for "the place
        # where the inscribed radius has fallen to 0.8 of its maximum" adapts to each row
        # automatically and keeps the spheres apart by construction.
        #
        # And reading positions off the sampled ridge makes the sheet inherit the ridge's
        # noise; solving for them keeps it exactly symmetric.
        half_width = float(np.abs(section.planar[:, 0]).max())
        probe = np.linspace(0.0, half_width, 400)
        _, profile = section.best_transverse(probe)
        peak_index = int(np.argmax(profile))
        peak = float(profile[peak_index])

        # Columns are placed by how far each one *reaches*, which is the only formulation
        # that cannot produce a swallowed sphere.
        #
        # Write reach(p) = p + r(p): how far along the row the sphere at p extends. Sphere k
        # is contained in an inner sphere j exactly when (p_k - p_j) + r_k <= r_j, and that
        # rearranges to reach(p_k) <= reach(p_j). So as long as every column reaches
        # strictly further than the one inside it, no column can be swallowed -- whatever
        # the section's shape. Placing by radius ratio instead only bounds it indirectly and
        # fails on a steep profile: a column asked for 0.82 of the peak radius landed 18.2
        # mm out with its radius 18.3 mm smaller, which is inside by a tenth of a millimetre.
        #
        # ``columns`` are fractions of the reach still available past the widest inscribed
        # circle, so 0 is the middle of the row and 1 its far edge.
        reach = probe + profile
        limit_index = int(np.argmax(reach))
        start = float(reach[peak_index])
        available = float(reach[limit_index]) - start

        targets = []
        for column in columns:
            fraction = float(column)
            if fraction <= 0.0 or available <= 1e-9:
                targets.append(float(probe[peak_index]))
                continue
            wanted = start + min(fraction, 1.0) * available
            segment = reach[peak_index:limit_index + 1]
            beyond = np.flatnonzero(segment >= wanted)
            if len(beyond) == 0:
                targets.append(float(probe[limit_index]))
                continue
            index = peak_index + int(beyond[0])
            if index == peak_index:
                targets.append(float(probe[peak_index]))
                continue
            low, high = reach[index - 1], reach[index]
            span = high - low
            blend = 0.0 if span < 1e-12 else (wanted - low) / span
            targets.append(float(probe[index - 1] + blend * (probe[index] - probe[index - 1])))

        spread = np.diff(np.asarray(targets))
        if len(spread) and spread.min() < 1e-3:
            raise ValueError(
                f"sheet {name}: the cut at y={height:.3f} is too narrow for "
                f"{len(columns)} columns -- they land within "
                f"{spread.min() * 1000:.1f} mm of each other, which would emit spheres "
                "inside one another. Use fewer columns here, or move the row to where the "
                "section is wider."
            )

        planar, clearance = section.best_transverse(np.array(targets))
        centers = section.to_world(planar)
        half_span = float(np.abs(ridge[:, 0]).max())

        labels = (
            [f"c{index}" for index in range(len(columns))] if column_names is None
            else list(column_names)
        )
        if len(labels) != len(columns):
            raise ValueError("column_names must have one entry per column")

        for column_index, (center, radius) in enumerate(zip(centers, clearance)):
            label = labels[column_index]
            sphere_name = (
                f"{label}_{name}_r{row}" if column_names is not None
                else f"{name}_r{row}{label}"
            )
            spheres.append(
                SphereSpec(
                    name=sphere_name,
                    center=center,
                    radius=float(radius * radius_scale),
                    group=group or name,
                )
            )
            grid[(row, column_index)] = sphere_name
        diagnostics.append({
            "row": row, "height": float(height),
            "medial_half_span": half_span,
            "radii": [float(value) for value in clearance],
        })
    sheet.last_diagnostics = diagnostics
    return spheres, grid


def stub(
    body: Body,
    parent: str,
    child: str,
    *,
    name: str,
    group: str = "",
    beyond: float = 0.0,
    vertex_mask: np.ndarray | None = None,
    radius_scale: float = 1.0,
) -> SphereSpec:
    """A single closing sphere at (or just past) a joint where the model stops."""
    origin, axis, length = body.bone(parent, child)
    center = origin + (length + float(beyond)) * axis
    return SphereSpec(
        name=name,
        center=center,
        radius=float(inscribed_radius(body, center[None])[0] * radius_scale),
        group=group or name,
    )


def mirror(spec: TopologySpec, *, only_prefix: str = "L_") -> TopologySpec:
    """Reflect every ``L_``-prefixed sphere and connection across x = 0.

    Mirroring at spec level rather than after assembly is what keeps the result exactly
    symmetric: reflecting names cannot drift, whereas reflecting coordinates and matching
    them back up by proximity -- which is what the tool's mirror-edit does -- accumulates
    the millimetre errors visible in the hand-authored templates.
    """
    existing = {sphere.name for sphere in spec.spheres}
    added: list[SphereSpec] = []
    for sphere in spec.spheres:
        if not sphere.name.startswith(only_prefix):
            continue
        twin = _mirror_name(sphere.name)
        if twin in existing:
            continue
        existing.add(twin)
        added.append(sphere.mirrored(twin))
    spec.spheres.extend(added)

    seen = {tuple(sorted(connection)) for connection in spec.connections}
    for connection in list(spec.connections):
        if not any(name.startswith(only_prefix) for name in connection):
            continue
        twin = tuple(_mirror_name(name) for name in connection)
        if tuple(sorted(twin)) in seen:
            continue
        seen.add(tuple(sorted(twin)))
        spec.connections.append(twin)
    return spec


# ------------------------------------------------------------------------ refinement


def snap_radii_to_skin(
    body: Body, centers: np.ndarray, *, margin: float = 0.0
) -> np.ndarray:
    """The largest radius at each centre that does not cross the skin, less ``margin``.

    A sphere at this radius is tangent to the surface from inside. It is the right starting
    point for every sphere in the template, and the reason the generator needs no protrusion
    term of its own.
    """
    return np.maximum(inscribed_radius(body, centers) - float(margin), 1e-3)


def relieve_protrusion(
    body: Body,
    centers: np.ndarray,
    radii: np.ndarray,
    connections: Sequence[Sequence[int]],
    *,
    vertex_mask: np.ndarray | None = None,
    target: float = 0.002,
    max_iterations: int = 40,
    shrink: float = 0.5,
    min_radius_fraction: float = 0.6,
    floor: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Shrink radii until no bumper pokes more than ``target`` through the skin.

    Individual spheres cannot protrude if their radii came from ``snap_radii_to_skin``, but
    the cones and slabs *between* them can, wherever the surface is concave along the way --
    the waist, the armpit, the inside of a knee. This finds the worst offending vertex,
    charges the protrusion to the spheres of the bumper that owns it, and shrinks those,
    repeating until the target is met.

    Shrinking is deliberately blunt. The alternative -- moving centres -- trades a
    protrusion for a gap somewhere else and needs the full optimiser; this is the cheap
    guaranteed-terminating pass that runs after it.
    """
    centers = np.ascontiguousarray(centers, dtype=np.float64)
    radii = np.array(radii, dtype=np.float64)
    # A floor, because shrinking is a blunt instrument and an unbounded one is worse. A
    # sphere charged repeatedly for a protrusion it only partly causes can end up at a
    # tenth of its inscribed radius, at which point it is swallowed by its neighbour, its
    # cones have no lateral surface, and the region it was covering is simply gone.
    # The floor is relative to the radii the caller *placed*, when it passes them, not to
    # the ones this function is handed. Applying a fraction at each stage compounds: 70% of
    # the fit, which was itself 70% of the placement, is half the sphere.
    floor = (
        np.maximum(np.asarray(floor, dtype=np.float64), 1e-3) if floor is not None
        else np.maximum(radii * float(min_radius_fraction), 1e-3)
    )
    graph = BumperGraph(len(centers), connections)
    points = body.vertices if vertex_mask is None else body.vertices[vertex_mask]

    history = []
    for iteration in range(int(max_iterations)):
        distance = graph.distances(points, centers, radii)
        owner = np.argmin(distance, axis=1)
        nearest = distance[np.arange(len(owner)), owner]
        worst_point = int(np.argmin(nearest))
        worst = float(-nearest[worst_point])
        history.append(worst)
        if worst <= float(target):
            return radii, {
                "iterations": iteration, "converged": True,
                "final_protrusion": worst, "history": history,
            }

        # Charge the excess to each sphere in proportion to how much of the offending
        # surface point it actually accounts for. A cone that pokes out near its thin end
        # is the thin sphere's fault, and shrinking the fat one as well only opens a gap
        # somewhere the geometry was correct -- which is what a flat split does: the blunt
        # version cost 5 mm of RMS and doubled the fraction of badly covered vertices.
        involved = np.array(graph.bumpers[owner[worst_point]].spheres)
        blame = _medial_weights(points[worst_point], centers[involved], radii[involved])
        excess = worst - float(target)
        radii[involved] -= float(shrink) * excess * blame / max(blame.max(), 1e-9)
        radii = np.maximum(radii, floor)
    return radii, {
        "iterations": int(max_iterations), "converged": False,
        "final_protrusion": history[-1] if history else float("nan"), "history": history,
    }


def _medial_weights(
    point: np.ndarray, centers: np.ndarray, radii: np.ndarray, steps: int = 64
) -> np.ndarray:
    """Barycentric weights of the medial point nearest to ``point``.

    Which sphere of a bumper is responsible for the surface at a given place is exactly the
    weight the minimiser of ``|p - c(lambda)| - r(lambda)`` puts on it. Solved by sampling
    the simplex, which is cheap because it is asked about one point at a time.
    """
    count = len(centers)
    if count == 1:
        return np.ones(1)
    if count == 2:
        parameter = np.linspace(0.0, 1.0, steps * 4)[:, None]
        medial = centers[0] + parameter * (centers[1] - centers[0])
        radius = radii[0] + parameter[:, 0] * (radii[1] - radii[0])
        best = int(np.argmin(np.linalg.norm(medial - point, axis=1) - radius))
        share = float(parameter[best, 0])
        return np.array([1.0 - share, share])
    grid = []
    for first in range(steps + 1):
        for second in range(steps + 1 - first):
            weight = np.zeros(count)
            weight[0] = first / steps
            weight[1] = second / steps
            weight[2] = 1.0 - weight[0] - weight[1]
            grid.append(weight)
    weights = np.array(grid)
    medial = weights[:, :3] @ centers[:3]
    radius = weights[:, :3] @ radii[:3]
    best = int(np.argmin(np.linalg.norm(medial - point, axis=1) - radius))
    return weights[best]


def refine(
    body: Body,
    centers: np.ndarray,
    radii: np.ndarray,
    connections: Sequence[Sequence[int]],
    *,
    max_offset: float = 0.010,
    protrusion_weight: float = 10.0,
    band_max: float = 0.05,
    vertex_mask: np.ndarray | None = None,
    max_outer: int = 40,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Hand the placement to ``src.sphere_fit.optimize_spheres`` for a least-squares pass.

    The generator's job is to get the structure and the neighbourhood right; this closes
    the remaining millimetres. ``max_offset`` caps how far a centre may travel, which keeps
    the binding that was chosen for it valid -- a sphere allowed to wander across a joint
    boundary would be rebound to a different bone on load and undo the animation reasoning
    that placed it.
    """
    from src.sphere_fit import optimize_spheres, select_band

    points = body.vertices if vertex_mask is None else body.vertices[vertex_mask]
    band = select_band(points, centers, radii, connections, band_max=band_max)
    if band.size < 4:
        raise ValueError("refine: the band is empty; the placement is nowhere near the body")
    new_centers, new_radii, info = optimize_spheres(
        centers, radii, points[band],
        max_offset=float(max_offset),
        protrusion_weight=float(protrusion_weight),
        max_outer=int(max_outer),
        connections=list(connections),
    )
    info["band_size"] = int(band.size)
    return new_centers, new_radii, info


def symmetrize(
    centers: np.ndarray, radii: np.ndarray, names: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Force exact bilateral symmetry, pairing spheres by name rather than by proximity.

    Every step that touches geometry breaks symmetry a little: the least-squares pass moves
    a left sphere and its right twin by different amounts because their assigned vertices
    differ by the mesh's own sampling, and the shrink pass charges a protrusion to one side
    only. Two millimetres is enough to fail a strict mirror check, and it is exactly the
    drift visible in the hand-authored templates.

    Pairing by name is what makes this exact. Matching a reflected centre to its nearest
    neighbour -- which is what the tool's mirror-edit does -- has to guess when two spheres
    are close together, and guesses wrong precisely where the spheres are dense.
    ``L_``-prefixed names pair with their ``R_`` twins; everything else is on the midline
    and is snapped to x = 0.
    """
    centers = np.array(centers, dtype=np.float64)
    radii = np.array(radii, dtype=np.float64)
    index_of = {name: index for index, name in enumerate(names)}

    for index, name in enumerate(names):
        if not name.startswith("L_"):
            continue
        twin = index_of.get("R_" + name[2:])
        if twin is None:
            continue
        reflected = centers[twin] * np.array([-1.0, 1.0, 1.0])
        mean = 0.5 * (centers[index] + reflected)
        centers[index] = mean
        centers[twin] = mean * np.array([-1.0, 1.0, 1.0])
        mean_radius = 0.5 * (radii[index] + radii[twin])
        radii[index] = radii[twin] = mean_radius

    for index, name in enumerate(names):
        if not (name.startswith("L_") or name.startswith("R_")):
            centers[index, 0] = 0.0
    return centers, radii


def blobs(
    body: Body,
    lower_corner,
    upper_corner,
    count: int,
    *,
    name: str,
    group: str = "",
    resolution: float = 0.006,
    min_separation: float = 0.045,
    connect: bool = True,
) -> tuple[list[SphereSpec], list[tuple[str, str]]]:
    """The largest spheres that fit inside the body within a box, greedily and separated.

    For a *bulge* -- a buttock, a belly, a breast -- neither of the other builders is right.
    A chain wants a bone to walk and a sheet wants a medial locus, and the bulge has
    neither: the largest circle inscribed in a pelvic cross-section sits 120 mm in front of
    the buttock, so every medial construction places the sphere in the wrong place. Measured
    consequence of using one anyway: a 48 mm hole across the seat that a *dense* 900-sphere
    medial cover does not close either, and, if the section is narrowed to the posterior
    vertices to force the issue, spheres of 15 mm radius hugging the skin.

    So this asks the question directly. Grid the box, keep the points inside the body, score
    each by its distance to the nearest surface vertex -- which is the radius a sphere there
    could carry without crossing the skin -- and take the best, then the best that is at
    least ``min_separation`` away, and so on. Inside-ness is the sign of the nearest
    vertex's normal, which is exact enough on a region chosen to be inside to begin with.
    """
    lower_corner = np.asarray(lower_corner, dtype=np.float64)
    upper_corner = np.asarray(upper_corner, dtype=np.float64)
    axes = [
        np.arange(lower_corner[axis], upper_corner[axis] + 1e-9, float(resolution))
        for axis in range(3)
    ]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

    from scipy.spatial import cKDTree

    tree = cKDTree(body.vertices)
    clearance, nearest = tree.query(grid)
    outward = np.einsum("ij,ij->i", grid - body.vertices[nearest], body.normals[nearest])
    inside = outward < 0.0
    grid, clearance = grid[inside], clearance[inside]
    if len(grid) == 0:
        raise ValueError(f"blobs {name}: the box contains no interior points")

    order = np.argsort(-clearance)
    chosen: list[int] = []
    for candidate in order:
        if len(chosen) >= int(count):
            break
        if any(
            np.linalg.norm(grid[candidate] - grid[picked]) < float(min_separation)
            for picked in chosen
        ):
            continue
        chosen.append(int(candidate))
    if len(chosen) < int(count):
        raise ValueError(
            f"blobs {name}: only {len(chosen)} of {count} spheres fit with "
            f"{min_separation * 1000:.0f} mm separation inside the given box"
        )

    # Order along the box's longest axis so the chain reads head to tail.
    extent = upper_corner - lower_corner
    principal = int(np.argmax(np.abs(extent)))
    chosen.sort(key=lambda index: grid[index][principal], reverse=extent[principal] < 0)

    spheres = [
        SphereSpec(
            name=f"{name}_{position}",
            center=grid[index].copy(),
            radius=float(clearance[index]),
            group=group or name,
        )
        for position, index in enumerate(chosen)
    ]
    links = (
        [(spheres[i].name, spheres[i + 1].name) for i in range(len(spheres) - 1)]
        if connect else []
    )
    return spheres, links


def ring(
    body: Body,
    heights: Sequence[float],
    angles: Sequence[float],
    *,
    name: str,
    column_names: Sequence[str],
    group: str = "",
    vertex_mask: np.ndarray | None = None,
    half_thickness: float = 0.018,
    radius_scale: float = 1.0,
    offsets: int = 60,
    min_offset_fraction: float = 0.35,
) -> tuple[list[SphereSpec], dict[tuple[int, int], str]]:
    """A cyclic ring of spheres per horizontal cut, rather than a collinear row.

    A row of spheres whose centres share a depth -- what a coronal medial sheet degenerates
    to in the section plane -- is the wrong family for a section that is only mildly
    elliptical. Solving the minimax placement of K disks against the true trunk outline, a
    collinear ladder and a free ring score very differently on the lower trunk: at y =
    -0.160 the ladder cannot get below 10.9 mm however many disks it is given, while a ring
    of four reaches 4.8 mm. The reason is that the ladder can only ever be wrong in one
    direction -- it has no disk anywhere near the front or back wall -- whereas a ring puts
    one against each.

    ``angles`` are directions in the section plane, in degrees, measured from the +x axis
    towards +z. They should come in mirror pairs so the ring is bilaterally symmetric; the
    caller names them and mirrors the left ones as usual.

    The ring's radius is not a parameter but a solve: the centres sit at a common offset
    from the section's widest inscribed point, and that offset is swept to minimise the
    worst distance from the section outline to the union of the disks. Sweeping one number
    keeps the result deterministic and exactly mirror-symmetric, which a free per-sphere
    optimisation would not be.
    """
    up = np.array([0.0, 1.0, 0.0])
    spheres: list[SphereSpec] = []
    grid: dict[tuple[int, int], str] = {}
    labels = list(column_names)
    if len(labels) != len(angles):
        raise ValueError("column_names must have one entry per angle")

    direction = np.stack(
        [np.cos(np.deg2rad(angles)), np.sin(np.deg2rad(angles))], axis=1
    )                                                                    # (K, 2)

    for row, height in enumerate(heights):
        section = section_at(
            body, np.array([0.0, float(height), 0.0]), up,
            half_thickness=half_thickness, vertex_mask=vertex_mask,
            reference_up=np.array([1.0, 0.0, 0.0]), symmetrize=True,
        )
        if len(section.planar) < 12:
            raise ValueError(f"ring {name}: too few vertices in the cut at y={height}")
        normals, plane_offsets = section.half_planes()

        # The outline, densely resampled, is what the union of disks has to reach.
        from scipy.spatial import ConvexHull

        hull = ConvexHull(section.planar)
        loop = section.planar[hull.vertices]
        outline = np.concatenate([
            loop[index] + np.linspace(0.0, 1.0, 24, endpoint=False)[:, None]
            * (loop[(index + 1) % len(loop)] - loop[index])
            for index in range(len(loop))
        ])

        # The anchor is pinned to the midline rather than taken as the global argmax of
        # the clearance field. The section is symmetrised so that field is symmetric in x
        # and its maximum lies on x = 0, but where the maximum is nearly flat the argmax
        # falls to floating-point noise and lands tens of millimetres off-centre -- which
        # threw whole rings sideways on three rows out of five.
        anchor_planar, anchor_clearance = section.best_transverse(np.array([0.0]))
        anchor = anchor_planar[0]
        widest = float(anchor_clearance[0])

        # A minimum offset, because a near-circular section makes the sweep collapse the
        # ring onto its own anchor: four spheres within 9 mm of each other, mutually
        # swallowed, buying nothing over the one sphere they degenerate to.
        best_offset, best_score = float(min_offset_fraction) * widest, np.inf
        for offset in np.linspace(float(min_offset_fraction) * widest, widest, int(offsets)):
            centers = anchor[None, :] + offset * direction
            clearance = (plane_offsets[None, :] - centers @ normals.T).min(axis=1)
            if (clearance <= 1e-4).any():
                continue
            distance = np.linalg.norm(
                outline[:, None, :] - centers[None, :, :], axis=2
            ) - clearance[None, :]
            score = float(np.abs(distance.min(axis=1)).max())
            if score < best_score:
                best_offset, best_score = float(offset), score

        centers = anchor[None, :] + best_offset * direction
        clearance = (plane_offsets[None, :] - centers @ normals.T).min(axis=1)
        world = section.to_world(centers)
        for column, (center, radius) in enumerate(zip(world, clearance)):
            sphere_name = f"{labels[column]}_{name}_r{row}"
            spheres.append(
                SphereSpec(
                    name=sphere_name, center=center,
                    radius=float(radius * radius_scale), group=group or name,
                )
            )
            grid[(row, column)] = sphere_name
    return spheres, grid
