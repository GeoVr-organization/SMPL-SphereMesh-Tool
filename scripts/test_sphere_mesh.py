"""Tests for src/sphere_mesh.py, the SphereMesh class the GUI tool and SMFitting share.

Pins the signed distances against the exact closed forms in src/topology/bumpers.py,
the vectorised signedDistances() against the per-point signedDistance(), and the
degenerate primitives (coincident centres, one sphere containing another, collinear
centres, no tangent plane) against what they reduce to.

Run with: uv run python scripts/test_sphere_mesh.py
"""
from __future__ import annotations

import contextlib
import io
import itertools
import json
import sys
import warnings
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.sphere_mesh import SphereMesh  # noqa: E402
from src.topology.bumpers import prysmoid_distances, round_cone_distances  # noqa: E402

# Metres. Everything here is exact maths, so the margins are rounding-level: the
# capsuloid's sqrt(|pa|^2 - t^2) and bumpers' closed form round differently by up to
# ~1e-9 m near the axis.
DISTANCE_TOLERANCE = 1e-8

TEMPLATE_PATHS = [
    REPOSITORY_ROOT / "data" / "template_male_1.0.json",
    REPOSITORY_ROOT / "data" / "template_female_1.0.json",
]
# fitted_smpl0.txt holds the two prysmoids with no tangent plane (s19,s21,s34 and
# s46,s47,s58) on which the old plane iteration oscillated.
EXAMPLE_PATHS = [
    REPOSITORY_ROOT / "SMFitting" / "examples" / "fitted_smpl0.txt",
    REPOSITORY_ROOT / "SMFitting" / "examples" / "template_male_2.0.txt",
]

FAILURES: list[str] = []


def check(condition: bool, label: str, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f"  ({detail})" if detail else ""))
    if not condition:
        FAILURES.append(label)


@contextlib.contextmanager
def quiet_degenerate_warnings():
    """Swallow the class's one-off degenerate-primitive warnings while loading meshes."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def load_json_mesh(path: Path) -> SphereMesh:
    """A SphereMesh over a GUI template's spheres and connections."""
    data = json.loads(path.read_text())
    centers = np.array([sphere["center"] for sphere in data["spheres"]], dtype=np.float64)
    radii = np.array([sphere["radius"] for sphere in data["spheres"]], dtype=np.float64)
    connections = [connection["indices"] for connection in data["connections"]]
    with quiet_degenerate_warnings():
        return SphereMesh.fromSpheresAndConnections(centers, radii, connections)


def load_text_mesh(path: Path) -> SphereMesh:
    """A SphereMesh read from a "Sphere Mesh 2.0" text file."""
    mesh = SphereMesh()
    with quiet_degenerate_warnings():
        mesh.loadFromText(path.read_text())
    return mesh


def all_meshes() -> list[tuple[str, SphereMesh]]:
    return ([(path.name, load_json_mesh(path)) for path in TEMPLATE_PATHS]
            + [(path.name, load_text_mesh(path)) for path in EXAMPLE_PATHS])


def primitive_spheres(primitive) -> list:
    spheres = [primitive.sA, primitive.sB]
    if isinstance(primitive, SphereMesh.Prysmoid):
        spheres.append(primitive.sC)
    return spheres


def probe_points(primitive, rng: np.random.Generator) -> np.ndarray:
    """Random points around a primitive, plus points on its axes, edges and midplane.

    The box reaches 3 cm past the spheres, so it covers the corners where two side
    planes meet; the on-axis and on-midplane points are where rounding used to give NaN.
    """
    spheres = primitive_spheres(primitive)
    centers = np.array([sphere.center for sphere in spheres])
    radii = np.array([sphere.radius for sphere in spheres])
    low = (centers - radii[:, None]).min(axis=0) - 0.03
    high = (centers + radii[:, None]).max(axis=0) + 0.03
    random_points = rng.uniform(low, high, size=(600, 3))

    fractions = np.linspace(-0.2, 1.2, 15)[:, None]
    on_edges = [centers[first] + fractions * (centers[second] - centers[first])
                for first, second in itertools.combinations(range(len(spheres)), 2)]
    weights = rng.dirichlet(np.ones(len(spheres)), size=40)
    inside_hull = weights @ centers
    return np.vstack([random_points, *on_edges, inside_hull, centers])


def exact_distances(primitive, points: np.ndarray) -> np.ndarray:
    """bumpers.py's closed forms for this primitive."""
    spheres = primitive_spheres(primitive)
    centers = np.array([sphere.center for sphere in spheres])
    radii = np.array([sphere.radius for sphere in spheres])
    if len(spheres) == 2:
        return round_cone_distances(points, centers[:1], radii[:1], centers[1:], radii[1:])[:, 0]
    return prysmoid_distances(points, centers[None], radii[None])[:, 0]


def test_distances_against_exact() -> None:
    print("\nSigned distances: exact, vectorised, finite")
    rng = np.random.default_rng(7)
    for mesh_name, mesh in all_meshes():
        worst_exact = {"Capsuloid": 0.0, "Prysmoid": 0.0}
        worst_vectorised = 0.0
        non_finite = 0
        query_count = 0
        for primitive in [*mesh.capsuloids, *mesh.prysmoids]:
            points = probe_points(primitive, rng)
            scalar = np.array([primitive.signedDistance(point) for point in points])
            vectorised = primitive.signedDistances(points)
            exact = exact_distances(primitive, points)
            kind = "Prysmoid" if isinstance(primitive, SphereMesh.Prysmoid) else "Capsuloid"
            worst_exact[kind] = max(worst_exact[kind], float(np.max(np.abs(scalar - exact))))
            worst_vectorised = max(worst_vectorised, float(np.max(np.abs(scalar - vectorised))))
            non_finite += int(np.count_nonzero(~np.isfinite(scalar)))
            query_count += len(points)
        check(non_finite == 0, f"{mesh_name}: every distance finite",
              f"{non_finite} NaN/inf of {query_count}")
        check(max(worst_exact.values()) < DISTANCE_TOLERANCE,
              f"{mesh_name}: signedDistance equals bumpers' exact distance",
              f"max |diff| capsuloid {worst_exact['Capsuloid']:.1e} m, "
              f"prysmoid {worst_exact['Prysmoid']:.1e} m")
        check(worst_vectorised < DISTANCE_TOLERANCE,
              f"{mesh_name}: signedDistances equals signedDistance per point",
              f"max |diff| {worst_vectorised:.1e} m")


def test_mesh_union_and_closest_sphere() -> None:
    print("\nWhole mesh: signedDistances is the union closestSphere answers from")
    rng = np.random.default_rng(11)
    for mesh_name, mesh in all_meshes():
        centers = np.array([sphere.center for sphere in mesh.spheres])
        points = rng.uniform(centers.min(axis=0) - 0.05, centers.max(axis=0) + 0.05, size=(400, 3))
        union = mesh.signedDistances(points)
        per_primitive = np.min([[primitive.signedDistance(point) for point in points]
                                for primitive in [*mesh.capsuloids, *mesh.prysmoids]], axis=0)
        closest_sphere_distances = []
        contribution_sums = []
        for point in points:
            sphere, contributions = mesh.closestSphere(point)
            closest_sphere_distances.append(sphere.signedDistance(point))
            contribution_sums.append(sum(weight for _, weight in contributions))
        check(np.max(np.abs(union - per_primitive)) < DISTANCE_TOLERANCE,
              f"{mesh_name}: signedDistances equals min over primitives",
              f"max |diff| {np.max(np.abs(union - per_primitive)):.1e} m")
        # The face's closest sphere comes out of a projection, so it carries a few
        # more rounding steps than the plane distance itself.
        closest_gap = float(np.max(np.abs(np.array(closest_sphere_distances) - union)))
        check(closest_gap < 1e-7,
              f"{mesh_name}: closestSphere's sphere is at the union distance",
              f"max |diff| {closest_gap:.1e} m")
        contribution_gap = float(np.max(np.abs(np.array(contribution_sums) - 1.0)))
        check(contribution_gap < 1e-9, f"{mesh_name}: contributions sum to 1",
              f"max |sum - 1| {contribution_gap:.1e}")


def test_tangent_planes_and_geometry() -> None:
    print("\nTangent planes and buildGeometry")
    for mesh_name, mesh in all_meshes():
        worst_tangency = 0.0
        wrong_side = 0
        faceless = []
        worst_surface = 0.0
        worst_outside = 0.0
        non_finite_vertices = 0
        for prysmoid in mesh.prysmoids:
            spheres = primitive_spheres(prysmoid)
            if not prysmoid.hasFaces:
                faceless.append(",".join(f"s{sphere.index}" for sphere in spheres))
            else:
                for plane in (prysmoid.upperPlane, prysmoid.lowerPlane):
                    heights = [plane.signedDistance(sphere.center) + sphere.radius for sphere in spheres]
                    worst_tangency = max(worst_tangency, max(abs(height) for height in heights))
                upper_side = float(np.dot(prysmoid.upperPlane.normal, prysmoid.midplane.normal))
                lower_side = float(np.dot(prysmoid.lowerPlane.normal, prysmoid.midplane.normal))
                wrong_side += int(not (upper_side > 0.0 > lower_side))
            vertices, triangles = prysmoid.buildGeometry(2, 32)
            non_finite_vertices += int(np.count_nonzero(~np.isfinite(vertices)))
            vertex_distances = prysmoid.signedDistances(vertices)
            if prysmoid.hasFaces:
                worst_surface = max(worst_surface, float(np.max(np.abs(vertex_distances[:6]))))
            worst_outside = max(worst_outside, float(np.max(vertex_distances)))
        contained_area = 0.0
        for capsuloid in mesh.capsuloids:
            vertices, triangles = capsuloid.buildGeometry(16, 32)
            non_finite_vertices += int(np.count_nonzero(~np.isfinite(vertices)))
            if capsuloid.isContained:
                # The capsuloid is its containing sphere, which is drawn on its own;
                # its cone collapses onto the axis and must not show.
                corners = vertices[triangles]
                areas = np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0],
                                                corners[:, 2] - corners[:, 0]), axis=1)
                contained_area = max(contained_area, float(areas.max()))
                continue
            worst_surface = max(worst_surface, float(np.max(np.abs(capsuloid.signedDistances(vertices)))))
        contained_count = sum(capsuloid.isContained for capsuloid in mesh.capsuloids)
        if contained_count:
            check(contained_area < 1e-15,
                  f"{mesh_name}: {contained_count} contained capsuloids draw no area",
                  f"max triangle area x2 {contained_area:.1e} m^2")
        check(worst_tangency < 1e-12, f"{mesh_name}: both face planes touch all three spheres",
              f"max |n.c + r - k| {worst_tangency:.1e} m")
        check(wrong_side == 0, f"{mesh_name}: upper face above the midplane, lower below",
              f"{wrong_side} prysmoids wrong")
        check(non_finite_vertices == 0, f"{mesh_name}: buildGeometry vertices finite")
        check(worst_surface < DISTANCE_TOLERANCE and worst_outside < DISTANCE_TOLERANCE,
              f"{mesh_name}: drawn vertices lie on the surface they draw",
              f"cone and face vertices max |d| {worst_surface:.1e} m, "
              f"prysmoid vertices max d {worst_outside:.1e} m")
        if mesh_name == "fitted_smpl0.txt":
            check(faceless == ["s19,s21,s34", "s46,s47,s58"],
                  f"{mesh_name}: the two prysmoids without a tangent plane are recognised",
                  ", ".join(faceless))
        for resolution in (1, 2, 3):
            spheres = mesh.getInterpolatedSpheres(resolution)
            finite = all(np.all(np.isfinite(sphere.center)) and np.isfinite(sphere.radius)
                         for sphere in spheres)
            check(finite, f"{mesh_name}: getInterpolatedSpheres({resolution}) all finite",
                  f"{len(spheres)} spheres")


def test_index_order() -> None:
    print("\nPrysmoid distances do not depend on how its spheres are numbered")
    rng = np.random.default_rng(3)
    mesh = load_json_mesh(TEMPLATE_PATHS[0])
    worst = 0.0
    for prysmoid in mesh.prysmoids:
        spheres = primitive_spheres(prysmoid)
        points = probe_points(prysmoid, rng)
        reference = prysmoid.signedDistances(points)
        for permutation in itertools.permutations(range(3)):
            relabelled = [SphereMesh.Sphere(new_index, spheres[old_index].center, spheres[old_index].radius)
                          for new_index, old_index in enumerate(permutation)]
            permuted = SphereMesh.Prysmoid(*relabelled, 1.0)
            worst = max(worst, float(np.max(np.abs(permuted.signedDistances(points) - reference))))
    check(worst < DISTANCE_TOLERANCE, "all 6 relabellings of every template prysmoid agree",
          f"max |diff| {worst:.1e} m")


def build_quietly(factory):
    """Run factory, returning (result, warning messages, stderr text)."""
    stderr = io.StringIO()
    with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stderr(stderr):
        warnings.simplefilter("always")
        result = factory()
    return result, [str(warning.message) for warning in caught], stderr.getvalue()


def test_degenerate_primitives() -> None:
    print("\nDegenerate primitives reduce to what they contain")
    rng = np.random.default_rng(5)
    points = rng.uniform(-0.3, 0.3, size=(500, 3))
    Sphere = SphereMesh.Sphere

    big = Sphere(0, (0.0, 0.0, 0.0), 0.10)
    small_inside = Sphere(1, (0.03, 0.0, 0.0), 0.05)
    capsuloid, messages, stderr = build_quietly(lambda: SphereMesh.Capsuloid(big, small_inside))
    gap = float(np.max(np.abs(capsuloid.signedDistances(points) - big.signedDistances(points))))
    scalar_gap = max(abs(capsuloid.signedDistance(point) - big.signedDistance(point)) for point in points)
    check(capsuloid.isContained and max(gap, scalar_gap) < DISTANCE_TOLERANCE,
          "contained capsuloid is the containing sphere", f"max |diff| {max(gap, scalar_gap):.1e} m")
    check(len(messages) == 1 and stderr == "", "contained capsuloid warns once, prints nothing",
          f"{len(messages)} warnings, stderr {stderr!r}")
    _, contributions = capsuloid.closestSphereWithContributions(points[0])
    check(contributions == [(0, 1.0), (1, 0.0)], "contained capsuloid contributions: all on the big sphere",
          str(contributions))

    coincident_small = Sphere(1, (0.0, 0.0, 0.0), 0.04)
    capsuloid, messages, stderr = build_quietly(lambda: SphereMesh.Capsuloid(coincident_small, Sphere(2, (0.0, 0.0, 0.0), 0.08)))
    distances = capsuloid.signedDistances(points)
    vertices, _ = capsuloid.buildGeometry(16, 32)
    check(np.all(np.isfinite(distances)) and np.all(np.isfinite(vertices))
          and np.max(np.abs(distances - (np.linalg.norm(points, axis=1) - 0.08))) < DISTANCE_TOLERANCE,
          "coincident capsuloid is its bigger sphere, geometry finite")

    sphere_a = Sphere(0, (0.0, 0.0, 0.0), 0.05)
    sphere_b = Sphere(1, (0.1, 0.0, 0.0), 0.04)
    sphere_c = Sphere(2, (0.25, 0.0, 0.0), 0.06)
    prysmoid, messages, stderr = build_quietly(lambda: SphereMesh.Prysmoid(sphere_a, sphere_b, sphere_c, 1.0))
    edge_union = np.min([edge.signedDistances(points) for edge in prysmoid.edgeCapsuloids], axis=0)
    exact = exact_distances(prysmoid, points)
    check(not prysmoid.hasFaces and np.max(np.abs(prysmoid.signedDistances(points) - edge_union)) < DISTANCE_TOLERANCE
          and np.max(np.abs(prysmoid.signedDistances(points) - exact)) < DISTANCE_TOLERANCE,
          "collinear prysmoid is its edge capsuloids, matches bumpers",
          f"max |diff| to bumpers {np.max(np.abs(prysmoid.signedDistances(points) - exact)):.1e} m")
    vertices, triangles = prysmoid.buildGeometry(2, 32)
    check(np.all(np.isfinite(vertices)) and stderr == "" and len(messages) == 1,
          "collinear prysmoid: finite geometry, one warning, nothing printed",
          f"{len(messages)} warnings")

    no_plane = load_text_mesh(EXAMPLE_PATHS[0])
    faceless = [prysmoid for prysmoid in no_plane.prysmoids if not prysmoid.hasFaces][0]
    probe = probe_points(faceless, rng)
    gap = float(np.max(np.abs(faceless.signedDistances(probe) - exact_distances(faceless, probe))))
    check(gap < DISTANCE_TOLERANCE, "prysmoid without a tangent plane matches bumpers",
          f"max |diff| {gap:.1e} m")


def main() -> int:
    test_distances_against_exact()
    test_mesh_union_and_closest_sphere()
    test_tangent_planes_and_geometry()
    test_index_order()
    test_degenerate_primitives()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
