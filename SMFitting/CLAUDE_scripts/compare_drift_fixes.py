"""Why do s51/s23 (the pelvis spheres) poke out of the lower back after the
fit, and which change to the fit stops it?

Part 1 - the vertices the fitted s51/s23 swallow: where they are, which
primitive the assignment gave them, and what d they get with main.py's abs()
argmin against the true (signed) minimum.

Part 2 - the whole fit under several variants, all scored the same way:
signed distance of every vertex to the fitted sphere mesh (minimum over
primitives, negative = the sphere mesh sticks out past the skin there), and
how far each fitted sphere sticks out of the surface (PyVista implicit
distance, as protrusion_by_sphere.py).

    .venv/Scripts/python.exe SMFitting/CLAUDE_scripts/compare_drift_fixes.py
"""

import contextlib
import io

import numpy as np
import pyvista as pv

from fitting_common import SphereMesh, describe, loadFittingInputs, solveFit

watchedSpheres = (51, 23)
sphereMesh, meshVertices, meshNormals, meshTriangles = loadFittingInputs()
originalCentres = np.array([sphere.center for sphere in sphereMesh.spheres])
originalRadii = np.array([sphere.radius for sphere in sphereMesh.spheres])


def buildMesh(centres, radii):
    """A SphereMesh with the template's connections over new spheres."""
    builtMesh = SphereMesh()
    builtMesh.spheres = [SphereMesh.Sphere(index, centre, radius)
                         for index, (centre, radius) in enumerate(zip(centres, radii))]
    with contextlib.redirect_stderr(io.StringIO()):
        builtMesh.capsuloids = [SphereMesh.Capsuloid(builtMesh.spheres[c.sA.index], builtMesh.spheres[c.sB.index])
                                for c in sphereMesh.capsuloids]
        builtMesh.prysmoids = [SphereMesh.Prysmoid(builtMesh.spheres[p.sA.index], builtMesh.spheres[p.sB.index],
                                                   builtMesh.spheres[p.sC.index], 1.0)
                               for p in sphereMesh.prysmoids]
    return builtMesh


def queryVertices(mesh, useAbs):
    """Per vertex: (winning primitive, contributions, d, alignment), by abs() or
    signed argmin. alignment is n . (p - c) / |p - c| against the closest
    sphere: near 1 when the vertex normal points away from it, as tangency
    assumes; near 0 or below for a vertex the sphere meets side-on."""
    primitives = [*mesh.capsuloids, *mesh.prysmoids]
    results = []
    for vertex, normal in zip(meshVertices, meshNormals):
        signedDistances = np.array([primitive.signedDistance(vertex) for primitive in primitives])
        winner = primitives[int(np.argmin(np.abs(signedDistances) if useAbs else signedDistances))]
        closest, contributions = winner.closestSphereWithContributions(vertex)
        offset = vertex - closest.center
        alignment = float(normal @ offset / max(np.linalg.norm(offset), 1e-12))
        results.append((winner, [(int(index), float(weight)) for index, weight in contributions],
                        closest.signedDistance(vertex), alignment))
    return results


def keptContributions(queries, minimumAlignment):
    """Contributions per vertex, emptied for vertices whose normal disagrees
    with their closest sphere (an empty row adds nothing to the solve)."""
    if minimumAlignment is None:
        return [q[1] for q in queries]
    return [q[1] if q[3] >= minimumAlignment else [] for q in queries]


surface = pv.PolyData(meshVertices, np.hstack([np.full((len(meshTriangles), 1), 3), meshTriangles]).ravel())
goldenAngle = np.pi * (3.0 - np.sqrt(5.0))
fibonacciHeights = 1.0 - 2.0 * (np.arange(600) + 0.5) / 600
fibonacciRings = np.sqrt(1.0 - fibonacciHeights ** 2)
unitDirections = np.column_stack((fibonacciRings * np.cos(goldenAngle * np.arange(600)),
                                  fibonacciRings * np.sin(goldenAngle * np.arange(600)), fibonacciHeights))


def sphereProtrusions(centres, radii):
    """How far each sphere sticks out of the surface, metres (negative = inside)."""
    samplePoints = (centres[:, None, :] + radii[:, None, None] * unitDirections[None]).reshape(-1, 3)
    distances = np.asarray(pv.PolyData(samplePoints).compute_implicit_distance(surface)["implicit_distance"])
    return distances.reshape(len(centres), -1).max(axis=1)


# The implicit distance comes out negative inside for this surface; checked in
# protrusion_by_sphere.py with a torso probe.

# ---------------------------------------------------------------- part 1
baselineQueries = queryVertices(sphereMesh, useAbs=True)
baselineCentres, baselineRadii = solveFit(sphereMesh, meshVertices, meshNormals, [q[1] for q in baselineQueries])
fittedMesh = buildMesh(baselineCentres, baselineRadii)
fittedAbsQueries = queryVertices(fittedMesh, useAbs=True)
fittedSignedQueries = queryVertices(fittedMesh, useAbs=False)

print("PART 1 - vertices the fitted pelvis spheres swallow")
for sphereIndex in watchedSpheres:
    fittedSphere = fittedMesh.spheres[sphereIndex]
    insideDepth = np.array([-fittedSphere.signedDistance(vertex) for vertex in meshVertices])
    swallowed = np.flatnonzero(insideDepth > 0.01)
    originalSphere = sphereMesh.spheres[sphereIndex]
    alreadyInside = sum(originalSphere.signedDistance(meshVertices[i]) < -0.01 for i in swallowed)
    print(f"\ns{sphereIndex}: {len(swallowed)} vertices lie > 1 cm inside the FITTED sphere "
          f"(deepest {insideDepth.max() * 100:.1f} cm); {alreadyInside} of them were > 1 cm inside the original")
    if len(swallowed) == 0:
        continue
    print(f"  their centroid {np.round(meshVertices[swallowed].mean(axis=0), 3)}, "
          f"mean normal {np.round(meshNormals[swallowed].mean(axis=0) / np.linalg.norm(meshNormals[swallowed].mean(axis=0)), 2)}")
    assignedTo = {}
    for i in swallowed:
        name = describe(baselineQueries[i][0])
        assignedTo[name] = assignedTo.get(name, 0) + 1
    print(f"  assigned in the fit to: {sorted(assignedTo.items(), key=lambda item: -item[1])[:6]}")
    givesWeight = sum(any(index == sphereIndex and weight > 1e-9 for index, weight in baselineQueries[i][1])
                      for i in swallowed)
    print(f"  of which give s{sphereIndex} any weight: {givesWeight}")
    absDistances = np.array([fittedAbsQueries[i][2] for i in swallowed])
    signedDistances = np.array([fittedSignedQueries[i][2] for i in swallowed])
    print(f"  d after the fit, as main.py section 11 shows it (abs argmin): median {np.median(absDistances) * 1000:+.1f} mm, "
          f"min {absDistances.min() * 1000:+.1f} mm")
    print(f"  d after the fit, signed minimum (true union):              median {np.median(signedDistances) * 1000:+.1f} mm, "
          f"min {signedDistances.min() * 1000:+.1f} mm")

# ---------------------------------------------------------------- part 2


def score(name, centres, radii):
    mesh = buildMesh(centres, radii)
    distances = np.array([q[2] for q in queryVertices(mesh, useAbs=False)])
    protrusions = sphereProtrusions(centres, radii)
    worstSphere = int(np.argmax(protrusions))
    watched = "  ".join(
        f"s{index} {np.linalg.norm(centres[index] - originalCentres[index]) * 100:4.1f}cm/"
        f"{protrusions[index] * 100:+4.1f}" for index in watchedSpheres)
    print(f"{name:34s} {np.mean(np.abs(distances)) * 1000:6.2f} {np.sum(distances < -0.01):6d} "
          f"{np.sum(distances > 0.01):6d} {distances.min() * 100:+6.1f} "
          f"{protrusions.max() * 100:+5.1f}@s{worstSphere:<2d} {np.sum(protrusions > 0.02):5d}   {watched}")


def iterateFit(rounds, useAbs, centreRegularization=0.05, minimumAlignment=None):
    """Assign against the current fit, refit (anchored to the template), repeat."""
    centres, radii = originalCentres, originalRadii
    for _ in range(rounds):
        queries = queryVertices(buildMesh(centres, radii), useAbs)
        centres, radii = solveFit(sphereMesh, meshVertices, meshNormals,
                                  keptContributions(queries, minimumAlignment),
                                  centreRegularization=centreRegularization)
    return centres, radii


print("\n\nPART 2 - fit variants, scored by the signed distance of every vertex to the result")
print(f"{'variant':34s} {'mean|d|':>6s} {'d<-1cm':>6s} {'d>+1cm':>6s} {'min d':>6s} "
      f"{'worst sphere':>11s} {'>2cm':>5s}   centre shift / protrusion (cm)")
print(f"{'':34s} {'mm':>6s} {'verts':>6s} {'verts':>6s} {'cm':>6s} {'out, cm':>11s} {'sph.':>5s}")
score("template (no fit)", originalCentres, originalRadii)
score("main.py now (abs, reg 0.05)", baselineCentres, baselineRadii)
for centreRegularization in (0.5, 1.0, 2.0):
    centres, radii = solveFit(sphereMesh, meshVertices, meshNormals, [q[1] for q in baselineQueries],
                              centreRegularization=centreRegularization)
    score(f"centre reg {centreRegularization}", centres, radii)
signedQueries = queryVertices(sphereMesh, useAbs=False)
centres, radii = solveFit(sphereMesh, meshVertices, meshNormals, [q[1] for q in signedQueries])
score("signed argmin, 1 pass", centres, radii)
score("abs argmin, 4 passes (refit loop)", *iterateFit(4, useAbs=True))
score("signed argmin, 4 passes (refit loop)", *iterateFit(4, useAbs=False))
score("signed, 4 passes, centre reg 0.5", *iterateFit(4, useAbs=False, centreRegularization=0.5))

# Rejecting vertices whose normal disagrees with their closest sphere.
for rejectionThreshold in (0.3, 0.5):
    rejectedAbs = sum(q[3] < rejectionThreshold for q in baselineQueries)
    rejectedSigned = sum(q[3] < rejectionThreshold for q in signedQueries)
    print(f"-- reject alignment < {rejectionThreshold}: drops {rejectedAbs} vertices (abs) / "
          f"{rejectedSigned} (signed) of {len(meshVertices)}")
    centres, radii = solveFit(sphereMesh, meshVertices, meshNormals,
                              keptContributions(baselineQueries, rejectionThreshold))
    score(f"abs, reject <{rejectionThreshold}, 1 pass", centres, radii)
    centres, radii = solveFit(sphereMesh, meshVertices, meshNormals,
                              keptContributions(signedQueries, rejectionThreshold))
    score(f"signed, reject <{rejectionThreshold}, 1 pass", centres, radii)
    score(f"signed, reject <{rejectionThreshold}, 4 passes",
          *iterateFit(4, useAbs=False, minimumAlignment=rejectionThreshold))
    score(f"signed, reject <{rejectionThreshold}, 4 p, reg 0.5",
          *iterateFit(4, useAbs=False, centreRegularization=0.5, minimumAlignment=rejectionThreshold))
