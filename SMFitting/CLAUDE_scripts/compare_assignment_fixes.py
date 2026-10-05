"""Re-run the whole fit under three vertex-to-primitive assignment rules and
compare what the solve does to the spheres. SphereMesh.py is NOT modified:
the alternatives are evaluated here, from the prysmoid's own planes.

  current    SphereMesh as it is: a prysmoid's face distance is always taken
             from upperPlane, even for a point behind the midplane.
  cull       the commented-out midplane check put back: behind the midplane a
             prysmoid face reports +inf (edges still answer).
  two-sided  behind the midplane the face distance comes from lowerPlane, with
             side planes built from lowerPlane's normal: the prysmoid as the
             convex hull of its three spheres.

    .venv/Scripts/python.exe SMFitting/CLAUDE_scripts/compare_assignment_fixes.py
"""

import numpy as np

from fitting_common import SphereMesh, loadFittingInputs, solveFit

Plane = SphereMesh.Prysmoid.Plane


def lowerSidePlanes(prysmoid):
    """m1, m2, m0 for the lower face, in edgeCapsuloidAt()'s test order.

    The same construction as the constructor's, with lowerPlane's normal and
    the opposite direction sign, so for equal radii they equal the upper ones.
    """
    lowerNormal, direction = prysmoid.lowerPlane.normal, -prysmoid.direction
    sphereA, sphereB, sphereC = prysmoid.sA, prysmoid.sB, prysmoid.sC
    return (
        (Plane(np.cross(sphereC.center - sphereA.center, lowerNormal) * direction, sphereC.center), (sphereA, sphereC)),
        (Plane(np.cross(sphereA.center - sphereB.center, lowerNormal) * direction, sphereA.center), (sphereA, sphereB)),
        (Plane(np.cross(sphereB.center - sphereC.center, lowerNormal) * direction, sphereB.center), (sphereB, sphereC)),
    )


def lowerFaceQuery(prysmoid, point):
    """(signedDistance, closestSphere, contributions) against the lower face."""
    for sidePlane, (firstSphere, secondSphere) in lowerSidePlanes(prysmoid):
        if sidePlane.isBehind(point):
            edge = SphereMesh.Capsuloid(firstSphere, secondSphere)
            closest, contributions = edge.closestSphereWithContributions(point)
            return closest.signedDistance(point), closest, contributions

    lowerNormal = prysmoid.lowerPlane.normal
    faceDistance = prysmoid.lowerPlane.signedDistance(point)
    closestPoint = point - faceDistance * lowerNormal
    cosTilt = float(np.dot(lowerNormal, prysmoid.midplane.normal))
    pointOnMidplane = closestPoint - (prysmoid.midplane.signedDistance(closestPoint) / cosTilt) * lowerNormal
    barycentricCoords = prysmoid.computeBarycentricCoordinates(pointOnMidplane)
    radius = sum(weight * sphere.radius for weight, sphere in
                 zip(barycentricCoords, (prysmoid.sA, prysmoid.sB, prysmoid.sC)))
    closest = SphereMesh.Sphere(-1, pointOnMidplane, radius)
    contributions = [(prysmoid.sA.index, barycentricCoords[0]),
                     (prysmoid.sB.index, barycentricCoords[1]),
                     (prysmoid.sC.index, barycentricCoords[2])]
    return faceDistance, closest, contributions


def primitiveQuery(primitive, point, rule):
    """(signedDistance used by the argmin, contributions, distance of the result)."""
    isPrysmoid = isinstance(primitive, SphereMesh.Prysmoid)
    behindFace = (isPrysmoid and primitive.midplane.isBehind(point)
                  and primitive.edgeCapsuloidAt(point) is None)
    if behindFace and rule == "cull":
        return float("inf"), [], np.nan
    if isPrysmoid and primitive.midplane.isBehind(point) and rule == "two-sided":
        distance, closest, contributions = lowerFaceQuery(primitive, point)
        return distance, contributions, closest.signedDistance(point)
    closest, contributions = primitive.closestSphereWithContributions(point)
    return primitive.signedDistance(point), contributions, closest.signedDistance(point)


def assign(sphereMesh, meshVertices, rule):
    primitives = [*sphereMesh.capsuloids, *sphereMesh.prysmoids]
    contributionsPerVertex, signedDistances = [], []
    for vertex in meshVertices:
        best = min((primitiveQuery(primitive, vertex, rule) for primitive in primitives), key=lambda q: q[0])
        contributionsPerVertex.append([(int(index), float(weight)) for index, weight in best[1]])
        signedDistances.append(best[2])
    return contributionsPerVertex, np.asarray(signedDistances)


sphereMesh, meshVertices, meshNormals, _ = loadFittingInputs()
originalRadii = np.array([sphere.radius for sphere in sphereMesh.spheres])
originalCentres = np.array([sphere.center for sphere in sphereMesh.spheres])
watchedSpheres = (33, 22, 36, 34, 38)

print(f"{'rule':10s} {'unassigned':>10s} {'|d0|>5cm':>9s} {'max r/r0':>9s} {'max |dc|':>9s}  "
      + "  ".join(f"r_s{index}" for index in watchedSpheres))
for rule in ("current", "cull", "two-sided"):
    contributionsPerVertex, signedDistances = assign(sphereMesh, meshVertices, rule)
    fittedCentres, fittedRadii = solveFit(sphereMesh, meshVertices, meshNormals, contributionsPerVertex)
    unassignedCount = sum(1 for contributions in contributionsPerVertex if not contributions)
    farCount = int(np.sum(np.abs(np.nan_to_num(signedDistances)) > 0.05))
    radiusRatio = fittedRadii / originalRadii
    centreShift = np.linalg.norm(fittedCentres - originalCentres, axis=1)
    print(f"{rule:10s} {unassignedCount:10d} {farCount:9d} "
          f"{radiusRatio.max():6.2f}@s{radiusRatio.argmax():<2d} {centreShift.max():.3f}@s{centreShift.argmax():<2d}  "
          + "  ".join(f"{fittedRadii[index]:.4f}" for index in watchedSpheres))
print("\noriginal r " + " " * 43 + "  ".join(f"{originalRadii[index]:.4f}" for index in watchedSpheres))
