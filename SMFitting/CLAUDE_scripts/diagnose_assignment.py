"""Why do far-away vertices end up assigned to a sphere?

For every vertex, re-runs SphereMesh.closestSphere()'s argmin and records
whether the winning primitive is a prysmoid the vertex sits *behind* (on the
far side of its midplane, i.e. facing its lower tangent plane, which
Prysmoid.signedDistance never looks at). Then prints, for one sphere's
outlier contributors, the candidates the argmin compared.

    .venv/Scripts/python.exe SMFitting/CLAUDE_scripts/diagnose_assignment.py 33
"""

import sys
from collections import Counter

import numpy as np

from fitting_common import SphereMesh, describe, loadFittingInputs

targetSphere = int(sys.argv[1]) if len(sys.argv) > 1 else 33
sphereMesh, meshVertices, meshNormals, meshTriangles = loadFittingInputs()
allPrimitives = [*sphereMesh.capsuloids, *sphereMesh.prysmoids]

# Orientation of the prysmoids that use the target sphere.
for prysmoid in sphereMesh.prysmoids:
    indices = (prysmoid.sA.index, prysmoid.sB.index, prysmoid.sC.index)
    if targetSphere in indices:
        print(f"{describe(prysmoid)}: upperPlane normal {np.round(prysmoid.upperPlane.normal, 3)}, "
              f"midplane normal {np.round(prysmoid.midplane.normal, 3)}")

behindWinnerCount = 0
behindWinnerSpheres = Counter()
targetOutliers = []
for vertexIndex, vertex in enumerate(meshVertices):
    distances = np.array([primitive.signedDistance(vertex) for primitive in allPrimitives])
    winnerSlot = int(np.argmin(distances))
    winner = allPrimitives[winnerSlot]
    if isinstance(winner, SphereMesh.Prysmoid) and winner.midplane.isBehind(vertex) \
            and winner.edgeCapsuloidAt(vertex) is None:
        behindWinnerCount += 1
        for sphere in (winner.sA, winner.sB, winner.sC):
            behindWinnerSpheres[sphere.index] += 1
        if targetSphere in (winner.sA.index, winner.sB.index, winner.sC.index):
            targetOutliers.append((vertexIndex, distances))

print(f"\n{behindWinnerCount} of {len(meshVertices)} vertices are won by a prysmoid face they sit BEHIND.")
print("Spheres those vertices are credited to (count):")
print("  " + ", ".join(f"s{index}:{count}" for index, count in behindWinnerSpheres.most_common()))

print(f"\nOf those, {len(targetOutliers)} involve s{targetSphere}. The first few, with the "
      f"three best candidates the argmin saw:")
for vertexIndex, distances in targetOutliers[:6]:
    vertex = meshVertices[vertexIndex]
    print(f"  vertex {vertexIndex} at {np.round(vertex, 3)}, normal {np.round(meshNormals[vertexIndex], 2)}")
    for slot in np.argsort(distances)[:3]:
        primitive = allPrimitives[slot]
        sideNote = ""
        if isinstance(primitive, SphereMesh.Prysmoid):
            sideNote = " (behind midplane)" if primitive.midplane.isBehind(vertex) else " (in front)"
        print(f"      {describe(primitive):28s} d = {distances[slot]:+.4f}{sideNote}")
