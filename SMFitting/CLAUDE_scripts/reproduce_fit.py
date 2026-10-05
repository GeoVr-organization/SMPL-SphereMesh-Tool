"""Reproduce main.py sections 4-9 headlessly and dump the per-vertex
assignment, so an over-grown sphere can be traced back to its vertices.

Run from python/ with the venv interpreter:
    .venv/Scripts/python.exe SMFitting/CLAUDE_scripts/reproduce_fit.py
Writes output/fit_state.pkl, which analyze_sphere.py reads.
"""

import pickle

import numpy as np

from fitting_common import describe, loadFittingInputs, outputDirectory, solveFit

sphereMesh, meshVertices, meshNormals, meshTriangles = loadFittingInputs()
vertexCount, sphereCount = len(meshVertices), len(sphereMesh.spheres)

# Per vertex: signed distance, contributions, and which primitive won.
signedDistances = np.full(vertexCount, np.nan)
contributionsPerVertex = []
winningPrimitiveDescriptions = []
for vertexIndex, vertex in enumerate(meshVertices):
    closest, contributions = sphereMesh.closestSphere(vertex)
    signedDistances[vertexIndex] = closest.signedDistance(vertex) if contributions else np.nan
    contributionsPerVertex.append([(int(sphereIndex), float(weight)) for sphereIndex, weight in contributions])

    # Re-run the argmin to know *which* primitive answered.
    bestDistance, bestPrimitive = float("inf"), None
    for primitive in [*sphereMesh.capsuloids, *sphereMesh.prysmoids]:
        distance = primitive.signedDistance(vertex)
        if distance < bestDistance:
            bestDistance, bestPrimitive = distance, primitive
    winningPrimitiveDescriptions.append(describe(bestPrimitive))

fittedCentres, fittedRadii = solveFit(sphereMesh, meshVertices, meshNormals, contributionsPerVertex)
originalCentres = np.array([sphere.center for sphere in sphereMesh.spheres])
originalRadii = np.array([sphere.radius for sphere in sphereMesh.spheres])

print(f"{vertexCount} vertices, {sphereCount} spheres, "
      f"{len(sphereMesh.prysmoids)} prysmoids, {len(sphereMesh.capsuloids)} capsuloids")
print("\nTop 8 fitted spheres by radius growth (fitted - original):")
print(f"{'s':>4} {'r0':>8} {'r_fit':>8} {'ratio':>6} {'|dc|':>8}  centre0")
for sphereIndex in np.argsort(fittedRadii - originalRadii)[::-1][:8]:
    print(f"{sphereIndex:>4} {originalRadii[sphereIndex]:8.4f} {fittedRadii[sphereIndex]:8.4f} "
          f"{fittedRadii[sphereIndex] / originalRadii[sphereIndex]:6.2f} "
          f"{np.linalg.norm(fittedCentres[sphereIndex] - originalCentres[sphereIndex]):8.4f}  "
          f"{np.round(originalCentres[sphereIndex], 4)}")
print("\nTop 5 by absolute fitted radius:")
for sphereIndex in np.argsort(fittedRadii)[::-1][:5]:
    print(f"  s{sphereIndex}: r0 {originalRadii[sphereIndex]:.4f} -> r_fit {fittedRadii[sphereIndex]:.4f}")

outputDirectory.mkdir(exist_ok=True)
with open(outputDirectory / "fit_state.pkl", "wb") as stateFile:
    pickle.dump(dict(
        meshVertices=meshVertices, meshNormals=meshNormals, meshTriangles=meshTriangles,
        signedDistances=signedDistances, contributionsPerVertex=contributionsPerVertex,
        winningPrimitiveDescriptions=winningPrimitiveDescriptions,
        fittedCentres=fittedCentres, fittedRadii=fittedRadii,
        originalCentres=originalCentres, originalRadii=originalRadii,
        prysmoidTriples=[(p.sA.index, p.sB.index, p.sC.index) for p in sphereMesh.prysmoids],
        capsuloidPairs=[(c.sA.index, c.sB.index) for c in sphereMesh.capsuloids],
    ), stateFile)
