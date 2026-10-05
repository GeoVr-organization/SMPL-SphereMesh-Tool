"""Trace one over-grown sphere back to the vertices that drive it.

Needs output/fit_state.pkl from reproduce_fit.py. The sphere index is the
first argument (default 33):
    .venv/Scripts/python.exe SMFitting/CLAUDE_scripts/analyze_sphere.py 33
"""

import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

outputDirectory = Path(__file__).resolve().parent / "output"
with open(outputDirectory / "fit_state.pkl", "rb") as stateFile:
    state = pickle.load(stateFile)

targetSphere = int(sys.argv[1]) if len(sys.argv) > 1 else 33
meshVertices, meshNormals, meshTriangles = state["meshVertices"], state["meshNormals"], state["meshTriangles"]
originalCentres, originalRadii = state["originalCentres"], state["originalRadii"]
fittedCentres, fittedRadii = state["fittedCentres"], state["fittedRadii"]
contributionsPerVertex = state["contributionsPerVertex"]
winningPrimitives = state["winningPrimitiveDescriptions"]

print("Body extents (rotated frame, z up):")
for axisName, axisIndex in (("x", 0), ("y", 1), ("z", 2)):
    print(f"  {axisName}: {meshVertices[:, axisIndex].min():+.3f} .. {meshVertices[:, axisIndex].max():+.3f}")

targetCentre, targetRadius = originalCentres[targetSphere], originalRadii[targetSphere]
print(f"\ns{targetSphere}: centre {np.round(targetCentre, 4)} r {targetRadius:.4f}"
      f" -> fitted centre {np.round(fittedCentres[targetSphere], 4)} r {fittedRadii[targetSphere]:.4f}")

print("\nPrimitives using it:")
for triple in state["prysmoidTriples"]:
    if targetSphere in triple:
        print(f"  prysmoid {triple}")
for pair in state["capsuloidPairs"]:
    if targetSphere in pair:
        print(f"  capsuloid {pair}")

contributorIndices, contributorWeights = [], []
for vertexIndex, contributions in enumerate(contributionsPerVertex):
    for sphereIndex, weight in contributions:
        if sphereIndex == targetSphere and abs(weight) > 1e-9:
            contributorIndices.append(vertexIndex)
            contributorWeights.append(weight)
contributorIndices = np.array(contributorIndices)
contributorWeights = np.array(contributorWeights)
print(f"\n{len(contributorIndices)} vertices give s{targetSphere} a nonzero weight "
      f"(sum of weights {contributorWeights.sum():.1f}, {np.sum(contributorWeights > 0.5)} with weight > 0.5)")

print("\nWinning primitive among its contributors:")
for primitive, count in Counter(winningPrimitives[i] for i in contributorIndices).most_common():
    print(f"  {primitive}: {count}")

# The radius this vertex alone asks s{target} to have if it were the only sphere:
# r = (p - c) . n. The fit solves a weighted blend, but this is the pull.
offsets = meshVertices[contributorIndices] - targetCentre
impliedRadius = np.einsum("ij,ij->i", offsets, meshNormals[contributorIndices])
distanceToCentre = np.linalg.norm(offsets, axis=1)
normalAlignment = impliedRadius / distanceToCentre  # cos(angle between p-c and n)

print("\nImplied radius (p - c).n over contributors, weighted percentiles:")
for percentile in (5, 25, 50, 75, 95, 99):
    print(f"  p{percentile}: {np.percentile(impliedRadius, percentile):.4f}")

# Connected pieces of the contributor patch on the surface.
contributorSet = np.zeros(len(meshVertices), bool)
contributorSet[contributorIndices] = True
edgeList = np.vstack((meshTriangles[:, [0, 1]], meshTriangles[:, [1, 2]], meshTriangles[:, [2, 0]]))
edgeList = edgeList[contributorSet[edgeList[:, 0]] & contributorSet[edgeList[:, 1]]]
adjacency = coo_matrix((np.ones(len(edgeList)), (edgeList[:, 0], edgeList[:, 1])), shape=(len(meshVertices),) * 2)
_, componentLabels = connected_components(adjacency, directed=False)
componentsByLabel = defaultdict(list)
for vertexIndex in contributorIndices:
    componentsByLabel[componentLabels[vertexIndex]].append(vertexIndex)
print(f"\nThe contributor patch splits into {len(componentsByLabel)} surface-connected pieces:")
contributorPosition = {vertexIndex: k for k, vertexIndex in enumerate(contributorIndices)}
for label, members in sorted(componentsByLabel.items(), key=lambda item: -len(item[1])):
    memberPositions = [contributorPosition[m] for m in members]
    centroid = meshVertices[members].mean(axis=0)
    meanNormal = meshNormals[members].mean(axis=0)
    print(f"  {len(members):4d} verts  centroid {np.round(centroid, 3)}  mean normal {np.round(meanNormal / np.linalg.norm(meanNormal), 2)}"
          f"  weight sum {contributorWeights[memberPositions].sum():6.1f}"
          f"  implied r median {np.median(impliedRadius[memberPositions]):.4f}"
          f"  max {impliedRadius[memberPositions].max():.4f}"
          f"  d0 median {np.nanmedian(state['signedDistances'][members]):+.4f}")
    if len(componentsByLabel) <= 12 and len(members) >= 1:
        primitivesHere = Counter(winningPrimitives[m] for m in members).most_common(4)
        print(f"         via {primitivesHere}")

# Where the s{target} weight sits relative to the sphere: which side of the body.
print(f"\nContributor positions relative to s{targetSphere} centre (dx, dy, dz) quartiles:")
for axisName, axisIndex in (("dx", 0), ("dy", 1), ("dz", 2)):
    print(f"  {axisName}: {np.round(np.percentile(offsets[:, axisIndex], [0, 25, 50, 75, 100]), 3)}")

np.save(outputDirectory / f"contributors_s{targetSphere}.npy", contributorIndices)
