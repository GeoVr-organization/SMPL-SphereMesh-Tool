"""Is a sphere's centre shift something the data asks for, or a drift along a
direction the solve can barely see?

For each sphere given (default 51 23), with the assignment main.py makes
(SphereMesh.closestSphere, abs() argmin):
  - lsqr's stop reason, and the exact regularized solution next to it
  - the centre shift against the axis of each capsuloid the sphere ends
  - how stiff the solve is along the shift: the vertex residual a unit move
    of the centre costs, against moving it the other ways
  - a refit with that sphere's centre pinned harder, and what the vertices
    lose for it

    .venv/Scripts/python.exe SMFitting/CLAUDE_scripts/diagnose_drift.py 51 23
"""

import sys

import numpy as np
import scipy.sparse.linalg as sparseLinalg
from scipy.sparse import diags, lil_matrix, vstack

from fitting_common import loadFittingInputs

watchedSpheres = [int(argument) for argument in sys.argv[1:]] or [51, 23]
sphereMesh, meshVertices, meshNormals, _ = loadFittingInputs()
vertexCount, sphereCount = len(meshVertices), len(sphereMesh.spheres)

contributionsPerVertex = []
for vertex in meshVertices:
    _, contributions = sphereMesh.closestSphere(vertex)
    contributionsPerVertex.append([(int(index), float(weight)) for index, weight in contributions])

systemMatrix = lil_matrix((vertexCount, 4 * sphereCount))
rightHandSide = np.einsum("ij,ij->i", meshVertices, meshNormals)
for vertexIndex, contributions in enumerate(contributionsPerVertex):
    normal = meshNormals[vertexIndex]
    for sphereIndex, weight in contributions:
        systemMatrix[vertexIndex, 4 * sphereIndex: 4 * sphereIndex + 3] = weight * normal
        systemMatrix[vertexIndex, 4 * sphereIndex + 3] = weight
systemMatrix = systemMatrix.tocsr()

initialUnknowns = np.zeros(4 * sphereCount)
for sphereIndex, sphere in enumerate(sphereMesh.spheres):
    initialUnknowns[4 * sphereIndex: 4 * sphereIndex + 3] = sphere.center
    initialUnknowns[4 * sphereIndex + 3] = sphere.radius


def solve(regularizationWeights, exact=False):
    """Regularized least squares, as main.py (lsqr) or exactly (normal equations)."""
    regularizationMatrix = diags(regularizationWeights)
    regularizedMatrix = vstack([systemMatrix, regularizationMatrix]).tocsr()
    regularizedRightHandSide = np.concatenate([rightHandSide, regularizationMatrix @ initialUnknowns])
    if exact:
        normalMatrix = (regularizedMatrix.T @ regularizedMatrix).tocsc()
        return sparseLinalg.spsolve(normalMatrix, regularizedMatrix.T @ regularizedRightHandSide), None
    result = sparseLinalg.lsqr(regularizedMatrix, regularizedRightHandSide)
    return result[0], result


def vertexResidualRms(unknowns):
    return float(np.sqrt(np.mean((systemMatrix @ unknowns - rightHandSide) ** 2)))


mainRegularization = np.tile([0.05, 0.05, 0.05, 0.05], sphereCount)
lsqrUnknowns, lsqrResult = solve(mainRegularization)
exactUnknowns, _ = solve(mainRegularization, exact=True)
print(f"lsqr: istop {lsqrResult[1]} after {lsqrResult[2]} iterations "
      f"(1/2 = converged, 7 = hit the iteration limit), cond(A) estimate {lsqrResult[6]:.1f}")
print(f"largest |lsqr - exact| over all unknowns: {np.abs(lsqrUnknowns - exactUnknowns).max():.2e}")
print(f"vertex residual RMS: original {vertexResidualRms(initialUnknowns) * 1000:.2f} mm, "
      f"fitted {vertexResidualRms(lsqrUnknowns) * 1000:.2f} mm")

for sphereIndex in watchedSpheres:
    columns = slice(4 * sphereIndex, 4 * sphereIndex + 3)
    originalCentre = initialUnknowns[columns]
    centreShift = lsqrUnknowns[columns] - originalCentre
    shiftLength = np.linalg.norm(centreShift)
    shiftDirection = centreShift / shiftLength
    print(f"\n=== s{sphereIndex}: centre {np.round(originalCentre, 3)} r {initialUnknowns[4 * sphereIndex + 3]:.4f}"
          f" -> shift {np.round(centreShift, 3)} ({shiftLength * 100:.1f} cm), "
          f"r {lsqrUnknowns[4 * sphereIndex + 3]:.4f}")

    for capsuloid in sphereMesh.capsuloids:
        if sphereIndex not in (capsuloid.sA.index, capsuloid.sB.index):
            continue
        otherSphere = capsuloid.sB if capsuloid.sA.index == sphereIndex else capsuloid.sA
        axisDirection = originalCentre - otherSphere.center
        axisLength = np.linalg.norm(axisDirection)
        axisDirection /= axisLength
        otherShift = lsqrUnknowns[4 * otherSphere.index: 4 * otherSphere.index + 3] - otherSphere.center
        print(f"  capsuloid with s{otherSphere.index} at {np.round(otherSphere.center, 3)} "
              f"(axis {axisLength * 100:.0f} cm): cos(shift, axis) {shiftDirection @ axisDirection:+.3f}, "
              f"s{otherSphere.index} shifts {np.linalg.norm(otherShift) * 100:.1f} cm")

    # Stiffness: residual a 1 m move of the centre alone costs, along the shift
    # and along the stiffest direction of this sphere's 3x3 normal block.
    centreColumns = systemMatrix[:, columns].toarray()
    centreBlock = centreColumns.T @ centreColumns
    eigenvalues, eigenvectors = np.linalg.eigh(centreBlock)
    alongShift = float(shiftDirection @ centreBlock @ shiftDirection)
    print(f"  stiffness of the centre (sum of w^2 (n.u)^2 over its vertices): along the shift {alongShift:.2f}, "
          f"eigen {np.round(eigenvalues, 2)}; softest direction {np.round(eigenvectors[:, 0], 2)} "
          f"(cos with shift {abs(eigenvectors[:, 0] @ shiftDirection):.2f}); regularization adds 0.05^2 = 0.0025")

    # Pin this sphere's centre 20x harder and see what the vertices lose.
    pinnedRegularization = mainRegularization.copy()
    pinnedRegularization[4 * sphereIndex: 4 * sphereIndex + 3] = 1.0
    pinnedUnknowns, _ = solve(pinnedRegularization)
    pinnedShift = pinnedUnknowns[columns] - originalCentre
    print(f"  centre pinned with weight 1.0: shift {np.round(pinnedShift, 3)} ({np.linalg.norm(pinnedShift) * 100:.1f} cm), "
          f"r {pinnedUnknowns[4 * sphereIndex + 3]:.4f}, vertex residual RMS "
          f"{vertexResidualRms(pinnedUnknowns) * 1000:.2f} mm (free: {vertexResidualRms(lsqrUnknowns) * 1000:.2f} mm)")
