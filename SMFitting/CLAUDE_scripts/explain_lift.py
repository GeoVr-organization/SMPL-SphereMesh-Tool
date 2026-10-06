"""What lifts a sphere in the fit: solve its 4 unknowns alone, with every
other sphere frozen at the template, and list the vertices that push hardest.

The upward push of vertex i at the template is -w_i * n_iz * residual_i
(residual = c.n + r - p.n over its blend); positive means it wants the
centre higher. Run with the abs() argmin main.py uses and with the signed one.

    .venv/Scripts/python.exe SMFitting/CLAUDE_scripts/explain_lift.py 51
"""

import sys

import numpy as np

from fitting_common import loadFittingInputs, solveFit

targetSphere = int(sys.argv[1]) if len(sys.argv) > 1 else 51
sphereMesh, meshVertices, meshNormals, _ = loadFittingInputs()
originalCentres = np.array([sphere.center for sphere in sphereMesh.spheres])
originalRadii = np.array([sphere.radius for sphere in sphereMesh.spheres])
primitives = [*sphereMesh.capsuloids, *sphereMesh.prysmoids]

for ruleName, useAbs in (("abs argmin (main.py)", True), ("signed argmin", False)):
    contributionsPerVertex = []
    for vertex in meshVertices:
        distances = np.array([primitive.signedDistance(vertex) for primitive in primitives])
        winner = primitives[int(np.argmin(np.abs(distances) if useAbs else distances))]
        _, contributions = winner.closestSphereWithContributions(vertex)
        contributionsPerVertex.append([(int(index), float(weight)) for index, weight in contributions])

    rows, targetWeights = [], []
    for vertexIndex, contributions in enumerate(contributionsPerVertex):
        weight = sum(w for index, w in contributions if index == targetSphere)
        if weight > 1e-9:
            rows.append(vertexIndex)
            targetWeights.append(weight)
    rows, targetWeights = np.array(rows), np.array(targetWeights)
    normals, positions = meshNormals[rows], meshVertices[rows]

    # Residual at the template: blended support c.n + r minus p.n.
    residuals = np.array([
        sum(w * (originalCentres[index] @ meshNormals[row] + originalRadii[index]) for index, w in contributionsPerVertex[row])
        - positions[k] @ normals[k] for k, row in enumerate(rows)])
    upwardPush = -targetWeights * normals[:, 2] * residuals

    # s{target} alone: others frozen, same 0.05 regularization. Its row is
    # w (c.n + r) = (its template term) - residual.
    targetTerms = targetWeights * (normals @ originalCentres[targetSphere] + originalRadii[targetSphere])
    localMatrix = np.column_stack((targetWeights[:, None] * normals, targetWeights))
    localRightHandSide = targetTerms - residuals
    regularization = 0.05
    stackedMatrix = np.vstack((localMatrix, regularization * np.eye(4)))
    stackedRightHandSide = np.concatenate((localRightHandSide, regularization * np.append(originalCentres[targetSphere], originalRadii[targetSphere])))
    localSolution = np.linalg.lstsq(stackedMatrix, stackedRightHandSide, rcond=None)[0]

    centreOffsets = positions - originalCentres[targetSphere]
    print(f"\n=== {ruleName}: {len(rows)} vertices give s{targetSphere} weight (sum {targetWeights.sum():.1f})")
    print(f"  s{targetSphere} solved alone: centre shift {np.round(localSolution[:3] - originalCentres[targetSphere], 3)}, "
          f"r {originalRadii[targetSphere]:.4f} -> {localSolution[3]:.4f}")
    print(f"  net upward push {upwardPush.sum():+.4f}; from vertices facing down (n_z < -0.3): "
          f"{upwardPush[normals[:, 2] < -0.3].sum():+.4f} over {np.sum(normals[:, 2] < -0.3)} vertices")
    insideTemplate = residuals > 0.005
    print(f"  {insideTemplate.sum()} of them sit > 5 mm inside the template blend (residual > 0); "
          f"their push {upwardPush[insideTemplate].sum():+.4f}, mean normal "
          f"{np.round(normals[insideTemplate].mean(axis=0), 2) if insideTemplate.any() else '-'}, "
          f"mean offset from s{targetSphere} {np.round(centreOffsets[insideTemplate].mean(axis=0), 3) if insideTemplate.any() else '-'}")
    print("  top 6 pushers (vertex, weight, normal, offset from centre, residual mm, push):")
    for k in np.argsort(upwardPush)[::-1][:6]:
        print(f"    {rows[k]:5d}  w {targetWeights[k]:.2f}  n {np.round(normals[k], 2)}  "
              f"off {np.round(centreOffsets[k], 3)}  res {residuals[k] * 1000:+6.1f}  push {upwardPush[k]:+.4f}")

    fittedCentres, fittedRadii = solveFit(sphereMesh, meshVertices, meshNormals, contributionsPerVertex)
    for capsuloid in sphereMesh.capsuloids:
        pair = (capsuloid.sA.index, capsuloid.sB.index)
        if targetSphere not in pair:
            continue
        otherIndex = pair[0] if pair[1] == targetSphere else pair[1]
        centreDistance = np.linalg.norm(fittedCentres[targetSphere] - fittedCentres[otherIndex])
        radiusGap = abs(fittedRadii[targetSphere] - fittedRadii[otherIndex])
        print(f"  after the full fit, capsuloid (s{pair[0]}, s{pair[1]}): centres {centreDistance * 100:.1f} cm apart, "
              f"radii differ by {radiusGap * 100:.1f} cm -> "
              + ("one sphere swallows the other (degenerate)" if radiusGap >= centreDistance else "still a cone"))
