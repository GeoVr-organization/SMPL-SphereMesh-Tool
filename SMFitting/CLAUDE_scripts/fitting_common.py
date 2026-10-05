"""What every script in CLAUDE_scripts shares: main.py's inputs (the SMPL
star-pose sphere mesh and surface, both rotated by meshRotation) and its
regularized least-squares solve, reproduced outside the notebook."""

import sys
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as sparseLinalg
from scipy.sparse import diags, lil_matrix, vstack

# SMFitting/, the folder above this one: SphereMesh.py and examples/ live there.
fittingDirectory = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(fittingDirectory))
from SphereMesh import SphereMesh  # noqa: E402

outputDirectory = Path(__file__).resolve().parent / "output"

sphereMeshPath = fittingDirectory / "examples/SMPL_male_star.txt"
objPath = fittingDirectory / "examples/SMPL_starpose.obj"

# main.py section 4: +90 degrees about x, (x, y, z) -> (x, -z, y).
meshRotation = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def loadObjMesh(path):
    """Same as main.py's loadObjMesh (the starpose .obj carries normals)."""
    vertices, fileNormals, triangles, cornerNormalPairs = [], [], [], []
    with open(path) as objFile:
        for line in objFile:
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
            elif line.startswith("vn "):
                fileNormals.append([float(value) for value in line.split()[1:4]])
            elif line.startswith("f "):
                corners = []
                for corner in line.split()[1:]:
                    fields = corner.split("/")
                    vertexIndex = int(fields[0])
                    vertexIndex = vertexIndex - 1 if vertexIndex > 0 else len(vertices) + vertexIndex
                    corners.append(vertexIndex)
                    if len(fields) == 3 and fields[2]:
                        normalIndex = int(fields[2])
                        normalIndex = normalIndex - 1 if normalIndex > 0 else len(fileNormals) + normalIndex
                        cornerNormalPairs.append((vertexIndex, normalIndex))
                for fanIndex in range(1, len(corners) - 1):
                    triangles.append([corners[0], corners[fanIndex], corners[fanIndex + 1]])
    vertices = np.asarray(vertices, dtype=float)
    triangles = np.asarray(triangles, dtype=int)
    vertexNormals = np.zeros_like(vertices)
    cornerVertexIndices, cornerNormalIndices = np.asarray(cornerNormalPairs).T
    np.add.at(vertexNormals, cornerVertexIndices, np.asarray(fileNormals, dtype=float)[cornerNormalIndices])
    vertexNormals /= np.linalg.norm(vertexNormals, axis=1, keepdims=True)
    return vertices, triangles, vertexNormals


def loadFittingInputs():
    """(sphereMesh, meshVertices, meshNormals, meshTriangles), as main.py has them."""
    sphereMesh = SphereMesh()
    sphereMesh.loadFromText(sphereMeshPath.read_text())
    sphereMesh.rotate(meshRotation)
    objVertices, meshTriangles, objNormals = loadObjMesh(objPath)
    return sphereMesh, objVertices @ meshRotation.T, objNormals @ meshRotation.T, meshTriangles


def describe(primitive):
    """'prysmoid(s33,s34,s48)' / 'capsuloid(s3,s33)'."""
    indices = [primitive.sA.index, primitive.sB.index]
    if hasattr(primitive, "sC"):
        indices.append(primitive.sC.index)
    kind = "prysmoid" if hasattr(primitive, "sC") else "capsuloid"
    return f"{kind}(" + ",".join(f"s{index}" for index in indices) + ")"


def solveFit(sphereMesh, meshVertices, meshNormals, contributionsPerVertex,
             centreRegularization=0.05, radiusRegularization=0.05):
    """main.py sections 8-9: (fittedCentres, fittedRadii) from the lsqr solve."""
    vertexCount, sphereCount = len(meshVertices), len(sphereMesh.spheres)
    systemMatrix = lil_matrix((vertexCount, 4 * sphereCount))
    rightHandSide = np.zeros(vertexCount)
    for vertexIndex in range(vertexCount):
        position, normal = meshVertices[vertexIndex], meshNormals[vertexIndex]
        rightHandSide[vertexIndex] = position @ normal
        for sphereIndex, weight in contributionsPerVertex[vertexIndex]:
            systemMatrix[vertexIndex, 4 * sphereIndex + 0] = weight * normal[0]
            systemMatrix[vertexIndex, 4 * sphereIndex + 1] = weight * normal[1]
            systemMatrix[vertexIndex, 4 * sphereIndex + 2] = weight * normal[2]
            systemMatrix[vertexIndex, 4 * sphereIndex + 3] = weight

    initialUnknowns = np.zeros(4 * sphereCount)
    for sphereIndex, sphere in enumerate(sphereMesh.spheres):
        initialUnknowns[4 * sphereIndex: 4 * sphereIndex + 3] = sphere.center
        initialUnknowns[4 * sphereIndex + 3] = sphere.radius
    regularizationMatrix = diags(np.tile([centreRegularization] * 3 + [radiusRegularization], sphereCount))
    regularizedMatrix = vstack([systemMatrix.tocsr(), regularizationMatrix]).tocsr()
    regularizedRightHandSide = np.concatenate([rightHandSide, regularizationMatrix @ initialUnknowns])
    solution = sparseLinalg.lsqr(regularizedMatrix, regularizedRightHandSide)[0].reshape(-1, 4)
    return solution[:, :3], solution[:, 3]
