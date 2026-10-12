# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "scipy",
# ]
# ///
"""Fit a sphere mesh to a surface from the command line, with main.py's
linear method (sections 6, 8 and 9) and without the notebook.

Every vertex of the .obj is handed to the primitive of the sphere mesh
closest to it, which splits it among two or three constituent spheres. One
regularized least-squares solve of the tangency condition (p - c) . n = r
over all the vertices then moves every centre and radius at once.

    python SMFitting/FitSphereMesh.py SMFitting/examples/fitted_smpl0.txt SMFitting/examples/SMPL_starpose.obj
    python SMFitting/FitSphereMesh.py input.txt surface.obj -o fitted.txt --centre-regularization 0.5
    python SMFitting/FitSphereMesh.py input.txt surface.obj -o fitted.txt --save-obj

--save-obj also writes the fitted sphere mesh as a Wavefront .obj: the
surfaces the notebook draws for its spheres, capsuloids and prysmoids, built
by their own buildGeometry().

The sphere mesh and the .obj must share one frame, as they do for main.py.
main.py's meshRotation is left out here: it turns the tangency equations and
the regularization (the same weight on x, y and z) alike, so the solve gives
back the same spheres rotated, and main.py rotates them back before saving.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as sparseLinalg
from scipy.sparse import diags, lil_matrix, vstack

# SphereMesh.py and ObjMesh.py sit next to this script, so the script runs
# from any working directory.
fittingDirectory = Path(__file__).resolve().parent
if str(fittingDirectory) not in sys.path:
    sys.path.insert(0, str(fittingDirectory))

from ObjMesh import loadObjMesh  # noqa: E402
from SphereMesh import SphereMesh  # noqa: E402

# main.py section 8's P and P_r: how hard each centre coordinate and each
# radius is held to its value in the input sphere mesh.
DEFAULT_CENTRE_REGULARIZATION = 0.25
DEFAULT_RADIUS_REGULARIZATION = 0.25

# main.py's numTheta and numPhi slider defaults: how finely every sphere and
# capsuloid surface of the --save-obj export is triangulated.
DEFAULT_NUM_THETA = 16
DEFAULT_NUM_PHI = 32
# The sliders' minimum; below it a sphere has no non-flat triangle.
MINIMUM_SUBDIVISIONS = 4

# Twice a triangle's area, in m^2, below which the export drops it:
# Sphere.buildGeometry() puts a fan of zero-area triangles at each pole.
DEGENERATE_TRIANGLE_LIMIT = 1e-12


def loadSphereMesh(sphereMeshPath):
    """A SphereMesh read from a "Sphere Mesh 2.0" text file."""
    sphereMesh = SphereMesh()
    sphereMesh.loadFromText(Path(sphereMeshPath).read_text())
    return sphereMesh


def assignVertices(sphereMesh, vertices):
    """Section 6: every vertex's constituent-sphere contributions and signed distance.

    Returns (contributionsPerVertex, signedDistances). A vertex no primitive
    answers gets no contributions and a NaN distance, as in the notebook's
    table.
    """
    contributionsPerVertex = []
    signedDistances = np.full(len(vertices), np.nan)
    for vertexIndex, vertex in enumerate(vertices):
        closestSphere, contributions = sphereMesh.closestSphere(vertex)
        if contributions:
            signedDistances[vertexIndex] = closestSphere.signedDistance(vertex)
        contributionsPerVertex.append(
            [(int(sphereIndex), float(contribution))
             for sphereIndex, contribution in contributions]
        )
    return contributionsPerVertex, signedDistances


def buildLinearSystem(vertices, normals, contributionsPerVertex, sphereCount):
    """Section 8: the tangency system A x = b, one row per vertex.

    Row i reads sum_j w_ij (c_j . n_i + r_j) = p_i . n_i over the spheres
    vertex i was handed to, and x holds (cx, cy, cz, r) for every sphere.
    """
    systemMatrix = lil_matrix((len(vertices), 4 * sphereCount))
    rightHandSide = np.einsum("ij,ij->i", vertices, normals)
    for vertexIndex, contributions in enumerate(contributionsPerVertex):
        normal = normals[vertexIndex]
        for sphereIndex, weight in contributions:
            systemMatrix[vertexIndex, 4 * sphereIndex + 0] = weight * normal[0]
            systemMatrix[vertexIndex, 4 * sphereIndex + 1] = weight * normal[1]
            systemMatrix[vertexIndex, 4 * sphereIndex + 2] = weight * normal[2]
            systemMatrix[vertexIndex, 4 * sphereIndex + 3] = weight
    return systemMatrix.tocsr(), rightHandSide


def solveRegularized(systemMatrix, rightHandSide, sphereMesh,
                     centreRegularization, radiusRegularization):
    """Section 8's regularized lsqr solve: the fitted (cx, cy, cz, r) of every sphere.

    The regularization rows pull every unknown toward its value in the input
    sphere mesh, which is also all that holds a sphere no vertex reaches.
    Returns (an (M, 4) array of fitted unknowns, lsqr's full result tuple).
    """
    sphereCount = len(sphereMesh.spheres)
    initialUnknowns = np.zeros(4 * sphereCount)
    for sphereIndex, sphere in enumerate(sphereMesh.spheres):
        initialUnknowns[4 * sphereIndex: 4 * sphereIndex + 3] = sphere.center
        initialUnknowns[4 * sphereIndex + 3] = sphere.radius

    regularizationMatrix = diags(
        np.tile([centreRegularization] * 3 + [radiusRegularization], sphereCount)
    )
    regularizedMatrix = vstack([systemMatrix, regularizationMatrix]).tocsr()
    regularizedRightHandSide = np.concatenate(
        [rightHandSide, regularizationMatrix @ initialUnknowns]
    )
    lsqrResult = sparseLinalg.lsqr(regularizedMatrix, regularizedRightHandSide)
    return lsqrResult[0].reshape(sphereCount, 4), lsqrResult


def buildFittedSphereMesh(sphereMesh, fittedUnknowns):
    """Section 9: the input's spheres, prysmoids and capsuloids over the fitted centres and radii."""
    fittedSphereMesh = SphereMesh()
    fittedSphereMesh.spheres = [
        SphereMesh.Sphere(sphereIndex, sphereUnknowns[:3], sphereUnknowns[3])
        for sphereIndex, sphereUnknowns in enumerate(fittedUnknowns)
    ]
    fittedSpheres = fittedSphereMesh.spheres
    fittedSphereMesh.prysmoids = [
        SphereMesh.Prysmoid(fittedSpheres[prysmoid.sA.index],
                            fittedSpheres[prysmoid.sB.index],
                            fittedSpheres[prysmoid.sC.index],
                            prysmoid.direction)
        for prysmoid in sphereMesh.prysmoids
    ]
    fittedSphereMesh.capsuloids = [
        SphereMesh.Capsuloid(fittedSpheres[capsuloid.sA.index],
                             fittedSpheres[capsuloid.sB.index])
        for capsuloid in sphereMesh.capsuloids
    ]
    return fittedSphereMesh


def primitiveObjectName(primitive):
    """'sphere_s4', 'capsuloid_s3_s33' or 'prysmoid_s33_s34_s48': a primitive's object in the .obj."""
    if isinstance(primitive, SphereMesh.Sphere):
        sphereIndices = [primitive.index]
    elif isinstance(primitive, SphereMesh.Prysmoid):
        sphereIndices = [primitive.sA.index, primitive.sB.index, primitive.sC.index]
    else:
        sphereIndices = [primitive.sA.index, primitive.sB.index]
    primitiveKind = primitive.name.replace(" Class", "").lower()
    return primitiveKind + "".join(f"_s{sphereIndex}" for sphereIndex in sphereIndices)


def outwardSurface(primitive, numTheta, numPhi):
    """A primitive's buildGeometry() surface, zero-area triangles dropped and the rest wound outward.

    Sphere and Capsuloid wind their triangles inward and Prysmoid its two
    faces outward, which the notebook never shows (plotly lights both sides,
    pygfx draws side="both") but an .obj viewer shades by. A triangle faces
    out when its normal points away from the centre of the sphere its
    primitive puts closest to it; one buried inside the envelope has no
    outside and keeps whichever side that test gives it.
    """
    vertices, triangles = primitive.buildGeometry(numTheta, numPhi)
    corners = vertices[triangles]
    faceNormals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    isProperTriangle = np.linalg.norm(faceNormals, axis=1) > DEGENERATE_TRIANGLE_LIMIT
    triangles = triangles[isProperTriangle]
    faceNormals = faceNormals[isProperTriangle]
    centroids = corners[isProperTriangle].mean(axis=1)

    if isinstance(primitive, SphereMesh.Sphere):
        insidePoints = np.broadcast_to(primitive.center, centroids.shape)
    else:
        insidePoints = np.array(
            [primitive.closestSphere(centroid)[0].center for centroid in centroids])
    facesInward = np.einsum("ij,ij->i", faceNormals, centroids - insidePoints) < 0.0
    triangles[facesInward] = triangles[facesInward][:, ::-1]
    return vertices, triangles


def saveSphereMeshObj(sphereMesh, objOutputPath, numTheta, numPhi):
    """Write every sphere, capsuloid and prysmoid surface to one Wavefront .obj, an object each.

    These are the shells the notebook draws, overlapping wherever the
    primitives do, not one merged envelope. Returns (vertex count, triangle
    count).
    """
    primitives = [*sphereMesh.spheres, *sphereMesh.capsuloids, *sphereMesh.prysmoids]
    vertexCount = 0
    triangleCount = 0
    with open(objOutputPath, "w") as objFile:
        objFile.write(f"# FitSphereMesh.py: {len(sphereMesh.spheres)} spheres, "
                      f"{len(sphereMesh.capsuloids)} capsuloids, "
                      f"{len(sphereMesh.prysmoids)} prysmoids, "
                      f"numTheta {numTheta}, numPhi {numPhi}\n")
        for primitive in primitives:
            vertices, triangles = outwardSurface(primitive, numTheta, numPhi)
            objFile.write(f"o {primitiveObjectName(primitive)}\n")
            np.savetxt(objFile, vertices, fmt="v %.6f %.6f %.6f")
            # .obj indices start at 1 and run on across the objects.
            np.savetxt(objFile, triangles + vertexCount + 1, fmt="f %d %d %d")
            vertexCount += len(vertices)
            triangleCount += len(triangles)
    return vertexCount, triangleCount


def printDistanceStatistics(label, signedDistances):
    """One line of signed-distance statistics (metres, negative inside), NaNs skipped."""
    print(f"{label}: min {np.nanmin(signedDistances):+.4f}, "
          f"max {np.nanmax(signedDistances):+.4f}, "
          f"mean {np.nanmean(signedDistances):+.4f}, "
          f"std {np.nanstd(signedDistances):.4f}, "
          f"mean |d| {np.nanmean(np.abs(signedDistances)):.4f}")


def main():
    """Read the inputs, fit, write the fitted sphere mesh and report how the fit went."""
    parser = argparse.ArgumentParser(
        description="Fit a sphere mesh (Sphere Mesh 2.0 .txt) to a surface "
                    "(.obj) with main.py's regularized linear least squares.")
    parser.add_argument("sphereMeshPath", type=Path, metavar="SPHEREMESH_TXT",
                        help="sphere mesh to start from, Sphere Mesh 2.0 text format")
    parser.add_argument("objPath", type=Path, metavar="SURFACE_OBJ",
                        help="surface to fit to, in the same frame as the sphere mesh")
    parser.add_argument("-o", "--output", type=Path, dest="outputPath",
                        metavar="FITTED_TXT",
                        help="fitted sphere mesh to write "
                             "(default: fitted_<obj name>.txt next to the .obj)")
    parser.add_argument("--centre-regularization", type=float,
                        default=DEFAULT_CENTRE_REGULARIZATION,
                        dest="centreRegularization", metavar="WEIGHT",
                        help="weight holding each centre coordinate to its input "
                             f"value (main.py's P, default {DEFAULT_CENTRE_REGULARIZATION})")
    parser.add_argument("--radius-regularization", type=float,
                        default=DEFAULT_RADIUS_REGULARIZATION,
                        dest="radiusRegularization", metavar="WEIGHT",
                        help="weight holding each radius to its input value "
                             f"(main.py's P_r, default {DEFAULT_RADIUS_REGULARIZATION})")
    parser.add_argument("--no-report", action="store_true", dest="skipReport",
                        help="skip measuring the fitted sphere mesh against the "
                             "surface, which takes as long as the fit itself")
    parser.add_argument("--save-obj", nargs="?", const="", dest="objOutputPath",
                        metavar="SPHEREMESH_OBJ",
                        help="also write the fitted sphere mesh's surfaces as a "
                             "Wavefront .obj (default path: the fitted .txt's, "
                             "ending in .obj); put it after the two inputs")
    parser.add_argument("--num-theta", type=int, default=DEFAULT_NUM_THETA,
                        dest="numTheta", metavar="COUNT",
                        help="--save-obj subdivisions from pole to pole of every "
                             f"sphere and capsuloid (default {DEFAULT_NUM_THETA})")
    parser.add_argument("--num-phi", type=int, default=DEFAULT_NUM_PHI,
                        dest="numPhi", metavar="COUNT",
                        help="--save-obj subdivisions around every sphere and "
                             f"capsuloid (default {DEFAULT_NUM_PHI})")
    arguments = parser.parse_args()

    outputPath = arguments.outputPath or (
        arguments.objPath.parent / f"fitted_{arguments.objPath.stem}.txt")
    objOutputPath = None
    if arguments.objOutputPath is not None:
        objOutputPath = (Path(arguments.objOutputPath) if arguments.objOutputPath
                         else outputPath.with_suffix(".obj"))
        # "-o surface.txt --save-obj" would otherwise land on the input surface.
        if objOutputPath.resolve() == arguments.objPath.resolve():
            parser.error(f"--save-obj would overwrite the input surface {arguments.objPath}")
        if min(arguments.numTheta, arguments.numPhi) < MINIMUM_SUBDIVISIONS:
            parser.error(f"--num-theta and --num-phi must be at least {MINIMUM_SUBDIVISIONS}")

    sphereMesh = loadSphereMesh(arguments.sphereMeshPath)
    if not sphereMesh.capsuloids and not sphereMesh.prysmoids:
        parser.error(f"{arguments.sphereMeshPath} has no capsuloids or prysmoids, "
                     "so no vertex can be handed to a sphere")
    sphereCount = len(sphereMesh.spheres)

    objVertices, _objTriangles, objNormals = loadObjMesh(arguments.objPath)
    # loadObjMesh gives a vertex no face touches a NaN normal, and one NaN
    # in b would spoil the whole lsqr solve.
    hasNormal = np.all(np.isfinite(objNormals), axis=1)
    meshVertices = objVertices[hasNormal]
    meshNormals = objNormals[hasNormal]
    print(f"{arguments.sphereMeshPath}: {sphereCount} spheres, "
          f"{len(sphereMesh.prysmoids)} prysmoids, {len(sphereMesh.capsuloids)} capsuloids")
    print(f"{arguments.objPath}: {len(meshVertices)} vertices"
          + (f" ({int((~hasNormal).sum())} without a normal, skipped)"
             if not hasNormal.all() else ""))

    startTime = time.perf_counter()
    contributionsPerVertex, inputDistances = assignVertices(sphereMesh, meshVertices)
    assignmentSeconds = time.perf_counter() - startTime

    systemMatrix, rightHandSide = buildLinearSystem(
        meshVertices, meshNormals, contributionsPerVertex, sphereCount)
    fittedUnknowns, lsqrResult = solveRegularized(
        systemMatrix, rightHandSide, sphereMesh,
        arguments.centreRegularization, arguments.radiusRegularization)
    fittedSphereMesh = buildFittedSphereMesh(sphereMesh, fittedUnknowns)

    outputPath.parent.mkdir(parents=True, exist_ok=True)
    fittedSphereMesh.saveToFile(str(outputPath))

    lsqrStopReason, lsqrIterationCount = lsqrResult[1], lsqrResult[2]
    print(f"vertex assignment {assignmentSeconds:.1f} s; lsqr stopped with "
          f"istop {lsqrStopReason} after {lsqrIterationCount} iterations "
          f"(regularization: centre {arguments.centreRegularization}, "
          f"radius {arguments.radiusRegularization})")

    # A sphere whose four columns are empty is held only by the
    # regularization, so it comes out exactly as it went in.
    vertexCountPerSphere = np.asarray(
        (systemMatrix[:, 3::4] != 0).sum(axis=0)).ravel()
    unreachedSphereIndices = np.flatnonzero(vertexCountPerSphere == 0)
    if len(unreachedSphereIndices) > 0:
        print(f"{len(unreachedSphereIndices)} spheres no vertex reaches, kept as they were: "
              + ", ".join(f"s{sphereIndex}" for sphereIndex in unreachedSphereIndices))

    fittedRadii = fittedUnknowns[:, 3]
    nonPositiveRadiusIndices = np.flatnonzero(fittedRadii <= 0.0)
    if len(nonPositiveRadiusIndices) > 0:
        print("WARNING: non-positive fitted radius on "
              + ", ".join(f"s{sphereIndex} ({fittedRadii[sphereIndex]:.4f})"
                          for sphereIndex in nonPositiveRadiusIndices))

    radiusChanges = fittedRadii - np.array([sphere.radius for sphere in sphereMesh.spheres])
    centreShifts = np.linalg.norm(
        fittedUnknowns[:, :3] - np.array([sphere.center for sphere in sphereMesh.spheres]),
        axis=1)
    print(f"largest centre shift {centreShifts.max():.4f} (s{centreShifts.argmax()}), "
          f"largest radius change {radiusChanges[np.abs(radiusChanges).argmax()]:+.4f} "
          f"(s{np.abs(radiusChanges).argmax()})")

    printDistanceStatistics("signed distance, input ", inputDistances)
    if not arguments.skipReport:
        _, fittedDistances = assignVertices(fittedSphereMesh, meshVertices)
        printDistanceStatistics("signed distance, fitted", fittedDistances)

    print(f"saved {outputPath}")

    if objOutputPath is not None:
        objOutputPath.parent.mkdir(parents=True, exist_ok=True)
        objVertexCount, objTriangleCount = saveSphereMeshObj(
            fittedSphereMesh, objOutputPath, arguments.numTheta, arguments.numPhi)
        print(f"saved {objOutputPath}: {objVertexCount} vertices, "
              f"{objTriangleCount} triangles (numTheta {arguments.numTheta}, "
              f"numPhi {arguments.numPhi})")


if __name__ == "__main__":
    main()
