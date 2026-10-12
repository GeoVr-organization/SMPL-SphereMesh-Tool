"""Sphere-mesh primitives: spheres, capsuloids (sphere pairs) and prysmoids
(sphere triples), with signed-distance queries.

This is the one SphereMesh implementation: the GUI tool (src/) renders and
measures with it, and SMFitting/SphereMesh.py hands this module to the
notebooks there.
"""

from __future__ import annotations

import sys
import warnings
from typing import Sequence

import numpy as np
from numpy import cross, dot, sqrt

# A 3D vector / point, stored as a float numpy array of shape (3,).
Vec3 = np.ndarray


def vec3(values: Sequence[float] | np.ndarray) -> Vec3:
    """Build a float64 3-component vector from anything array-like."""
    return np.asarray(values, dtype=np.float64)


def length(vector: Vec3) -> float:
    return float(np.linalg.norm(vector))


def normalize(vector: Vec3) -> Vec3:
    return vector / np.linalg.norm(vector)


def clamp(value: float, lowerBound: float, upperBound: float) -> float:
    return float(np.clip(value, lowerBound, upperBound))


# A triangle of sphere centres counts as collinear when |AB x AC| falls below
# this fraction of |AB| |AC|, i.e. the sine of its widest-open corner angle.
collinearTolerance = 1e-9

# A prysmoid has tangent faces only while the out-of-plane part of their unit
# normal, squared, stays above this; below it the faces stand almost
# perpendicular to the triangle and are slivers no wider than its edges.
faceTiltTolerance = 1e-6


# Tracing for closestSphere(). Set debugClosestSphere to False to silence it.
# The verbose flag adds a line per Capsuloid.closestSphere() call, which
# signedDistance() makes once per primitive per query, so it is only worth
# turning on when the axis maths itself is in question.
debugClosestSphere = True
debugClosestSphereVerbose = False





def formatVector(vector: Vec3) -> str:
    return "(" + ", ".join(f"{component: .4f}" for component in vector) + ")"


def describePrimitive(primitive) -> str:
    """A primitive as its type plus the indices of its constituent spheres."""
    indices = [primitive.sA.index, primitive.sB.index]
    if hasattr(primitive, "sC"):
        indices.append(primitive.sC.index)
    kind = primitive.name.replace(" Class", "").lower()
    return f"{kind}(" + ",".join(f"s{index}" for index in indices) + ")"


class SphereMesh:
    def __init__(self) -> None:
        self.name: str = "SphereMesh Class"

        self.spheres: list[SphereMesh.Sphere] = []
        self.capsuloids: list[SphereMesh.Capsuloid] = []
        self.prysmoids: list[SphereMesh.Prysmoid] = []

    def loadFromText(self, text: str) -> None:
        """Populate the mesh from the "Sphere Mesh 2.0" text format.

        Line 0 is the format banner and line 1 the element counts, followed by
        the spheres, then the prysmoids, then the capsuloids.
        """
        lines = text.splitlines()

        # The format counts quads too, but SphereMesh holds no quads, so that
        # trailing block is left unread.
        nSpheres, nPrysmoids, nCapsuloids, nQuads = (
            int(count) for count in lines[1].split()
        )

        for sphereIndex in range(nSpheres):
            x, y, z, radius = (
                float(field) for field in lines[2 + sphereIndex].split()
            )
            self.spheres.append(
                SphereMesh.Sphere(sphereIndex, (x, y, z), radius))

        for prysmoidIndex in range(nPrysmoids):
            indexA, indexB, indexC = (
                int(field)
                for field in lines[2 + nSpheres + prysmoidIndex].split()
            )
            self.prysmoids.append(
                SphereMesh.Prysmoid(self.spheres[indexA],
                                    self.spheres[indexB],
                                    self.spheres[indexC],
                                    1.0))

        for capsuloidIndex in range(nCapsuloids):
            indexA, indexB = (
                int(field)
                for field in
                lines[2 + nSpheres + nPrysmoids + capsuloidIndex].split()
            )
            self.capsuloids.append(
                SphereMesh.Capsuloid(self.spheres[indexA],
                                     self.spheres[indexB]))

    @classmethod
    def fromSpheresAndConnections(
            cls, centers: np.ndarray, radii: np.ndarray,
            connections: Sequence[Sequence[int]]) -> "SphereMesh":
        """A mesh over the given spheres: a capsuloid for every 2-index
        connection and a prysmoid for every 3-index one.

        Sphere i gets index i, as in the GUI tool's JSON templates; the
        prysmoids get direction 1.0, as loadFromText() gives them.
        """
        mesh = cls()
        mesh.spheres = [SphereMesh.Sphere(sphereIndex, center, radius)
                        for sphereIndex, (center, radius)
                        in enumerate(zip(centers, radii))]
        for connection in connections:
            connectionSpheres = [mesh.spheres[int(sphereIndex)]
                                 for sphereIndex in connection]
            if len(connectionSpheres) == 2:
                mesh.capsuloids.append(SphereMesh.Capsuloid(*connectionSpheres))
            elif len(connectionSpheres) == 3:
                mesh.prysmoids.append(SphereMesh.Prysmoid(*connectionSpheres, 1.0))
            else:
                raise ValueError(f"A connection joins 2 or 3 spheres, not "
                                 f"{len(connectionSpheres)}: {list(connection)}")
        return mesh

    def saveToFile(self, filename: str) -> None:
        """Write the mesh to a text file in the "Sphere Mesh 2.0" format."""
        with open(filename, "w") as file:
            file.write("Sphere Mesh 2.0\n")
            file.write(f"{len(self.spheres)} {len(self.prysmoids)} {len(self.capsuloids)} 0\n")
            for sphere in self.spheres:
                file.write(f"{sphere.center[0]} {sphere.center[1]} {sphere.center[2]} {sphere.radius}\n")
            for prysmoid in self.prysmoids:
                file.write(f"{prysmoid.sA.index} {prysmoid.sB.index} {prysmoid.sC.index}\n")
            for capsuloid in self.capsuloids:
                file.write(f"{capsuloid.sA.index} {capsuloid.sB.index}\n")

            

    def rotate (self, rotationMatrix: np.ndarray) -> None:
        """Rotate the mesh about the origin by a 3x3 rotation matrix.

        Capsuloids and prysmoids cache geometry derived from their spheres'
        centres when they are built (the capsuloid axis, the prysmoid's tangent
        and side planes), so after moving the centres they are rebuilt from the
        same, now rotated, spheres instead of keeping the unrotated geometry.
        """
        for sphere in self.spheres:
            sphere.center = rotationMatrix @ sphere.center

        self.capsuloids = [
            SphereMesh.Capsuloid(capsuloid.sA, capsuloid.sB)
            for capsuloid in self.capsuloids
        ]
        self.prysmoids = [
            SphereMesh.Prysmoid(prysmoid.sA, prysmoid.sB, prysmoid.sC,
                                prysmoid.direction)
            for prysmoid in self.prysmoids
        ]

    def closestSphere(
            self, point: Sequence[float] | np.ndarray
    ) -> tuple[SphereMesh.Sphere | None, list[tuple[int, float]]]:
        """The sphere of the mesh's envelope nearest to `point`.

        Every capsuloid and prysmoid is asked for its signed distance, the
        nearest one wins, and that primitive is asked for the sphere it puts
        closest to the point.

        Returns (sphere, contributions), where contributions pairs the index of
        each constituent sphere the result interpolates with how much of it
        that sphere accounts for: two entries for a capsuloid, three for a
        prysmoid, summing to 1 in both cases.

        The constituent spheres are not candidates in their own right - one
        only shows up through the primitives it belongs to. A prysmoid answers
        from whichever of its two faces the point is on, so only a mesh with
        no primitives gives back (None, []).
        """
        queryPoint = vec3(point)



        nearestPrimitive = None
        nearestDistance = float('inf')
        for primitive in [*self.capsuloids, *self.prysmoids]:
            distance = primitive.signedDistance(queryPoint)
            isNearest = distance < nearestDistance

            if isNearest:
                nearestDistance = distance
                nearestPrimitive = primitive

        if nearestPrimitive is None:
            print("Nearest primitive is None", file=sys.stderr)
            return None, []



        sphere, contributions = nearestPrimitive.closestSphereWithContributions(
            queryPoint)



        return sphere, contributions

    def signedDistances(self, points: np.ndarray) -> np.ndarray:
        """Signed distance of every row of an (N, 3) array to the mesh's envelope.

        The minimum over every capsuloid and prysmoid, the same comparison
        closestSphere() picks its primitive by; like there, the constituent
        spheres only count through their primitives. +inf for a mesh with
        no primitives.
        """
        points = np.asarray(points, dtype=np.float64)
        distances = np.full(len(points), np.inf)
        for primitive in [*self.capsuloids, *self.prysmoids]:
            np.minimum(distances, primitive.signedDistances(points),
                       out=distances)
        return distances

    def getInterpolatedSpheres(self, resolution: int) -> list[SphereMesh.Sphere]:
        """Spheres sampled inside every capsuloid and prysmoid of the mesh.

        `resolution` is the number of spheres per edge. Edges are split into
        resolution + 1 equal steps and the corners are skipped, so the
        constituent spheres are never repeated:

        - capsuloid: resolution points on the axis,
          sA.center + i / (resolution + 1) * (sB.center - sA.center),
          for i = 1..resolution
        - prysmoid: the same resolution points along each of its three edges,
          plus an inner triangle of resolution * (resolution + 1) / 2 points:
          the interior of the barycentric grid with resolution + 2 steps.
          1 gives 3 + 1 (the centroid), 2 gives 6 + 3, 3 gives 9 + 6.

        That inner grid is one step finer than the edges because the centroid
        is not on the edges' grid for resolution 1; of the scales that fit the
        counts it keeps the gaps between neighbouring points the most even.

        Each point becomes a sphere through its primitive's closestSphere().
        Edges shared with a capsuloid or another prysmoid are sampled by each
        of them, so those spheres coincide.
        """
        capsuloidSpheres, prysmoidSpheres = self.getInterpolatedSpheresBySource(
            resolution)
        return capsuloidSpheres + prysmoidSpheres

    def getInterpolatedSpheresBySource(
            self, resolution: int
    ) -> tuple[list[SphereMesh.Sphere], list[SphereMesh.Sphere]]:
        """The spheres of getInterpolatedSpheres(), split by what sampled them.

        Returns (capsuloidSpheres, prysmoidSpheres), so a caller that wants to
        tell the two apart - to colour them, say - does not have to rely on
        where the boundary falls in the concatenated list.
        """
        capsuloidSpheres: list[SphereMesh.Sphere] = []
        prysmoidSpheres: list[SphereMesh.Sphere] = []
        if resolution <= 0:
            return capsuloidSpheres, prysmoidSpheres

        stepCount = resolution + 1

        for capsuloid in self.capsuloids:
            for stepIndex in range(1, stepCount):
                point = (capsuloid.sA.center
                         + (stepIndex / stepCount)
                         * (capsuloid.sB.center - capsuloid.sA.center))
                sphere, _ = capsuloid.closestSphere(point)
                capsuloidSpheres.append(sphere)

        innerStepCount = resolution + 2

        for prysmoid in self.prysmoids:
            # Grid points sit exactly on the midplane, where midplane.isBehind()
            # is settled by ~1e-18 rounding and can send closestSphere() down
            # its radius-0 branch. Nudging them to the front side avoids that.
            # Collinear centres have no midplane, and no face to land on.
            midplaneNudge = (1e-9 * prysmoid.midplane.normal
                             if prysmoid.midplane is not None else 0.0)
            cornerA = prysmoid.sA.center
            cornerB = prysmoid.sB.center
            cornerC = prysmoid.sC.center

            for edgeStart, edgeEnd in ((cornerA, cornerB),
                                       (cornerB, cornerC),
                                       (cornerC, cornerA)):
                for stepIndex in range(1, stepCount):
                    point = (edgeStart
                             + (stepIndex / stepCount) * (edgeEnd - edgeStart))
                    sphere, _ = prysmoid.closestSphere(point + midplaneNudge)
                    prysmoidSpheres.append(sphere)

            for weightA in range(1, innerStepCount):
                for weightB in range(1, innerStepCount - weightA):
                    weightC = innerStepCount - weightA - weightB
                    point = (weightA * cornerA
                             + weightB * cornerB
                             + weightC * cornerC) / innerStepCount
                    sphere, _ = prysmoid.closestSphere(point + midplaneNudge)
                    prysmoidSpheres.append(sphere)

        return capsuloidSpheres, prysmoidSpheres

    class Sphere:
        def __init__(self, index: int, position: Sequence[float] | np.ndarray,
                     radius: float) -> None:
            self.name: str = "Sphere Class"
            self.index: int = index
            self.center: Vec3 = vec3(position)
            self.radius: float = float(radius)

        def signedDistance(self, point: Vec3) -> float:
            distance = length(point - self.center) - self.radius
            return distance

        def signedDistances(self, points: np.ndarray) -> np.ndarray:
            """signedDistance() for every row of an (N, 3) array at once."""
            return np.linalg.norm(points - self.center, axis=1) - self.radius

        def buildGeometry(self, numTheta: int, numPhi: int) -> tuple[np.ndarray, np.ndarray]:
            """Build a mesh of vertices and triangles for the sphere."""
            theta = np.linspace(0, np.pi, numTheta)
            phi = np.linspace(0, 2 * np.pi, numPhi)
            theta, phi = np.meshgrid(theta, phi)

            x = self.radius * np.sin(theta) * np.cos(phi) + self.center[0]
            y = self.radius * np.sin(theta) * np.sin(phi) + self.center[1]
            z = self.radius * np.cos(theta) + self.center[2]

            vertices = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=-1)

            triangles = []
            for i in range(numPhi - 1):
                for j in range(numTheta - 1):
                    idx0 = i * numTheta + j
                    idx1 = (i + 1) * numTheta + j
                    idx2 = (i + 1) * numTheta + (j + 1)
                    idx3 = i * numTheta + (j + 1)

                    triangles.append([idx0, idx1, idx2])
                    triangles.append([idx0, idx2, idx3])

            triangles = np.array(triangles)

            return vertices, triangles

    class Capsuloid:
        def __init__(self, sA: "SphereMesh.Sphere", sB: "SphereMesh.Sphere") -> None:
            self.name: str = "Capsuloid Class"

            self.sA: SphereMesh.Sphere = sA if sA.index < sB.index else sB
            self.sB: SphereMesh.Sphere = sB if sA.index < sB.index else sA

            self.axis: Vec3 = self.sB.center - self.sA.center
            axisLengthSquared = float(dot(self.axis, self.axis))
            self.axisLength: float = np.sqrt(axisLengthSquared)

            radiusDelta = self.sB.radius - self.sA.radius

            # One sphere containing the other - coincident centres included -
            # leaves no tangent cone: every sphere interpolated between the
            # two lies inside the bigger one, so the capsuloid is exactly
            # that sphere. closestSphere() answers with it.
            self.isContained: bool = self.axisLength <= abs(radiusDelta)
            self.containingSphere: SphereMesh.Sphere = (
                self.sB if radiusDelta > 0.0 else self.sA)
            if self.isContained:
                warnings.warn(
                    f"Degenerate {describePrimitive(self)}: "
                    f"s{self.containingSphere.index} contains the other sphere, "
                    "so the capsuloid is that sphere alone", stacklevel=2)

            # Coincident centres have no axis; any unit vector keeps
            # buildGeometry() finite, and its triangles all have zero area.
            self.axisNorm: Vec3 = (self.axis / self.axisLength
                                   if self.axisLength > 0.0
                                   else np.array([1.0, 0.0, 0.0]))

            self.dR: float = (radiusDelta / self.axisLength
                              if self.axisLength > 0.0 else 0.0)

            radiusDeltaSquared = radiusDelta * radiusDelta
            diff = axisLengthSquared - radiusDeltaSquared
            self.k: float = (0.0 if self.isContained
                             else radiusDelta / (np.sqrt(diff)))

            sinAlpha = min(abs(self.dR), 1.0)

            self.cosAlpha: float = float(np.sqrt(1 - sinAlpha * sinAlpha))

        def closestSphere(self, point: Vec3) -> tuple["SphereMesh.Sphere", float]:
            if self.isContained:
                containingT = (self.axisLength
                               if self.containingSphere is self.sB else 0.0)
                return (SphereMesh.Sphere(-1, self.containingSphere.center,
                                          self.containingSphere.radius),
                        containingT)

            pointToA = point - self.sA.center
            axialT = float(dot(pointToA, self.axisNorm))
            # |pa|^2 - t^2 is never negative mathematically, but for a point on
            # the axis it rounds to ~-1e-17 and the sqrt returns NaN.
            perpendicularDistance = sqrt(
                max(dot(pointToA, pointToA) - axialT * axialT, 0.0))

            slantedT = axialT + self.k * perpendicularDistance

            t = clamp(slantedT, 0.0, self.axisLength)

            if debugClosestSphereVerbose:
                # A clamped t means the point sits off the end of the cone, so
                # the answer is an end sphere rather than an interpolated one.
                print(f"    {describePrimitive(self)} axialT {axialT: .6f} "
                      f"perp {perpendicularDistance: .6f} k {self.k: .6f} "
                      f"-> t {slantedT: .6f}"
                      + (f" CLAMPED to {t: .6f} of {self.axisLength:.6f}"
                         if t != slantedT else f" of {self.axisLength:.6f}"))

            closest = SphereMesh.Sphere(
                index=-1,
                position=self.sA.center + t * self.axisNorm,
                radius=self.sA.radius + t * self.dR,
            )

            return closest, t

        def closestSphereWithContributions(
                self, point: Vec3
        ) -> tuple["SphereMesh.Sphere", list[tuple[int, float]]]:
            """closestSphere(), plus how much each end sphere accounts for it.

            closestSphere() places the result at t along the axis and gives it
            radius sA.radius + t * dR, which for t / axisLength = u is exactly
            the lerp of sA and sB at u in both centre and radius. So (1 - u, u)
            describes the result completely, the way a prysmoid's barycentric
            coordinates describe its own.
            """
            closest, t = self.closestSphere(point)

            # A contained capsuloid is its containing sphere whole, and its
            # axis may have no length to divide by.
            if self.isContained:
                contributionB = 1.0 if self.containingSphere is self.sB else 0.0
            else:
                contributionB = t / self.axisLength

            return closest, [(self.sA.index, 1.0 - contributionB),
                             (self.sB.index, contributionB)]

        def signedDistance(self, point: Vec3) -> float:
            closest, _ = self.closestSphere(point)

            return closest.signedDistance(point)

        def signedDistances(self, points: np.ndarray) -> np.ndarray:
            """signedDistance() for every row of an (N, 3) array at once."""
            if self.isContained:
                return self.containingSphere.signedDistances(points)

            pointsToA = points - self.sA.center
            axialT = pointsToA @ self.axisNorm
            perpendicularDistances = np.sqrt(np.maximum(
                np.einsum("ij,ij->i", pointsToA, pointsToA) - axialT * axialT,
                0.0))
            t = np.clip(axialT + self.k * perpendicularDistances,
                        0.0, self.axisLength)

            closestCenters = self.sA.center + t[:, None] * self.axisNorm
            closestRadii = self.sA.radius + t * self.dR
            return (np.linalg.norm(points - closestCenters, axis=1)
                    - closestRadii)

        def buildGeometry(self, numTheta: int, numPhi: int) -> tuple[np.ndarray, np.ndarray]:
            """Build a mesh of vertices and triangles for the capsuloid.

            Only the lateral surface is built: the tangent cone spanning the two
            tangent circles, without the spherical caps on sA and sB. A point of
            the envelope sits at -dR along the axis and cosAlpha out along the
            radius from its interpolated centre, both in units of that centre's
            radius.
            """
            # An orthonormal pair spanning the plane perpendicular to the axis.
            referenceVector = (np.array([1.0, 0.0, 0.0])
                               if abs(self.axisNorm[0]) < 0.9
                               else np.array([0.0, 1.0, 0.0]))
            radialU = normalize(cross(self.axisNorm, referenceVector))
            radialV = cross(self.axisNorm, radialU)

            phi = np.linspace(0, 2 * np.pi, numPhi)
            radialDirections = (np.cos(phi)[:, None] * radialU
                                + np.sin(phi)[:, None] * radialV)

            # How far the tangent cone leans off the axis. Clamped so a
            # degenerate capsuloid still yields a mesh; equal to self.cosAlpha
            # for a well-formed one.
            cosThetaTangent = float(np.clip(-self.dR, -1.0, 1.0))
            coneCosAlpha = float(np.sqrt(1.0 - cosThetaTangent * cosThetaTangent))

            rowCount = max(2, numTheta)

            # Rows run from the sB tangent circle to the sA tangent circle. Each
            # row is a circle: a point on the axis, an offset along the axis and
            # a radius. The endpoint rows are the tangent circles themselves.
            lateralT = np.linspace(self.axisLength, 0.0, rowCount)
            lateralRadius = self.sA.radius + lateralT * self.dR

            rowOrigins = self.sA.center[None, :] + lateralT[:, None] * self.axisNorm
            rowAxialOffsets = -lateralRadius * self.dR
            rowRadii = lateralRadius * coneCosAlpha

            vertices = (rowOrigins[None, :, :]
                        + rowAxialOffsets[None, :, None] * self.axisNorm
                        + rowRadii[None, :, None] * radialDirections[:, None, :])
            vertices = vertices.reshape(-1, 3)

            triangles = []
            for i in range(numPhi - 1):
                for j in range(rowCount - 1):
                    idx0 = i * rowCount + j
                    idx1 = (i + 1) * rowCount + j
                    idx2 = (i + 1) * rowCount + (j + 1)
                    idx3 = i * rowCount + (j + 1)

                    triangles.append([idx0, idx1, idx2])
                    triangles.append([idx0, idx2, idx3])

            triangles = np.array(triangles)

            return vertices, triangles

    class Prysmoid:
        def __init__(self, sA: "SphereMesh.Sphere", sB: "SphereMesh.Sphere",
                     sC: "SphereMesh.Sphere", direction: float) -> None:
            self.name: str = "Prysmoid Class"

            orderedSpheres = sorted((sA, sB, sC), key=lambda sphere: sphere.index)
            self.sA: SphereMesh.Sphere = orderedSpheres[0]
            self.sB: SphereMesh.Sphere = orderedSpheres[1]
            self.sC: SphereMesh.Sphere = orderedSpheres[2]

            Plane = SphereMesh.Prysmoid.Plane

            self.direction: float = float(direction)

            # The capsuloids over the three edges, in the m1, m2, m0 order of
            # the side planes facing them: AC, AB, BC.
            self.edgeCapsuloids: tuple[SphereMesh.Capsuloid, ...] = (
                SphereMesh.Capsuloid(self.sA, self.sC),
                SphereMesh.Capsuloid(self.sA, self.sB),
                SphereMesh.Capsuloid(self.sB, self.sC),
            )

            self.upperPlane: SphereMesh.Prysmoid.Plane | None = self.computeUpperPlane(
                self.sA, self.sB, self.sC, direction)
            # The opposite tangent plane. It is not the mirror of upperPlane:
            # offsetting each centre by -upperPlane.normal * radius only stays
            # planar when all three radii are equal, so it has to be solved for.
            self.lowerPlane: SphereMesh.Prysmoid.Plane | None = self.computeUpperPlane(
                self.sA, self.sB, self.sC, -direction)

            # Without tangent planes - collinear centres, or one sphere
            # bulging past where a plane could touch all three - no point is
            # nearest to the inside of the triangle, so the prysmoid is
            # exactly the union of its three edge capsuloids.
            self.hasFaces: bool = (self.upperPlane is not None
                                   and self.lowerPlane is not None)

            triangleNormal = cross(self.sB.center - self.sA.center,
                                   self.sC.center - self.sA.center)
            self.midplane: SphereMesh.Prysmoid.Plane | None = (
                Plane(triangleNormal, self.sA.center)
                if self.isTriangle(self.sA, self.sB, self.sC) else None)

            if direction < 0 and self.midplane is not None:
                self.midplane.flip()

            if not self.hasFaces:
                warnings.warn(
                    f"Degenerate {describePrimitive(self)}: no plane touches "
                    "all three spheres, so the prysmoid is its three edge "
                    "capsuloids alone", stacklevel=2)
                self.m0 = self.m1 = self.m2 = None
                self.lowerM0 = self.lowerM1 = self.lowerM2 = None
                return

            self.m0: SphereMesh.Prysmoid.Plane = Plane(
                cross(self.sB.center - self.sC.center,
                      self.upperPlane.normal) * direction,
                self.sB.center)
            self.m1: SphereMesh.Prysmoid.Plane = Plane(
                cross(self.sC.center - self.sA.center,
                      self.upperPlane.normal) * direction,
                self.sC.center)
            self.m2: SphereMesh.Prysmoid.Plane = Plane(
                cross(self.sA.center - self.sB.center,
                      self.upperPlane.normal) * direction,
                self.sA.center)

            # The same side planes for the lower face. lowerPlane was solved
            # with -direction, so the sign flips too and they still face in.
            lowerDirection = -direction
            self.lowerM0: SphereMesh.Prysmoid.Plane = Plane(
                cross(self.sB.center - self.sC.center,
                      self.lowerPlane.normal) * lowerDirection,
                self.sB.center)
            self.lowerM1: SphereMesh.Prysmoid.Plane = Plane(
                cross(self.sC.center - self.sA.center,
                      self.lowerPlane.normal) * lowerDirection,
                self.sC.center)
            self.lowerM2: SphereMesh.Prysmoid.Plane = Plane(
                cross(self.sA.center - self.sB.center,
                      self.lowerPlane.normal) * lowerDirection,
                self.sA.center)

        @staticmethod
        def isTriangle(sA: "SphereMesh.Sphere", sB: "SphereMesh.Sphere",
                       sC: "SphereMesh.Sphere") -> bool:
            """Whether the three centres span a triangle rather than a line or a point."""
            edgeB = sB.center - sA.center
            edgeC = sC.center - sA.center
            return bool(length(cross(edgeB, edgeC))
                        > collinearTolerance * length(edgeB) * length(edgeC))

        @staticmethod
        def computeUpperPlane(sA: "SphereMesh.Sphere", sB: "SphereMesh.Sphere",
                              sC: "SphereMesh.Sphere",
                              direction: float) -> "SphereMesh.Prysmoid.Plane | None":
            """The plane touching all three spheres on the `direction` side of
            the triangle of their centres, or None when no plane does.

            A plane n . x = k touches every sphere from the same side when
            n . c + r = k for each of them, which pins the part of the unit
            normal n lying in the triangle's plane with two linear equations,

                n . (cB - cA) = rA - rB,    n . (cC - cA) = rA - rC,

            and leaves the rest of n out of that plane, + for one face and -
            for the other. When the in-plane part alone has length 1 or more
            there is no such plane. This closed form replaced a fixed-point
            iteration that stopped ~0.03 mm short of tangency and oscillated,
            "diverged", on prysmoids without a plane.
            """
            if not SphereMesh.Prysmoid.isTriangle(sA, sB, sC):
                return None

            edgeB = sB.center - sA.center
            edgeC = sC.center - sA.center
            triangleNormal = cross(edgeB, edgeC)

            # Cramer's rule on the 2x2 Gram system for the in-plane part
            # weightB * edgeB + weightC * edgeC. Its determinant is
            # |edgeB x edgeC|^2, which isTriangle() keeps away from zero.
            gramBB = dot(edgeB, edgeB)
            gramBC = dot(edgeB, edgeC)
            gramCC = dot(edgeC, edgeC)
            determinant = dot(triangleNormal, triangleNormal)
            radiusDropB = sA.radius - sB.radius
            radiusDropC = sA.radius - sC.radius
            weightB = (gramCC * radiusDropB - gramBC * radiusDropC) / determinant
            weightC = (gramBB * radiusDropC - gramBC * radiusDropB) / determinant
            inPlaneNormal = weightB * edgeB + weightC * edgeC

            outOfPlaneSquared = 1.0 - float(dot(inPlaneNormal, inPlaneNormal))
            if outOfPlaneSquared <= faceTiltTolerance:
                return None

            normal = (inPlaneNormal
                      + np.sign(direction) * sqrt(outOfPlaneSquared)
                      * normalize(triangleNormal))
            return SphereMesh.Prysmoid.Plane(normal, sA.center + normal * sA.radius)

        def faceAt(self, point: Vec3) -> tuple["SphereMesh.Prysmoid.Plane",
                                                tuple["SphereMesh.Prysmoid.Plane", ...]]:
            """The tangent plane facing point, and its side planes in m1, m2, m0 order.

            Which face counts as upper depends only on the sphere index order,
            so a point behind the midplane is answered by the lower face rather
            than dropped; measured against upperPlane it would come out deeply
            negative and win every nearest-primitive comparison. Only defined
            when hasFaces.
            """
            if self.midplane.isBehind(point):
                return self.lowerPlane, (self.lowerM1, self.lowerM2, self.lowerM0)
            return self.upperPlane, (self.m1, self.m2, self.m0)

        def nearestEdgeAt(
                self, point: Vec3
        ) -> "tuple[SphereMesh.Capsuloid, float] | None":
            """The edge capsuloid owning this point and the point's signed
            distance to it, or None for the face.

            Past any of the three side planes of the face facing the point,
            the nearest surface is an edge of the prysmoid rather than its
            tangent plane: no sphere inside the triangle is then closer than
            the best one on its rim. That edge is the nearest of the
            capsuloids whose side planes the point is behind - near a corner
            it is behind two, and the first of them can be the far edge.
            """
            if self.hasFaces:
                _, sidePlanes = self.faceAt(point)
                candidateEdges = [
                    edgeCapsuloid
                    for sidePlane, edgeCapsuloid in zip(sidePlanes, self.edgeCapsuloids)
                    if sidePlane.isBehind(point)]
                if not candidateEdges:
                    return None
            else:
                candidateEdges = list(self.edgeCapsuloids)

            candidateDistances = [edgeCapsuloid.signedDistance(point)
                                  for edgeCapsuloid in candidateEdges]
            nearestIndex = min(range(len(candidateEdges)),
                               key=candidateDistances.__getitem__)
            return candidateEdges[nearestIndex], candidateDistances[nearestIndex]

        def edgeCapsuloidAt(self, point: Vec3) -> "SphereMesh.Capsuloid | None":
            """The edge capsuloid owning this point, or None for the face; see nearestEdgeAt()."""
            nearestEdge = self.nearestEdgeAt(point)
            return None if nearestEdge is None else nearestEdge[0]

        def signedDistance(self, point: Vec3) -> float:
            nearestEdge = self.nearestEdgeAt(point)
            if nearestEdge is not None:
                return nearestEdge[1]

            tangentPlane, _ = self.faceAt(point)
            return tangentPlane.signedDistance(point)

        def signedDistances(self, points: np.ndarray) -> np.ndarray:
            """signedDistance() for every row of an (N, 3) array at once."""
            edgeDistances = np.min(
                [edgeCapsuloid.signedDistances(points)
                 for edgeCapsuloid in self.edgeCapsuloids], axis=0)
            if not self.hasFaces:
                return edgeDistances

            # The same choices faceAt() and edgeCapsuloidAt() make, per row.
            isBehindMidplane = self.midplane.signedDistances(points) < 0.0
            isInsideUpperFace = np.all(
                [sidePlane.signedDistances(points) >= 0.0
                 for sidePlane in (self.m1, self.m2, self.m0)], axis=0)
            isInsideLowerFace = np.all(
                [sidePlane.signedDistances(points) >= 0.0
                 for sidePlane in (self.lowerM1, self.lowerM2, self.lowerM0)],
                axis=0)

            faceDistances = np.where(isBehindMidplane,
                                     self.lowerPlane.signedDistances(points),
                                     self.upperPlane.signedDistances(points))
            isOnFace = np.where(isBehindMidplane, isInsideLowerFace,
                                isInsideUpperFace)
            return np.where(isOnFace, faceDistances, edgeDistances)

        def closestSphereWithContributions(
                self, point: Vec3
        ) -> tuple["SphereMesh.Sphere", list[tuple[int, float]]]:
            """closestSphere(), plus how much each sphere accounts for it.

            On the face those are the barycentric coordinates of the result in
            the triangle of the three centres - the very weights closestSphere
            interpolates the radius with. Past one of the sides it is whatever
            the owning edge capsuloid reports, over that edge's two spheres.
            """
            edgeCapsuloid = self.edgeCapsuloidAt(point)
            if edgeCapsuloid is not None:
                return edgeCapsuloid.closestSphereWithContributions(point)

            closest, barycentricCoords = self.closestSphereOnFace(point)

            return closest, [(self.sA.index, barycentricCoords[0]),
                             (self.sB.index, barycentricCoords[1]),
                             (self.sC.index, barycentricCoords[2])]

        def closestSphere(self, point: Vec3) -> tuple["SphereMesh.Sphere", float]:
            edgeCapsuloid = self.edgeCapsuloidAt(point)
            if edgeCapsuloid is not None:
                return edgeCapsuloid.closestSphere(point)

            closest, _ = self.closestSphereOnFace(point)
            return closest, closest.radius

        def closestSphereOnFace(
                self, point: Vec3
        ) -> tuple["SphereMesh.Sphere", tuple[float, float, float]]:
            """The sphere the tangent face puts closest to point, and the
            barycentric coordinates its radius is blended from.

            Only meaningful inside all three side planes of the face facing the
            point; both callers above settle that first. They share this so
            the coordinates come straight out of the construction instead of
            being read back off the sphere it returns.
            """
            tangentPlane, _ = self.faceAt(point)

            #closest point on the tangent plane
            distance = tangentPlane.signedDistance(point)
            closestPoint = point - distance * tangentPlane.normal

            #project closest point onto middle plane along the direction of the tangent plane normal
            midplaneDistance = self.midplane.signedDistance(closestPoint)
            # midplaneDistance is measured along midplane.normal, but the step is
            # taken along tangentPlane.normal, which is tilted from it; dividing
            # by the cosine of that tilt makes the step land exactly on the
            # midplane. For the lower face the cosine is negative, which flips
            # the step to match.
            cosTilt = float(dot(tangentPlane.normal, self.midplane.normal))
            closestPointOnMidplane = (closestPoint
                                      - (midplaneDistance / cosTilt)
                                      * tangentPlane.normal)
            #triangular interpolation of the radius based on the barycentric coordinates of the closest point on the midplane
            barycentricCoords = self.computeBarycentricCoordinates(closestPointOnMidplane)
            radius = (barycentricCoords[0] * self.sA.radius +
                      barycentricCoords[1] * self.sB.radius +
                      barycentricCoords[2] * self.sC.radius)
            return (SphereMesh.Sphere(-1, closestPointOnMidplane, radius),
                    barycentricCoords)

        def computeBarycentricCoordinates(self, point: Vec3) -> tuple[float, float, float]:
            v0 = self.sB.center - self.sA.center
            v1 = self.sC.center - self.sA.center
            v2 = point - self.sA.center

            d00 = dot(v0, v0)
            d01 = dot(v0, v1)
            d11 = dot(v1, v1)
            d20 = dot(v2, v0)
            d21 = dot(v2, v1)

            denom = d00 * d11 - d01 * d01
            if denom == 0:
                return 1.0, 0.0, 0.0  # Degenerate case

            v = (d11 * d20 - d01 * d21) / denom
            w = (d00 * d21 - d01 * d20) / denom
            u = 1.0 - v - w

            return u, v, w

        def buildGeometry(self, numTheta: int, numPhi: int) -> tuple[np.ndarray, np.ndarray]:
            """Build a mesh of vertices and triangles for the prysmoid.

            The two tangent faces plus the three edge capsuloids; a prysmoid
            without faces is its edge capsuloids alone.
            """
            if self.hasFaces:
                vertices = np.array([
                    self.sA.center + self.upperPlane.normal * self.sA.radius,
                    self.sB.center + self.upperPlane.normal * self.sB.radius,
                    self.sC.center + self.upperPlane.normal * self.sC.radius,
                    self.sA.center + self.lowerPlane.normal * self.sA.radius,
                    self.sB.center + self.lowerPlane.normal * self.sB.radius,
                    self.sC.center + self.lowerPlane.normal * self.sC.radius,
                ])
                # The lower face is wound the other way so both normals point out.
                triangles = np.array([
                    [0, 1, 2],
                    [5, 4, 3]
                ])
            else:
                vertices = np.empty((0, 3))
                triangles = np.empty((0, 3), dtype=int)

            capsuloidCA, capsuloidAB, capsuloidBC = self.edgeCapsuloids
            verticesAB, trianglesAB = capsuloidAB.buildGeometry(numTheta, numPhi)
            verticesBC, trianglesBC = capsuloidBC.buildGeometry(numTheta, numPhi)
            verticesCA, trianglesCA = capsuloidCA.buildGeometry(numTheta, numPhi)

            faceVertexCount = len(vertices)
            vertices = np.vstack((vertices, verticesAB, verticesBC, verticesCA))
            triangles = np.vstack((triangles,
                                   trianglesAB + faceVertexCount,
                                   trianglesBC + faceVertexCount + len(verticesAB),
                                   trianglesCA + faceVertexCount + len(verticesAB) + len(verticesBC)))

            return vertices, triangles


        class Plane:
            def __init__(self, direction: Vec3, point: Vec3) -> None:
                self.name: str = "Plane Class"
                self.normal: Vec3 = normalize(direction)
                self.k: float = float(dot(self.normal, point))

            @classmethod
            def fromNormalAndOffset(cls, normal: Vec3,
                                    k: float) -> "SphereMesh.Prysmoid.Plane":
                plane = cls.__new__(cls)
                plane.name = "Plane Class"
                plane.normal = normal
                plane.k = float(k)
                return plane

            def flip(self) -> None:
                self.normal = -self.normal
                self.k = -self.k

            def isBehind(self, point: Vec3) -> bool:
                return bool(dot(self.normal, point) < self.k)

            def signedDistance(self, point: Vec3) -> float:
                return float(dot(self.normal, point) - self.k)

            def signedDistances(self, points: np.ndarray) -> np.ndarray:
                """signedDistance() for every row of an (N, 3) array at once."""
                return points @ self.normal - self.k
