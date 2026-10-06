"""Wavefront .obj surfaces for the SMFitting notebooks: loading an .obj, and
the two stand-ins that draw it through PlotlyView.renderPrimitives - a clear
shell (ObjSurface) and a signed-distance heatmap (DistanceHeatmapSurface)."""

import numpy as np


def loadObjMesh(objPath):
    """The vertices, triangles and per-vertex normals of a Wavefront .obj.

    Corners are reduced to their vertex index, so the v, v/vt, v//vn and
    v/vt/vn forms all read the same, and a face with more than three
    corners is fanned into triangles.

    .obj normals belong to face corners, not to vertices: arm.obj gives one
    normal per face, shared by its three corners. A vertex's normal is the
    normalized sum of the normals its corners name. A file without normals
    (SMPL_male.obj) gets them from the triangles instead, as the
    area-weighted sum of the incident face normals. A vertex no face
    touches gets NaN.
    """
    vertices = []
    fileNormals = []
    triangles = []
    # (vertex index, normal index) for every face corner that names a normal.
    cornerNormalPairs = []
    with open(objPath, "r") as objFile:
        for line in objFile:
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
            elif line.startswith("vn "):
                fileNormals.append(
                    [float(value) for value in line.split()[1:4]]
                )
            elif line.startswith("f "):
                # .obj indices start at 1, and a negative one counts back
                # from the last vertex (or normal) read so far.
                corners = []
                for corner in line.split()[1:]:
                    fields = corner.split("/")
                    vertexIndex = int(fields[0])
                    vertexIndex = (
                        vertexIndex - 1
                        if vertexIndex > 0
                        else len(vertices) + vertexIndex
                    )
                    corners.append(vertexIndex)
                    if len(fields) == 3 and fields[2]:
                        normalIndex = int(fields[2])
                        normalIndex = (
                            normalIndex - 1
                            if normalIndex > 0
                            else len(fileNormals) + normalIndex
                        )
                        cornerNormalPairs.append((vertexIndex, normalIndex))
                for fanIndex in range(1, len(corners) - 1):
                    triangles.append(
                        [corners[0], corners[fanIndex], corners[fanIndex + 1]]
                    )

    vertices = np.asarray(vertices, dtype=float)
    triangles = np.asarray(triangles, dtype=int)

    vertexNormals = np.zeros_like(vertices)
    if cornerNormalPairs:
        cornerVertexIndices, cornerNormalIndices = np.asarray(cornerNormalPairs).T
        np.add.at(
            vertexNormals,
            cornerVertexIndices,
            np.asarray(fileNormals, dtype=float)[cornerNormalIndices],
        )
    else:
        # Unnormalized cross products are twice the face area long, which
        # is what weights each face by its area in the sum.
        faceNormals = np.cross(
            vertices[triangles[:, 1]] - vertices[triangles[:, 0]],
            vertices[triangles[:, 2]] - vertices[triangles[:, 0]],
        )
        for cornerSlot in range(3):
            np.add.at(vertexNormals, triangles[:, cornerSlot], faceNormals)

    normalLengths = np.linalg.norm(vertexNormals, axis=1, keepdims=True)
    vertexNormals = np.divide(
        vertexNormals,
        normalLengths,
        out=np.full_like(vertexNormals, np.nan),
        where=normalLengths > 0,
    )
    return vertices, triangles, vertexNormals


class ObjSurface:
    """A loaded .obj standing in for a primitive in renderPrimitives.

    It carries its own plotly settings so it renders as a thin shell of
    clear plastic - smooth-shaded, barely tinted white and glossy - with
    the sphere mesh it was fitted to staying visible inside it.
    """

    def __init__(self, vertices, triangles, name, opacity):
        self.vertices = vertices
        self.triangles = triangles
        self.name = name
        # A wireframe over a few thousand triangles would bury the shell,
        # and the spheres inside it, under a mess of lines.
        self.drawWireframe = False
        self.traceOverrides = dict(
            opacity=opacity,
            flatshading=False,
            lighting=dict(
                ambient=0.55,
                diffuse=0.6,
                specular=0.9,
                roughness=0.15,
                fresnel=1.6,
            ),
        )

    def buildGeometry(self, numTheta, numPhi):
        return self.vertices, self.triangles


# Two hue families that meet at 0 with a hard break, so even a tiny d shows
# its sign: cool for inside the sphere mesh (cyan near 0, deepening to
# dark blue), warm for outside (yellow near 0, deepening to dark red).
# Each list runs from the band next to 0 out to the band at the limit.
insideColorAnchors = ["#8ee8f2", "#2fa4e7", "#1f5fd0", "#0d2a85"]
outsideColorAnchors = ["#ffe45c", "#ffa63d", "#e8492e", "#8c1010"]
# Above the cutoff a vertex is out of tolerance; grey says so instead of
# letting it pass for the strongest in-range red.
cutoffColor = "#4a4a4a"

# Nearly flat colour: a high ambient floor with only a little diffuse on
# top, and no highlight or rim, so a band reads as the same colour all the
# way round the surface instead of darkening into the greys of the cutoff.
heatmapLighting = dict(
    ambient=0.85, diffuse=0.2, specular=0.0, roughness=1.0, fresnel=0.0
)


def sampleColorRamp(anchors, count):
    """count colours evenly spaced along the hex anchors, in RGB."""
    anchorRgb = np.array(
        [[int(anchor[k:k + 2], 16) for k in (1, 3, 5)] for anchor in anchors],
        dtype=float,
    )
    positions = np.linspace(0.0, len(anchors) - 1, count)
    colors = []
    for position in positions:
        lower = min(int(position), len(anchors) - 2)
        fraction = position - lower
        rgb = (1 - fraction) * anchorRgb[lower] + fraction * anchorRgb[lower + 1]
        colors.append("#" + "".join(f"{int(round(c)):02x}" for c in rgb))
    return colors


class DistanceHeatmapSurface:
    """A loaded .obj coloured per vertex by its signed distance.

    Stands in for a primitive in renderPrimitives, like ObjSurface. The
    scale is fixed at +-distanceLimit and symmetric about 0, split into
    flat bands bandWidth wide: cool bands below 0, warm bands above, with a
    hard break at 0. Only the positive side is cut off: a d above
    distanceLimit is drawn in cutoffColor, while a d below -distanceLimit
    stays in the deepest blue band. hoverTexts holds one label per vertex,
    which is what plotly shows on hover.
    """

    def __init__(
        self, vertices, triangles, distances, hoverTexts, name, opacity,
        distanceLimit, bandWidth=0.005
    ):
        self.vertices = vertices
        self.triangles = triangles
        self.name = name
        self.drawWireframe = False

        bandCount = max(1, int(round(distanceLimit / bandWidth)))
        insideBands = sampleColorRamp(insideColorAnchors, bandCount)
        outsideBands = sampleColorRamp(outsideColorAnchors, bandCount)

        # The colour axis runs from -distanceLimit to a margin past
        # +distanceLimit, which holds the grey cutoff cap. Plotly clamps
        # anything beyond the axis to its end colour.
        axisMin = -distanceLimit
        axisMax = 1.1 * distanceLimit

        def axisPosition(value):
            return (value - axisMin) / (axisMax - axisMin)

        # Bands from the most negative up: deepest blue ... cyan | yellow
        # ... dark red | grey. Repeating each edge with the next colour
        # makes every step hard, so a band is one flat colour, not a blend.
        bandEdges = np.linspace(-distanceLimit, distanceLimit, 2 * bandCount + 1)
        bandColors = insideBands[::-1] + outsideBands + [cutoffColor]
        # The grey cap is the last band, from the limit to the axis end.
        colorEdges = [axisPosition(edge) for edge in bandEdges] + [1.0]
        colorscale = []
        for bandIndex, bandColor in enumerate(bandColors):
            colorscale += [
                [colorEdges[bandIndex], bandColor],
                [colorEdges[bandIndex + 1], bandColor],
            ]

        # intensity takes precedence over the flat colour renderPrimitives
        # assigns, so the per-type colour never shows on this trace.
        self.traceOverrides = dict(
            intensity=distances,
            intensitymode="vertex",
            colorscale=colorscale,
            cmin=axisMin,
            cmax=axisMax,
            showscale=True,
            colorbar=dict(
                title="d [mm]",
                thickness=12,
                len=0.6,
                tickvals=bandEdges,
                ticktext=[f"{1000 * edge:+.0f}" for edge in bandEdges],
            ),
            opacity=opacity,
            flatshading=False,
            lighting=heatmapLighting,
            text=hoverTexts,
            hovertemplate="%{text}<extra></extra>",
        )

    def buildGeometry(self, numTheta, numPhi):
        return self.vertices, self.triangles
