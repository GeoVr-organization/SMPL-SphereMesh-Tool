# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "pandas",
#     "plotly",
# ]
#
# # The SMPL figures carry ~120 primitive meshes: about 5 MB at the default
# # numTheta/numPhi and ~40 MB at their maximum, past marimo's 8 MB default.
# [tool.marimo.runtime]
# output_max_bytes = 50_000_000
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # SphereMesh `buildGeometry` test

    Three sphere meshes, one per primitive type:

    1. **one sphere** — a single `Sphere`
    2. **one capsuloid** — two spheres + the `Capsuloid` joining them
    3. **one prysmoid** — three spheres + the `Prysmoid` spanning them

    Each primitive's `buildGeometry(numTheta, numPhi)` output is rendered
    directly, so whatever the function returns is what you see.

    The plots are interactive: drag to orbit, scroll to zoom, right-drag to
    pan, and click a legend entry to hide that primitive.
    """)
    return


@app.cell(hide_code=True)
def _():
    import sys
    import marimo as mo
    from marimo._runtime.context import get_context
    get_context().marimo_config["runtime"]["output_max_bytes"] = 50_000_000
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import matplotlib.pyplot as plt

    notebookDirectory = Path(__file__).parent
    if str(notebookDirectory) not in sys.path:
        sys.path.insert(0, str(notebookDirectory))

    from SphereMesh import SphereMesh
    from scipy.sparse import lil_matrix
    import scipy.sparse.linalg as spla
    from scipy.sparse import vstack, diags


    return SphereMesh, diags, go, lil_matrix, mo, np, pd, plt, spla, vstack


@app.cell
def _():
    # Every example file the notebook reads, keyed by the mesh names that the
    # section 4 dropdown and fittedMesh of section 5 use.
    sphereMeshPathsByMeshName = {
        "bunny": "examples/bunny.txt",  # TODO da rifare il file bunny.txt
        "smpl_arm": "examples/arm2.txt",
        #"smpl": "examples/smol_star.txt",
        "smpl": "examples/SMPL_male_star.txt",
    
    }
    # The surface each sphere mesh was fitted to, shown from section 5 on.
    objPathsByMeshName = {
        "smpl_arm": "examples/arm.obj",
        #"smpl": "examples/SMPL_male.obj",
        "smpl": "examples/SMPL_starpose.obj",
    }
    return objPathsByMeshName, sphereMeshPathsByMeshName


@app.cell
def _(mo):
    numThetaSlider = mo.ui.slider(4, 48, value=16, label="numTheta", show_value=True)
    numPhiSlider = mo.ui.slider(4, 96, value=32, label="numPhi", show_value=True)
    opacitySlider = mo.ui.slider(
        0.1, 1.0, step=0.05, value=1.0, label="opacity", show_value=True
    )
    showSpheresCheckbox = mo.ui.checkbox(value=True, label="show the constituent spheres")
    showWireframeCheckbox = mo.ui.checkbox(value=False, label="show the wireframe")
    showAxesCheckbox = mo.ui.checkbox(value=True, label="show the axes")

    mo.vstack(
        [
            mo.hstack([numThetaSlider, numPhiSlider], justify="start", gap=2),
            opacitySlider,
            mo.hstack(
                [showSpheresCheckbox, showWireframeCheckbox, showAxesCheckbox],
                justify="start",
                gap=2,
            ),
        ]
    )
    return (
        numPhiSlider,
        numThetaSlider,
        opacitySlider,
        showAxesCheckbox,
        showSpheresCheckbox,
        showWireframeCheckbox,
    )


@app.cell(hide_code=True)
def _(go, np):
    # Colour per primitive type, so it is obvious which mesh came from where.
    primitiveColors = {
        "Sphere Class": "#ee380a",
        "Capsuloid Class": "#2f76b9",
        "Prysmoid Class": "#9545d6",
        # Interpolated spheres keep the colour of the primitive they came from.
        "Capsuloid Interpolation": "#2f76b9",
        "Prysmoid Interpolation": "#9545d6",
        "Surface Mesh": "#ffffff",
        "Closest Sphere": "#07beb8",
    }

    class MergedSpheres:
        """Many spheres drawn as one mesh, so they cost a single plotly trace.

        Stands in for a primitive in renderPrimitives: it has a name (colour,
        legend and hover label) and a buildGeometry().
        """

        def __init__(self, spheres, name):
            self.spheres = list(spheres)
            self.name = name

        def buildGeometry(self, numTheta, numPhi):
            allVertices = []
            allTriangles = []
            vertexOffset = 0
            for sphere in self.spheres:
                vertices, triangles = sphere.buildGeometry(numTheta, numPhi)
                allVertices.append(vertices)
                allTriangles.append(triangles + vertexOffset)
                vertexOffset += len(vertices)
            return np.vstack(allVertices), np.vstack(allTriangles)

    class HighlightedSphere:
        """One sphere drawn opaque over dimmed context, labelled by what it is.

        The hover carries the contributions SphereMesh.closestSphere() returns -
        which constituent spheres the result interpolates, and by how much -
        in place of the raw coordinates every other trace shows, since those
        weights are the answer being looked at.
        """

        def __init__(self, sphere, name, contributions):
            self.sphere = sphere
            self.name = name
            contributionLines = "".join(
                f"<br>s{sphereIndex} {contribution:.3f}"
                for sphereIndex, contribution in contributions
            )
            self.drawWireframe = False
            self.traceOverrides = dict(
                opacity=1.0,
                hovertemplate=f"{name}{contributionLines}<extra></extra>",
            )

        def buildGeometry(self, numTheta, numPhi):
            return self.sphere.buildGeometry(numTheta, numPhi)

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

    def buildWireframeSegments(vertices, triangles):
        """One polyline per unique triangle edge, NaN-separated for Scatter3d."""
        edges = np.vstack(
            (triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]])
        )
        edges = np.unique(np.sort(edges, axis=1), axis=0)

        segments = np.full((len(edges) * 3, 3), np.nan, dtype=vertices.dtype)
        segments[0::3] = vertices[edges[:, 0]]
        segments[1::3] = vertices[edges[:, 1]]
        return segments

    # Where every figure's camera starts.
    defaultCameraEye = dict(x=1.6, y=-1.6, z=0.9)

    # Pure Lambertian shading: an ambient fill plus diffuse, with no specular
    # highlight or fresnel rim, so brightness depends only on how a surface
    # faces the light. Plotly's own default (ambient 0.8 + diffuse 0.8)
    # saturates to full brightness on nearly every face, which shows no shape.
    # The light is fixed in the scene, not to the camera, so it is put along
    # the starting eye direction: the opening view is lit from the front, and
    # orbiting round shows the far side at the ambient level. Plotly caps each
    # light coordinate at 1e5, hence the rescale.
    _lightScale = 1e5 / max(abs(value) for value in defaultCameraEye.values())
    lambertianShading = dict(
        lighting=dict(
            ambient=0.4, diffuse=0.6, specular=0.0, roughness=1.0, fresnel=0.0
        ),
        lightposition={
            axis: value * _lightScale for axis, value in defaultCameraEye.items()
        },
    )

    def renderPrimitives(
        primitives, title, numTheta, numPhi, opacity, showWireframe, showAxes,
        showLegend=True, backgroundColor=None, uiRevision=None,
        primitiveColor=None, lambertian=False
    ):
        """Render each primitive's buildGeometry() output into one 3D figure.

        backgroundColor paints the whole figure (paper, plot and scene) and
        switches the wireframe to a light colour, for a dark-themed plot.
        uiRevision defaults to the title; pass it explicitly when the figure has
        no title, so the camera survives a slider change.
        primitiveColor replaces the per-type colours with one, for a figure
        where colour already encodes something else.
        lambertian shades every mesh with lambertianShading instead of plotly's
        default lighting; a primitive's own traceOverrides still win.
        """
        traces = []
        drawnLabels = set()
        wireframeColor = "#fbfcfc" if backgroundColor else "black"

        for primitive in primitives:
            vertices, triangles = primitive.buildGeometry(numTheta, numPhi)
            # Half the bytes of float64/int64 in the output marimo ships to the
            # browser, at far more precision than a plot can show.
            vertices = np.asarray(vertices, dtype=np.float32)
            triangles = np.asarray(triangles, dtype=np.int32)

            color = primitiveColor or primitiveColors.get(primitive.name, "#747070")
            # Only the first mesh of a type carries the legend entry; the group
            # makes a click on it toggle every mesh of that type.
            showTraceLegend = primitive.name not in drawnLabels
            drawnLabels.add(primitive.name)

            meshArguments = dict(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=triangles[:, 0],
                j=triangles[:, 1],
                k=triangles[:, 2],
                color=color,
                opacity=opacity,
                flatshading=True,
                name=primitive.name,
                legendgroup=primitive.name,
                showlegend=showTraceLegend,
                hovertemplate=(
                    f"{primitive.name}<br>"
                    "x %{x:.3f}<br>y %{y:.3f}<br>z %{z:.3f}<extra></extra>"
                ),
            )
            if lambertian:
                meshArguments.update(lambertianShading)
            # A primitive may bring its own plotly settings, which is how the
            # .obj shell gets its transparency and gloss.
            meshArguments.update(getattr(primitive, "traceOverrides", {}))
            traces.append(go.Mesh3d(**meshArguments))

            if showWireframe and getattr(primitive, "drawWireframe", True):
                segments = buildWireframeSegments(vertices, triangles)
                traces.append(
                    go.Scatter3d(
                        x=segments[:, 0],
                        y=segments[:, 1],
                        z=segments[:, 2],
                        mode="lines",
                        line=dict(color=wireframeColor, width=1),
                        legendgroup=primitive.name,
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

        figure = go.Figure(data=traces)
        figure.update_layout(
            title=title,
            height=600,
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(x=0.0, y=1.0),
            showlegend=showLegend,
            # Keeps the camera where the user left it when a slider re-runs the cell.
            uirevision=uiRevision if uiRevision is not None else title,
            scene=dict(
                # 'data' gives all three axes the same unit length, so nothing is
                # stretched the way an auto aspect ratio would stretch it.
                aspectmode="data",
                xaxis_title="x",
                yaxis_title="y",
                zaxis_title="z",
                xaxis=dict(visible=showAxes),
                yaxis=dict(visible=showAxes),
                zaxis=dict(visible=showAxes),
                camera=dict(eye=defaultCameraEye),
            ),
        )

        if backgroundColor:
            figure.update_layout(
                paper_bgcolor=backgroundColor,
                plot_bgcolor=backgroundColor,
                font=dict(color="#fbfcfc"),
            )
            figure.update_scenes(bgcolor=backgroundColor)

        return figure

    return (
        DistanceHeatmapSurface,
        HighlightedSphere,
        MergedSpheres,
        ObjSurface,
        loadObjMesh,
        renderPrimitives,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 - One sphere
    """)
    return


@app.cell
def _(SphereMesh):
    # ---- mesh 1: a single sphere ----------------------------------------
    meshSingleSphere = SphereMesh()

    singelspheremesh=  """ Sphere Mesh 2.0
                        1 0 0 0
                        0.0 0.0 0.0 1.0"""

    meshSingleSphere.loadFromText(singelspheremesh)
    singleSpherePrimitives = list(meshSingleSphere.spheres)
    return meshSingleSphere, singleSpherePrimitives


@app.cell
def _(
    numPhiSlider,
    numThetaSlider,
    opacitySlider,
    renderPrimitives,
    showAxesCheckbox,
    showWireframeCheckbox,
    singleSpherePrimitives,
):
    renderPrimitives(
        singleSpherePrimitives,
        "SphereMesh with 1 sphere",
        numThetaSlider.value,
        numPhiSlider.value,
        opacitySlider.value,
        showWireframeCheckbox.value,
        showAxesCheckbox.value,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 — one capsuloid
    """)
    return


@app.cell
def _(SphereMesh, showSpheresCheckbox):
    # ---- mesh 2: two spheres joined by one capsuloid ---------------------
    meshCapsuloid = SphereMesh()

    capspheremesh=  """ Sphere Mesh 2.0
                        2 0 1 0
                        0.0 0.0 0.0 0.9
                        1.5 0.5 0.0 0.4
                        0 1"""

    meshCapsuloid.loadFromText(capspheremesh)

    capsuloidPrimitives = list(meshCapsuloid.capsuloids)
    if showSpheresCheckbox.value:
        capsuloidPrimitives = list(meshCapsuloid.spheres) + capsuloidPrimitives
    return capsuloidPrimitives, meshCapsuloid


@app.cell
def _(
    capsuloidPrimitives,
    numPhiSlider,
    numThetaSlider,
    opacitySlider,
    renderPrimitives,
    showAxesCheckbox,
    showWireframeCheckbox,
):
    renderPrimitives(
        capsuloidPrimitives,
        "SphereMesh with 1 capsuloid",
        numThetaSlider.value,
        numPhiSlider.value,
        opacitySlider.value,
        showWireframeCheckbox.value,
        showAxesCheckbox.value,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 — one prysmoid
    """)
    return


@app.cell
def _(SphereMesh, showSpheresCheckbox):
    # ---- mesh 3: three spheres spanned by one prysmoid -------------------
    meshPrysmoid = SphereMesh()

    trispheremesh=  """ Sphere Mesh 2.0
                        3 1 0 0
                        0.0 0.0 0.0 0.9
                        2.5 0.0 0.0 0.6
                        1.0 3.2 0.3 0.45
                        0 1 2"""

    meshPrysmoid.loadFromText(trispheremesh)


    prysmoidPrimitives = list(meshPrysmoid.prysmoids)
    if showSpheresCheckbox.value:
        prysmoidPrimitives = list(meshPrysmoid.spheres) + prysmoidPrimitives
    return meshPrysmoid, prysmoidPrimitives


@app.cell
def _(
    numPhiSlider,
    numThetaSlider,
    opacitySlider,
    prysmoidPrimitives,
    renderPrimitives,
    showAxesCheckbox,
    showWireframeCheckbox,
):
    renderPrimitives(
        prysmoidPrimitives,
        "SphereMesh with 1 prysmoid",
        numThetaSlider.value,
        numPhiSlider.value,
        opacitySlider.value,
        showWireframeCheckbox.value,
        showAxesCheckbox.value,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 — pick a sphere mesh
    """)
    return


@app.cell
def _(SphereMesh, np, sphereMeshPathsByMeshName):
    # One rotation for these sphere meshes and for the .obj surfaces of section
    # 5, so each surface stays aligned with its sphere mesh: +90 degrees about
    # x, (x, y, z) -> (x, -z, y).
    meshRotation = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])

    text=""
    with open(sphereMeshPathsByMeshName["bunny"], "r") as f:
        text = f.read()
    bunnySM=SphereMesh()
    bunnySM.loadFromText(text)

    with open(sphereMeshPathsByMeshName["smpl"], "r") as f:
        text = f.read()
    SmplSM=SphereMesh()
    SmplSM.loadFromText(text)
    SmplSM.rotate(meshRotation)


    with open(sphereMeshPathsByMeshName["smpl_arm"], "r") as f:
        text = f.read()
    armSM=SphereMesh()
    armSM.loadFromText(text)
    armSM.rotate(meshRotation)
    return SmplSM, armSM, bunnySM, meshRotation


@app.cell
def _(
    SmplSM,
    armSM,
    bunnySM,
    meshCapsuloid,
    meshPrysmoid,
    meshSingleSphere,
    mo,
):
    meshesByName = {
        "single-sphere": meshSingleSphere,
        "capsuloid": meshCapsuloid,
        "prysmoid": meshPrysmoid,
        "bunny": bunnySM,
        "smpl": SmplSM,
        "smpl_arm": armSM,
    }

    meshSelector = mo.ui.dropdown(
        options=list(meshesByName),
        value="single-sphere",
        label="sphere mesh",
    )
    showInterpolationCheckbox = mo.ui.checkbox(
        value=False, label="Show interpolation"
    )
    return meshSelector, meshesByName, showInterpolationCheckbox


@app.cell(hide_code=True)
def _(mo, showInterpolationCheckbox):
    # Reading the checkbox re-runs this cell on every toggle, which builds a
    # fresh slider: turning interpolation back on always restarts from 1.
    showInterpolationCheckbox.value

    spheresPerPrimitiveSlider = mo.ui.slider(
        0, 20, value=1, label="Spheres per primitive", show_value=True
    )
    return (spheresPerPrimitiveSlider,)


@app.cell
def _(
    MergedSpheres,
    meshSelector,
    meshesByName,
    mo,
    numPhiSlider,
    numThetaSlider,
    opacitySlider,
    renderPrimitives,
    showInterpolationCheckbox,
    showSpheresCheckbox,
    showWireframeCheckbox,
    spheresPerPrimitiveSlider,
):
    selectedMesh = meshesByName[meshSelector.value]

    if showInterpolationCheckbox.value:
        # The interpolated spheres replace the capsuloid/prysmoid surfaces,
        # which would otherwise enclose them and hide them from view. They are
        # kept in two groups so each keeps the colour of the primitive it was
        # sampled from.
        capsuloidInterpolation, prysmoidInterpolation = (
            selectedMesh.getInterpolatedSpheresBySource(
                spheresPerPrimitiveSlider.value
            )
        )
        interpolatedSphereCount = (
            len(capsuloidInterpolation) + len(prysmoidInterpolation)
        )
        selectedPrimitives = [
            MergedSpheres(spheres, name)
            for spheres, name in (
                (capsuloidInterpolation, "Capsuloid Interpolation"),
                (prysmoidInterpolation, "Prysmoid Interpolation"),
            )
            if spheres
        ]
    else:
        interpolatedSphereCount = 0
        selectedPrimitives = (
            list(selectedMesh.capsuloids) + list(selectedMesh.prysmoids)
        )

    # Spheres are all a single-sphere mesh has, so never leave the plot empty.
    if showSpheresCheckbox.value or not selectedPrimitives:
        selectedPrimitives = list(selectedMesh.spheres) + selectedPrimitives

    selectedFigure = renderPrimitives(
        selectedPrimitives,
        None,
        numThetaSlider.value,
        numPhiSlider.value,
        opacitySlider.value,
        showWireframeCheckbox.value,
        showAxes=False,
        showLegend=False,
        backgroundColor="#1d252b",
        # Re-frames on a new mesh, but holds the camera through slider changes.
        uiRevision=meshSelector.value,
    )

    controlItems = [mo.md("**Sphere mesh**"), meshSelector, showInterpolationCheckbox]
    if showInterpolationCheckbox.value:
        controlItems += [
            spheresPerPrimitiveSlider,
            mo.md(f"{interpolatedSphereCount} interpolated spheres"),
        ]
    selectorControls = mo.vstack(controlItems)
    mo.hstack(
        [selectorControls, selectedFigure], widths=[1, 3], align="start", gap=2
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 — a fitted sphere mesh inside its surface
    """)
    return


@app.cell(hide_code=True)
def _(loadObjMesh, meshRotation, meshesByName, objPathsByMeshName):
    # The one example every block from here to the end shows. Change the name
    # to switch: it picks both the sphere mesh and the surface it was fitted to.
    fittedMesh = "smpl"

    spheremesh = meshesByName[fittedMesh]
    _objVertices, mesh_triangles, _objNormals = loadObjMesh(
        objPathsByMeshName[fittedMesh]
    )
    # The rotation section 4 gave the sphere meshes, so the surface stays
    # aligned with them. The rows are points, hence the transpose; a rotation
    # carries normals exactly as it carries points, and keeps the winding.
    mesh_vertices = _objVertices @ meshRotation.T
    mesh_normals = _objNormals @ meshRotation.T
    return fittedMesh, mesh_normals, mesh_triangles, mesh_vertices, spheremesh


@app.cell
def _(mo):
    surfaceShowInterpolationCheckbox = mo.ui.checkbox(
        value=False, label="Show interpolation"
    )
    surfaceOpacitySlider = mo.ui.slider(
        0.0, 1.0, step=0.05, value=0.2, label="surface opacity", show_value=True
    )
    showClosestSphereCheckbox = mo.ui.checkbox(
        value=False, label="Show closest sphere"
    )
    # While a vertex is being queried everything but the answer is context, and
    # these two say how far each half of it fades back. They stand in for
    # surfaceOpacitySlider and the notebook-wide opacity slider, which is why
    # the surface one is hidden below while this mode is on.
    closestSphereMeshOpacitySlider = mo.ui.slider(
        0.0, 1.0, step=0.05, value=0.6,
        label="sphere mesh opacity", show_value=True,
    )
    closestSurfaceOpacitySlider = mo.ui.slider(
        0.0, 1.0, step=0.05, value=0.3,
        label="surface opacity", show_value=True,
    )
    return (
        closestSphereMeshOpacitySlider,
        closestSurfaceOpacitySlider,
        showClosestSphereCheckbox,
        surfaceOpacitySlider,
        surfaceShowInterpolationCheckbox,
    )


@app.cell
def _(mo, surfaceShowInterpolationCheckbox):
    # Rebuilt on every toggle, like the slider of the block above, so turning
    # interpolation back on always restarts from 1.
    surfaceShowInterpolationCheckbox.value

    surfaceSpheresPerPrimitiveSlider = mo.ui.slider(
        0, 20, value=1, label="Spheres per primitive", show_value=True
    )
    return (surfaceSpheresPerPrimitiveSlider,)


@app.cell
def _(mesh_vertices, mo, showClosestSphereCheckbox):
    # Rebuilt whenever the checkbox is toggled, so the pick restarts at
    # vertex 0, and sized to the example's vertex count.
    showClosestSphereCheckbox.value

    surfaceVertexCount = len(mesh_vertices)
    vertexIndexSlider = mo.ui.slider(
        0, surfaceVertexCount - 1, value=0, label="vertex", show_value=True
    )
    return (vertexIndexSlider,)


@app.cell
def _(
    HighlightedSphere,
    MergedSpheres,
    ObjSurface,
    closestSphereMeshOpacitySlider,
    closestSurfaceOpacitySlider,
    fittedMesh,
    go,
    mesh_triangles,
    mesh_vertices,
    mo,
    numPhiSlider,
    numThetaSlider,
    opacitySlider,
    renderPrimitives,
    showClosestSphereCheckbox,
    showSpheresCheckbox,
    showWireframeCheckbox,
    spheremesh,
    surfaceOpacitySlider,
    surfaceShowInterpolationCheckbox,
    surfaceSpheresPerPrimitiveSlider,
    vertexIndexSlider,
):
    if surfaceShowInterpolationCheckbox.value:
        capsuloidFittedSpheres, prysmoidFittedSpheres = (
            spheremesh.getInterpolatedSpheresBySource(
                surfaceSpheresPerPrimitiveSlider.value
            )
        )
        fittedSphereCount = (
            len(capsuloidFittedSpheres) + len(prysmoidFittedSpheres)
        )
        fittedPrimitives = [
            MergedSpheres(spheres, name)
            for spheres, name in (
                (capsuloidFittedSpheres, "Capsuloid Interpolation"),
                (prysmoidFittedSpheres, "Prysmoid Interpolation"),
            )
            if spheres
        ]
    else:
        fittedSphereCount = 0
        fittedPrimitives = (
            list(spheremesh.capsuloids) + list(spheremesh.prysmoids)
        )

    if showSpheresCheckbox.value or not fittedPrimitives:
        fittedPrimitives = list(spheremesh.spheres) + fittedPrimitives

    # Everything drawn so far is context for the query below, so it is dimmed
    # to let the one sphere the query returns stand out opaque on top of it.
    # Each half of that context fades by its own slider while the query is on.
    contextOpacity = (
        closestSphereMeshOpacitySlider.value
        if showClosestSphereCheckbox.value
        else opacitySlider.value
    )
    shellOpacity = (
        closestSurfaceOpacitySlider.value
        if showClosestSphereCheckbox.value
        else surfaceOpacitySlider.value
    )

    # Before the shell, so plotly blends the shell over what it encloses.
    fittedPrimitives = fittedPrimitives + [
        ObjSurface(mesh_vertices, mesh_triangles, "Surface Mesh", shellOpacity)
    ]

    closestSphereContributions = []
    if showClosestSphereCheckbox.value:
        queriedVertex = mesh_vertices[vertexIndexSlider.value]
        closestSphere, closestSphereContributions = spheremesh.closestSphere(
            queriedVertex
        )
        if closestSphere is not None:
            # Last of all, so nothing translucent is blended over the answer.
            fittedPrimitives = fittedPrimitives + [
                HighlightedSphere(
                    closestSphere, "Closest Sphere", closestSphereContributions
                )
            ]

    fittedFigure = renderPrimitives(
        fittedPrimitives,
        None,
        numThetaSlider.value,
        numPhiSlider.value,
        contextOpacity,
        showWireframeCheckbox.value,
        showAxes=False,
        showLegend=False,
        backgroundColor="#1d252b",
        uiRevision=fittedMesh,
    )

    if showClosestSphereCheckbox.value:
        fittedFigure.add_trace(
            go.Scatter3d(
                x=[queriedVertex[0]],
                y=[queriedVertex[1]],
                z=[queriedVertex[2]],
                mode="markers",
                marker=dict(color="#ff7f0e", size=5),
                name="Vertex",
                showlegend=False,
                hovertemplate=(
                    f"vertex {vertexIndexSlider.value}<br>"
                    "x %{x:.3f}<br>y %{y:.3f}<br>z %{z:.3f}<extra></extra>"
                ),
            )
        )

    fittedControlItems = [mo.md(f"**Sphere mesh** `{fittedMesh}`")]
    # The closest-sphere sliders take this one's job while that mode is on, so
    # showing both at once would put two "surface opacity" controls side by
    # side, only one of which does anything.
    if not showClosestSphereCheckbox.value:
        fittedControlItems.append(surfaceOpacitySlider)
    fittedControlItems.append(surfaceShowInterpolationCheckbox)

    if surfaceShowInterpolationCheckbox.value:
        fittedControlItems += [
            surfaceSpheresPerPrimitiveSlider,
            mo.md(f"{fittedSphereCount} interpolated spheres"),
        ]

    fittedControlItems.append(showClosestSphereCheckbox)
    if showClosestSphereCheckbox.value:
        contributionReport = "".join(
            f"- `s{sphereIndex}` {contribution:.3f}\n"
            for sphereIndex, contribution in closestSphereContributions
        )
        fittedControlItems += [
            closestSphereMeshOpacitySlider,
            closestSurfaceOpacitySlider,
            vertexIndexSlider,
            mo.md(
                f"**Closest sphere**\n\n{contributionReport}"
                if contributionReport
                else "**Closest sphere**\n\nno primitive faces this vertex"
            ),
        ]

    mo.hstack(
        [mo.vstack(fittedControlItems), fittedFigure],
        widths=[1, 3],
        align="start",
        gap=2,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 — signed distance of every surface vertex

    `SphereMesh.closestSphere()` run on every vertex of the example's surface:
    its position, its normal, its signed distance `d` to the sphere mesh (negative
    inside), and the constituent spheres that closest sphere interpolates, with
    how much each one accounts for. Sort by `d` to find where the fit is worst.
    """)
    return


@app.cell
def _(mesh_normals, mesh_vertices, np, pd, spheremesh):
    # ---- signed distance and contributions for every surface vertex -------
    _distances = []
    _contributionColumn = []
    for _vertex in mesh_vertices:
        _sphere, _contributions = spheremesh.closestSphere(_vertex)
        # No contributions means no primitive surface answered: the mesh has
        # none (sphere is None), or a prysmoid won from behind its midplane and
        # handed back a radius-0 placeholder whose d = 0 would mean nothing.
        _distances.append(
            _sphere.signedDistance(_vertex) if _contributions else np.nan
        )
        _contributionColumn.append(
            [
                (sphereIndex, float(contribution))
                for sphereIndex, contribution in _contributions
            ]
        )

    dfVertices = pd.DataFrame(
        {
            "px": mesh_vertices[:, 0],
            "py": mesh_vertices[:, 1],
            "pz": mesh_vertices[:, 2],
            "nx": mesh_normals[:, 0],
            "ny": mesh_normals[:, 1],
            "nz": mesh_normals[:, 2],
            "d": _distances,
            "contributions": _contributionColumn,
        }
    )
    # Same numbering as the vertex slider above, so a row can be looked at there.
    dfVertices.index.name = "vertex"
    dfVertices
    return (dfVertices,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7 — signed distance heatmap

    The surface coloured by the `d` of the table above, in flat 5 mm bands.
    **Cool** colours (cyan near 0, deepening to dark blue) where it is inside
    the sphere mesh, so the spheres stick out through it; **warm** colours
    (yellow near 0, deepening to dark red) where it is outside, so the spheres
    fall short of it. The two meet at 0 with a hard break, so even a tiny `d`
    shows its sign. The scale is fixed and symmetric about 0. A vertex above
    its top is **dark grey**; one below its bottom stays in the deepest blue.
    The sphere mesh itself is hidden here. Hover a vertex to read its number,
    `d` and contributions.
    """)
    return


@app.cell
def _(mo):
    heatmapSurfaceOpacitySlider = mo.ui.slider(
        0.0, 1.0, step=0.05, value=1.0, label="surface opacity", show_value=True
    )
    return (heatmapSurfaceOpacitySlider,)


@app.cell
def _(
    DistanceHeatmapSurface,
    dfVertices,
    fittedMesh,
    heatmapSurfaceOpacitySlider,
    mesh_triangles,
    mesh_vertices,
    mo,
    numPhiSlider,
    numThetaSlider,
    renderPrimitives,
    showWireframeCheckbox,
):
    # The colour scale spans +-this; a d above it is drawn dark grey.
    heatmapDistanceLimit = 0.035

    _hoverTexts = [
        f"vertex {vertexIndex}<br>d {distance:.4f}"
        + "".join(
            f"<br>{sphereName} {weight:.3f}" for sphereName, weight in contributions
        )
        for vertexIndex, distance, contributions in zip(
            dfVertices.index,
            dfVertices["d"],
            dfVertices["contributions"],
        )
    ]

    # Only the surface: the sphere mesh is hidden in this view, so nothing but
    # the distance colours is on screen.
    heatmapPrimitives = [
        DistanceHeatmapSurface(
            mesh_vertices,
            mesh_triangles,
            dfVertices["d"].to_numpy(),
            _hoverTexts,
            "Signed Distance",
            heatmapSurfaceOpacitySlider.value,
            heatmapDistanceLimit,
        )
    ]
    _cutoffCount = int((dfVertices["d"] > heatmapDistanceLimit).sum())

    heatmapFigure = renderPrimitives(
        heatmapPrimitives,
        None,
        numThetaSlider.value,
        numPhiSlider.value,
        1.0,
        showWireframeCheckbox.value,
        showAxes=False,
        showLegend=False,
        backgroundColor="#1d252b",
        uiRevision=f"heatmap {fittedMesh}",
        lambertian=True,
    )

    # Controls in one row on top, so the render below gets the full width.
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.md(f"**Sphere mesh** `{fittedMesh}`"),
                    heatmapSurfaceOpacitySlider,
                    mo.md(
                        f"scale fixed at ±{heatmapDistanceLimit} · "
                        f"dark grey: {_cutoffCount} of {len(dfVertices)} "
                        f"vertices with d > {heatmapDistanceLimit}"
                    ),
                ],
                justify="start",
                align="center",
                gap=2,
                wrap=True,
            ),
            heatmapFigure,
        ],
        gap=1,
    )
    return (heatmapDistanceLimit,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8 — equation to resolve

    ## Derivation of the Linear System

    Consider a vertex with coords $\mathbf p_i$ and normal $\hat{\mathbf n}_i$. The tangency condition with a sphere of center $\mathbf c$ and radius $r$ is

    $$
    (\mathbf p_i-\mathbf c)\cdot\hat{\mathbf n}_i=r.
    $$

    Expanding:

    $$
    \mathbf p_i\cdot\hat{\mathbf n}_i
    -
    \mathbf c\cdot\hat{\mathbf n}_i
    =
    r,
    $$

    $$
    \mathbf c\cdot\hat{\mathbf n}_i+r
    =
    \mathbf p_i\cdot\hat{\mathbf n}_i
    $$

    Since each vertex is associated with a combination of 2 or 3 spheres. We introduce a weight $w_{ij}$ for each sphere $j$, with

    $$
    \sum_{j=1}^{m} w_{ij}=1.
    $$

    Using the weights $w_{ij}$, the condition can be written compactly as

    $$
    \sum_{j=1}^{m}
    w_{ij}
    \left(
    \mathbf c_j\cdot\hat{\mathbf n}_i+r_j
    \right)
    =
    \mathbf p_i\cdot\hat{\mathbf n}_i.
    $$

    Considering all \(N\) points, we can collect these equations into a single linear system

    $$
    \boxed{
    A\mathbf{x}=\mathbf{b}
    }
    $$

    where \(A\) contains the coefficients of the unknown sphere centers and radii,

    $$
    \underbrace{
    \begin{bmatrix}
    w_{11}n_{1x} & w_{11}n_{1y} & w_{11}n_{1z} & w_{11}

    &
    \cdots
    &
    w_{1m}n_{1x} & w_{1m}n_{1y} & w_{1m}n_{1z} & w_{1m}
    \\
    \vdots & \vdots & \vdots & \vdots
    &
    \ddots
    &
    \vdots & \vdots & \vdots & \vdots
    \\
    w_{N1}n_{Nx} & w_{N1}n_{Ny} & w_{N1}n_{Nz} & w_{N1}
    &
    \cdots
    &
    w_{Nm}n_{Nx} & w_{Nm}n_{Ny} & w_{Nm}n_{Nz} & w_{Nm}
    \end{bmatrix}
    }_{A\;(N\times4m)},
    $$

    the unknown vector is

    $$
    \underbrace{
    \begin{bmatrix}
    c_{1x}\\
    c_{1y}\\
    c_{1z}\\
    r_1\\
    \vdots\\
    c_{mx}\\
    c_{my}\\
    c_{mz}\\
    r_m
    \end{bmatrix}
    }_{\mathbf{x}\;(4m\times1)},
    $$

    and the right-hand side contains one equation for each point:

    $$
    \underbrace{
    \begin{bmatrix}
    \mathbf p_1\cdot\hat{\mathbf n}_1\\
    \vdots\\
    \mathbf p_N\cdot\hat{\mathbf n}_N
    \end{bmatrix}
    }_{\mathbf{b}\;(N\times1)}.
    $$

    Thus,

    $$
    A\in\mathbb{R}^{N\times4m},
    \qquad
    \mathbf{x}\in\mathbb{R}^{4m},
    \qquad
    \mathbf{b}\in\mathbb{R}^{N}.
    $$
    """)
    return


@app.cell
def _(dfVertices, lil_matrix, mesh_vertices, np, spheremesh):
    N=len(mesh_vertices)
    M=len(spheremesh.spheres)

    A = lil_matrix((N, 4 * M))
    b = np.zeros(N)
    for _i in range(N):
        data=dfVertices.iloc[_i]
        p=np.array([data.px, data.py, data.pz])
        n=np.array([data.nx, data.ny, data.nz])
        b[_i]=np.dot(p,n)

        for w in data.contributions:
            index=w[0]
            weight=w[1]
            A[_i, 4 * index + 0] = weight * n[0]
            A[_i, 4 * index + 1] = weight * n[1]
            A[_i, 4 * index + 2] = weight * n[2]
            A[_i, 4 * index + 3] = weight
 
    return A, M, b


@app.cell
def _(A, M, b, diags, np, spheremesh, vstack):
    #Regularization
    P, P_r = 0.05, 0.05 

    x0 = np.zeros(4 * M)
    for j, s in enumerate(spheremesh.spheres):
        x0[4*j : 4*j+3] = np.asarray(s.center, dtype=float)
        x0[4*j + 3]     = s.radius

    D = diags(np.tile([P, P, P, P_r], M))       # (4M × 4M), diagonale
    A_reg = vstack([A.tocsr(), D]).tocsr()      # (N + 4M) × 4M
    b_reg = np.concatenate([b, D @ x0])
    return A_reg, b_reg


@app.cell
def _(A_reg, b_reg, spla):
    res=spla.lsqr(A_reg.tocsr(), b_reg)
    return (res,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9 - Reconstructing the spheremesh
    """)
    return


@app.cell
def _(M, res, spheremesh):
    fitted_SM_text="Sphere Mesh 2.0\n"
    fitted_SM_text+=f"{M} {len(spheremesh.prysmoids)} {len(spheremesh.capsuloids)} 0 \n"

    for _i in range(M):
        fitted_SM_text+=f"{res[0][4*_i+0]} {res[0][4*_i+1]} {res[0][4*_i+2]} {res[0][4*_i+3]}\n"

    for _i in range(len(spheremesh.prysmoids)):
        fitted_SM_text+=f"{spheremesh.prysmoids[_i].sA.index} {spheremesh.prysmoids[_i].sB.index} {spheremesh.prysmoids[_i].sC.index}\n"

    for _i in range(len(spheremesh.capsuloids)):
        fitted_SM_text+=f"{spheremesh.capsuloids[_i].sA.index} {spheremesh.capsuloids[_i].sB.index}\n"
    return (fitted_SM_text,)


@app.cell
def _(fitted_SM_text):
    print(fitted_SM_text)
    return


@app.cell
def _(SphereMesh, fitted_SM_text):
    fittedSM=SphereMesh()
    fittedSM.loadFromText(fitted_SM_text)
    return (fittedSM,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10 — the fitted sphere mesh inside its surface

    The view of section 5 again, for the sphere mesh rebuilt from the
    least-squares solution: the same surface drawn over it as a shell of clear
    plastic, with its own interpolation and closest-sphere controls.
    """)
    return


@app.cell
def _(mo):
    fittedShellInterpolationCheckbox = mo.ui.checkbox(
        value=False, label="Show interpolation"
    )
    fittedShellOpacitySlider = mo.ui.slider(
        0.0, 1.0, step=0.05, value=0.2, label="surface opacity", show_value=True
    )
    fittedShellClosestSphereCheckbox = mo.ui.checkbox(
        value=False, label="Show closest sphere"
    )
    # Stand in for fittedShellOpacitySlider and the notebook-wide opacity slider
    # while a vertex is being queried, as in section 5.
    fittedShellClosestMeshOpacitySlider = mo.ui.slider(
        0.0, 1.0, step=0.05, value=0.6,
        label="sphere mesh opacity", show_value=True,
    )
    fittedShellClosestSurfaceOpacitySlider = mo.ui.slider(
        0.0, 1.0, step=0.05, value=0.3,
        label="surface opacity", show_value=True,
    )
    return (
        fittedShellClosestMeshOpacitySlider,
        fittedShellClosestSphereCheckbox,
        fittedShellClosestSurfaceOpacitySlider,
        fittedShellInterpolationCheckbox,
        fittedShellOpacitySlider,
    )


@app.cell
def _(fittedShellInterpolationCheckbox, mo):
    # Rebuilt on every toggle, so turning interpolation back on restarts at 1.
    fittedShellInterpolationCheckbox.value

    fittedShellSpheresPerPrimitiveSlider = mo.ui.slider(
        0, 20, value=1, label="Spheres per primitive", show_value=True
    )
    return (fittedShellSpheresPerPrimitiveSlider,)


@app.cell
def _(fittedShellClosestSphereCheckbox, mesh_vertices, mo):
    # Rebuilt whenever the checkbox is toggled, so the pick restarts at vertex 0.
    fittedShellClosestSphereCheckbox.value

    fittedShellVertexIndexSlider = mo.ui.slider(
        0, len(mesh_vertices) - 1, value=0, label="vertex", show_value=True
    )
    return (fittedShellVertexIndexSlider,)


@app.cell
def _(
    HighlightedSphere,
    MergedSpheres,
    ObjSurface,
    fittedMesh,
    fittedSM,
    fittedShellClosestMeshOpacitySlider,
    fittedShellClosestSphereCheckbox,
    fittedShellClosestSurfaceOpacitySlider,
    fittedShellInterpolationCheckbox,
    fittedShellOpacitySlider,
    fittedShellSpheresPerPrimitiveSlider,
    fittedShellVertexIndexSlider,
    go,
    mesh_triangles,
    mesh_vertices,
    mo,
    numPhiSlider,
    numThetaSlider,
    opacitySlider,
    renderPrimitives,
    showSpheresCheckbox,
    showWireframeCheckbox,
):
    # Section 5's render, for fittedSM. Every name here is private (_), since
    # section 5 already defines the public ones.
    if fittedShellInterpolationCheckbox.value:
        _capsuloidSpheres, _prysmoidSpheres = (
            fittedSM.getInterpolatedSpheresBySource(
                fittedShellSpheresPerPrimitiveSlider.value
            )
        )
        _sphereCount = len(_capsuloidSpheres) + len(_prysmoidSpheres)
        _primitives = [
            MergedSpheres(spheres, name)
            for spheres, name in (
                (_capsuloidSpheres, "Capsuloid Interpolation"),
                (_prysmoidSpheres, "Prysmoid Interpolation"),
            )
            if spheres
        ]
    else:
        _sphereCount = 0
        _primitives = list(fittedSM.capsuloids) + list(fittedSM.prysmoids)

    if showSpheresCheckbox.value or not _primitives:
        _primitives = list(fittedSM.spheres) + _primitives

    # Everything drawn so far is context for the query below, so it is dimmed
    # to let the one sphere the query returns stand out opaque on top of it.
    _contextOpacity = (
        fittedShellClosestMeshOpacitySlider.value
        if fittedShellClosestSphereCheckbox.value
        else opacitySlider.value
    )
    _shellOpacity = (
        fittedShellClosestSurfaceOpacitySlider.value
        if fittedShellClosestSphereCheckbox.value
        else fittedShellOpacitySlider.value
    )

    # Before the shell, so plotly blends the shell over what it encloses.
    _primitives = _primitives + [
        ObjSurface(mesh_vertices, mesh_triangles, "Surface Mesh", _shellOpacity)
    ]

    _contributions = []
    if fittedShellClosestSphereCheckbox.value:
        _queriedVertex = mesh_vertices[fittedShellVertexIndexSlider.value]
        _closestSphere, _contributions = fittedSM.closestSphere(_queriedVertex)
        if _closestSphere is not None:
            # Last of all, so nothing translucent is blended over the answer.
            _primitives = _primitives + [
                HighlightedSphere(_closestSphere, "Closest Sphere", _contributions)
            ]

    _figure = renderPrimitives(
        _primitives,
        None,
        numThetaSlider.value,
        numPhiSlider.value,
        _contextOpacity,
        showWireframeCheckbox.value,
        showAxes=False,
        showLegend=False,
        backgroundColor="#1d252b",
        uiRevision=f"fitted shell {fittedMesh}",
    )

    if fittedShellClosestSphereCheckbox.value:
        _figure.add_trace(
            go.Scatter3d(
                x=[_queriedVertex[0]],
                y=[_queriedVertex[1]],
                z=[_queriedVertex[2]],
                mode="markers",
                marker=dict(color="#ff7f0e", size=5),
                name="Vertex",
                showlegend=False,
                hovertemplate=(
                    f"vertex {fittedShellVertexIndexSlider.value}<br>"
                    "x %{x:.3f}<br>y %{y:.3f}<br>z %{z:.3f}<extra></extra>"
                ),
            )
        )

    _controlItems = [mo.md(f"**Fitted sphere mesh** `{fittedMesh}`")]
    # The closest-sphere sliders take this one's job while that mode is on.
    if not fittedShellClosestSphereCheckbox.value:
        _controlItems.append(fittedShellOpacitySlider)
    _controlItems.append(fittedShellInterpolationCheckbox)

    if fittedShellInterpolationCheckbox.value:
        _controlItems += [
            fittedShellSpheresPerPrimitiveSlider,
            mo.md(f"{_sphereCount} interpolated spheres"),
        ]

    _controlItems.append(fittedShellClosestSphereCheckbox)
    if fittedShellClosestSphereCheckbox.value:
        _contributionReport = "".join(
            f"- `s{sphereIndex}` {contribution:.3f}\n"
            for sphereIndex, contribution in _contributions
        )
        _controlItems += [
            fittedShellClosestMeshOpacitySlider,
            fittedShellClosestSurfaceOpacitySlider,
            fittedShellVertexIndexSlider,
            mo.md(
                f"**Closest sphere**\n\n{_contributionReport}"
                if _contributionReport
                else "**Closest sphere**\n\nno primitive faces this vertex"
            ),
        ]

    mo.hstack(
        [mo.vstack(_controlItems), _figure],
        widths=[1, 3],
        align="start",
        gap=2,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11 — fitted sphere mesh heatmap

    The sphere mesh rebuilt from the least-squares solution, drawn like the
    heatmap of section 7: the surface coloured by its signed distance `d`, now
    measured against the **fitted** spheres, on the same fixed scale so the two
    views compare directly.
    """)
    return


@app.cell
def _(mo):
    fittedSMSphereMeshOpacitySlider = mo.ui.slider(
        0.0, 1.0, step=0.05, value=0.3, label="sphere mesh opacity", show_value=True
    )
    fittedSMSurfaceOpacitySlider = mo.ui.slider(
        0.0, 1.0, step=0.05, value=1.0, label="surface opacity", show_value=True
    )
    return fittedSMSphereMeshOpacitySlider, fittedSMSurfaceOpacitySlider


@app.cell
def _(
    DistanceHeatmapSurface,
    fittedMesh,
    fittedSM,
    fittedSMSphereMeshOpacitySlider,
    fittedSMSurfaceOpacitySlider,
    heatmapDistanceLimit,
    mesh_triangles,
    mesh_vertices,
    mo,
    np,
    numPhiSlider,
    numThetaSlider,
    renderPrimitives,
    showSpheresCheckbox,
    showWireframeCheckbox,
):
    # Signed distance of every surface vertex to the fitted sphere mesh, the
    # same way section 6 measures it against the original one.
    _distances = []
    _contributionColumn = []
    for _vertex in mesh_vertices:
        _sphere, _contributions = fittedSM.closestSphere(_vertex)
        _distances.append(
            _sphere.signedDistance(_vertex) if _contributions else np.nan
        )
        _contributionColumn.append(_contributions)
    fittedSMDistances = np.asarray(_distances)

    fittedSMPrimitives = list(fittedSM.capsuloids) + list(fittedSM.prysmoids)
    if showSpheresCheckbox.value or not fittedSMPrimitives:
        fittedSMPrimitives = list(fittedSM.spheres) + fittedSMPrimitives

    _hoverTexts = [
        f"vertex {vertexIndex}<br>d {distance:.4f}"
        + "".join(
            f"<br>{sphereName} {weight:.3f}" for sphereName, weight in contributions
        )
        for vertexIndex, (distance, contributions) in enumerate(
            zip(fittedSMDistances, _contributionColumn)
        )
    ]

    # Last, so plotly blends the surface over the sphere mesh it encloses.
    fittedSMPrimitives = fittedSMPrimitives + [
        DistanceHeatmapSurface(
            mesh_vertices,
            mesh_triangles,
            fittedSMDistances,
            _hoverTexts,
            "Signed Distance",
            fittedSMSurfaceOpacitySlider.value,
            heatmapDistanceLimit,
        )
    ]
    _cutoffCount = int((fittedSMDistances > heatmapDistanceLimit).sum())

    fittedSMFigure = renderPrimitives(
        fittedSMPrimitives,
        None,
        numThetaSlider.value,
        numPhiSlider.value,
        fittedSMSphereMeshOpacitySlider.value,
        showWireframeCheckbox.value,
        showAxes=False,
        showLegend=False,
        backgroundColor="#1d252b",
        uiRevision=f"fitted {fittedMesh}",
        # No primitiveColor: the sphere mesh keeps its per-type colours, red
        # spheres, blue capsuloids and purple prysmoids.
        lambertian=True,
    )

    # Controls in one row on top, so the render below gets the full width.
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.md(f"**Fitted sphere mesh** `{fittedMesh}`"),
                    fittedSMSphereMeshOpacitySlider,
                    fittedSMSurfaceOpacitySlider,
                    mo.md(
                        f"scale fixed at ±{heatmapDistanceLimit} · "
                        f"dark grey: {_cutoffCount} of {len(fittedSMDistances)} "
                        f"vertices with d > {heatmapDistanceLimit}"
                    ),
                ],
                justify="start",
                align="center",
                gap=2,
                wrap=True,
            ),
            fittedSMFigure,
        ],
        gap=1,
    )
    return (fittedSMDistances,)


@app.cell
def _(A_reg, M, go, np, plt):
    #render the linear system matrix
    # Every entry of A is a weight in [0, 1] times a component of a unit normal,
    # so the scale is fixed at [-1, 1], symmetric about 0: the zero entries -
    # most of the matrix, since a vertex involves at most three spheres - all
    # land on coolwarm's neutral grey, with blue below and red above.
    _coolwarm = plt.colormaps["coolwarm"]
    linearSystemColorscale = [
        [
            float(position),
            "rgb({}, {}, {})".format(
                *(round(255 * channel) for channel in _coolwarm(position)[:3])
            ),
        ]
        for position in np.linspace(0.0, 1.0, 21)
    ]

    # One label per column of A, in the order of the unknown vector x.
    unknownLabels = [
        f"s{sphereIndex} {unknown}"
        for sphereIndex in range(M)
        for unknown in ("cx", "cy", "cz", "r")
    ]

    linearSystemFigure = go.Figure(
        go.Heatmap(
            # float32 halves what marimo ships to the browser.
            z=A_reg.toarray().astype(np.float32),
            x=unknownLabels,
            colorscale=linearSystemColorscale,
            zmin=-1.0,
            zmax=1.0,
            colorbar=dict(tickformat=".2f", dtick=0.25, thickness=14),
            hovertemplate="vertex %{y}<br>%{x}<br>%{z:.3f}<extra></extra>",
        )
    )
    linearSystemFigure.update_layout(
        title=f"A  ({A_reg.shape[0]} × {A_reg.shape[1]})",
        height=800,
        margin=dict(l=60, r=20, t=50, b=60),
        xaxis=dict(title="unknown", tickangle=-90),
        # Row 0 on top, the way a matrix is written.
        yaxis=dict(title="vertex", autorange="reversed"),
        # Keeps a zoom in place when an upstream cell re-runs.
        uirevision="linear system matrix",
    )
    linearSystemFigure
    return


@app.cell
def _(fittedSMDistances, go, np):
    #some stats of fittedSMDistances

    print(f"fittedSMDistances: min {np.nanmin(fittedSMDistances):.4f}, max {np.nanmax(fittedSMDistances):.4f}, mean {np.nanmean(fittedSMDistances):.4f}, std {np.nanstd(fittedSMDistances):.4f}")
    #hystogram of fittedSMDistances
    histogramFigure = go.Figure(
        go.Histogram(
            x=fittedSMDistances,
            nbinsx=50,
            marker_color="#1f77b4",
            opacity=0.75,
        )
    )
    sum_of_distances = np.nansum(fittedSMDistances)
    print(f"Sum of fittedSMDistances: {sum_of_distances:.4f}")

    histogramFigure.update_layout(
        title="Histogram of fittedSMDistances",
        xaxis_title="Signed Distance",
        yaxis_title="Count",
        bargap=0.1,
        height=400,
        margin=dict(l=60, r=20, t=50, b=60),
    )

    return


if __name__ == "__main__":
    app.run()
