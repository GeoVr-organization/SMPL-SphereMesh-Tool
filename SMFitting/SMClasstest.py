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
# # Saving SphereMesh.py, ObjMesh.py or PlotlyView.py reloads the module and
# # re-runs the cells that use it, in the editor only (marimo run ignores it).
# auto_reload = "autorun"
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
    from ObjMesh import ObjSurface, loadObjMesh
    from PlotlyView import HighlightedSphere, MergedSpheres, renderPrimitives
    from scipy.sparse import lil_matrix
    import scipy.sparse.linalg as spla
    from scipy.sparse import vstack, diags


    return (
        HighlightedSphere,
        MergedSpheres,
        ObjSurface,
        Path,
        SphereMesh,
        go,
        loadObjMesh,
        mo,
        np,
        renderPrimitives,
    )


@app.cell
def _(Path):
    # Every example file the notebook reads, keyed by the mesh names that the
    # section 4 dropdown and fittedMesh of section 5 use.
    sphereMeshPathsByMeshName = {
        "bunny": "examples/bunny.txt",  # TODO da rifare il file bunny.txt
        "smpl_arm": "examples/arm2.txt",
        #"smpl": "examples/smol_star.txt",
        #"smpl": "examples/SMPL_male_star.txt",
        #"smpl": "examples/new_wp.txt",
        "smpl": "examples/fitted_smpl.txt",

    }
    # The surface each sphere mesh was fitted to, shown from section 5 on.
    objPathsByMeshName = {
        "smpl_arm": "examples/arm.obj",
        #"smpl": "examples/SMPL_male.obj",
        "smpl": "examples/SMPL_starpose_NOcrotch.obj",
    }

    #quick check if the files exist
    for meshName, path in sphereMeshPathsByMeshName.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"Sphere mesh file for {meshName} not found at {path}")

    for objName, path in objPathsByMeshName.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"OBJ file for {objName} not found at {path}")
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
    return fittedMesh, mesh_triangles, mesh_vertices, spheremesh


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


if __name__ == "__main__":
    app.run()
