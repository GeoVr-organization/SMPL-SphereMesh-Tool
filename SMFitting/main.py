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
    # SphereMesh fitting

    Fits a sphere mesh to the surface it was built for, with one regularized
    least-squares solve of the tangency condition, and measures how close the
    result gets:

    - **4** — pick a sphere mesh
    - **5–7** — the input sphere mesh inside its surface, the signed distance
      of every surface vertex to it, and that distance as a heatmap
    - **8–9** — the linear system, its solution, and the sphere mesh rebuilt
      from it
    - **10–11** — the fitted sphere mesh inside its surface, and its heatmap
    - **after 11** — diagnostics (the matrix, its empty columns, distance
      statistics, the vertices whose normal disagrees with their closest
      sphere) and the save to `fitted_<name>.txt`

    Sections 1–3, one sphere, one capsuloid and one prysmoid on their own, are
    in `SMClasstest.py`.

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
    from ObjMesh import DistanceHeatmapSurface, ObjSurface, loadObjMesh
    from PlotlyView import HighlightedSphere, MergedSpheres, renderPrimitives
    from scipy.sparse import lil_matrix
    import scipy.sparse.linalg as spla
    from scipy.sparse import vstack, diags


    return (
        DistanceHeatmapSurface,
        HighlightedSphere,
        MergedSpheres,
        ObjSurface,
        Path,
        SphereMesh,
        diags,
        go,
        lil_matrix,
        loadObjMesh,
        mo,
        np,
        pd,
        plt,
        renderPrimitives,
        spla,
        vstack,
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
        showSpheresCheckbox,
        showWireframeCheckbox,
    )


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
def _(SmplSM, armSM, bunnySM, mo):
    # The single-sphere, capsuloid and prysmoid test meshes live in
    # SMClasstest.py now, so the picker offers only the fitted examples.
    meshesByName = {
        "bunny": bunnySM,
        "smpl": SmplSM,
        "smpl_arm": armSM,
    }

    meshSelector = mo.ui.dropdown(
        options=list(meshesByName),
        value="smpl",
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
    heatmapDistanceLimit = 0.02

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
    P, P_r = 0.25, 0.25 

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
def _(A_reg, M, np):
    #print number of columns with no contributions
    # The last 4M rows of A_reg are the regularization diagonal, which puts a
    # 0.05 in every column, so only the vertex rows above it are checked.
    _vertexRows = A_reg[: A_reg.shape[0] - 4 * M]
    _nonZeroCounts = np.asarray((_vertexRows != 0).sum(axis=0)).ravel()
    _emptyColumns = np.flatnonzero(_nonZeroCounts == 0)
    _unknownNames = ("cx", "cy", "cz", "r")

    print(f"{len(_emptyColumns)} of {_vertexRows.shape[1]} columns of A are all zero:")
    for _column in _emptyColumns:
        print(f"  column {_column}: s{_column // 4} {_unknownNames[_column % 4]}")

    # The all-zero columns are listed above, so the emptiest ones worth naming
    # are the emptiest columns that still have a vertex in them. A sphere's
    # four columns usually tie, so every column at the minimum is listed.
    _nonEmptyCounts = _nonZeroCounts[_nonZeroCounts > 0]
    if len(_nonEmptyCounts) > 0:
        _minimumCount = _nonEmptyCounts.min()
        _emptiestColumns = np.flatnonzero(_nonZeroCounts == _minimumCount)
        print(
            f"the emptiest non-empty columns have {_minimumCount} non-zero values "
            f"({len(_emptiestColumns)} columns):"
        )
        for _column in _emptiestColumns:
            print(f"  column {_column}: s{_column // 4} {_unknownNames[_column % 4]}")
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


@app.cell
def _(
    ObjSurface,
    dfVertices,
    fittedMesh,
    go,
    mesh_normals,
    mesh_triangles,
    mesh_vertices,
    mo,
    np,
    numPhiSlider,
    numThetaSlider,
    renderPrimitives,
    showWireframeCheckbox,
    spheremesh,
):
    #solving butt problem

    #show only vertecies in which the original closest sphere - vertex position direction disagree with the vertex normal

    # A vertex is shown when n . (p - c) / |p - c| falls below this, with c the
    # centre of its closest sphere in the original mesh. 1 means the normal
    # points straight away from that sphere, as the tangency equation of
    # section 8 assumes; 0.5 is 60 degrees off.
    alignmentThreshold = 0.5

    # The closest sphere's centre is its contributions' blend of the constituent
    # centres - exactly, for a capsuloid's lerp and for a prysmoid's barycentric
    # point on the midplane - so section 6's table already holds it.
    constituentCentres = np.array([sphere.center for sphere in spheremesh.spheres])
    _contributionsPerVertex = list(dfVertices["contributions"])
    _closestCentres = np.array([
        sum(weight * constituentCentres[sphereIndex] for sphereIndex, weight in contributions)
        if contributions else np.full(3, np.nan)
        for contributions in _contributionsPerVertex
    ])
    _offsets = mesh_vertices - _closestCentres
    vertexAlignment = (
        np.einsum("ij,ij->i", mesh_normals, _offsets)
        / np.linalg.norm(_offsets, axis=1)
    )
    # A vertex without contributions has a NaN alignment, which this skips.
    misalignedVertexIndices = np.flatnonzero(vertexAlignment < alignmentThreshold)

    # How much of the solve those vertices hand each constituent sphere.
    _weightBySphere = {}
    for _vertexIndex in misalignedVertexIndices:
        for _sphereIndex, _weight in _contributionsPerVertex[_vertexIndex]:
            _weightBySphere[_sphereIndex] = _weightBySphere.get(_sphereIndex, 0.0) + _weight
    _involvedSphereIndices = sorted(
        (sphereIndex for sphereIndex, weight in _weightBySphere.items() if weight > 0.01),
        key=lambda sphereIndex: -_weightBySphere[sphereIndex],
    )

    # Only the spheres those vertices pull on, translucent, inside a faint
    # shell cut down to the box around them (plus 5 cm), so the view opens on
    # them rather than on the whole body.
    if _involvedSphereIndices:
        _involvedRadii = np.array(
            [spheremesh.spheres[sphereIndex].radius for sphereIndex in _involvedSphereIndices]
        )
        _involvedCentres = constituentCentres[_involvedSphereIndices]
        _boxMin = (_involvedCentres - _involvedRadii[:, None]).min(axis=0) - 0.05
        _boxMax = (_involvedCentres + _involvedRadii[:, None]).max(axis=0) + 0.05
        _vertexInBox = np.all((mesh_vertices >= _boxMin) & (mesh_vertices <= _boxMax), axis=1)
        _keptTriangles = mesh_triangles[_vertexInBox[mesh_triangles].all(axis=1)]
        # Re-indexed onto the vertices they use: plotly sizes the scene to every
        # vertex it is given, referenced or not.
        _shellVertexIndices, _shellCorners = np.unique(_keptTriangles.ravel(), return_inverse=True)
        _shellVertices = mesh_vertices[_shellVertexIndices]
        _shellTriangles = _shellCorners.reshape(-1, 3)
    else:
        _shellVertices, _shellTriangles = mesh_vertices, mesh_triangles
    _primitives = [spheremesh.spheres[sphereIndex] for sphereIndex in _involvedSphereIndices]
    _primitives.append(ObjSurface(_shellVertices, _shellTriangles, "Surface Mesh", 0.15))
    misalignmentFigure = renderPrimitives(
        _primitives,
        None,
        numThetaSlider.value,
        numPhiSlider.value,
        0.15,
        showWireframeCheckbox.value,
        showAxes=False,
        showLegend=True,
        backgroundColor="#1d252b",
        uiRevision=f"misaligned {fittedMesh}",
        lambertian=True,
    )
    # Front (-y) and a little below, closer than the notebook's usual eye:
    # these vertices sit between the legs, with their normals facing down.
    misalignmentFigure.update_layout(scene_camera=dict(eye=dict(x=0.45, y=-1.25, z=-0.2)))

    def _segments(starts, ends):
        """NaN-separated start-end pairs, one polyline per segment for Scatter3d."""
        points = np.full((3 * len(starts), 3), np.nan)
        points[0::3] = starts
        points[1::3] = ends
        return points

    _positions = mesh_vertices[misalignedVertexIndices]
    _normals = mesh_normals[misalignedVertexIndices]
    _centres = _closestCentres[misalignedVertexIndices]
    _alignments = vertexAlignment[misalignedVertexIndices]

    # What the tangency equation assumes the normal to be, next to what it is.
    _centreSegments = _segments(_centres, _positions)
    misalignmentFigure.add_trace(go.Scatter3d(
        x=_centreSegments[:, 0], y=_centreSegments[:, 1], z=_centreSegments[:, 2],
        mode="lines", line=dict(color="#c3c2b7", width=2),
        name="closest-sphere centre → vertex", hoverinfo="skip",
    ))
    _normalSegments = _segments(_positions, _positions + 0.05 * _normals)
    misalignmentFigure.add_trace(go.Scatter3d(
        x=_normalSegments[:, 0], y=_normalSegments[:, 1], z=_normalSegments[:, 2],
        mode="lines", line=dict(color="#ff7f0e", width=4),
        name="vertex normal (5 cm)", hoverinfo="skip",
    ))

    # One blue, light to dark: the lightest, most visible on the dark
    # background, are the vertices that disagree the most.
    misalignmentFigure.add_trace(go.Scatter3d(
        x=_positions[:, 0], y=_positions[:, 1], z=_positions[:, 2],
        mode="markers",
        marker=dict(
            size=8,
            color=_alignments,
            colorscale=[[0.0, "#cde2fb"], [1.0, "#2a78d6"]],
            cmin=-1.0,
            cmax=alignmentThreshold,
            colorbar=dict(title="n·(p−c)/|p−c|", thickness=12, len=0.6),
        ),
        name="misaligned vertex",
        text=[
            f"vertex {vertexIndex}<br>n·(p−c)/|p−c| {alignment:+.2f}"
            + "".join(
                f"<br>s{sphereIndex} {weight:.3f}"
                for sphereIndex, weight in _contributionsPerVertex[vertexIndex]
            )
            for vertexIndex, alignment in zip(misalignedVertexIndices, _alignments)
        ],
        hovertemplate="%{text}<extra></extra>",
    ))

    # Each involved sphere named at its centre.
    _labelCentres = constituentCentres[_involvedSphereIndices]
    misalignmentFigure.add_trace(go.Scatter3d(
        x=_labelCentres[:, 0], y=_labelCentres[:, 1], z=_labelCentres[:, 2],
        mode="text",
        text=[f"s{sphereIndex}" for sphereIndex in _involvedSphereIndices],
        textfont=dict(color="#fbfcfc", size=13),
        showlegend=False,
        hoverinfo="skip",
    ))

    mo.vstack(
        [
            mo.md(
                f"**{len(misalignedVertexIndices)}** of {len(mesh_vertices)} vertices "
                f"have n·(p−c)/|p−c| < {alignmentThreshold} against their closest "
                f"sphere in the original mesh. The weight they hand the solve: "
                + ", ".join(
                    f"`s{sphereIndex}` {_weightBySphere[sphereIndex]:.1f}"
                    for sphereIndex in _involvedSphereIndices
                )
            ),
            misalignmentFigure,
        ],
        gap=1,
    )
    return


@app.cell
def _(SphereMesh, fittedMesh, fitted_SM_text, meshRotation):
    #save the fitted sphere mesh to a file
    #rotate back to the original orientation, so it can be loaded into marimo
    # This is necessary because the fitting process is done in a different coordinate system.
    # A rotation's inverse is its transpose. It is applied to a copy rebuilt
    # from fitted_SM_text, never to fittedSM: marimo may run sections 10 and 11
    # after this cell, and they must draw fittedSM in the plot's orientation.
    # Saved into examples/, where sphereMeshPathsByMeshName reads it, so the
    # next run starts from this fit.
    _savedSM = SphereMesh()
    _savedSM.loadFromText(fitted_SM_text)
    _savedSM.rotate(meshRotation.T)
    _savedSM.saveToFile(f"examples/fitted_{fittedMesh}.txt")
    return


if __name__ == "__main__":
    app.run()
