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

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # SphereMesh fitting

    Fits a sphere mesh to the surface it was built for, with one regularized
    least-squares solve of the tangency condition, and measures how close the
    result gets:

    - **4** — the input sphere mesh on its own
    - **5–7** — the input sphere mesh inside its surface, the signed distance
      of every surface vertex to it, and that distance as a heatmap
    - **8–9** — the linear system, its solution, and the sphere mesh rebuilt
      from it
    - **10–11** — the fitted sphere mesh inside its surface, and its heatmap
    - **after 11** — diagnostics (the matrix, its empty columns, distance
      statistics, the vertices whose normal disagrees with their closest
      sphere) and the save to `fitted_<name>.txt`
    - **12** — the hernia loop: one more tangency row for every vertex still
      over 1 cm inside the fitted sphere mesh, solved again until none is
      left, and the save to `fitted_<name>_hernias.txt`

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
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    notebookDirectory = Path(__file__).parent
    if str(notebookDirectory) not in sys.path:
        sys.path.insert(0, str(notebookDirectory))

    from SphereMesh import SphereMesh
    from ObjMesh import DistanceHeatmapSurface, ObjSurface, heatmapBandColors, loadObjMesh
    from PlotlyView import HighlightedSphere, MergedSpheres, renderPrimitives
    from scipy.sparse import lil_matrix
    from scipy.sparse import vstack, diags


    return (
        BoundaryNorm,
        DistanceHeatmapSurface,
        HighlightedSphere,
        ListedColormap,
        MergedSpheres,
        ObjSurface,
        Path,
        Poly3DCollection,
        SphereMesh,
        diags,
        go,
        heatmapBandColors,
        lil_matrix,
        loadObjMesh,
        mo,
        np,
        pd,
        plt,
        renderPrimitives,
        vstack,
    )


@app.cell
def _(Path):
    # The one sphere mesh the notebook works on.
    #sphereMeshPath = "examples/smol_star.txt"
    #sphereMeshPath = "examples/SMPL_male_star.txt"
    #sphereMeshPath = "examples/new_wp.txt"
    sphereMeshPath = "examples/template_male_1.0.txt"
    # The surface it was fitted to, shown from section 5 on.
    #objPath = "examples/SMPL_male.obj"
    objPath = "examples/SMPL_starpose_NOcrotch.obj"

    #quick check if the files exist
    if not Path(sphereMeshPath).exists():
        raise FileNotFoundError(f"Sphere mesh file not found at {sphereMeshPath}")

    if not Path(objPath).exists():
        raise FileNotFoundError(f"OBJ file not found at {objPath}")
    return objPath, sphereMeshPath


@app.cell
def _():
    # Rendering parameters, shared by every plot.
    numTheta = 16
    numPhi = 32
    opacity = 1.0
    showSpheres = True  # the constituent spheres, next to the capsuloids and prysmoids
    showWireframe = False
    return numPhi, numTheta, opacity, showSpheres, showWireframe


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 — the sphere mesh
    """)
    return


@app.cell
def _(SphereMesh, np, sphereMeshPath):
    # One rotation for the sphere mesh and for the .obj surface of section 5,
    # so the surface stays aligned with its sphere mesh: +90 degrees about
    # x, (x, y, z) -> (x, -z, y).
    meshRotation = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])

    with open(sphereMeshPath, "r") as f:
        text = f.read()
    SmplSM=SphereMesh()
    SmplSM.loadFromText(text)
    SmplSM.rotate(meshRotation)
    return SmplSM, meshRotation


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 — a fitted sphere mesh inside its surface
    """)
    return


@app.cell(hide_code=True)
def _(SmplSM, loadObjMesh, meshRotation, objPath):
    # The sphere mesh every block from here to the end shows, and the surface it
    # was fitted to. The name labels the plots and names the saved fit.
    fittedMesh = "smpl"

    spheremesh = SmplSM
    _objVertices, mesh_triangles, _objNormals = loadObjMesh(objPath)
    # The rotation section 4 gave the sphere mesh, so the surface stays
    # aligned with it. The rows are points, hence the transpose; a rotation
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
    # surfaceOpacitySlider and the notebook-wide opacity, which is why
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
    numPhi,
    numTheta,
    opacity,
    renderPrimitives,
    showClosestSphereCheckbox,
    showSpheres,
    showWireframe,
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

    if showSpheres or not fittedPrimitives:
        fittedPrimitives = list(spheremesh.spheres) + fittedPrimitives

    # Everything drawn so far is context for the query below, so it is dimmed
    # to let the one sphere the query returns stand out opaque on top of it.
    # Each half of that context fades by its own slider while the query is on.
    contextOpacity = (
        closestSphereMeshOpacitySlider.value
        if showClosestSphereCheckbox.value
        else opacity
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
        numTheta,
        numPhi,
        contextOpacity,
        showWireframe,
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
def _(mesh_normals, mesh_vertices, pd, spheremesh):
    # ---- signed distance and contributions for every surface vertex -------
    _distances = []
    _contributionColumn = []
    for _vertex in mesh_vertices:
        _sphere, _contributions = spheremesh.closestSphere(_vertex)

        _distances.append(
            _sphere.signedDistance(_vertex) if _contributions else print (f"no contributions for vertex {_vertex}")
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
    The sphere mesh itself is hidden here. Three fixed views, front, 3/4 and
    back, each triangle in the band of the mean `d` of its corners.
    """)
    return


@app.cell
def _(
    BoundaryNorm,
    ListedColormap,
    Poly3DCollection,
    dfVertices,
    fittedMesh,
    heatmapBandColors,
    mesh_triangles,
    mesh_vertices,
    np,
    plt,
):
    # The colour scale spans +-this; a d above it is drawn dark grey.
    heatmapDistanceLimit = 0.02

    # The flat 5 mm bands and colours of DistanceHeatmapSurface, in mm. The
    # bands lie between neighbouring edges, and the last colour is the grey
    # above the last edge; a d below the first edge stays in the first band.
    _bandColors = heatmapBandColors(heatmapDistanceLimit)
    _bandEdges = np.linspace(
        -1000 * heatmapDistanceLimit, 1000 * heatmapDistanceLimit, len(_bandColors)
    )
    _colormap = ListedColormap(_bandColors)
    _norm = BoundaryNorm(_bandEdges, _colormap.N, extend="max")

    # Each triangle takes the mean d of its three corners, in mm.
    _distances = dfVertices["d"].to_numpy(dtype=float)
    _faceDistances = 1000 * _distances[mesh_triangles].mean(axis=1)
    plt.style.use('dark_background')
    heatmapFigure = plt.figure(figsize=(13, 4.6), dpi=200, layout="constrained")
    # The body's front is -y, which azimuth -90 looks at.
    for _index, (_view, _azimuth, _elevation) in enumerate(
        (("front", -90, 0), ("3/4", -45, 20), ("back", 90, 0))
    ):
        _axis = heatmapFigure.add_subplot(
            1, 3, _index + 1, projection="3d", proj_type="ortho"
        )
        _surface = Poly3DCollection(
            mesh_vertices[mesh_triangles],
            array=_faceDistances,
            cmap=_colormap,
            norm=_norm,
            linewidths=0.3,
        )
        _axis.add_collection3d(_surface)
        _surface.set_edgecolor("face")  # hides the seams between triangles
        _axis.auto_scale_xyz(*mesh_vertices.T)
        _axis.set_box_aspect(np.ptp(mesh_vertices, axis=0), zoom=1.3)
        _axis.view_init(_elevation, _azimuth)
        _axis.set_axis_off()
        _axis.set_title(_view)
        _axis.set_facecolor("dimgray")
    heatmapFigure.colorbar(
        _surface, ax=heatmapFigure.axes, ticks=_bandEdges, format="%+.0f",
        label="d [mm]", shrink=0.8,
    )
    heatmapFigure.suptitle(
        f"Sphere mesh {fittedMesh} · scale fixed at ±{1000 * heatmapDistanceLimit:.0f} mm · "
        f"dark grey: {int((_distances > heatmapDistanceLimit).sum())} of "
        f"{len(_distances)} vertices with d > {1000 * heatmapDistanceLimit:.0f} mm"
    )
    plt.close(heatmapFigure)
    heatmapFigure
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
def _(A_reg, b_reg, np):
    res=np.linalg.lstsq(A_reg.toarray(), b_reg, rcond=None)
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
def _(SphereMesh, fitted_SM_text):
    fittedSM=SphereMesh()
    fittedSM.loadFromText(fitted_SM_text)
    return (fittedSM,)


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
        0.0, 1.0, step=0.05, value=0.8, label="sphere mesh opacity", show_value=True
    )
    fittedSMSurfaceOpacitySlider = mo.ui.slider(
        0.0, 1.0, step=0.05, value=1.0, label="surface opacity", show_value=True
    )
    return fittedSMSphereMeshOpacitySlider, fittedSMSurfaceOpacitySlider


@app.cell
def _(fittedSM, mesh_vertices, np):
    # A vertex with a d below this, deeper inside the fitted sphere mesh than
    # 1 cm, gets a white sphere of this radius centred on it. A sphere rather
    # than a point marker: a marker sits exactly on the opaque surface, which
    # clips most of it.
    insideDistanceThreshold = -0.01
    insideMarkerRadius = 0.005

    # Signed distance of every surface vertex to the fitted sphere mesh, the
    # same way section 6 measures it against the original one. Apart from the
    # figure below, so its sliders re-run only the render: in the same cell,
    # every slider change would measure again, and re-run every cell that
    # uses these distances.
    _distances = []
    _contributionColumn = []
    for _vertex in mesh_vertices:
        _sphere, _contributions = fittedSM.closestSphere(_vertex)
        _distances.append(
            _sphere.signedDistance(_vertex) if _contributions else np.nan
        )
        _contributionColumn.append(_contributions)
    fittedSMDistances = np.asarray(_distances)

    fittedSMHoverTexts = [
        f"vertex {vertexIndex}<br>d {distance:.4f}"
        + "".join(
            f"<br>{sphereName} {weight:.3f}" for sphereName, weight in contributions
        )
        for vertexIndex, (distance, contributions) in enumerate(
            zip(fittedSMDistances, _contributionColumn)
        )
    ]
    return (
        fittedSMDistances,
        fittedSMHoverTexts,
        insideDistanceThreshold,
        insideMarkerRadius,
    )


@app.cell
def _(
    DistanceHeatmapSurface,
    MergedSpheres,
    SphereMesh,
    fittedMesh,
    fittedSM,
    fittedSMDistances,
    fittedSMHoverTexts,
    fittedSMSphereMeshOpacitySlider,
    fittedSMSurfaceOpacitySlider,
    heatmapDistanceLimit,
    insideDistanceThreshold,
    insideMarkerRadius,
    mesh_triangles,
    mesh_vertices,
    mo,
    np,
    numPhi,
    numTheta,
    renderPrimitives,
    showSpheres,
    showWireframe,
):
    fittedSMPrimitives = list(fittedSM.capsuloids) + list(fittedSM.prysmoids)
    if showSpheres or not fittedSMPrimitives:
        fittedSMPrimitives = list(fittedSM.spheres) + fittedSMPrimitives

    # Last, so plotly blends the surface over the sphere mesh it encloses.
    fittedSMPrimitives = fittedSMPrimitives + [
        DistanceHeatmapSurface(
            mesh_vertices,
            mesh_triangles,
            fittedSMDistances,
            fittedSMHoverTexts,
            "Signed Distance",
            fittedSMSurfaceOpacitySlider.value,
            heatmapDistanceLimit,
        )
    ]
    _cutoffCount = int((fittedSMDistances > heatmapDistanceLimit).sum())

    # A NaN d (no closest sphere) compares False, so it is never marked.
    _markedVertexIndices = np.flatnonzero(fittedSMDistances < insideDistanceThreshold)
    if len(_markedVertexIndices) > 0:
        _markerSpheres = MergedSpheres(
            [
                SphereMesh.Sphere(
                    vertexIndex, mesh_vertices[vertexIndex], insideMarkerRadius
                )
                for vertexIndex in _markedVertexIndices
            ],
            "Marked Vertex",
        )
        # Opaque, unlit white. The notebook's light is fixed in the scene, so a
        # lit white sphere turns grey when seen from the side it does not reach.
        _markerSpheres.drawWireframe = False
        _markerSpheres.traceOverrides = dict(
            color="#ffffff",
            opacity=1.0,
            lighting=dict(
                ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0
            ),
        )
        fittedSMPrimitives = fittedSMPrimitives + [_markerSpheres]

    fittedSMFigure = renderPrimitives(
        fittedSMPrimitives,
        None,
        numTheta,
        numPhi,
        fittedSMSphereMeshOpacitySlider.value,
        showWireframe,
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
                        f"vertices with d > {heatmapDistanceLimit} · "
                        f"white: {len(_markedVertexIndices)} vertices "
                        f"with d < {insideDistanceThreshold}"
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
    return


@app.cell
def _(dfVertices, fittedSMDistances, go, mo, np, pd):
    #some stats of fittedSMDistances vs the original SMPL distances

    # Signed distance of every surface vertex, in mm: to the original sphere
    # mesh (section 6) and to the fitted one (section 11).
    _distancesByName = {
        "before the fit": 1000 * dfVertices["d"].to_numpy(dtype=float),
        "after the fit": 1000 * fittedSMDistances,
    }
    # Line colour and its 10% wash, categorical slots 1 and 2.
    _colorsByName = {
        "before the fit": ("#2a78d6", "rgba(42, 120, 214, 0.1)"),
        "after the fit": ("#eb6834", "rgba(235, 104, 52, 0.1)"),
    }

    # The same bins for both, so their counts compare bin for bin.
    _allDistances = np.concatenate(list(_distancesByName.values()))
    _binEdges = np.linspace(np.nanmin(_allDistances), np.nanmax(_allDistances), 51)

    histogramFigure = go.Figure()
    for _name, _distances in _distancesByName.items():
        _counts, _ = np.histogram(_distances[~np.isnan(_distances)], bins=_binEdges)
        # An outline histogram: a step per bin, from 0 up at the first edge and
        # back down to 0 at the last.
        histogramFigure.add_trace(
            go.Scatter(
                x=np.concatenate([[_binEdges[0]], _binEdges]),
                y=np.concatenate([[0], _counts, [0]]),
                name=_name,
                mode="lines",
                line=dict(color=_colorsByName[_name][0], width=2, shape="hv"),
                fill="tozeroy",
                fillcolor=_colorsByName[_name][1],
                hovertemplate=(
                    f"{_name}<br>from %{{x:.1f}} mm: %{{y}} vertices<extra></extra>"
                ),
            )
        )

    # 0 is the sign break: negative is inside the sphere mesh.
    histogramFigure.add_vline(x=0, line_width=1, line_color="#c3c2b7")
    histogramFigure.update_layout(
        title=dict(
            text="Signed distance of the surface vertices, before and after the fit",
            font=dict(color="#0b0b0b", size=15),
        ),
        xaxis_title="signed distance [mm], negative inside the sphere mesh",
        yaxis_title="vertices",
        height=400,
        margin=dict(l=60, r=20, t=80, b=60),
        paper_bgcolor="#fcfcfb",
        plot_bgcolor="#fcfcfb",
        font=dict(color="#52514e"),
        legend=dict(orientation="h", x=0, y=1.1, font=dict(color="#0b0b0b")),
    )
    histogramFigure.update_xaxes(gridcolor="#e1e0d9", linecolor="#c3c2b7", zeroline=False)
    histogramFigure.update_yaxes(gridcolor="#e1e0d9", linecolor="#c3c2b7", zeroline=False)

    _statistics = pd.DataFrame(
        {
            _name: {
                "min": np.nanmin(_distances),
                "max": np.nanmax(_distances),
                "mean": np.nanmean(_distances),
                "std": np.nanstd(_distances),
                "sum": np.nansum(_distances),
            }
            for _name, _distances in _distancesByName.items()
        }
    ).round(2)
    mo.vstack([histogramFigure, mo.md("**Signed distance, in mm**"), _statistics])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 12 — the ernias problem

    il fitting produce alcune "ernie" ossia alcune sfere escono in maniera spropositata in alcuni punti del corpo. il prossimo passo del fitting è quello di inserire vincoli aggiuntivi ai vertici che sono dentro l'ernia ovvero che hanno signed distance $d < -\varepsilon$, con $\varepsilon = 1$ cm: sono i vertici bianchi della sezione 11, dentro la sphere mesh, dove le sfere escono dalla pelle.

    Ogni vertice in ernia aggiunge in fondo al sistema regolarizzato una riga di tangenza come quelle della sezione 8, scritta però con le contributions della sphere mesh **fittata**, cioè del primitivo che ora lo ingloba:

    $$
    \lambda\sum_{j} w_{ij}\left(\mathbf c_j\cdot\hat{\mathbf n}_i+r_j\right)=\lambda\,\mathbf p_i\cdot\hat{\mathbf n}_i
    $$

    con $\lambda$ = `herniaWeight`. Si risolve di nuovo, si rimisura $d$ e si ripete finché non resta nessuna ernia, o per al massimo `maxHerniaIterations` volte. Le righe si **accumulano**: un vertice che resta in ernia riceve un'altra riga, e il suo peso cresce. Tenere solo le righe dell'ultima iterazione fa oscillare il ciclo, perché le ernie già sistemate perdono la loro riga e tornano.
    """)
    return


@app.cell
def _(
    A_reg,
    M,
    SphereMesh,
    b_reg,
    fittedSM,
    lil_matrix,
    mesh_normals,
    mesh_vertices,
    np,
    spheremesh,
    vstack,
):
    #copiando lo stile di codice del primo fitting, svolgiamo questi passaggi:
    #1 misura d di ogni vertice rispetto alla sphere mesh corrente (la prima volta la fitted
    #  sphere mesh della sezione 9) e filtra i vertici con d < -0.01 (cioè quelli che sono
    #  all'interno della mesh sferica, dove le sfere escono dalla pelle)
    #2 trova sphere e contributions per ogni vertice filtrato rispetto alla sphere mesh corrente,
    #  non all'originale: il primitivo che ora ingloba il vertice può non essere quello della
    #  sua riga della sezione 8
    #3 come prima ogni nuovo vincolo è una riga come quelle della matrice A:
    #  la sommatoria su j di w_j * (c_j · n + r_j) = p · n, dove il peso w_j moltiplica anche
    #  il raggio. riga e termine noto sono moltiplicati entrambi per herniaWeight
    #4 questo aggiunge Y righe in fondo ad A_reg e Y valori in fondo a b_reg (non ad A e b:
    #  senza le righe della regolarizzazione le sfere non sono più tenute vicine all'originale),
    #  dove Y è il numero di vertici filtrati
    #5 il sistema è risolto con np.linalg.lstsq, la sphere mesh è ricostruita dalla soluzione e
    #  si riparte dal punto 1, finché non resta nessun vertice con d < -0.01 o al massimo
    #  maxHerniaIterations volte
    #6 le righe delle iterazioni precedenti restano nel sistema: se si tengono solo quelle
    #  dell'ultima, le ernie già sistemate perdono la loro riga, tornano, e il ciclo oscilla
    #il salvataggio nello stesso formato del file della mesh sferica originale è nella cella sotto

    # A vertex deeper than this inside the sphere mesh, in m, is a hernia: the
    # white markers of section 11.
    herniaThreshold = -0.01
    # Multiplies every hernia row and its b entry; a tangency row of A has 1.
    herniaWeight = 1.0
    maxHerniaIterations = 10

    # Prysmoids first, then capsuloids, as the text format lists them, so every
    # solution is rebuilt into a sphere mesh with the same topology.
    sphereMeshConnections = [
        (prysmoid.sA.index, prysmoid.sB.index, prysmoid.sC.index)
        for prysmoid in spheremesh.prysmoids
    ] + [
        (capsuloid.sA.index, capsuloid.sB.index)
        for capsuloid in spheremesh.capsuloids
    ]

    herniaSM = fittedSM
    _herniaRowBlocks = []
    _herniaRightHandSides = []
    # One entry per solve; iteration 0 is the fit of section 9.
    herniaHistory = []
    for _iteration in range(maxHerniaIterations + 1):
        # The d of section 11 for every vertex at once: signedDistances() takes
        # the same minimum over the primitives that closestSphere() does.
        _distances = herniaSM.signedDistances(mesh_vertices)
        _herniaVertexIndices = np.flatnonzero(_distances < herniaThreshold)
        herniaHistory.append(
            {
                "iteration": _iteration,
                "hernia rows in the solve": sum(
                    len(_rightHandSide) for _rightHandSide in _herniaRightHandSides
                ),
                "hernias": len(_herniaVertexIndices),
                "min d [mm]": 1000 * _distances.min(),
            }
        )
        if len(_herniaVertexIndices) == 0 or _iteration == maxHerniaIterations:
            break

        _herniaRows = lil_matrix((len(_herniaVertexIndices), 4 * M))
        _herniaRightHandSide = np.zeros(len(_herniaVertexIndices))
        for _rowIndex, _vertexIndex in enumerate(_herniaVertexIndices):
            _vertexPosition = mesh_vertices[_vertexIndex]
            _vertexNormal = mesh_normals[_vertexIndex]
            _herniaRightHandSide[_rowIndex] = herniaWeight * np.dot(
                _vertexPosition, _vertexNormal
            )

            _, _contributions = herniaSM.closestSphere(_vertexPosition)
            for _sphereIndex, _weight in _contributions:
                _rowWeight = herniaWeight * _weight
                _herniaRows[_rowIndex, 4 * _sphereIndex + 0] = _rowWeight * _vertexNormal[0]
                _herniaRows[_rowIndex, 4 * _sphereIndex + 1] = _rowWeight * _vertexNormal[1]
                _herniaRows[_rowIndex, 4 * _sphereIndex + 2] = _rowWeight * _vertexNormal[2]
                _herniaRows[_rowIndex, 4 * _sphereIndex + 3] = _rowWeight
        _herniaRowBlocks.append(_herniaRows.tocsr())
        _herniaRightHandSides.append(_herniaRightHandSide)

        _herniaA = vstack([A_reg] + _herniaRowBlocks).tocsr()
        _herniaB = np.concatenate([b_reg] + _herniaRightHandSides)
        _herniaSolution = np.linalg.lstsq(_herniaA.toarray(), _herniaB, rcond=None)[0]

        _sphereBlocks = _herniaSolution.reshape(M, 4)
        herniaSM = SphereMesh.fromSpheresAndConnections(
            _sphereBlocks[:, :3], _sphereBlocks[:, 3], sphereMeshConnections
        )
    return herniaHistory, herniaSM, herniaThreshold, sphereMeshConnections


@app.cell
def _(
    Path,
    SphereMesh,
    fittedMesh,
    herniaHistory,
    herniaSM,
    herniaThreshold,
    meshRotation,
    mo,
    pd,
    sphereMeshConnections,
    sphereMeshPath,
):
    # Next to the sphere mesh the fit started from, under a new name.
    herniaSavePath = Path(sphereMeshPath).with_name(f"fitted_{fittedMesh}_hernias.txt")

    # The text files hold a sphere mesh in its own frame, before section 4's
    # rotation, so the fit is turned back first: the inverse of a rotation is
    # its transpose. A copy, so herniaSM itself stays in the notebook's frame.
    _savedSM = SphereMesh.fromSpheresAndConnections(
        [sphere.center for sphere in herniaSM.spheres],
        [sphere.radius for sphere in herniaSM.spheres],
        sphereMeshConnections,
    )
    _savedSM.rotate(meshRotation.T)
    _savedSM.saveToFile(str(herniaSavePath))

    _herniasLeft = herniaHistory[-1]["hernias"]
    _outcome = (
        f"no vertex left with d < {herniaThreshold}"
        if _herniasLeft == 0
        else f"stopped with {_herniasLeft} vertices still at d < {herniaThreshold}"
    )
    mo.vstack(
        [
            mo.md(
                f"**Hernia loop**: {len(herniaHistory) - 1} solves, {_outcome}. "
                f"Saved to `{herniaSavePath}`."
            ),
            pd.DataFrame(herniaHistory).round(1),
        ]
    )
    return


@app.cell
def _(mo):
    # This figure's own sliders. Sharing section 11's made every change re-run
    # both figures, and every cell downstream of section 11's.
    herniaSMSphereMeshOpacitySlider = mo.ui.slider(
        0.0, 1.0, step=0.05, value=0.8, label="sphere mesh opacity", show_value=True
    )
    herniaSMSurfaceOpacitySlider = mo.ui.slider(
        0.0, 1.0, step=0.05, value=1.0, label="surface opacity", show_value=True
    )
    return herniaSMSphereMeshOpacitySlider, herniaSMSurfaceOpacitySlider


@app.cell
def _(herniaSM, mesh_vertices, np):
    # Signed distance of every surface vertex to the sphere mesh after the
    # hernia loop, the same way section 6 measures it against the original one.
    # Apart from the figure below, like section 11's, so its sliders re-run
    # only the render.
    _distances = []
    _contributionColumn = []
    for _vertex in mesh_vertices:
        _sphere, _contributions = herniaSM.closestSphere(_vertex)
        _distances.append(
            _sphere.signedDistance(_vertex) if _contributions else np.nan
        )
        _contributionColumn.append(_contributions)
    final_fittedSMDistances = np.asarray(_distances)

    final_fittedSMHoverTexts = [
        f"vertex {vertexIndex}<br>d {distance:.4f}"
        + "".join(
            f"<br>{sphereName} {weight:.3f}" for sphereName, weight in contributions
        )
        for vertexIndex, (distance, contributions) in enumerate(
            zip(final_fittedSMDistances, _contributionColumn)
        )
    ]
    return final_fittedSMDistances, final_fittedSMHoverTexts


@app.cell
def _(
    DistanceHeatmapSurface,
    MergedSpheres,
    SphereMesh,
    final_fittedSMDistances,
    final_fittedSMHoverTexts,
    fittedMesh,
    heatmapDistanceLimit,
    herniaSM,
    herniaSMSphereMeshOpacitySlider,
    herniaSMSurfaceOpacitySlider,
    insideDistanceThreshold,
    insideMarkerRadius,
    mesh_triangles,
    mesh_vertices,
    mo,
    np,
    numPhi,
    numTheta,
    renderPrimitives,
    showSpheres,
    showWireframe,
):
    # Section 11's heatmap, drawn for herniaSM, the sphere mesh the hernia loop
    # ends with, with the same scale and white markers.

    final_fittedSMPrimitives = list(herniaSM.capsuloids) + list(herniaSM.prysmoids)
    if showSpheres or not final_fittedSMPrimitives:
        final_fittedSMPrimitives = list(herniaSM.spheres) + final_fittedSMPrimitives

    # Last, so plotly blends the surface over the sphere mesh it encloses.
    final_fittedSMPrimitives = final_fittedSMPrimitives + [
        DistanceHeatmapSurface(
            mesh_vertices,
            mesh_triangles,
            final_fittedSMDistances,
            final_fittedSMHoverTexts,
            "Signed Distance",
            herniaSMSurfaceOpacitySlider.value,
            heatmapDistanceLimit,
        )
    ]
    _cutoffCount = int((final_fittedSMDistances > heatmapDistanceLimit).sum())

    # A NaN d (no closest sphere) compares False, so it is never marked.
    _markedVertexIndices = np.flatnonzero(final_fittedSMDistances < insideDistanceThreshold)
    if len(_markedVertexIndices) > 0:
        _markerSpheres = MergedSpheres(
            [
                SphereMesh.Sphere(
                    vertexIndex, mesh_vertices[vertexIndex], insideMarkerRadius
                )
                for vertexIndex in _markedVertexIndices
            ],
            "Marked Vertex",
        )
        # Opaque, unlit white. The notebook's light is fixed in the scene, so a
        # lit white sphere turns grey when seen from the side it does not reach.
        _markerSpheres.drawWireframe = False
        _markerSpheres.traceOverrides = dict(
            color="#ffffff",
            opacity=1.0,
            lighting=dict(
                ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0
            ),
        )
        final_fittedSMPrimitives = final_fittedSMPrimitives + [_markerSpheres]

    final_fittedSMFigure = renderPrimitives(
        final_fittedSMPrimitives,
        None,
        numTheta,
        numPhi,
        herniaSMSphereMeshOpacitySlider.value,
        showWireframe,
        showAxes=False,
        showLegend=False,
        backgroundColor="#1d252b",
        uiRevision=f"hernias {fittedMesh}",
        # No primitiveColor: the sphere mesh keeps its per-type colours, red
        # spheres, blue capsuloids and purple prysmoids.
        lambertian=True,
    )

    # Controls in one row on top, so the render below gets the full width.
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.md(f"**After the hernia loop** `{fittedMesh}`"),
                    herniaSMSphereMeshOpacitySlider,
                    herniaSMSurfaceOpacitySlider,
                    mo.md(
                        f"scale fixed at ±{heatmapDistanceLimit} · "
                        f"dark grey: {_cutoffCount} of {len(final_fittedSMDistances)} "
                        f"vertices with d > {heatmapDistanceLimit} · "
                        f"white: {len(_markedVertexIndices)} vertices "
                        f"with d < {insideDistanceThreshold}"
                    ),
                ],
                justify="start",
                align="center",
                gap=2,
                wrap=True,
            ),
            final_fittedSMFigure,
        ],
        gap=1,
    )
    return


@app.cell
def _(
    dfVertices,
    final_fittedSMDistances,
    fittedSMDistances,
    go,
    herniaHistory,
    mo,
    np,
    pd,
):
    # The histogram of section 11 with herniaSM added as a third series.

    # The solves the hernia loop took to reach herniaSM; iteration 0 is section 9's fit.
    _herniaIterationCount = herniaHistory[-1]["iteration"]
    _herniaSeriesName = (
        f"after {_herniaIterationCount} hernia "
        f"{'iteration' if _herniaIterationCount == 1 else 'iterations'}"
    )

    # Signed distance of every surface vertex, in mm: to the original sphere
    # mesh (section 6), to the fitted one (section 11) and to herniaSM
    # (section 12).
    _distancesByName = {
        "before the fit": 1000 * dfVertices["d"].to_numpy(dtype=float),
        "after the fit": 1000 * fittedSMDistances,
        _herniaSeriesName: 1000 * final_fittedSMDistances,
    }
    # Line colour and its 10% wash, categorical slots 1, 2 and 3.
    _colorsByName = {
        "before the fit": ("#2a78d6", "rgba(42, 120, 214, 0.1)"),
        "after the fit": ("#eb6834", "rgba(235, 104, 52, 0.1)"),
        _herniaSeriesName: ("#1baf7a", "rgba(27, 175, 122, 0.1)"),
    }

    # The same bins for all three, so their counts compare bin for bin.
    _allDistances = np.concatenate(list(_distancesByName.values()))
    _binEdges = np.linspace(np.nanmin(_allDistances), np.nanmax(_allDistances), 51)

    # Underscored: section 11's cell already defines histogramFigure, and marimo
    # refuses a name defined in two cells.
    _histogramFigure = go.Figure()
    for _name, _distances in _distancesByName.items():
        _counts, _ = np.histogram(_distances[~np.isnan(_distances)], bins=_binEdges)
        # An outline histogram: a step per bin, from 0 up at the first edge and
        # back down to 0 at the last.
        _histogramFigure.add_trace(
            go.Scatter(
                x=np.concatenate([[_binEdges[0]], _binEdges]),
                y=np.concatenate([[0], _counts, [0]]),
                name=_name,
                mode="lines",
                line=dict(color=_colorsByName[_name][0], width=2, shape="hv"),
                fill="tozeroy",
                fillcolor=_colorsByName[_name][1],
                hovertemplate=(
                    f"{_name}<br>from %{{x:.1f}} mm: %{{y}} vertices<extra></extra>"
                ),
            )
        )

    # 0 is the sign break: negative is inside the sphere mesh.
    _histogramFigure.add_vline(x=0, line_width=1, line_color="#c3c2b7")
    _histogramFigure.update_layout(
        title=dict(
            text=(
                "Signed distance of the surface vertices, before and after the "
                "fit, and after the hernia loop"
            ),
            font=dict(color="#0b0b0b", size=15),
        ),
        xaxis_title="signed distance [mm], negative inside the sphere mesh",
        yaxis_title="vertices",
        height=400,
        margin=dict(l=60, r=20, t=80, b=60),
        paper_bgcolor="#fcfcfb",
        plot_bgcolor="#fcfcfb",
        font=dict(color="#52514e"),
        legend=dict(orientation="h", x=0, y=1.1, font=dict(color="#0b0b0b")),
    )
    _histogramFigure.update_xaxes(gridcolor="#e1e0d9", linecolor="#c3c2b7", zeroline=False)
    _histogramFigure.update_yaxes(gridcolor="#e1e0d9", linecolor="#c3c2b7", zeroline=False)

    _statistics = pd.DataFrame(
        {
            _name: {
                "min": np.nanmin(_distances),
                "max": np.nanmax(_distances),
                "mean": np.nanmean(_distances),
                "std": np.nanstd(_distances),
                "sum": np.nansum(_distances),
            }
            for _name, _distances in _distancesByName.items()
        }
    ).round(2)
    mo.vstack([_histogramFigure, mo.md("**Signed distance, in mm**"), _statistics])
    return


if __name__ == "__main__":
    app.run()
