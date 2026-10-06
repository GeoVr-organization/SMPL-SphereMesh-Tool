"""plotly rendering for SphereMesh primitives, for the marimo notebooks.

`renderPrimitives` draws anything with a name and a buildGeometry() into one
interactive 3D figure: the SphereMesh primitives themselves, and the stand-ins
here (MergedSpheres, HighlightedSphere) and in ObjMesh.py (ObjSurface,
DistanceHeatmapSurface). SphereMeshView.py is the pygfx counterpart.
"""

import numpy as np
import plotly.graph_objects as go

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
