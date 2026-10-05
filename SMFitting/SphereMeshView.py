"""pygfx rendering for SphereMesh primitives.

`renderPrimitives` returns a jupyter RenderCanvas, which is an anywidget and so
embeds directly in a marimo notebook with mouse orbit/zoom/pan. `renderToImage`
renders the identical scene offscreen and hands back a numpy RGBA image, for
tests and for any context without a live widget.
"""

from __future__ import annotations

import contextlib
from typing import Any, Generator, Iterable, Sequence

import numpy as np
import pygfx as gfx

# Colour per primitive type, so it is obvious which mesh came from where.
primitiveColors: dict[str, str] = {
    "Sphere Class": "#ee380a",
    "Capsuloid Class": "#2f76b9",
    "Prysmoid Class": "#9545d6",
}

defaultColor = "#7f7f7f"


@contextlib.contextmanager
def suppressJupyterRfbOutputWidget() -> Generator[None]:
    """Stop RemoteFrameBuffer.__init__ from display()-ing its output context.

    jupyter_rfb/widget.py builds an ipywidgets.Output and calls display() on it,
    to surface prints and exceptions raised during a draw. marimo patches
    IPython's display() to append to the cell output, then renders whatever
    arrives through its anywidget path, which reads the widget's `_esm`. A plain
    ipywidget has none, so marimo sends an empty js-url and the canvas slot
    shows:

        [anywidget] No ESM spec for model <id>: nothing ever provided this
        widget's code

    Nothing else calls display() during construction, and the output context
    still collects draw errors -- see readDrawErrors().
    """
    import jupyter_rfb.widget as jupyterRfbWidget

    originalDisplay = jupyterRfbWidget.display
    jupyterRfbWidget.display = lambda *args, **kwargs: None
    try:
        yield
    finally:
        jupyterRfbWidget.display = originalDisplay


def readDrawErrors(canvas: Any) -> list[dict[str, Any]]:
    """Whatever the canvas' undisplayed output context captured during draws."""
    outputContext = getattr(canvas, "_output_context", None)
    return list(getattr(outputContext, "outputs", ()))


def buildPrimitiveGroup(primitives: Iterable[Any], numTheta: int, numPhi: int,
                        opacity: float, showWireframe: bool) -> gfx.Group:
    """One gfx.Mesh per primitive, from its buildGeometry() output."""
    group = gfx.Group()

    for primitive in primitives:
        vertices, triangles = primitive.buildGeometry(numTheta, numPhi)

        geometry = gfx.Geometry(
            positions=np.ascontiguousarray(vertices, dtype=np.float32),
            indices=np.ascontiguousarray(triangles, dtype=np.uint32),
        )

        # side="both" because these surfaces are open (the capsuloid cone has no
        # caps, the prysmoid is two loose faces), so backfaces must still draw.
        material = gfx.MeshPhongMaterial(
            color=primitiveColors.get(primitive.name, defaultColor),
            flat_shading=True,
            side="both",
        )
        material.opacity = float(opacity)
        group.add(gfx.Mesh(geometry, material))

        if showWireframe:
            wireframeMaterial = gfx.MeshBasicMaterial(
                color="#101010",
                wireframe=True,
                wireframe_thickness=1.0,
                side="both",
            )
            group.add(gfx.Mesh(geometry, wireframeMaterial))

    return group


def buildScene(primitives: Iterable[Any], numTheta: int, numPhi: int,
               opacity: float, showWireframe: bool,
               showAxes: bool) -> tuple[gfx.Scene, gfx.Group]:
    """The full scene: background, lights, the primitives and optional axes."""
    scene = gfx.Scene()
    scene.add(gfx.Background(None, gfx.BackgroundMaterial("#e8edf2", "#fbfcfd")))

    group = buildPrimitiveGroup(primitives, numTheta, numPhi, opacity,
                                showWireframe)
    scene.add(group)

    if showAxes:
        boundingBox = group.get_world_bounding_box()
        if boundingBox is not None:
            # A third of the diagonal: long enough to read the orientation at
            # the world origin, short enough not to swamp the primitives.
            axesSize = 0.33 * float(np.linalg.norm(boundingBox[1] - boundingBox[0]))
            if axesSize > 0.0:
                scene.add(gfx.AxesHelper(size=axesSize))

    scene.add(gfx.AmbientLight(intensity=0.7))

    return scene, group


def buildCamera(group: gfx.Group,
                viewDirection: Sequence[float] = (-1.6, -1.6, -0.9),
                up: Sequence[float] = (0.0, 0.0, 1.0)) -> gfx.PerspectiveCamera:
    """A perspective camera framing the group, with a headlight attached.

    The default up axis is +z, matching how these meshes were viewed before.
    Pass up=(0, 1, 0) for the body-scale templates, which are y-up.
    """
    camera = gfx.PerspectiveCamera(45)
    # scale > 1 pulls back a little so nothing is clipped by the viewport edge.
    camera.show_object(group, view_dir=tuple(viewDirection), up=tuple(up),
                       scale=1.25)

    # Parenting the light to the camera keeps whatever you orbit towards lit.
    headlight = gfx.DirectionalLight(intensity=2.6)
    headlight.local.position = (0.0, 0.0, 1.0)
    camera.add(headlight)

    return camera


def renderPrimitives(primitives: Iterable[Any], numTheta: int = 16,
                     numPhi: int = 32, opacity: float = 1.0,
                     showWireframe: bool = False, showAxes: bool = True,
                     size: tuple[int, int] = (900, 600),
                     up: Sequence[float] = (0.0, 0.0, 1.0)) -> Any:
    """Render into a jupyter RenderCanvas (an anywidget) and return it.

    The returned canvas keeps the renderer, scene, camera and controller alive
    on itself, so the caller only has to hold on to the canvas.
    """
    from rendercanvas.jupyter import RenderCanvas

    primitives = list(primitives)

    with suppressJupyterRfbOutputWidget():
        canvas = RenderCanvas(size=size)
    renderer = gfx.renderers.WgpuRenderer(canvas)

    scene, group = buildScene(primitives, numTheta, numPhi, opacity,
                              showWireframe, showAxes)
    camera = buildCamera(group, up=up)
    scene.add(camera)

    controller = gfx.OrbitController(camera, register_events=renderer)

    canvas.request_draw(lambda: renderer.render(scene, camera))

    # Keep every piece referenced, or the scene is collected out from under the
    # widget once this function returns.
    canvas._sphereMeshView = (renderer, scene, group, camera, controller)

    return canvas


def renderToImage(primitives: Iterable[Any], numTheta: int = 16,
                  numPhi: int = 32, opacity: float = 1.0,
                  showWireframe: bool = False, showAxes: bool = True,
                  size: tuple[int, int] = (900, 600),
                  up: Sequence[float] = (0.0, 0.0, 1.0)) -> np.ndarray:
    """Render the same scene offscreen and return an RGBA uint8 image."""
    from rendercanvas.offscreen import RenderCanvas as OffscreenRenderCanvas

    primitives = list(primitives)

    canvas = OffscreenRenderCanvas(size=size, pixel_ratio=1)
    renderer = gfx.renderers.WgpuRenderer(canvas)

    scene, group = buildScene(primitives, numTheta, numPhi, opacity,
                              showWireframe, showAxes)
    camera = buildCamera(group, up=up)
    scene.add(camera)

    canvas.request_draw(lambda: renderer.render(scene, camera))

    return np.asarray(canvas.draw())
