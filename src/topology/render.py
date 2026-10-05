"""Off-screen pictures of a topology, so a run can be looked at without opening the GUI.

The validator reports numbers; a person judging a candidate wants to *see* where the
spheres sit, which slabs exist and where the sphere-mesh misses the skin. Two kinds of
picture cover that:

* **structure views** -- the body as a translucent grey mesh with the collider drawn on
  top: spheres coloured per anatomical group (a mirrored pair such as ``L_arm`` /
  ``R_arm`` shares one hue, so left and right read as the same structure), cones drawn
  with ``src.spheremesh.Spheremesh`` and slabs with ``Trispheremesh`` -- the very
  geometry the tool renders, so what is shown is what the user will get; a quad is drawn
  as its two split triangles exactly as the tool would show it;
* **heat views** -- the opaque body coloured by the exact signed distance to the
  sphere-mesh (``src.topology.bumpers``), red where the skin protrudes outside the
  collider and blue where the collider leaves a gap, clipped to +/-20 mm so the scale
  stays readable from one topology to the next.

Views are orthographic: the front looks along ``-z`` (the body faces ``+z``), the back
along ``+z``, and the left along ``-x`` (``+x`` is the body's left side in this SMPL
model, so the left view shows the left arm nearest the camera). The parallel scale is
fitted to the collider and body bounds, never to a default, because a T-pose is wider
than the portrait window is and an auto-fit clips the hands.

Rendering needs ``pyvista`` with an off-screen backend, which this machine has.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .anatomy import Body, load_body
from .bumpers import BumperGraph
from .export import merge_quads

__all__ = ["render_topology", "group_colour", "load_topology_json"]

WINDOW_SIZE = (900, 1400)
BODY_COLOUR = (0.78, 0.78, 0.78)
DEFAULT_GROUP_COLOUR = (0.30, 0.55, 0.85)
HEAT_LIMIT_MM = 20.0

# One hue per anatomical stem; ``L_`` / ``R_`` are stripped before the lookup so mirrored
# groups share it. Anything unknown draws from ``_FALLBACK_COLOURS`` in a stable order.
_STEM_COLOURS: dict[str, tuple[float, float, float]] = {
    "trunk": (0.85, 0.45, 0.20),
    "torso": (0.85, 0.45, 0.20),
    "pelvis": (0.80, 0.35, 0.35),
    "neck": (0.60, 0.40, 0.75),
    "head": (0.55, 0.55, 0.25),
    "arm": (0.20, 0.60, 0.80),
    "upperarm": (0.20, 0.60, 0.80),
    "forearm": (0.25, 0.70, 0.65),
    "hand": (0.30, 0.45, 0.70),
    "leg": (0.35, 0.65, 0.30),
    "thigh": (0.35, 0.65, 0.30),
    "shin": (0.55, 0.75, 0.30),
    "foot": (0.45, 0.55, 0.35),
    "hip": (0.75, 0.55, 0.30),
    "glute": (0.75, 0.55, 0.30),
}
_FALLBACK_COLOURS: tuple[tuple[float, float, float], ...] = (
    (0.90, 0.60, 0.15),
    (0.15, 0.55, 0.60),
    (0.60, 0.30, 0.55),
    (0.40, 0.60, 0.20),
    (0.70, 0.35, 0.25),
    (0.25, 0.40, 0.70),
)


def _group_stem(group: str) -> str:
    return group[2:] if group[:2] in ("L_", "R_") else group


def group_colour(group: str | None, fallback_index: dict[str, int]) -> tuple[float, float, float]:
    """Colour of a group name; mirrored ``L_``/``R_`` groups share it."""
    if group is None:
        return DEFAULT_GROUP_COLOUR
    stem = _group_stem(str(group))
    if stem in _STEM_COLOURS:
        return _STEM_COLOURS[stem]
    if stem not in fallback_index:
        fallback_index[stem] = len(fallback_index)
    return _FALLBACK_COLOURS[fallback_index[stem] % len(_FALLBACK_COLOURS)]


def _body_polydata(body: Body):
    import pyvista as pv

    faces = np.asarray(body.faces, dtype=np.int64)
    cells = np.hstack([np.full((len(faces), 1), 3, dtype=np.int64), faces]).ravel()
    return pv.PolyData(np.asarray(body.vertices, dtype=np.float64), cells)


def _scene_bounds(body: Body, centers: np.ndarray, radii: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    low = np.minimum(body.vertices.min(axis=0), (centers - radii[:, None]).min(axis=0))
    high = np.maximum(body.vertices.max(axis=0), (centers + radii[:, None]).max(axis=0))
    return low, high


def _apply_view(plotter, view: str, low: np.ndarray, high: np.ndarray, padding: float = 0.06) -> None:
    """Orthographic camera for ``view`` that frames the whole scene in a portrait window."""
    focus = (low + high) / 2.0
    extent = high - low
    distance = float(np.linalg.norm(extent)) * 2.0 + 1.0
    if view == "front":
        position = focus + np.array([0.0, 0.0, distance])
        visible_width = extent[0]
    elif view == "back":
        position = focus - np.array([0.0, 0.0, distance])
        visible_width = extent[0]
    elif view == "left":
        position = focus + np.array([distance, 0.0, 0.0])
        visible_width = extent[2]
    elif view == "right":
        position = focus - np.array([distance, 0.0, 0.0])
        visible_width = extent[2]
    else:
        raise ValueError(f"unknown view {view!r}; expected front, back, left or right")

    plotter.camera_position = [tuple(position), tuple(focus), (0.0, 1.0, 0.0)]
    plotter.enable_parallel_projection()
    # ``parallel_scale`` is half the visible height; the width follows from the aspect, so
    # whichever of the two the body exceeds sets the scale.
    aspect = WINDOW_SIZE[0] / WINDOW_SIZE[1]
    half_height = max(extent[1] / 2.0, (visible_width / 2.0) / aspect)
    plotter.camera.parallel_scale = float(half_height) * (1.0 + padding)
    plotter.camera.clipping_range = (0.01, distance * 4.0)
    # ``screenshot`` reuses the last frame when only the camera changed; force the frame.
    plotter.render()


def _draw_collider(
    plotter,
    centers: np.ndarray,
    radii: np.ndarray,
    connections: Sequence[Sequence[int]],
    colours: list[tuple[float, float, float]],
    names: Sequence[str] | None,
) -> int:
    """Spheres, cones and slabs as the tool draws them. Returns the number skipped."""
    import pyvista as pv

    from src.spheremesh import Spheremesh, Trispheremesh

    def label(index: int) -> str:
        return f"{index}" if names is None else f"{index} ({names[index]})"

    for index, (center, radius) in enumerate(zip(centers, radii)):
        sphere = pv.Sphere(
            radius=float(radius), center=tuple(center),
            theta_resolution=32, phi_resolution=32,
        )
        plotter.add_mesh(sphere, color=colours[index], opacity=0.55, smooth_shading=True)

    skipped = 0
    for connection in connections:
        connection = tuple(int(index) for index in connection)
        colour = colours[connection[0]]
        triangles: list[tuple[int, int, int]]
        if len(connection) == 2:
            triangles = []
        elif len(connection) == 3:
            triangles = [connection]  # type: ignore[list-item]
        elif len(connection) == 4:
            first, second, third, fourth = connection
            triangles = [(first, second, third), (first, third, fourth)]
        else:
            print(f"[warn] connection {connection} has {len(connection)} spheres; skipped")
            skipped += 1
            continue

        try:
            if len(connection) == 2:
                first, second = connection
                primitive = Spheremesh(
                    center1=tuple(centers[first]), radius1=float(radii[first]),
                    center2=tuple(centers[second]), radius2=float(radii[second]),
                    resolution=32,
                )
                plotter.add_mesh(primitive, color=colour, opacity=0.45, smooth_shading=True)
            else:
                for first, second, third in triangles:
                    primitive = Trispheremesh(
                        center1=tuple(centers[first]), radius1=float(radii[first]),
                        center2=tuple(centers[second]), radius2=float(radii[second]),
                        center3=tuple(centers[third]), radius3=float(radii[third]),
                        resolution=32,
                    )
                    plotter.add_mesh(primitive, color=colour, opacity=0.45, smooth_shading=True)
        except ValueError as error:
            print(
                f"[warn] skipped connection {' '.join(label(index) for index in connection)}: "
                f"{error}"
            )
            skipped += 1
    return skipped


def render_topology(
    body: Body,
    centers: np.ndarray,
    radii: np.ndarray,
    connections: Iterable[Sequence[int]],
    output_dir,
    stem: str,
    *,
    names: Sequence[str] | None = None,
    groups: Sequence[str] | None = None,
    signed: np.ndarray | None = None,
    views: Sequence[str] = ("front", "back", "left"),
    heat: bool = True,
) -> list[Path]:
    """Write ``{stem}_{view}.png`` per view and, with ``heat``, the two heat maps.

    ``signed`` is the per-vertex signed distance to the sphere-mesh in metres (negative
    where the skin protrudes); when omitted it is computed exactly from the connection
    list with ``BumperGraph``. ``groups`` colours the spheres; ``names`` only labels the
    warnings for primitives that could not be built. Returns the paths written.
    """
    import pyvista as pv

    centers = np.atleast_2d(np.asarray(centers, dtype=np.float64))
    radii = np.asarray(radii, dtype=np.float64).reshape(-1)
    connections = [tuple(int(index) for index in connection) for connection in connections]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fallback_index: dict[str, int] = {}
    colours = [
        group_colour(None if groups is None else groups[index], fallback_index)
        for index in range(len(centers))
    ]
    body_mesh = _body_polydata(body)
    low, high = _scene_bounds(body, centers, radii)
    written: list[Path] = []

    plotter = pv.Plotter(off_screen=True, window_size=list(WINDOW_SIZE))
    plotter.set_background("white")
    plotter.add_mesh(body_mesh, color=BODY_COLOUR, opacity=0.35, smooth_shading=True)
    skipped = _draw_collider(plotter, centers, radii, connections, colours, names)
    if skipped:
        print(f"[warn] {stem}: {skipped} primitive(s) not drawn")
    for view in views:
        _apply_view(plotter, view, low, high)
        path = output_dir / f"{stem}_{view}.png"
        plotter.screenshot(str(path))
        written.append(path)
    plotter.close()

    if heat:
        if signed is None:
            graph = BumperGraph(len(centers), connections)
            signed = graph.distances(body.vertices, centers, radii).min(axis=1)
        signed_mm = np.asarray(signed, dtype=np.float64).reshape(-1) * 1000.0
        heat_mesh = body_mesh.copy()
        heat_mesh["signed distance (mm)"] = signed_mm

        plotter = pv.Plotter(off_screen=True, window_size=list(WINDOW_SIZE))
        plotter.set_background("white")
        plotter.add_mesh(
            heat_mesh,
            scalars="signed distance (mm)",
            cmap="RdBu",
            clim=(-HEAT_LIMIT_MM, HEAT_LIMIT_MM),
            smooth_shading=True,
            scalar_bar_args={
                "title": "signed distance (mm)   red: protrusion   blue: gap",
                "vertical": False,
                "position_x": 0.15,
                "position_y": 0.02,
                "width": 0.7,
                "height": 0.06,
                "color": "black",
                "title_font_size": 18,
                "label_font_size": 16,
            },
        )
        for view in ("front", "back"):
            _apply_view(plotter, view, low, high)
            path = output_dir / f"{stem}_heat_{view}.png"
            plotter.screenshot(str(path))
            written.append(path)
        plotter.close()

    return written


def load_topology_json(path) -> dict:
    """Read a tool JSON back into arrays, rebuilding quads from ``quad_groups``.

    Returns ``centers``, ``radii``, ``connections`` (quads whole again) and, when the
    file was written by ``scripts/build_topology.py``, ``names`` and ``groups``.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    centers = np.array([sphere["center"] for sphere in payload["spheres"]], dtype=np.float64)
    radii = np.array([sphere["radius"] for sphere in payload["spheres"]], dtype=np.float64)
    connections = [tuple(int(index) for index in entry["indices"]) for entry in payload["connections"]]
    if payload.get("quad_groups"):
        connections = merge_quads(connections, payload["quad_groups"])
    provenance = payload.get("generated_by") or {}
    return {
        "centers": centers,
        "radii": radii,
        "connections": connections,
        "names": provenance.get("sphere_names"),
        "groups": provenance.get("groups"),
        "gender": provenance.get("gender", "male"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a topology JSON off-screen.")
    parser.add_argument("json", type=Path, help="tool JSON (quad_groups honoured)")
    parser.add_argument("--stem", default=None, help="file stem (default: JSON stem)")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent.parent.parent / "data" / "generated" / "renders",
    )
    parser.add_argument("--gender", default=None, help="override the body gender")
    parser.add_argument("--no-heat", action="store_true")
    parser.add_argument("--views", nargs="+", default=["front", "back", "left"])
    args = parser.parse_args()

    topology = load_topology_json(args.json)
    body = load_body(args.gender or topology["gender"])
    written = render_topology(
        body, topology["centers"], topology["radii"], topology["connections"],
        args.output_dir, args.stem or args.json.stem,
        names=topology["names"], groups=topology["groups"],
        views=args.views, heat=not args.no_heat,
    )
    for path in written:
        print(f"{path}  {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
