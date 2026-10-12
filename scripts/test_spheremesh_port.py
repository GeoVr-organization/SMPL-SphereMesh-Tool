"""End-to-end check of the SphereMesh port on a real MainWindow.

The GUI draws and measures connections with src/sphere_mesh.py (the class SMFitting uses).
Checks that every connection's actor holds exactly SphereMesh's buildGeometry() surface after
load, pose changes, a simulated drag, the Inflate slider and deleting a sphere; that the
distances equal the exact closed forms of src/topology/bumpers.py; that degenerate
connections neither crash nor flood the console; that actors created after clip planes were
set are clipped; and that the heatmap has SMFitting's style and stays current.

Runs with ui_widgets.set_non_interactive(True) so no modal waits for a human.

Run with: uv run python scripts/test_spheremesh_port.py
"""
from __future__ import annotations

import contextlib
import io
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pyvista as pv
import torch as tc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.managed_sphere import SPHERE_WIDGET_PHI_RESOLUTION, SPHERE_WIDGET_THETA_RESOLUTION  # noqa: E402
from src.smpl_models import MALE, load_smpl_model  # noqa: E402
from src.sphere_connection import SPHEREMESH_NUM_PHI, SPHEREMESH_NUM_THETA  # noqa: E402
from src.sphere_mesh import SphereMesh  # noqa: E402
from src.topology.bumpers import prysmoid_distances, round_cone_distances  # noqa: E402
from src.ui_widgets import (  # noqa: E402
    HEATMAP_AMBIENT,
    HEATMAP_CUTOFF_TEXT_NAME,
    HEATMAP_DIFFUSE,
    HEATMAP_DISTANCE_LIMIT_MM,
    HEATMAP_SCALARS_NAME,
    MainWindow,
    set_non_interactive,
)

# Metres. The drawn points are SphereMesh's own float64 vertices, so they match exactly.
GEOMETRY_TOLERANCE = 1e-12
# Metres. The exact closed forms and SphereMesh round differently near a cone's axis.
DISTANCE_TOLERANCE = 1e-8
failures: list[str] = []


def check(condition: bool, label: str, detail: str = "") -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))
    if not condition:
        failures.append(label)


def fresh_primitive(manager, connection):
    """A SphereMesh primitive over the connection's spheres as they are now, built here."""
    spheres = [
        SphereMesh.Sphere(sphere_index, manager.spheres[sphere_index].get_center(),
                          manager.spheres[sphere_index].get_radius())
        for sphere_index in connection.sphere_indices
    ]
    if len(spheres) == 2:
        return SphereMesh.Capsuloid(*spheres)
    return SphereMesh.Prysmoid(*spheres, 1.0)


def geometry_gap(manager) -> tuple[float, int, int]:
    """(max |drawn - buildGeometry| over every built connection, topology mismatches, built count)."""
    worst = 0.0
    topology_mismatches = 0
    built = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for connection in manager.connections:
            if connection.spheremesh is None:
                continue
            built += 1
            vertices, triangles = fresh_primitive(manager, connection).buildGeometry(
                SPHEREMESH_NUM_THETA, SPHEREMESH_NUM_PHI
            )
            drawn = connection.spheremesh
            drawn_triangles = drawn.faces.reshape(-1, 4)[:, 1:]
            if drawn.n_points != len(vertices) or not np.array_equal(drawn_triangles, triangles):
                topology_mismatches += 1
                continue
            worst = max(worst, float(np.max(np.abs(np.asarray(drawn.points) - vertices))))
    return worst, topology_mismatches, built


def check_geometry(manager, label: str) -> None:
    worst, topology_mismatches, built = geometry_gap(manager)
    check(
        built == len(manager.connections) and topology_mismatches == 0 and worst <= GEOMETRY_TOLERANCE,
        f"{label}: every actor is SphereMesh's buildGeometry surface",
        f"{built}/{len(manager.connections)} built, max |diff| {worst:.1e} m, "
        f"{topology_mismatches} topology mismatches",
    )


def exact_union_distances(manager, points) -> np.ndarray:
    """bumpers.py's exact union over every connection, plus spheres in no connection."""
    distances = np.full(len(points), np.inf)
    in_connection = set()
    for connection in manager.connections:
        indices = list(connection.sphere_indices)
        in_connection.update(indices)
        centers = np.array([manager.spheres[i].get_center() for i in indices])
        radii = np.array([manager.spheres[i].get_radius() for i in indices])
        if len(indices) == 2:
            primitive_distances = round_cone_distances(points, centers[:1], radii[:1], centers[1:], radii[1:])
        else:
            primitive_distances = prysmoid_distances(points, centers[None], radii[None])
        distances = np.minimum(distances, primitive_distances[:, 0])
    for sphere_index, sphere in enumerate(manager.spheres):
        if sphere_index not in in_connection:
            distances = np.minimum(distances, np.linalg.norm(points - sphere.get_center(), axis=1) - sphere.get_radius())
    return distances


def body_mesh(window):
    return pv.wrap(window.mesh_actor.GetMapper().GetInput())


def heatmap_gap_mm(window) -> float:
    """max |drawn heatmap - exact union distance| in mm."""
    mesh = body_mesh(window)
    drawn_mm = np.asarray(mesh.point_data[HEATMAP_SCALARS_NAME])
    exact_mm = 1000.0 * exact_union_distances(window.sphere_manager, np.asarray(mesh.points, dtype=np.float64))
    return float(np.max(np.abs(drawn_mm - exact_mm)))


def simulate_drag(manager, sphere_index: int, offset) -> None:
    """What a sphere widget's end-of-interaction callback does: widget moved, manager told."""
    sphere = manager.spheres[sphere_index]
    new_center = sphere.get_center() + np.asarray(offset, dtype=np.float64)
    sphere.widget.SetCenter(tuple(new_center))
    manager._on_sphere_interaction(sphere_index, new_center)


def connector_holds_centers(manager, connection) -> bool:
    """The thin connector's bounds contain its spheres' centres (5 mm tube radius of slack)."""
    actor = connection._connector_actor
    if actor is None:
        return True
    bounds = np.array(actor.GetMapper().GetInput().GetBounds()).reshape(3, 2)
    return all(
        np.all(manager.spheres[i].get_center() >= bounds[:, 0] - 0.006)
        and np.all(manager.spheres[i].get_center() <= bounds[:, 1] + 0.006)
        for i in connection.sphere_indices
    )


def is_clipped(actor) -> bool:
    planes = actor.GetMapper().GetClippingPlanes()
    return planes is not None and planes.GetNumberOfItems() > 0


def main() -> int:
    set_non_interactive(True)
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow(load_smpl_model(MALE), MALE)
    window.resize(1400, 900)
    window.show()
    app.processEvents()
    manager = window.sphere_manager
    random_generator = np.random.default_rng(13)
    manager.update_sphere_positions()
    print(f"Loaded {manager.get_sphere_count()} spheres, {len(manager.connections)} connections")

    print("\nClip planes reach actors created after them")
    check(all(not c._actors for c in manager.connections), "spheremesh actors are created lazily",
          f"{sum(bool(c._actors) for c in manager.connections)} built before Show Spheremeshes")
    window._add_clip_planes_axis("x")
    window.checkbox_show_spheremeshes.setChecked(True)
    app.processEvents()
    lazy_actors = [actor for c in manager.connections for actor in c._actors]
    check(bool(lazy_actors) and all(is_clipped(actor) for actor in lazy_actors),
          "Show Spheremeshes: the new actors are clipped", f"{len(lazy_actors)} actors")
    cone = next(c for c in manager.connections if len(c.sphere_indices) == 2)
    check(manager.try_add_connection(tuple(cone.sphere_indices)[::-1]), "Add Connection with clip planes on")
    added = manager.connections[-1]
    check(is_clipped(added._connector_actor) and all(is_clipped(actor) for actor in added._actors),
          "Add Connection: connector and spheremesh clipped")
    manager.delete_connection(added)
    window._reset_clip_planes()
    window.resize(1400, 900)  # the clip-plane panel calls adjustSize()
    app.processEvents()

    print("\nThe look")
    check_geometry(manager, "after Show Spheremeshes")
    spheremesh_actors = [actor for c in manager.connections for actor in c._actors]
    check(all(actor.GetProperty().GetInterpolation() == 0 for actor in spheremesh_actors),
          "spheremeshes are flat shaded", "VTK_FLAT = 0")
    widgets = [sphere.widget for sphere in manager.spheres]
    check(
        all(widget.GetThetaResolution() == SPHERE_WIDGET_THETA_RESOLUTION
            and widget.GetPhiResolution() == SPHERE_WIDGET_PHI_RESOLUTION
            and widget.GetSphereProperty().GetInterpolation() == 0 for widget in widgets),
        "sphere widgets are 31 x 16 and flat, like Sphere.buildGeometry(16, 32)",
    )
    # The drawn 2-row cones are the same surface as SMFitting's default 16 rows: the
    # 16-row mesh shares the end rows and its inner rows lie on the straight generators.
    worst_end_rows = 0.0
    worst_inner_rows = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for connection in manager.connections:
            primitive = fresh_primitive(manager, connection)
            capsuloids = [primitive] if len(connection.sphere_indices) == 2 else list(primitive.edgeCapsuloids)
            for capsuloid in capsuloids:
                coarse, _ = capsuloid.buildGeometry(SPHEREMESH_NUM_THETA, SPHEREMESH_NUM_PHI)
                fine, _ = capsuloid.buildGeometry(16, SPHEREMESH_NUM_PHI)
                coarse = coarse.reshape(SPHEREMESH_NUM_PHI, 2, 3)
                fine = fine.reshape(SPHEREMESH_NUM_PHI, 16, 3)
                worst_end_rows = max(worst_end_rows, float(np.max(np.abs(fine[:, [0, -1]] - coarse))))
                generator = coarse[:, 1] - coarse[:, 0]
                generator_length = np.linalg.norm(generator, axis=1, keepdims=True)
                unit_generator = np.divide(generator, generator_length, out=np.zeros_like(generator),
                                           where=generator_length > 0)
                to_inner = fine[:, 1:-1] - coarse[:, None, 0]
                along = np.einsum("pri,pi->pr", to_inner, unit_generator)
                off_line = np.linalg.norm(to_inner - along[..., None] * unit_generator[:, None], axis=2)
                worst_inner_rows = max(worst_inner_rows, float(off_line.max()))
    check(worst_end_rows < 1e-12 and worst_inner_rows < 1e-12,
          "2-row cones are the same surface as 16-row ones",
          f"end rows max |diff| {worst_end_rows:.1e} m, inner rows off the generators {worst_inner_rows:.1e} m")

    print("\nDistances")
    points = random_generator.uniform(-0.6, 0.6, size=(3000, 3)) + np.array([0.0, 0.2, 0.0])
    worst_connection = 0.0
    for connection in manager.connections:
        indices = list(connection.sphere_indices)
        centers = np.array([manager.spheres[i].get_center() for i in indices])
        radii = np.array([manager.spheres[i].get_radius() for i in indices])
        if len(indices) == 2:
            exact = round_cone_distances(points, centers[:1], radii[:1], centers[1:], radii[1:])[:, 0]
        else:
            exact = prysmoid_distances(points, centers[None], radii[None])[:, 0]
        worst_connection = max(worst_connection, float(np.max(np.abs(connection.get_distances(points) - exact))))
    scalar_gap = max(
        abs(connection.get_distance(point) - connection.get_distances(point[None])[0])
        for connection in manager.connections[:10] for point in points[:50]
    )
    check(worst_connection < DISTANCE_TOLERANCE and scalar_gap < DISTANCE_TOLERANCE,
          "connection distances equal bumpers' exact distances",
          f"max |diff| {worst_connection:.1e} m; get_distance vs get_distances {scalar_gap:.1e} m")

    print("\nPose and shape changes")
    theta = tc.zeros(72, dtype=tc.float64)
    theta[3 * 3 + 0] = 0.4    # spine1
    theta[16 * 3 + 2] = -0.8  # left shoulder
    theta[18 * 3 + 1] = -1.0  # left elbow
    theta[1 * 3 + 0] = -0.6   # left hip
    tick_seconds = []
    for _ in range(3):
        start = time.perf_counter()
        window._on_theta_changed(theta)
        window._on_theta_changed(tc.zeros(72, dtype=tc.float64))
        tick_seconds.append((time.perf_counter() - start) / 2)
    window._on_theta_changed(theta)
    check_geometry(manager, "posed")
    print(f"      one theta tick with spheremeshes shown: {1000 * min(tick_seconds):.0f} ms "
          f"(FK, body, {len(manager.connections)} connections, render)")
    window._on_theta_changed(tc.zeros(72, dtype=tc.float64))

    print("\nDrag, Inflate slider")
    dragged_index = next(c for c in manager.connections if len(c.sphere_indices) == 3).sphere_indices[0]
    simulate_drag(manager, dragged_index, (0.0, 0.03, 0.02))
    check_geometry(manager, "after a drag")
    window.checkbox_inflate.setChecked(True)
    window.inflate_slider.setValue(150)
    app.processEvents()
    check_geometry(manager, "Inflate slider at 1.50")
    window.inflate_slider.setValue(120)
    app.processEvents()
    check_geometry(manager, "Inflate slider moved to 1.20 while on")
    window.checkbox_inflate.setChecked(False)
    app.processEvents()
    check_geometry(manager, "Inflate off")

    print("\nDeleting a sphere re-indexes against the new list")
    connected = sorted({i for c in manager.connections for i in c.sphere_indices})
    deleted_index = connected[len(connected) // 3]
    survivors_expected = sum(1 for c in manager.connections if deleted_index not in c.sphere_indices)
    manager.selected_connection = None
    manager.selected_sphere_index = deleted_index
    window.info_panel._on_delete_clicked()
    app.processEvents()
    check(len(manager.connections) == survivors_expected, "connections through the sphere dropped",
          f"{len(manager.connections)} left")
    check_geometry(manager, "after delete")
    check(all(connector_holds_centers(manager, c) for c in manager.connections),
          "every connector runs between its own spheres")
    window.checkbox_show_sphere_widgets.setChecked(False)
    app.processEvents()
    hidden_delete_index = sorted({i for c in manager.connections for i in c.sphere_indices})[2]
    manager.selected_connection = None
    manager.selected_sphere_index = hidden_delete_index
    window.info_panel._on_delete_clicked()
    app.processEvents()
    check(all(not c._connector_actor.GetVisibility() for c in manager.connections if c._connector_actor),
          "with sphere widgets hidden, re-indexed connectors stay hidden")
    window.checkbox_show_sphere_widgets.setChecked(True)
    app.processEvents()

    print("\nDegenerate connections")
    first_center = manager.spheres[0].get_center()
    coincident_index = manager.add_sphere(first_center, 0.03)
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        refused = not manager.try_add_connection((0, coincident_index))
    check(refused, "a cone between coincident centres is refused")
    contained_index = manager.add_sphere(first_center + np.array([0.004, 0.0, 0.0]), 0.002)
    collinear_index = manager.add_sphere(first_center + np.array([0.3, 0.0, 0.0]), 0.02)
    far_index = manager.add_sphere(first_center + np.array([0.6, 0.0, 0.0]), 0.03)
    stderr = io.StringIO()
    with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stderr(stderr):
        # Python's own default: each distinct warning is shown once, not on every rebuild.
        warnings.simplefilter("default")
        contained_added = manager.try_add_connection((0, contained_index))
        collinear_added = manager.try_add_connection((0, collinear_index, far_index))
        for _ in range(5):
            for connection in manager.connections[-2:]:
                connection.update_mesh()
    finite = all(np.all(np.isfinite(np.asarray(c.spheremesh.points))) for c in manager.connections[-2:])
    check(contained_added and collinear_added and finite,
          "a contained cone and a collinear triangle are built, all finite")
    # Only SphereMesh's own warnings; VTK's numpy_support adds a NumPy deprecation of its own.
    degenerate_texts = [str(warning.message) for warning in caught
                        if str(warning.message).startswith("Degenerate")]
    check(stderr.getvalue() == "" and len(degenerate_texts) == 2
          and any(f"capsuloid(s0,s{contained_index})" in text for text in degenerate_texts)
          and any(f"prysmoid(s0,s{collinear_index},s{far_index})" in text for text in degenerate_texts),
          "one warning per degenerate primitive over all rebuilds, nothing printed",
          " | ".join(degenerate_texts))
    check_geometry(manager, "with degenerate connections")
    collinear_connection = manager.connections[-1]
    cells_without_faces = collinear_connection.spheremesh.n_cells
    simulate_drag(manager, collinear_index, (0.0, 0.1, 0.0))
    check(collinear_connection.spheremesh.n_cells == cells_without_faces + 2,
          "lifting a sphere off the line gives the triangle its two faces, in place",
          f"{cells_without_faces} -> {collinear_connection.spheremesh.n_cells} cells")
    check_geometry(manager, "after the triangle gained its faces")

    print("\nHeatmap")
    start = time.perf_counter()
    window.checkbox_show_heatmap.setChecked(True)
    app.processEvents()
    heatmap_seconds = time.perf_counter() - start
    check(heatmap_gap_mm(window) < 1e-5, "heatmap is the exact signed union distance",
          f"max |diff| {heatmap_gap_mm(window):.1e} mm, toggled on in {1000 * heatmap_seconds:.0f} ms")
    body_property = window.mesh_actor.GetProperty()
    check(abs(body_property.GetAmbient() - HEATMAP_AMBIENT) < 1e-12
          and abs(body_property.GetDiffuse() - HEATMAP_DIFFUSE) < 1e-12 and body_property.GetSpecular() == 0.0,
          "nearly flat heatmap lighting")
    mapper = window.mesh_actor.GetMapper()
    lookup_table = mapper.GetLookupTable()
    check(mapper.GetScalarVisibility() == 1 and mapper.GetInterpolateScalarsBeforeMapping() == 1
          and lookup_table.GetNumberOfTableValues() == 8
          and tuple(lookup_table.GetRange()) == (-HEATMAP_DISTANCE_LIMIT_MM, HEATMAP_DISTANCE_LIMIT_MM),
          "8 flat 5 mm bands over +-20 mm, band edges kept sharp")
    check("d [mm]" in window.plotter.scalar_bars, "legend titled d [mm]")
    distances_mm = np.asarray(body_mesh(window).point_data[HEATMAP_SCALARS_NAME])
    cutoff_text = window.plotter.actors[HEATMAP_CUTOFF_TEXT_NAME].GetInput()
    expected_count = int(np.count_nonzero(distances_mm > HEATMAP_DISTANCE_LIMIT_MM))
    check(f"dark grey: {expected_count} of {len(distances_mm)}" in cutoff_text, "grey-vertex count shown",
          cutoff_text)

    calls = {"count": 0}
    original_get_heatmap = manager.get_heatmap

    def counting_get_heatmap(mesh):
        calls["count"] += 1
        return original_get_heatmap(mesh)

    manager.get_heatmap = counting_get_heatmap
    manager.select_next_sphere()
    manager.update_sphere_information_callback()
    check(calls["count"] == 0, "a selection change does not recompute it")
    simulate_drag(manager, dragged_index, (0.0, -0.02, 0.0))
    check(calls["count"] == 1 and heatmap_gap_mm(window) < 1e-5, "a drag recomputes it",
          f"{calls['count']} recomputations, max |diff| {heatmap_gap_mm(window):.1e} mm")
    window.add_sphere_clicked()
    check(calls["count"] == 2 and heatmap_gap_mm(window) < 1e-5,
          "Add Sphere recomputes it (a sphere in no connection counts)",
          f"max |diff| {heatmap_gap_mm(window):.1e} mm")
    start = time.perf_counter()
    window._on_theta_changed(theta)
    heatmap_tick = time.perf_counter() - start
    check(calls["count"] == 3 and heatmap_gap_mm(window) < 1e-5, "a pose change recomputes it",
          f"max |diff| {heatmap_gap_mm(window):.1e} mm; theta tick with heatmap {1000 * heatmap_tick:.0f} ms")
    window._on_theta_changed(tc.zeros(72, dtype=tc.float64))
    manager.get_heatmap = original_get_heatmap

    window.checkbox_show_heatmap.setChecked(False)
    app.processEvents()
    check(mapper.GetScalarVisibility() == 0 and HEATMAP_CUTOFF_TEXT_NAME not in window.plotter.actors
          and "d [mm]" not in window.plotter.scalar_bars, "turning it off removes colours, legend and count")
    check(abs(body_property.GetAmbient() - HEATMAP_AMBIENT) > 1e-6 or HEATMAP_AMBIENT == 0.0,
          "body lighting restored", f"ambient {body_property.GetAmbient():.2f}")

    window.close()
    app.quit()

    print()
    if failures:
        print(f"{len(failures)} FAILURE(S): " + ", ".join(failures))
        return 1
    print("SphereMesh port OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
