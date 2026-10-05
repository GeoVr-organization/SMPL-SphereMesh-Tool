"""End-to-end check of Divide (split a cone or triangle with a new sphere) on a real MainWindow.

Covers cone and triangle divides through the button handler, the edge split of every
triangle on a divided cone (with duplicate cones), exact geometry (the pieces cover exactly
the old volume, measured with the exact distances of src/topology/bumpers.py), the new
sphere's joint binding, Inflate, beta offsets, clip planes, save and reload, and selecting
cones and triangles by clicking in the 3D view (plain click, Ctrl+click, drag, empty click).

Runs with ui_widgets.set_non_interactive(True) so no modal waits for a human.

Run with: python scripts/test_divide_connection.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch as tc
import vtk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.smpl_models import MALE, load_smpl_model  # noqa: E402
from src.topology.bumpers import prysmoid_distances, round_cone_distances  # noqa: E402
from src.ui_widgets import MainWindow, set_non_interactive  # noqa: E402

GEOMETRY_TOLERANCE = 1e-9
failures: list[str] = []


def check(condition: bool, label: str, detail: str = "") -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))
    if not condition:
        failures.append(label)


# ---------------------------------------------------------------------------
# Topology helpers
# ---------------------------------------------------------------------------

def connection_tuples(manager) -> list[tuple[int, ...]]:
    return [tuple(connection.sphere_indices) for connection in manager.connections]


def find_connection(manager, sphere_indices):
    for connection in manager.connections:
        if tuple(connection.sphere_indices) == tuple(sphere_indices):
            return connection
    return None


def snapshot_primitives(manager, connections):
    """(centers, radii) of each connection, frozen now."""
    return [
        (
            np.array([manager.spheres[i].get_center() for i in connection.sphere_indices]),
            np.array([manager.spheres[i].get_radius() for i in connection.sphere_indices]),
        )
        for connection in connections
    ]


def union_distance(points, primitives) -> np.ndarray:
    """Exact signed distance to the union of cones and triangular slabs."""
    best_distances = np.full(len(points), np.inf)
    for centers, radii in primitives:
        if len(radii) == 2:
            distances = round_cone_distances(
                points, centers[0][None], radii[0:1], centers[1][None], radii[1:2]
            )
        else:
            distances = prysmoid_distances(points, centers[None], radii[None])
        best_distances = np.minimum(best_distances, distances[:, 0])
    return best_distances


def sample_points(primitives, random_generator) -> np.ndarray:
    """Uniform points around the primitives plus points close to their surface."""
    all_centers = np.vstack([centers for centers, _ in primitives])
    largest_radius = max(float(radii.max()) for _, radii in primitives)
    lower_corner = all_centers.min(axis=0) - 1.5 * largest_radius
    upper_corner = all_centers.max(axis=0) + 1.5 * largest_radius
    point_sets = [random_generator.uniform(lower_corner, upper_corner, size=(4000, 3))]
    for centers, radii in primitives:
        weights = random_generator.dirichlet(np.ones(len(radii)), size=400)
        directions = random_generator.normal(size=(400, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        surface_scale = random_generator.uniform(0.95, 1.05, size=400)
        point_sets.append(weights @ centers + directions * ((weights @ radii) * surface_scale)[:, None])
    return np.vstack(point_sets)


def triangle_normal(manager, sphere_indices) -> np.ndarray:
    first, second, third = (manager.spheres[i].get_center() for i in sphere_indices)
    return np.cross(second - first, third - first)


def divide_and_check(window, connection, label: str, random_generator) -> int:
    """Divide through the button handler and check the topology and geometry it produced."""
    manager = window.sphere_manager
    endpoint_indices = tuple(connection.sphere_indices)
    endpoint_set = set(endpoint_indices)
    is_cone = len(endpoint_indices) == 2
    if is_cone:
        affected_connections = [c for c in manager.connections if endpoint_set.issubset(c.sphere_indices)]
    else:
        affected_connections = [c for c in manager.connections if set(c.sphere_indices) == endpoint_set]
    affected_triangle_tuples = [
        tuple(c.sphere_indices) for c in affected_connections if len(c.sphere_indices) == 3
    ]
    unaffected_tuples_before = [
        tuple(c.sphere_indices) for c in manager.connections if c not in affected_connections
    ]
    old_primitives = snapshot_primitives(manager, affected_connections)
    sphere_count_before = manager.get_sphere_count()
    expected_center = np.mean([manager.spheres[i].get_center() for i in endpoint_indices], axis=0)
    expected_base_radius = float(np.mean([manager._base_radius(i) for i in endpoint_indices]))

    check(manager.select_connection(connection), f"{label}: selectable")
    check(window.divide_connection_button.isEnabled(), f"{label}: Divide enabled while selected")
    window.divide_connection_clicked()

    new_index = sphere_count_before
    check(manager.get_sphere_count() == sphere_count_before + 1, f"{label}: one sphere added")
    new_sphere = manager.spheres[new_index]
    check(
        np.allclose(new_sphere.get_center(), expected_center, atol=1e-12, rtol=0.0),
        f"{label}: new center is the mean",
    )
    check(
        abs(manager._base_radius(new_index) - expected_base_radius) < 1e-12,
        f"{label}: new radius is the mean",
        f"{manager._base_radius(new_index):.6f} vs {expected_base_radius:.6f}",
    )
    check(
        manager.get_selected_sphere_index() == new_index and manager.get_selected_connection() is None,
        f"{label}: new sphere selected, no connection selected",
    )
    check(not window.divide_connection_button.isEnabled(), f"{label}: Divide disabled afterwards")

    tuples_after = connection_tuples(manager)
    new_tuples = [t for t in tuples_after if new_index in t]
    kept_tuples = [t for t in tuples_after if new_index not in t]
    check(kept_tuples == unaffected_tuples_before, f"{label}: other connections unchanged, same order")

    if is_cone:
        first_endpoint, second_endpoint = endpoint_indices
        check(
            not any(endpoint_set.issubset(t) for t in tuples_after),
            f"{label}: no connection still spans the divided edge",
        )
        new_pair_sets = sorted(sorted(t) for t in new_tuples if len(t) == 2)
        expected_pair_sets = sorted(
            [sorted((first_endpoint, new_index)), sorted((second_endpoint, new_index))]
        )
        check(new_pair_sets == expected_pair_sets, f"{label}: exactly (a, m) and (m, b)", str(new_pair_sets))
        distinct_triangle_sets = {frozenset(t) for t in affected_triangle_tuples}
        new_triangle_tuples = [t for t in new_tuples if len(t) == 3]
        check(
            len(new_triangle_tuples) == 2 * len(distinct_triangle_sets),
            f"{label}: each triangle on the edge split in two",
            f"{len(distinct_triangle_sets)} triangle(s) -> {len(new_triangle_tuples)}",
        )
        for new_triangle in new_triangle_tuples:
            other_corners = set(new_triangle) - {new_index}
            parent_triangle = next(t for t in affected_triangle_tuples if other_corners.issubset(t))
            replaced_corner = (set(parent_triangle) - other_corners).pop()
            expected_triangle = tuple(new_index if i == replaced_corner else i for i in parent_triangle)
            check(
                new_triangle == expected_triangle,
                f"{label}: {parent_triangle} -> {new_triangle} keeps the corner order",
            )
    else:
        first_corner, second_corner, third_corner = endpoint_indices
        expected_tuples = [
            (first_corner, second_corner, new_index),
            (second_corner, third_corner, new_index),
            (third_corner, first_corner, new_index),
        ]
        check(new_tuples == expected_tuples, f"{label}: three triangles in order", str(new_tuples))
        original_normal = triangle_normal(manager, endpoint_indices)
        check(
            all(float(np.dot(triangle_normal(manager, t), original_normal)) > 0.0 for t in new_tuples),
            f"{label}: new triangles face the same way",
        )

    new_connections = [c for c in manager.connections if new_index in c.sphere_indices]
    check(
        all(c.spheremesh is not None for c in new_connections),
        f"{label}: every new piece has its geometry",
    )
    new_primitives = snapshot_primitives(manager, new_connections)
    points = sample_points(old_primitives, random_generator)
    largest_difference = float(
        np.max(np.abs(union_distance(points, old_primitives) - union_distance(points, new_primitives)))
    )
    check(
        largest_difference < GEOMETRY_TOLERANCE,
        f"{label}: pieces cover exactly the old volume",
        f"max |d_old - d_new| = {largest_difference:.2e} over {len(points)} points",
    )
    return new_index


# ---------------------------------------------------------------------------
# Picking helpers
# ---------------------------------------------------------------------------

def display_position(window, world_point) -> tuple[int, int]:
    renderer = window.plotter.renderer
    renderer.SetWorldPoint(float(world_point[0]), float(world_point[1]), float(world_point[2]), 1.0)
    renderer.WorldToDisplay()
    display_x, display_y, _ = renderer.GetDisplayPoint()
    return int(round(display_x)), int(round(display_y))


def click(window, app, position, control_key=False, release_position=None) -> None:
    """Inject a left press (and, after an optional drag, a release) into the interactor."""
    render_window_interactor = window.plotter.iren.interactor
    control_flag = 1 if control_key else 0
    render_window_interactor.SetEventInformation(position[0], position[1], control_flag, 0)
    render_window_interactor.LeftButtonPressEvent()
    if release_position is not None:
        steps = 8
        for step in range(1, steps + 1):
            moved_x = position[0] + (release_position[0] - position[0]) * step // steps
            moved_y = position[1] + (release_position[1] - position[1]) * step // steps
            render_window_interactor.SetEventInformation(moved_x, moved_y, control_flag, 0)
            render_window_interactor.MouseMoveEvent()
        position = release_position
    render_window_interactor.SetEventInformation(position[0], position[1], control_flag, 0)
    render_window_interactor.LeftButtonReleaseEvent()
    app.processEvents()


def picking_topology_payload() -> dict:
    """A clear cone, a clear triangle and a cone buried inside its two big spheres."""
    sphere_rows = [
        ([-0.60, 0.00, 0.0], 0.03),   # 0  clear cone (0, 1)
        ([-0.20, 0.00, 0.0], 0.03),   # 1
        ([0.20, -0.20, 0.0], 0.03),   # 2  clear triangle (2, 3, 4)
        ([0.60, -0.20, 0.0], 0.03),   # 3
        ([0.40, 0.20, 0.0], 0.03),    # 4
        ([-0.15, 0.50, 0.0], 0.20),   # 5  buried cone (5, 6): midpoint inside both spheres
        ([0.15, 0.50, 0.0], 0.20),    # 6
    ]
    return {
        "version": 2,
        "spheres": [{"center": center, "radius": radius} for center, radius in sphere_rows],
        "connections": [{"indices": [0, 1]}, {"indices": [2, 3, 4]}, {"indices": [5, 6]}],
    }


# ---------------------------------------------------------------------------
# Test sections
# ---------------------------------------------------------------------------

def pick_test_connections(manager):
    """Choose a shared-joint cone, a mixed-joint cone, a triangle and a shared triangle edge
    that do not touch each other's connections."""
    def parent_of(sphere_index):
        return manager.spheres[sphere_index].get_parent_joint()

    cones = [tuple(c.sphere_indices) for c in manager.connections if len(c.sphere_indices) == 2]
    triangles = [tuple(c.sphere_indices) for c in manager.connections if len(c.sphere_indices) == 3]
    shared_joint_cone = next(t for t in cones if parent_of(t[0]) == parent_of(t[1]))
    mixed_joint_cone = next(t for t in cones if parent_of(t[0]) != parent_of(t[1]))
    used_edges = {frozenset(shared_joint_cone), frozenset(mixed_joint_cone)}

    edge_to_triangles: dict[frozenset, list] = {}
    for triangle in triangles:
        for first_position in range(3):
            edge = frozenset((triangle[first_position], triangle[(first_position + 1) % 3]))
            edge_to_triangles.setdefault(edge, []).append(triangle)
    shared_edge = next(
        edge for edge, owners in edge_to_triangles.items()
        if len(owners) == 2 and edge not in used_edges
    )
    shared_edge_triangles = set(edge_to_triangles[shared_edge])
    lone_triangle = next(
        t for t in triangles
        if t not in shared_edge_triangles
        and not any(edge.issubset(t) for edge in used_edges | {shared_edge})
    )
    return shared_joint_cone, mixed_joint_cone, lone_triangle, tuple(sorted(shared_edge))


def main() -> int:
    set_non_interactive(True)
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow(load_smpl_model(MALE), MALE)
    window.resize(1400, 900)
    window.show()
    app.processEvents()
    manager = window.sphere_manager
    random_generator = np.random.default_rng(7)
    # Put every sphere at its FK position so "stays where it is" checks are meaningful.
    manager.update_sphere_positions()
    print(f"Loaded {manager.get_sphere_count()} spheres, {len(manager.connections)} connections")

    shared_joint_cone, mixed_joint_cone, lone_triangle, shared_edge = pick_test_connections(manager)
    print(
        f"Cases: shared-joint cone {shared_joint_cone}, mixed-joint cone {mixed_joint_cone}, "
        f"triangle {lone_triangle}, shared triangle edge {shared_edge}"
    )

    print("\nNothing selected")
    manager.clear_connection_selection()
    sphere_count = manager.get_sphere_count()
    window.divide_connection_clicked()
    check(manager.get_sphere_count() == sphere_count, "Divide without a selection does nothing")
    check(not window.divide_connection_button.isEnabled(), "Divide disabled without a selection")

    print("\nCone whose endpoints share a joint")
    endpoint_offsets = [manager.spheres[i].get_rest_offset() for i in shared_joint_cone]
    shared_parent = manager.spheres[shared_joint_cone[0]].get_parent_joint()
    shared_cone_sphere = divide_and_check(
        window, find_connection(manager, shared_joint_cone), "shared-joint cone", random_generator
    )
    new_sphere = manager.spheres[shared_cone_sphere]
    check(new_sphere.get_parent_joint() == shared_parent, "binds to the shared joint")
    check(
        np.allclose(new_sphere.get_rest_offset(), np.mean(endpoint_offsets, axis=0), atol=1e-12),
        "rest offset is the mean of the endpoints' offsets",
    )

    print("\nCone whose endpoints are on different joints")
    endpoint_parents = {manager.spheres[i].get_parent_joint() for i in mixed_joint_cone}
    mixed_cone_sphere = divide_and_check(
        window, find_connection(manager, mixed_joint_cone), "mixed-joint cone", random_generator
    )
    check(
        manager.spheres[mixed_cone_sphere].get_parent_joint() in endpoint_parents,
        "binds to one of the endpoints' joints",
        f"{manager.spheres[mixed_cone_sphere].get_parent_joint()} in {sorted(endpoint_parents)}",
    )

    print("\nTriangle")
    divide_and_check(window, find_connection(manager, lone_triangle), "triangle", random_generator)

    print("\nCone on an edge shared by two triangles, added twice (duplicates)")
    check(manager.try_add_connection(shared_edge), "cone on the shared edge added")
    check(manager.try_add_connection(shared_edge[::-1]), "duplicate cone (reversed) added")
    divide_and_check(window, find_connection(manager, shared_edge), "edge split", random_generator)

    print("\nPose following")
    centers_before = np.array([s.get_center() for s in manager.spheres])
    manager.update_sphere_positions()
    centers_after = np.array([s.get_center() for s in manager.spheres])
    check(
        float(np.abs(centers_after - centers_before).max()) < 1e-9,
        "FK at rest pose moves no sphere (new ones included)",
        f"max {float(np.abs(centers_after - centers_before).max()):.2e} m",
    )
    theta = tc.zeros(72, dtype=tc.float64)
    theta[3 * 3 + 0] = 0.4    # spine1
    theta[16 * 3 + 2] = -0.8  # left shoulder
    theta[18 * 3 + 1] = -1.0  # left elbow
    theta[1 * 3 + 0] = -0.6   # left hip
    window._on_theta_changed(theta)
    shared_midpoint = np.mean([manager.spheres[i].get_center() for i in shared_joint_cone], axis=0)
    check(
        np.allclose(manager.spheres[shared_cone_sphere].get_center(), shared_midpoint, atol=1e-9),
        "posed: shared-joint sphere stays at the endpoints' midpoint",
    )
    window._on_theta_changed(tc.zeros(72, dtype=tc.float64))

    print("\nInflate (scale 1.5)")
    manager.set_uniform_radius_override(True, 1.5, "scale")
    inflate_cone = next(
        tuple(c.sphere_indices) for c in manager.connections if len(c.sphere_indices) == 2
    )
    inflate_base_radius = float(np.mean([manager._base_radius(i) for i in inflate_cone]))
    inflate_sphere = divide_and_check(
        window, find_connection(manager, inflate_cone), "cone under Inflate", random_generator
    )
    check(
        abs(manager.spheres[inflate_sphere].get_radius() - 1.5 * inflate_base_radius) < 1e-12,
        "displayed radius is 1.5 x the mean base radius",
    )
    manager.set_uniform_radius_override(False)
    check(
        abs(manager.spheres[inflate_sphere].get_radius() - inflate_base_radius) < 1e-12,
        "base radius restored when Inflate is off",
    )

    print("\nBeta offsets")
    with tempfile.TemporaryDirectory() as offsets_directory:
        sphere_count = manager.get_sphere_count()
        # Small enough that no radius reaches the clamp floor in apply_beta_offsets.
        offset_rows = random_generator.normal(scale=0.004, size=(sphere_count, 4))
        with open(Path(offsets_directory) / "beta0.csv", "w", newline="") as offsets_file:
            offsets_file.write("sphere_id,dx,dy,dz,dr\n")
            for sphere_id, (dx, dy, dz, dr) in enumerate(offset_rows):
                offsets_file.write(f"{sphere_id},{dx},{dy},{dz},{dr}\n")
        manager.load_beta_offsets_from_dir(offsets_directory)
    beta_cone = next(tuple(c.sphere_indices) for c in manager.connections if len(c.sphere_indices) == 2)
    beta_sphere = divide_and_check(window, find_connection(manager, beta_cone), "cone with beta offsets", random_generator)
    check(
        manager._beta_template_pos.shape[0] == manager.get_sphere_count()
        and all(rows.shape[0] == manager.get_sphere_count() for rows in manager.offsets_pos.values())
        and all(rows.shape[0] == manager.get_sphere_count() for rows in manager.offsets_r.values()),
        "beta arrays grew with the sphere list",
    )
    manager.apply_beta_offsets(np.array([1.0]))
    beta_midpoint = np.mean([manager.spheres[i].get_center() for i in beta_cone], axis=0)
    beta_mean_radius = float(np.mean([manager.spheres[i].get_radius() for i in beta_cone]))
    check(
        np.allclose(manager.spheres[beta_sphere].get_center(), beta_midpoint, atol=1e-12)
        and abs(manager.spheres[beta_sphere].get_radius() - beta_mean_radius) < 1e-12,
        "with beta applied the new sphere stays at the mean center and radius",
    )
    manager.apply_beta_offsets(np.array([0.0]))
    manager.offsets_pos, manager.offsets_r = {}, {}
    manager._beta_template_pos = manager._beta_template_r = None

    print("\nClip planes")
    window._add_clip_planes_axis("x")
    clip_cone = next(tuple(c.sphere_indices) for c in manager.connections if len(c.sphere_indices) == 2)
    clip_sphere = divide_and_check(window, find_connection(manager, clip_cone), "cone with clip planes", random_generator)
    clipped_connectors = [
        c._connector_actor.GetMapper().GetClippingPlanes()
        for c in manager.connections
        if clip_sphere in c.sphere_indices and c._connector_actor is not None
    ]
    check(
        bool(clipped_connectors)
        and all(planes is not None and planes.GetNumberOfItems() > 0 for planes in clipped_connectors),
        "new connectors are clipped",
    )
    window._reset_clip_planes()

    print("\nSave and reload")
    with tempfile.TemporaryDirectory() as save_directory:
        save_path = Path(save_directory) / "divided.json"
        tuples_saved = connection_tuples(manager)
        centers_saved = np.array([s.get_center() for s in manager.spheres])
        parents_saved = [s.get_parent_joint() for s in manager.spheres]
        manager.save_to_json(str(save_path))
        manager.load_from_json(str(save_path))
    check(connection_tuples(manager) == tuples_saved, "connections survive save and reload")
    centers_loaded = np.array([s.get_center() for s in manager.spheres])
    check(
        centers_loaded.shape == centers_saved.shape
        and float(np.abs(centers_loaded - centers_saved).max()) <= 5e-4 + 1e-12,
        "centers survive (load rounds to 3 decimals)",
    )
    check([s.get_parent_joint() for s in manager.spheres] == parents_saved, "parent joints survive")

    print("\nClick picking")
    with tempfile.TemporaryDirectory() as picking_directory:
        picking_path = Path(picking_directory) / "picking.json"
        picking_path.write_text(json.dumps(picking_topology_payload()))
        manager.load_from_json(str(picking_path))
    window.resize(1400, 900)  # the clip-plane panel calls adjustSize()
    app.processEvents()
    window.plotter.view_xy()
    window.plotter.reset_camera(bounds=(-0.8, 0.8, -0.45, 0.8, -0.3, 0.3))
    window.plotter.render()
    app.processEvents()
    clear_cone = find_connection(manager, (0, 1))
    clear_triangle = find_connection(manager, (2, 3, 4))
    buried_cone = find_connection(manager, (5, 6))

    info_panel = window.info_panel
    click(window, app, display_position(window, [-0.40, 0.0, 0.0]))
    check(manager.get_selected_connection() is clear_cone, "click on a cone selects it")
    check(manager.get_selected_sphere_index() is None, "selecting a cone deselects the sphere")
    check(window.divide_connection_button.isEnabled(), "Divide enabled")
    connector_color = clear_cone._connector_actor.GetProperty().GetColor()
    check(np.allclose(connector_color, (0.0, 1.0, 1.0)), "selected connector turns cyan", str(connector_color))
    check(info_panel.spheremesh_id_label.text() == "Cone (0, 1)", "InfoPanel names the cone",
          info_panel.spheremesh_id_label.text())
    check(
        [info_panel.sphere1_box.title(), info_panel.sphere2_box.title()] == ["Sphere #0", "Sphere #1"]
        and info_panel.sphere2_box.isVisibleTo(info_panel)
        and not info_panel.sphere3_box.isVisibleTo(info_panel),
        "InfoPanel shows one box per sphere of the cone",
    )
    check(
        abs(info_panel.sphere2_radius.value() - manager.spheres[1].get_radius()) < 1e-4
        and abs(info_panel.sphere2_center_x.value() - manager.spheres[1].get_center()[0]) < 1e-3,
        "second box holds sphere 1's center and radius",
    )
    check(not info_panel.delete_button.isEnabled(), "Delete (spheres only) disabled for a cone")
    info_panel.sphere2_radius.setValue(0.045)
    check(abs(manager.spheres[1].get_radius() - 0.045) < 1e-9, "editing the second box edits sphere 1")
    check(manager.get_selected_connection() is clear_cone, "the cone stays selected while editing")
    info_panel.sphere2_radius.setValue(0.03)

    click(window, app, display_position(window, [0.4, -0.2 / 3.0, 0.0]))
    check(manager.get_selected_connection() is clear_triangle, "click on a triangle selects it")
    check(not np.allclose(clear_cone._connector_actor.GetProperty().GetColor(), (0.0, 1.0, 1.0)),
          "previous connection loses its highlight")
    check(info_panel.spheremesh_id_label.text() == "Triangle (2, 3, 4)", "InfoPanel names the triangle",
          info_panel.spheremesh_id_label.text())
    check(
        [box.title() for box in (info_panel.sphere1_box, info_panel.sphere2_box, info_panel.sphere3_box)]
        == ["Sphere #2", "Sphere #3", "Sphere #4"]
        and info_panel.sphere3_box.isVisibleTo(info_panel),
        "InfoPanel shows one box per sphere of the triangle",
    )

    camera_position_before = window.plotter.camera_position
    start = display_position(window, [-0.40, 0.0, 0.0])
    click(window, app, start, release_position=(start[0] + 40, start[1] + 10))
    check(window.plotter.camera_position != camera_position_before, "a drag still rotates the camera")
    check(manager.get_selected_connection() is clear_triangle, "a drag keeps the selection")
    window.plotter.view_xy()
    window.plotter.reset_camera(bounds=(-0.8, 0.8, -0.45, 0.8, -0.3, 0.3))
    window.plotter.render()

    click(window, app, display_position(window, [-0.60, 0.0, 0.0]))
    check(manager.get_selected_sphere_index() == 0, "click on a sphere selects the sphere")
    check(manager.get_selected_connection() is None, "selecting a sphere clears the connection")
    check(
        info_panel.spheremesh_id_label.text() == "Sphere #0"
        and info_panel.sphere1_box.title() == "Selected sphere"
        and not info_panel.sphere2_box.isVisibleTo(info_panel)
        and info_panel.connections_text.isVisibleTo(info_panel),
        "InfoPanel back to the single-sphere view",
    )

    buried_midpoint = display_position(window, [0.0, 0.5, 0.0])
    click(window, app, buried_midpoint)
    check(
        manager.get_selected_connection() is None and manager.get_selected_sphere_index() in (5, 6),
        "plain click on a buried cone hits a sphere",
    )
    renderers_before = [s.widget.GetCurrentRenderer() for s in manager.spheres]
    click(window, app, buried_midpoint, control_key=True)
    check(manager.get_selected_connection() is buried_cone, "Ctrl+click reaches the buried cone")
    window.move_selected_to_closest_joint_clicked()
    check(manager.get_selected_connection() is buried_cone, "'Selected -> closest joint' ignores a cone")
    check(
        [s.widget.GetCurrentRenderer() for s in manager.spheres] == renderers_before,
        "sphere widgets restored after Ctrl+click",
    )

    click(window, app, display_position(window, [0.0, -0.40, 0.0]))
    check(manager.get_selected_connection() is None, "click on empty space clears the connection")
    check(info_panel.spheremesh_id_label.text() == "Spheremesh: None", "InfoPanel reset")
    check(not window.divide_connection_button.isEnabled(), "Divide disabled again")

    click(window, app, display_position(window, [-0.40, 0.0, 0.0]))
    window.divide_connection_clicked()
    check(
        connection_tuples(manager)[:1] == [(0, 7)] and (7, 1) in connection_tuples(manager),
        "picked cone divided end to end",
        str(connection_tuples(manager)),
    )

    window.close()
    app.quit()

    print()
    if failures:
        print(f"{len(failures)} FAILURE(S): " + ", ".join(failures))
        return 1
    print("Divide OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
