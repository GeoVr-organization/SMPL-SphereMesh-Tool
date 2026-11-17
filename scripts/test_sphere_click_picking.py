"""End-to-end check that a click on stacked spheres takes the frontmost one, on a real MainWindow.

Every vtkSphereWidget picks only its own sphere, and the widgets run in the order they were
added, so a click used to go to the lowest-index sphere anywhere along the ray, even one
hidden behind another. The press observer now leaves only the frontmost sphere's widget
listening, for left clicks and drags (move) and right drags (scale).

Covers a grid of plain clicks over front/back pairs in both index orders and a partial
overlap, checked pixel by pixel against an analytic ray-sphere test; a left drag and a right
drag on a stacked pair; and that every widget listens again after each kind of click.

Clicks are simulated (VTK event injection), not physical input.

Run with: python scripts/test_sphere_click_picking.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.smpl_models import MALE, load_smpl_model  # noqa: E402
from src.ui_widgets import MainWindow, set_non_interactive  # noqa: E402
from test_divide_connection import click, display_position  # noqa: E402

# The widget draws a 31 x 16 polygon inscribed in the sphere, so its silhouette lies between
# about 0.976 r and r. Pixels whose analytic answer changes inside that band are skipped.
TESSELLATION_INNER_RADIUS_FACTOR = 0.97
GRID_STEP_PIXELS = 7
failures: list[str] = []


def check(condition: bool, label: str, detail: str = "") -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))
    if not condition:
        failures.append(label)


# The view looks down -Z, so a larger z is nearer the camera.
SPHERE_ROWS = [
    ([0.00, 0.00, -0.20], 0.12),   # 0  back, lower index than its front sphere
    ([0.00, 0.00, 0.20], 0.07),    # 1  front
    ([0.40, 0.00, 0.20], 0.07),    # 2  front, lower index than its back sphere
    ([0.40, 0.00, -0.20], 0.12),   # 3  back
    ([-0.40, 0.00, -0.10], 0.10),  # 4  back, partly covered
    ([-0.33, 0.03, 0.10], 0.07),   # 5  front, partly covering 4
]


def payload() -> dict:
    """The spheres above, no connections."""
    return {
        "version": 2,
        "spheres": [{"center": center, "radius": radius} for center, radius in SPHERE_ROWS],
        "connections": [],
    }


def reset_view(window, app) -> None:
    """Head-on XY view; resetting the camera also fixes the clipping range picks rely on."""
    window.resize(1400, 900)
    app.processEvents()
    window.plotter.view_xy()
    window.plotter.reset_camera(bounds=(-0.7, 0.7, -0.5, 0.5, -0.4, 0.4))
    window.plotter.render()
    app.processEvents()


def camera_ray(window, display_x, display_y) -> tuple[np.ndarray, np.ndarray]:
    """World-space origin and unit direction of the ray through a display pixel."""
    renderer = window.plotter.renderer
    ray_points = []
    for display_depth in (0.0, 1.0):
        renderer.SetDisplayPoint(float(display_x), float(display_y), display_depth)
        renderer.DisplayToWorld()
        world_x, world_y, world_z, world_w = renderer.GetWorldPoint()
        ray_points.append(np.array([world_x, world_y, world_z]) / world_w)
    direction = ray_points[1] - ray_points[0]
    return ray_points[0], direction / np.linalg.norm(direction)


def analytic_frontmost_sphere(spheres, ray_origin, ray_direction, radius_factor) -> "int | None":
    """Index of the sphere the ray enters first, with every radius scaled by radius_factor."""
    best_sphere_index, best_distance = None, np.inf
    for sphere_index, sphere in enumerate(spheres):
        to_center = sphere.get_center() - ray_origin
        along_ray = float(to_center @ ray_direction)
        squared_miss = float(to_center @ to_center) - along_ray * along_ray
        squared_radius = (sphere.get_radius() * radius_factor) ** 2
        if squared_miss > squared_radius:
            continue
        entry_distance = along_ray - np.sqrt(squared_radius - squared_miss)
        if 0.0 < entry_distance < best_distance:
            best_sphere_index, best_distance = sphere_index, entry_distance
    return best_sphere_index


class InteractionRecorder:
    """Wraps SphereManager._on_sphere_interaction to record which sphere each click reached."""

    def __init__(self, manager):
        self.sphere_indices: list[int] = []
        original_handler = manager._on_sphere_interaction

        def _recording_handler(sphere_index, center):
            self.sphere_indices.append(sphere_index)
            original_handler(sphere_index, center)

        manager._on_sphere_interaction = _recording_handler

    def last_and_clear(self) -> "int | None":
        """The sphere reached since the last call, or None; then forget it."""
        last_sphere_index = self.sphere_indices[-1] if self.sphere_indices else None
        self.sphere_indices = []
        return last_sphere_index


def all_widgets_listen(manager) -> bool:
    """Whether every sphere widget has its renderer back (no pause left behind)."""
    return all(sphere.widget.GetCurrentRenderer() is not None for sphere in manager.spheres)


def right_drag(window, app, position, release_position) -> None:
    """Inject a right press, a few moves and a right release into the interactor."""
    render_window_interactor = window.plotter.iren.interactor
    render_window_interactor.SetEventInformation(position[0], position[1], 0, 0)
    render_window_interactor.RightButtonPressEvent()
    steps = 8
    for step in range(1, steps + 1):
        moved_x = position[0] + (release_position[0] - position[0]) * step // steps
        moved_y = position[1] + (release_position[1] - position[1]) * step // steps
        render_window_interactor.SetEventInformation(moved_x, moved_y, 0, 0)
        render_window_interactor.MouseMoveEvent()
    render_window_interactor.SetEventInformation(release_position[0], release_position[1], 0, 0)
    render_window_interactor.RightButtonReleaseEvent()
    app.processEvents()


def main() -> int:
    set_non_interactive(True)
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow(load_smpl_model(MALE), MALE)
    window.show()
    app.processEvents()
    manager = window.sphere_manager

    with tempfile.TemporaryDirectory() as payload_directory:
        payload_path = Path(payload_directory) / "stacked_spheres.json"
        payload_path.write_text(json.dumps(payload()))
        manager.load_from_json(str(payload_path))
    reset_view(window, app)
    recorder = InteractionRecorder(manager)
    spheres = manager.spheres

    print("\nGrid of plain clicks against the analytic frontmost sphere")
    corner_low_x, corner_low_y = display_position(window, [-0.58, -0.17, 0.0])
    corner_high_x, corner_high_y = display_position(window, [0.58, 0.17, 0.0])
    clicked_pixels = 0
    skipped_pixels = 0
    mismatches: list[tuple[int, int, "int | None", "int | None"]] = []
    mismatches_on_stacked_pixels = 0
    stacked_pixels = 0
    for display_y in range(corner_low_y, corner_high_y + 1, GRID_STEP_PIXELS):
        for display_x in range(corner_low_x, corner_high_x + 1, GRID_STEP_PIXELS):
            ray_origin, ray_direction = camera_ray(window, display_x, display_y)
            expected_outer = analytic_frontmost_sphere(spheres, ray_origin, ray_direction, 1.0)
            expected_inner = analytic_frontmost_sphere(
                spheres, ray_origin, ray_direction, TESSELLATION_INNER_RADIUS_FACTOR
            )
            if expected_outer != expected_inner:
                skipped_pixels += 1
                continue
            spheres_on_ray = sum(
                analytic_frontmost_sphere([sphere], ray_origin, ray_direction, 1.0) is not None
                for sphere in spheres
            )
            recorder.last_and_clear()
            click(window, app, (display_x, display_y))
            reached_sphere_index = recorder.last_and_clear()
            clicked_pixels += 1
            if spheres_on_ray >= 2:
                stacked_pixels += 1
            if reached_sphere_index != expected_outer:
                mismatches.append((display_x, display_y, expected_outer, reached_sphere_index))
                if spheres_on_ray >= 2:
                    mismatches_on_stacked_pixels += 1
    check(clicked_pixels > 500 and stacked_pixels > 100, "setup: the grid covers stacked and single spheres",
          f"{clicked_pixels} pixels clicked, {stacked_pixels} with 2 spheres on the ray, "
          f"{skipped_pixels} silhouette pixels skipped")
    check(not mismatches, "every click reaches the analytic frontmost sphere (or none on empty pixels)",
          f"{len(mismatches)} of {clicked_pixels} wrong, {mismatches_on_stacked_pixels} of them on stacked pixels"
          + (f"; first: pixel ({mismatches[0][0]}, {mismatches[0][1]}) expected {mismatches[0][2]}, "
             f"got {mismatches[0][3]}" if mismatches else ""))
    wrong_pairs = sorted({(expected, reached) for _, _, expected, reached in mismatches}, key=str)
    if wrong_pairs:
        print(f"    (expected, reached) pairs among the misses: {wrong_pairs}")
    check(all_widgets_listen(manager), "every sphere widget listens again after the grid")

    print("\nLeft drag on a stacked pair")
    centers_before = [sphere.get_center() for sphere in spheres]
    front_center_pixel = display_position(window, SPHERE_ROWS[1][0])
    click(window, app, front_center_pixel, release_position=(front_center_pixel[0] + 40, front_center_pixel[1]))
    reached_sphere_index = recorder.last_and_clear()
    moved_by = [float(np.linalg.norm(sphere.get_center() - center_before))
                for sphere, center_before in zip(spheres, centers_before)]
    check(reached_sphere_index == 1 and manager.get_selected_sphere_index() == 1,
          "dragging over sphere 1 (front) reaches sphere 1, not 0 (behind, lower index)",
          f"reached {reached_sphere_index}, selected {manager.get_selected_sphere_index()}")
    check(moved_by[1] > 0.01 and moved_by[0] == 0.0, "the front sphere moves, the back one stays",
          f"sphere 1 moved {moved_by[1] * 1000:.1f} mm, sphere 0 moved {moved_by[0] * 1000:.1f} mm")
    check(all_widgets_listen(manager), "every sphere widget listens again after the drag")

    print("\nRight drag (scale) on a stacked pair")
    radii_before = [sphere.get_radius() for sphere in spheres]
    front_center_pixel = display_position(window, spheres[1].get_center())
    right_drag(window, app, front_center_pixel, (front_center_pixel[0], front_center_pixel[1] + 40))
    reached_sphere_index = recorder.last_and_clear()
    radius_change = [sphere.get_radius() - radius_before for sphere, radius_before in zip(spheres, radii_before)]
    check(reached_sphere_index == 1, "right-dragging over sphere 1 reaches sphere 1", f"reached {reached_sphere_index}")
    check(abs(radius_change[1]) > 1e-4 and radius_change[0] == 0.0, "the front sphere scales, the back one does not",
          f"sphere 1 radius {radius_change[1] * 1000:+.2f} mm, sphere 0 radius {radius_change[0] * 1000:+.2f} mm")
    check(all_widgets_listen(manager), "every sphere widget listens again after the right drag")

    print("\nCamera drags on empty space")
    empty_pixel = display_position(window, [0.2, 0.3, 0.0])
    camera_position_before = window.plotter.camera_position
    click(window, app, empty_pixel, release_position=(empty_pixel[0] + 30, empty_pixel[1] + 10))
    check(recorder.last_and_clear() is None and window.plotter.camera_position != camera_position_before,
          "a left drag on empty space rotates the camera and reaches no sphere")
    check(all_widgets_listen(manager), "every sphere widget listens again after a left camera drag")
    reset_view(window, app)
    empty_pixel = display_position(window, [0.2, 0.3, 0.0])
    camera_position_before = window.plotter.camera_position
    right_drag(window, app, empty_pixel, (empty_pixel[0], empty_pixel[1] + 30))
    check(recorder.last_and_clear() is None and window.plotter.camera_position != camera_position_before,
          "a right drag on empty space zooms the camera and reaches no sphere")
    check(all_widgets_listen(manager), "every sphere widget listens again after a right camera drag")
    reset_view(window, app)

    print("\nA click after a camera drag still picks the front sphere")
    click(window, app, display_position(window, spheres[2].get_center()))
    check(recorder.last_and_clear() == 2, "click over sphere 2 (front) reaches sphere 2")

    manager.clear_unsaved_changes()
    window.close()
    print(f"\n{'ALL PASSED' if not failures else f'{len(failures)} FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
