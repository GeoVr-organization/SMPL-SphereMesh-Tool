"""End-to-end check of the CSV pose and shape pickers on a real MainWindow.

Covers the "Random pose (CSV)" and "Random shape (CSV)" buttons and the row slider under
each:
- the rows match data/random_pose.csv and data/random_shapes.csv as numpy reads them
  (by position; the GUI reads poses by header name);
- a press never repeats the row shown;
- dragging a slider applies each row before the handle is released, checked against an
  independent forward pass of the body;
- position 0 and Reset give the rest pose / all betas 0;
- a hand edit marks the label "(edited)" without altering the stored row;
- none of it marks the template unsaved.
It also times one slider step.

Runs with ui_widgets.set_non_interactive(True) so no modal waits for a human.

Run with: python scripts/test_random_pose_shape.py
"""
from __future__ import annotations

import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pyvista as pv
import torch as tc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.app_paths import data_file  # noqa: E402
from src.smpl_forward_compat import my_function_forward  # noqa: E402
from src.smpl_models import MALE, load_smpl_model  # noqa: E402
from src.ui_widgets import (  # noqa: E402
    BETA_SLIDER_INT_MAX,
    BETA_SLIDER_INT_MIN,
    MainWindow,
    set_non_interactive,
)

MESH_TOLERANCE = 1e-6  # m; displayed body vertices vs an independent forward pass
LEFT_SHOULDER_JOINT = 16  # SMPL joint index, the one edited by hand below
failures: list[str] = []


def check(condition: bool, label: str, detail: str = "") -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))
    if not condition:
        failures.append(label)


def mesh_error(window, beta, theta) -> float:
    """Largest distance (m) between the displayed body and a fresh forward pass for beta, theta."""
    displayed_vertices = np.array(pv.wrap(window.mesh_actor.GetMapper().GetInput()).points)
    expected_vertices, _, _, _ = my_function_forward(window.smpl_model, beta, theta)
    return float(np.abs(displayed_vertices - expected_vertices.detach().cpu().numpy()).max())


def value_error(actual: tc.Tensor, expected: tc.Tensor) -> float:
    """Largest absolute difference between two parameter vectors."""
    return float((actual - expected).abs().max())


def drag_and_compare(window, app, row_picker, positions, expected_beta_theta):
    """Drag row_picker's handle through positions the way the mouse does and compare each step.

    QSlider's mouseMoveEvent calls setSliderPosition while the handle is held down; with
    tracking on, that emits valueChanged at once. Each comparison runs before any queued
    event is processed, so a pass means the step was applied synchronously, mid-drag.
    Returns (worst parameter error, worst mesh error in m, per-step times in ms).
    """
    worst_parameter_error = 0.0
    worst_mesh_error = 0.0
    step_times_ms = []
    row_picker.slider.setSliderDown(True)
    for position in positions:
        started = time.perf_counter()
        row_picker.slider.setSliderPosition(position)
        step_times_ms.append((time.perf_counter() - started) * 1000.0)
        expected_beta, expected_theta = expected_beta_theta(position)
        worst_parameter_error = max(
            worst_parameter_error,
            value_error(window.beta, expected_beta),
            value_error(window.theta, expected_theta),
        )
        worst_mesh_error = max(worst_mesh_error, mesh_error(window, expected_beta, expected_theta))
        app.processEvents()
    row_picker.slider.setSliderDown(False)
    app.processEvents()
    return worst_parameter_error, worst_mesh_error, step_times_ms


def timing_summary(step_times_ms) -> str:
    return f"median {statistics.median(step_times_ms):.1f} ms, max {max(step_times_ms):.1f} ms per step"


def main() -> int:
    set_non_interactive(True)
    random.seed(11)
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow(load_smpl_model(MALE), MALE)
    window.resize(1400, 900)
    window.show()
    app.processEvents()
    manager = window.sphere_manager
    theta_group = window.theta_slider_group
    beta_group = window.beta_slider_group
    theta_group.setChecked(True)  # expand both groups, as a user would
    beta_group.setChecked(True)
    app.processEvents()
    pose_picker = theta_group.random_pose_slider
    shape_picker = beta_group.random_shape_slider
    zero_theta = tc.zeros(72, dtype=tc.float64)
    zero_beta = tc.zeros(10, dtype=tc.float64)
    had_unsaved_changes = manager.has_unsaved_changes

    reference_poses = tc.from_numpy(np.loadtxt(data_file("random_pose.csv"), delimiter=",", skiprows=1, ndmin=2))
    reference_shapes = tc.from_numpy(np.loadtxt(data_file("random_shapes.csv"), delimiter=",", ndmin=2))
    pose_count = reference_poses.shape[0]
    shape_count = reference_shapes.shape[0]

    print("\nRows read from the CSVs")
    loaded_poses = tc.stack(theta_group.random_pose_rows)
    loaded_shapes = tc.stack(beta_group.random_shape_rows)
    check(loaded_poses.shape == reference_poses.shape, "pose rows", f"{tuple(loaded_poses.shape)}")
    check(value_error(loaded_poses, reference_poses) == 0.0, "pose rows equal numpy.loadtxt")
    check(loaded_shapes.shape == reference_shapes.shape, "shape rows", f"{tuple(loaded_shapes.shape)}")
    check(value_error(loaded_shapes, reference_shapes) == 0.0, "shape rows equal numpy.loadtxt")

    print("\nStartup")
    check(
        (pose_picker.slider.minimum(), pose_picker.slider.maximum()) == (0, pose_count),
        f"pose slider spans 0..{pose_count}",
    )
    check(
        (shape_picker.slider.minimum(), shape_picker.slider.maximum()) == (0, shape_count),
        f"shape slider spans 0..{shape_count}",
    )
    check(pose_picker.label.text() == "Pose: none (rest pose)", "pose label at startup", pose_picker.label.text())
    check(shape_picker.label.text() == "Shape: none (all betas 0)", "shape label at startup", shape_picker.label.text())
    check(
        pose_picker.slider.hasTracking() and shape_picker.slider.hasTracking(),
        "both sliders emit while dragging (tracking on)",
    )
    beta_ranges = {(slider.minimum(), slider.maximum()) for slider in beta_group.sliders_list}
    check(
        beta_ranges == {(-30, 30)} and (BETA_SLIDER_INT_MIN, BETA_SLIDER_INT_MAX) == (-30, 30),
        "beta sliders span -3.0..3.0",
        str(beta_ranges),
    )

    print("\nRandom pose button")
    shown_pose_rows = []
    worst_pose_error = 0.0
    for _ in range(40):
        theta_group.random_pose_button.click()
        app.processEvents()
        shown_row = pose_picker.slider.value()
        shown_pose_rows.append(shown_row)
        worst_pose_error = max(worst_pose_error, value_error(window.theta, reference_poses[shown_row - 1]))
    repeats = sum(1 for previous, current in zip(shown_pose_rows, shown_pose_rows[1:]) if previous == current)
    check(all(1 <= row <= pose_count for row in shown_pose_rows), "every press shows a CSV row")
    check(repeats == 0, "a press never repeats the row shown", f"{len(set(shown_pose_rows))} distinct rows in 40 presses")
    check(worst_pose_error == 0.0, "window theta is the row the slider shows", f"max |diff| = {worst_pose_error}")
    last_pose_row = shown_pose_rows[-1]
    check(
        pose_picker.label.text() == f"Pose {last_pose_row} / {pose_count}",
        "label names the row",
        pose_picker.label.text(),
    )
    body_error = mesh_error(window, window.beta, reference_poses[last_pose_row - 1])
    check(body_error < MESH_TOLERANCE, "body shows that pose", f"max vertex error {body_error:.1e} m")

    print("\nDragging the pose slider")
    pose_drag_positions = list(range(1, pose_count + 1)) + list(range(pose_count - 1, -1, -1))
    parameter_error, drag_mesh_error, pose_step_times_ms = drag_and_compare(
        window,
        app,
        pose_picker,
        pose_drag_positions,
        lambda position: (zero_beta, zero_theta if position == 0 else reference_poses[position - 1]),
    )
    check(parameter_error == 0.0, "theta follows every row mid-drag", f"max |diff| = {parameter_error}")
    check(
        drag_mesh_error < MESH_TOLERANCE,
        "body follows every row mid-drag, before the release",
        f"max vertex error {drag_mesh_error:.1e} m over {len(pose_drag_positions)} steps",
    )
    check(pose_picker.label.text() == "Pose: none (rest pose)", "position 0 is the rest pose", pose_picker.label.text())
    print(f"  pose slider step: {timing_summary(pose_step_times_ms)}")

    print("\nHand edit after a CSV pose")
    pose_picker.slider.setValue(5)
    app.processEvents()
    theta_group.joint_selector.setCurrentIndex(LEFT_SHOULDER_JOINT)
    theta_group.slider.setValue(theta_group.slider.value() + 3)
    app.processEvents()
    edited_theta_index = LEFT_SHOULDER_JOINT * 3
    expected_edited_theta = reference_poses[4].clone()
    expected_edited_theta[edited_theta_index] = theta_group.slider.value() / 10.0
    check(pose_picker.label.text() == f"Pose 5 / {pose_count} (edited)", "label marks the edit", pose_picker.label.text())
    check(
        value_error(window.theta, expected_edited_theta) == 0.0,
        "theta is row 5 with only the edited axis changed",
        f"theta[{edited_theta_index}] = {float(window.theta[edited_theta_index]):.2f}",
    )
    check(
        value_error(theta_group.random_pose_rows[4], reference_poses[4]) == 0.0,
        "the edit leaves stored row 5 untouched",
    )
    pose_picker.slider.setValue(6)
    pose_picker.slider.setValue(5)
    app.processEvents()
    check(value_error(window.theta, reference_poses[4]) == 0.0, "sliding away and back restores row 5")
    check(pose_picker.label.text() == f"Pose 5 / {pose_count}", "label drops (edited)", pose_picker.label.text())

    print("\nPose Reset")
    theta_group.reset()
    app.processEvents()
    check(pose_picker.slider.value() == 0, "Reset moves the pose slider to 0")
    check(pose_picker.label.text() == "Pose: none (rest pose)", "label after Reset", pose_picker.label.text())
    check(value_error(window.theta, zero_theta) == 0.0, "theta is all zeros")
    theta_group.slider.setValue(theta_group.slider.value() + 2)
    app.processEvents()
    check(pose_picker.label.text() == "Pose: none (edited)", "a hand edit at position 0 is marked", pose_picker.label.text())

    print("\nRandom shape button")
    theta_group.reset()
    pose_picker.slider.setValue(3)  # shape changes on a posed body
    app.processEvents()
    posed_theta = reference_poses[2]
    shown_shape_rows = []
    worst_shape_error = 0.0
    worst_slider_mismatch = 0
    for _ in range(30):
        beta_group.random_shape_button.click()
        app.processEvents()
        shown_row = shape_picker.slider.value()
        shown_shape_rows.append(shown_row)
        worst_shape_error = max(worst_shape_error, value_error(window.beta, reference_shapes[shown_row - 1]))
        expected_positions = [int(round(float(beta_value) * 10)) for beta_value in reference_shapes[shown_row - 1]]
        actual_positions = [slider.value() for slider in beta_group.sliders_list]
        worst_slider_mismatch = max(
            worst_slider_mismatch,
            max(abs(actual - expected) for actual, expected in zip(actual_positions, expected_positions)),
        )
    repeats = sum(1 for previous, current in zip(shown_shape_rows, shown_shape_rows[1:]) if previous == current)
    out_of_old_range = int((reference_shapes.abs() > 2.0).sum())
    check(all(1 <= row <= shape_count for row in shown_shape_rows), "every press shows a CSV row")
    check(repeats == 0, "a press never repeats the row shown", f"{len(set(shown_shape_rows))} distinct rows in 30 presses")
    check(worst_shape_error == 0.0, "window beta is the row the slider shows", f"max |diff| = {worst_shape_error}")
    check(
        worst_slider_mismatch == 0,
        "every beta slider sits at its CSV value, none pinned at an end",
        f"{out_of_old_range} of {reference_shapes.numel()} CSV betas lie beyond the old +-2.0",
    )
    last_shape_row = shown_shape_rows[-1]
    check(
        shape_picker.label.text() == f"Shape {last_shape_row} / {shape_count}",
        "label names the row",
        shape_picker.label.text(),
    )
    check(
        beta_group.slider_labels[1].text().startswith("Beta 2: "),
        "beta labels show the values",
        beta_group.slider_labels[1].text(),
    )
    check(value_error(window.theta, posed_theta) == 0.0, "the pose is kept", "pose row 3")

    print("\nDragging the shape slider")
    shape_drag_positions = list(range(1, shape_count + 1)) + list(range(shape_count - 1, -1, -1))
    parameter_error, drag_mesh_error, shape_step_times_ms = drag_and_compare(
        window,
        app,
        shape_picker,
        shape_drag_positions,
        lambda position: (zero_beta if position == 0 else reference_shapes[position - 1], posed_theta),
    )
    check(parameter_error == 0.0, "beta follows every row mid-drag", f"max |diff| = {parameter_error}")
    check(
        drag_mesh_error < MESH_TOLERANCE,
        "body follows every row mid-drag, before the release",
        f"max vertex error {drag_mesh_error:.1e} m over {len(shape_drag_positions)} steps",
    )
    check(shape_picker.label.text() == "Shape: none (all betas 0)", "position 0 is all betas 0", shape_picker.label.text())
    print(f"  shape slider step: {timing_summary(shape_step_times_ms)}")

    print("\nHand edits after a CSV shape")
    shape_picker.slider.setValue(4)
    app.processEvents()
    beta_group.sliders_list[0].setValue(beta_group.sliders_list[0].value() + 1)
    app.processEvents()
    check(shape_picker.label.text() == f"Shape 4 / {shape_count} (edited)", "a beta slider edit is marked", shape_picker.label.text())
    check(
        value_error(beta_group.random_shape_rows[3], reference_shapes[3]) == 0.0,
        "the edit leaves stored row 4 untouched",
    )
    shape_picker.slider.setValue(5)
    app.processEvents()
    beta_group.beta_line_edit.setText("0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0")
    beta_group.load_betas_from_text()
    app.processEvents()
    check(shape_picker.label.text() == f"Shape 5 / {shape_count} (edited)", "Load betas is marked as an edit", shape_picker.label.text())

    print("\nShape Reset")
    beta_group.reset()
    app.processEvents()
    check(shape_picker.slider.value() == 0, "Reset moves the shape slider to 0")
    check(shape_picker.label.text() == "Shape: none (all betas 0)", "label after Reset", shape_picker.label.text())
    check(value_error(window.beta, zero_beta) == 0.0, "beta is all zeros")

    print("\nTemplate state")
    check(
        manager.has_unsaved_changes == had_unsaved_changes,
        "pose and shape changes do not mark the template unsaved",
        f"before {had_unsaved_changes}, after {manager.has_unsaved_changes}",
    )

    theta_group.reset()
    app.processEvents()
    window.close()
    app.quit()

    print()
    if failures:
        print(f"{len(failures)} FAILURE(S): " + ", ".join(failures))
        return 1
    print("Random pose / shape OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
