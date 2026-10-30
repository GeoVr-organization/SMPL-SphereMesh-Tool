"""End-to-end check of LBS sphere binding (Theta Controls -> "LBS binding") on a real MainWindow.

Covers the default (on in the GUI, rigid in a bare SphereManager), the rigid path staying
exactly today's FK with the box off, both modes agreeing at the rest pose, the derived joint
weights, agreement with the SMPL body (a sphere carrying a vertex's skin weights follows that
vertex), the regressed-vs-FK joint offset, posed edits round-tripping through the InfoPanel
spinbox and Fit spheres, the toggle itself (unsaved flag, heatmap), weight lifetime (Delete,
Divide, Reassign all parents, beta change), saving, and slider cost in both modes.

Runs with ui_widgets.set_non_interactive(True) so no modal waits for a human. Clicks on the
checkbox are simulated Qt clicks; nothing here was looked at by eye.

Run with: python scripts/test_lbs_binding.py
"""
from __future__ import annotations

import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pyvista as pv
import torch as tc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.smpl_models import MALE, load_smpl_model  # noqa: E402
from src.sphere_manager import LBS_MAX_JOINTS, SphereManager  # noqa: E402
from src.ui_widgets import (  # noqa: E402
    FitSpheresDialog,
    MainWindow,
    _load_random_shape_rows,
    set_non_interactive,
)

POSE_ROWS_USED = (0, 5, 11)   # rows of data/random_pose.csv the pose checks run on
failures: list[str] = []


def check(condition: bool, label: str, detail: str = "") -> None:
    """Print one [PASS]/[FAIL] line and remember the failures."""
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))
    if not condition:
        failures.append(label)


def bound_sphere_indices(manager) -> list[int]:
    """Indices of the spheres FK moves (parent joint and rest offset set)."""
    return [
        sphere_index for sphere_index, sphere in enumerate(manager.spheres)
        if sphere.get_parent_joint() is not None and sphere.get_rest_offset() is not None
    ]


def sphere_centers(manager) -> np.ndarray:
    """Every sphere's current centre, in sphere order."""
    return np.array([sphere.get_center() for sphere in manager.spheres])


def rigid_centers(manager, sphere_indices) -> np.ndarray:
    """Today's FK, written out: J[p] + R[p] @ rest_offset with the regressed posed joints."""
    expected_centers = []
    for sphere_index in sphere_indices:
        sphere = manager.spheres[sphere_index]
        parent = sphere.get_parent_joint()
        expected_centers.append(
            manager.skeleton.joints[parent]
            + np.dot(manager.skeleton.get_joint_rotation(parent), sphere.get_rest_offset())
        )
    return np.array(expected_centers)


def set_lbs_checkbox(window, app, enabled: bool) -> None:
    """Click the Theta Controls checkbox if it is not already in the wanted state."""
    checkbox = window.theta_slider_group.lbs_binding_checkbox
    if checkbox.isChecked() != enabled:
        checkbox.click()
        app.processEvents()


def pose(window, app, theta) -> None:
    """Apply theta through the theta-slider handler."""
    window._on_theta_changed(theta.clone())
    app.processEvents()


def posed_body_points(window) -> np.ndarray:
    """The displayed (posed) SMPL body's vertices."""
    return np.asarray(pv.wrap(window.mesh_actor.GetMapper().GetInput()).points, dtype=np.float64)


def add_bound_sphere(manager, parent_joint: int, rest_center, joint_weights) -> int:
    """Append a sphere bound to parent_joint at rest_center, with its LBS weights forced."""
    sphere_index = manager.add_sphere(np.asarray(rest_center, dtype=np.float64), 0.01)
    sphere = manager.spheres[sphere_index]
    rest_offset = np.dot(
        manager.rest_skeleton.get_joint_rotation(parent_joint).T,
        np.asarray(rest_center) - manager.rest_skeleton.joints[parent_joint],
    )
    sphere.set_parent_joint(int(parent_joint), rest_offset)
    sphere.set_lbs_joint_weights(joint_weights)
    return sphere_index


def median_milliseconds(callback, repeat_count: int) -> float:
    """Median wall time of callback over repeat_count calls, in ms."""
    step_times_ms = []
    for _ in range(repeat_count):
        started = time.perf_counter()
        callback()
        step_times_ms.append((time.perf_counter() - started) * 1000.0)
    return statistics.median(step_times_ms)


def main() -> int:
    """Run every check on one MainWindow; 1 if any failed."""
    set_non_interactive(True)
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow(load_smpl_model(MALE), MALE)
    window.resize(1400, 900)
    window.show()
    window.theta_slider_group.setChecked(True)   # expand Theta Controls, as a user would
    app.processEvents()
    manager = window.sphere_manager
    checkbox = window.theta_slider_group.lbs_binding_checkbox
    pose_rows = [window.theta_slider_group.random_pose_rows[row] for row in POSE_ROWS_USED]
    rest_theta = tc.zeros(72, dtype=tc.float64)
    original_beta = window.beta.clone()
    manager.set_mirror_edits_enabled(False)   # edits below must not move twins
    print(f"Loaded {manager.get_sphere_count()} spheres, {len(manager.connections)} connections")

    # ---------------------------------------------------------------------------------
    print("\nDefault")
    check(checkbox.isChecked() and manager.lbs_binding_enabled, "the GUI starts with LBS on")
    check(not SphereManager(None, lambda: None).lbs_binding_enabled, "a bare SphereManager stays rigid")
    check(not checkbox.isHidden(), "the checkbox shows once Theta Controls is expanded")

    # ---------------------------------------------------------------------------------
    print("\nRest pose: both modes agree")
    manager.update_sphere_positions()   # loaded centres are rounded; start from FK
    manager.clear_unsaved_changes()
    bound_indices = bound_sphere_indices(manager)
    lbs_rest_centers = sphere_centers(manager)[bound_indices]
    set_lbs_checkbox(window, app, False)
    rigid_rest_centers = sphere_centers(manager)[bound_indices]
    rest_gap = float(np.abs(lbs_rest_centers - rigid_rest_centers).max())
    check(rest_gap < 1e-12, "at theta 0 LBS and rigid centres agree", f"max |diff| {rest_gap:.1e} m")
    check(not manager.has_unsaved_changes, "toggling does not set the unsaved flag")

    # ---------------------------------------------------------------------------------
    print("\nBox off: exactly today's rigid FK")
    for row_number, theta in zip(POSE_ROWS_USED, pose_rows):
        pose(window, app, theta)
        posed_centers = sphere_centers(manager)[bound_indices]
        check(
            np.array_equal(posed_centers, rigid_centers(manager, bound_indices)),
            f"CSV pose {row_number}: every centre equals J[p] + R[p] @ rest_offset bit for bit",
        )

    # ---------------------------------------------------------------------------------
    print("\nThe toggle, posed (via the checkbox)")
    call_counter = {"count": 0}
    original_get_heatmap = manager.get_heatmap

    def counting_get_heatmap(mesh):
        call_counter["count"] += 1
        return original_get_heatmap(mesh)

    manager.get_heatmap = counting_get_heatmap
    window.checkbox_show_heatmap.setChecked(True)
    app.processEvents()
    rigid_posed_centers = sphere_centers(manager)[bound_indices]
    call_counter["count"] = 0
    manager.clear_unsaved_changes()
    set_lbs_checkbox(window, app, True)
    lbs_posed_centers = sphere_centers(manager)[bound_indices]
    displacement = np.linalg.norm(lbs_posed_centers - rigid_posed_centers, axis=1)
    check(manager.lbs_binding_enabled, "clicking the checkbox switches the manager to LBS")
    check(
        displacement.max() > 1e-3,
        "switching LBS on moves posed spheres",
        f"max {displacement.max() * 1000:.1f} mm, median {np.median(displacement) * 1000:.1f} mm "
        f"over {len(bound_indices)} spheres (CSV pose {POSE_ROWS_USED[-1]})",
    )
    check(not manager.has_unsaved_changes, "toggling posed does not set the unsaved flag")
    check(call_counter["count"] >= 1, "the heatmap recomputes", f"{call_counter['count']} recomputation(s)")
    window.checkbox_show_heatmap.setChecked(False)
    manager.get_heatmap = original_get_heatmap
    app.processEvents()

    # ---------------------------------------------------------------------------------
    print("\nDerived joint weights")
    weight_rows = np.array([manager.spheres[i].get_lbs_joint_weights() for i in bound_indices])
    nonzero_counts = (weight_rows > 0).sum(axis=1)
    weight_sum_error = float(np.abs(weight_rows.sum(axis=1) - 1.0).max())
    check(nonzero_counts.max() <= LBS_MAX_JOINTS, f"at most {LBS_MAX_JOINTS} joints per sphere",
          f"joint counts {np.bincount(nonzero_counts, minlength=LBS_MAX_JOINTS + 1).tolist()} (index = count)")
    check(weight_sum_error < 1e-12, "weights sum to 1", f"max |sum - 1| {weight_sum_error:.1e}")
    strongest_in_gate = all(
        int(np.argmax(weight_rows[row])) in manager._lbs_gate_joints(manager.spheres[i].get_parent_joint())
        for row, i in enumerate(bound_indices)
    )
    outside_gate_weights = [
        weight_rows[row][joint]
        for row, i in enumerate(bound_indices)
        for joint in np.flatnonzero(weight_rows[row])
        if joint not in manager._lbs_gate_joints(manager.spheres[i].get_parent_joint())
    ]
    check(
        strongest_in_gate,
        "every sphere's strongest joint is inside its limb gate",
        f"{len(outside_gate_weights)} minor weights on joints outside the gate (minor influences of "
        f"admitted vertices), largest {max(outside_gate_weights, default=0.0):.3f}",
    )
    on_parent = np.array([
        weight_rows[row][manager.spheres[i].get_parent_joint()] for row, i in enumerate(bound_indices)
    ])
    print(f"      weight left on the rigid parent joint: median {np.median(on_parent):.2f}, "
          f"min {on_parent.min():.2f}; {int((on_parent < 0.5).sum())} spheres below 0.5")

    # ---------------------------------------------------------------------------------
    print("\nAgreement with the SMPL body")
    pose(window, app, rest_theta)
    skin_weights = manager.skinning_weights
    sorted_weights = np.sort(skin_weights, axis=1)
    probe_vertices = [int(np.argmax(sorted_weights[:, -2])), int(np.argmax(sorted_weights[:, -3]))]
    probe_sphere_indices = []
    for vertex in probe_vertices:
        dominant = int(np.argmax(skin_weights[vertex]))
        probe_sphere_indices.append(
            add_bound_sphere(manager, dominant, manager.rest_mesh_points[vertex], skin_weights[vertex])
        )
    shoulder_joint = 17   # right_shoulder: the largest regressed-vs-FK gap measured
    one_hot_weights = np.zeros(skin_weights.shape[1])
    one_hot_weights[shoulder_joint] = 1.0
    single_joint_sphere_index = add_bound_sphere(
        manager, shoulder_joint,
        manager.rest_skeleton.joints[shoulder_joint] + np.array([0.0, -0.03, 0.02]), one_hot_weights,
    )
    vertex_errors = []
    single_joint_errors = []
    single_joint_offsets_mm = []
    for theta in pose_rows:
        pose(window, app, theta)
        body_points = posed_body_points(window)
        for vertex, sphere_index in zip(probe_vertices, probe_sphere_indices):
            vertex_errors.append(float(np.linalg.norm(manager.spheres[sphere_index].get_center() - body_points[vertex])))
        single_joint_center = manager.spheres[single_joint_sphere_index].get_center()
        rigid_center = rigid_centers(manager, [single_joint_sphere_index])[0]
        joint_offset = manager.skeleton.world_transforms[shoulder_joint, :3, 3] - manager.skeleton.joints[shoulder_joint]
        single_joint_errors.append(float(np.abs(single_joint_center - rigid_center - joint_offset).max()))
        single_joint_offsets_mm.append(float(np.linalg.norm(joint_offset)) * 1000)
    check(
        max(vertex_errors) < 1e-9,
        "a sphere carrying a vertex's skin weights follows that posed vertex",
        f"vertices {probe_vertices}, max error {max(vertex_errors):.1e} m over {len(pose_rows)} poses",
    )
    check(
        max(single_joint_errors) < 1e-12,
        "one-hot on right_shoulder: LBS = rigid + (FK joint - regressed joint)",
        f"max error {max(single_joint_errors):.1e} m; that offset is "
        f"{', '.join(f'{offset:.1f}' for offset in single_joint_offsets_mm)} mm on the poses",
    )
    for sphere_index in sorted(probe_sphere_indices + [single_joint_sphere_index], reverse=True):
        manager._delete_sphere_at_index(sphere_index)

    # ---------------------------------------------------------------------------------
    print("\nPosed edits round-trip (InfoPanel spinbox, Fit spheres)")
    edit_theta = pose_rows[1]
    pose(window, app, edit_theta)
    weight_rows = {
        i: manager.spheres[i].get_lbs_joint_weights()
        for i in bound_sphere_indices(manager) if not manager.spheres[i].locked
    }
    # The sphere LBS moves furthest from its rigid bind: least weight left on its parent.
    edited_index = min(weight_rows, key=lambda i: weight_rows[i][manager.spheres[i].get_parent_joint()])
    info_panel = window.info_panel
    for mode_name, enabled in (("LBS", True), ("rigid", False)):
        set_lbs_checkbox(window, app, enabled)
        manager.selected_connection = None
        manager.selected_sphere_index = edited_index
        info_panel.update_selected_info()
        info_panel.sphere1_center_x.setValue(info_panel.sphere1_center_x.value() + 0.01)
        app.processEvents()
        target_center = np.array([
            info_panel.sphere1_center_x.value(), info_panel.sphere1_center_y.value(),
            info_panel.sphere1_center_z.value(),
        ])
        pose(window, app, edit_theta)
        round_trip_error = float(np.linalg.norm(manager.spheres[edited_index].get_center() - target_center))
        detail = f"sphere {edited_index}, error {round_trip_error:.1e} m"
        if enabled:
            sphere = manager.spheres[edited_index]
            parent = sphere.get_parent_joint()
            rigid_offset = np.dot(
                manager.skeleton.get_joint_rotation(parent).T, target_center - manager.skeleton.joints[parent]
            )
            skin_rotations, skin_translations = manager._skinning_transforms()
            blend_rotation, blend_translation = manager._lbs_blend(edited_index, skin_rotations, skin_translations)
            rigid_inverse_landing = blend_rotation @ manager.rest_skeleton.transform_point(rigid_offset, parent) + blend_translation
            detail += (f"; a rigid inverse would have landed "
                       f"{np.linalg.norm(rigid_inverse_landing - target_center) * 1000:.1f} mm away")
        check(round_trip_error < 1e-9, f"{mode_name}: a spinbox edit while posed stays put on the next pose", detail)
        if enabled:
            # A/B must be idempotent: switching off and on again re-derives nothing it keeps.
            set_lbs_checkbox(window, app, False)
            set_lbs_checkbox(window, app, True)
            toggle_error = float(np.linalg.norm(manager.spheres[edited_index].get_center() - target_center))
            check(toggle_error < 1e-9, "LBS: after that edit, toggling off and on leaves the sphere in place",
                  f"error {toggle_error:.1e} m")
    set_lbs_checkbox(window, app, True)

    fit_options = FitSpheresDialog(window).get_options()
    fit_options["max_outer"] = min(int(fit_options.get("max_outer", 10)), 10)   # keep the run short
    centers_before_fit = sphere_centers(manager)
    fit_info = manager.fit_spheres_to_mesh(pv.wrap(window.mesh_actor.GetMapper().GetInput()), **fit_options)
    fitted_centers = sphere_centers(manager)
    fit_displacement = float(np.linalg.norm(fitted_centers - centers_before_fit, axis=1).max())
    check(fit_info is not None and fit_displacement > 1e-3, "Fit spheres ran while posed",
          f"moved spheres up to {fit_displacement * 1000:.1f} mm")
    pose(window, app, edit_theta)
    bound_after_fit = bound_sphere_indices(manager)
    fit_error = float(np.abs(sphere_centers(manager)[bound_after_fit] - fitted_centers[bound_after_fit]).max())
    check(fit_error < 1e-9, "LBS: Fit spheres while posed stays put on the next pose", f"max error {fit_error:.1e} m")
    # The next beta tick re-derives every weight. The posed inverse solved for weights that
    # match the new rest centres, so most spheres move only as much as the body does; crotch
    # spheres whose two candidate weight sets derive each other still jump.
    nudged_beta = window.beta.clone()
    nudged_beta[0] += 0.01
    window._on_beta_changed(nudged_beta)
    app.processEvents()
    beta_tick_moves = np.linalg.norm(sphere_centers(manager)[bound_after_fit] - fitted_centers[bound_after_fit], axis=1)
    check(
        float(np.median(beta_tick_moves)) < 1e-3,
        "LBS: after a posed Fit, a beta[0] += 0.01 tick moves the median sphere less than 1 mm",
        f"median {np.median(beta_tick_moves) * 1000:.3f} mm, max {beta_tick_moves.max() * 1000:.2f} mm "
        f"(sphere {bound_after_fit[int(np.argmax(beta_tick_moves))]}), "
        f"{int((beta_tick_moves > 1e-3).sum())} spheres over 1 mm",
    )
    window._on_beta_changed(original_beta.clone())
    app.processEvents()

    # ---------------------------------------------------------------------------------
    print("\nWeight lifetime")
    pose(window, app, edit_theta)
    weights_by_sphere = {id(sphere): sphere.get_lbs_joint_weights() for sphere in manager.spheres}
    sphere_count_before_delete = len(manager.spheres)
    deleted_index = 3
    manager.selected_connection = None
    manager.selected_sphere_index = deleted_index
    info_panel._on_delete_clicked()
    app.processEvents()
    kept_weights = all(
        np.array_equal(sphere.get_lbs_joint_weights(), weights_by_sphere[id(sphere)]) for sphere in manager.spheres
    )
    check(len(manager.spheres) == sphere_count_before_delete - 1 and kept_weights,
          "after Delete every other sphere keeps its own weight row",
          f"{sphere_count_before_delete} -> {len(manager.spheres)} spheres")

    pose(window, app, rest_theta)
    shared_joint_cone = next(
        connection for connection in manager.connections
        if len(connection.sphere_indices) == 2
        and len({manager.spheres[i].get_parent_joint() for i in connection.sphere_indices}) == 1
    )
    endpoint_indices = tuple(shared_joint_cone.sphere_indices)
    manager.select_connection(shared_joint_cone)
    window.divide_connection_clicked()
    app.processEvents()
    new_index = len(manager.spheres) - 1
    check(manager.spheres[new_index].get_lbs_joint_weights() is None, "Divide: the new sphere has no weights yet")
    midpoint_gaps_mm = []
    for theta in pose_rows:
        pose(window, app, theta)
        midpoint = np.mean([manager.spheres[i].get_center() for i in endpoint_indices], axis=0)
        midpoint_gaps_mm.append(float(np.linalg.norm(manager.spheres[new_index].get_center() - midpoint)) * 1000)
    check(
        manager.spheres[new_index].get_lbs_joint_weights() is not None,
        "Divide: the next LBS pose derives them",
        f"posed gap from the endpoints' midpoint {', '.join(f'{gap:.2f}' for gap in midpoint_gaps_mm)} mm "
        f"(rigid: 0, shared joint)",
    )

    pose(window, app, rest_theta)
    dummy_weights = np.zeros(skin_weights.shape[1])
    dummy_weights[23] = 1.0
    manager.spheres[1].set_lbs_joint_weights(dummy_weights)
    window.reassign_all_parent_joints_clicked()
    app.processEvents()
    fresh_weights, _ = manager._derive_lbs_joint_weights(1)
    check(
        np.allclose(manager.spheres[1].get_lbs_joint_weights(), fresh_weights, atol=1e-12)
        and not np.array_equal(manager.spheres[1].get_lbs_joint_weights(), dummy_weights),
        "Reassign all parents re-derives the weights",
    )

    pose(window, app, pose_rows[0])
    weights_before_beta = {i: manager.spheres[i].get_lbs_joint_weights() for i in bound_sphere_indices(manager)}
    shape_rows = _load_random_shape_rows()
    window._on_beta_changed(shape_rows[0].clone())
    app.processEvents()
    fresh_error = max(
        float(np.abs(manager.spheres[i].get_lbs_joint_weights() - manager._derive_lbs_joint_weights(i)[0]).max())
        for i in bound_sphere_indices(manager)
    )
    beta_change = max(
        float(np.abs(manager.spheres[i].get_lbs_joint_weights() - weights_before_beta[i]).max())
        for i in bound_sphere_indices(manager)
    )
    check(
        fresh_error < 1e-12 and beta_change > 0.0,
        "a beta change re-derives every sphere's weights at the new shape",
        f"max |w - fresh| {fresh_error:.1e}; largest weight change from the old shape {beta_change:.3f}",
    )
    window._on_beta_changed(original_beta.clone())
    app.processEvents()

    # ---------------------------------------------------------------------------------
    print("\nSave at the rest pose")
    pose(window, app, rest_theta)
    with tempfile.TemporaryDirectory() as temporary_directory:
        saved_payloads = {}
        for mode_name, enabled in (("lbs", True), ("rigid", False)):
            set_lbs_checkbox(window, app, enabled)
            saved_path = Path(temporary_directory) / f"{mode_name}.json"
            manager.save_to_json(str(saved_path))
            saved_payloads[mode_name] = json.loads(saved_path.read_text())
    lbs_spheres, rigid_spheres = saved_payloads["lbs"]["spheres"], saved_payloads["rigid"]["spheres"]
    same_fields = [sorted(lbs_entry) for lbs_entry in lbs_spheres] == [sorted(rigid_entry) for rigid_entry in rigid_spheres]
    worst_center_gap = max(
        float(np.abs(np.array(lbs_entry["center"]) - np.array(rigid_entry["center"])).max())
        for lbs_entry, rigid_entry in zip(lbs_spheres, rigid_spheres)
    )
    other_fields_equal = all(
        {key: value for key, value in lbs_entry.items() if key != "center"}
        == {key: value for key, value in rigid_entry.items() if key != "center"}
        for lbs_entry, rigid_entry in zip(lbs_spheres, rigid_spheres)
    ) and saved_payloads["lbs"]["connections"] == saved_payloads["rigid"]["connections"]
    check(
        same_fields and other_fields_equal and worst_center_gap < 1e-12,
        "the JSON saved with LBS on matches the rigid one (no new fields)",
        f"max |centre diff| {worst_center_gap:.1e} m",
    )

    # ---------------------------------------------------------------------------------
    print("\nSlider cost (reported, not gated)")
    for mode_name, enabled in (("LBS", True), ("rigid", False)):
        set_lbs_checkbox(window, app, enabled)
        beta_step_ms = median_milliseconds(
            lambda: (window._on_beta_changed(shape_rows[1].clone()), window._on_beta_changed(original_beta.clone())), 4
        ) / 2
        theta_step_ms = median_milliseconds(
            lambda: (pose(window, app, pose_rows[0]), pose(window, app, pose_rows[2])), 4
        ) / 2
        print(f"      {mode_name:5s}: beta step {beta_step_ms:.1f} ms, theta step {theta_step_ms:.1f} ms (median)")
    check(True, "slider timings reported")

    set_lbs_checkbox(window, app, True)
    pose(window, app, rest_theta)
    window.close()
    app.quit()

    print()
    if failures:
        print(f"{len(failures)} FAILURE(S): " + ", ".join(failures))
        return 1
    print("LBS binding OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
