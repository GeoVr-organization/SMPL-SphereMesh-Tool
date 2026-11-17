"""End-to-end check of the star-pose startup on a real MainWindow.

Covers:
- the startup pose: hips turned about Z (left +, right -), shared with Theta Controls, the
  body equal to SMFitting/examples/SMPL_starpose.obj, the pose picker marked edited;
- the male startup template, data/template_male_1.3_starpose.txt, loaded as shown in it;
- binding under a pose: FK puts every sphere back where it was loaded, in LBS and rigid
  mode, and after Reassign all parents (a posed centre bound as a rest one would be posed
  twice, spreading the legs again);
- binding at the rest pose: the rest-frame code path, unchanged;
- Reset to the rest pose: the spheres follow the legs (rigid offsets from the parent joint
  unchanged), and back;
- switching to the female model and back.

Runs with ui_widgets.set_non_interactive(True) so no modal waits for a human.

Run with: python scripts/test_star_pose_startup.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyvista as pv
import torch as tc
from scipy.spatial import cKDTree

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))
sys.path.insert(0, str(REPOSITORY_ROOT / "SMFitting"))

import json  # noqa: E402

from PyQt6.QtWidgets import QApplication  # noqa: E402

from ObjMesh import loadObjMesh  # noqa: E402
from txt2json import convert_text_to_json_payload  # noqa: E402
from src.smpl_models import FEMALE, MALE, load_smpl_model, startup_spheremesh_template_path  # noqa: E402
from src.ui_widgets import (  # noqa: E402
    LEFT_HIP_JOINT,
    RIGHT_HIP_JOINT,
    STAR_POSE_HIP_ABDUCTION,
    MainWindow,
    default_theta_tensor,
    set_non_interactive,
)

STAR_POSE_OBJ_PATH = REPOSITORY_ROOT / "SMFitting" / "examples" / "SMPL_starpose.obj"
BODY_MATCH_TOLERANCE = 1e-5  # m; every star-pose OBJ vertex to the displayed body
STAY_PUT_TOLERANCE = 1e-6    # m; a sphere that FK should leave where it is
LOAD_ROUNDING_TOLERANCE = 0.87e-3  # m; loading rounds each centre coordinate to 3 decimals
failures: list[str] = []


def check(condition: bool, label: str, detail: str = "") -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))
    if not condition:
        failures.append(label)


def sphere_centres(manager) -> np.ndarray:
    """Every sphere's shown centre, (N, 3)."""
    return np.array([sphere.get_center() for sphere in manager.spheres], dtype=np.float64)


def largest_move(centres_after: np.ndarray, centres_before: np.ndarray) -> float:
    """Largest distance (m) any sphere moved."""
    return float(np.linalg.norm(centres_after - centres_before, axis=1).max())


def offsets_from_parent_joints(manager) -> np.ndarray:
    """Each sphere's centre in its parent joint's frame on the shown skeleton: rigid-invariant."""
    skeleton = manager.skeleton
    return np.array([
        skeleton.get_joint_rotation(sphere.get_parent_joint()).T
        @ (sphere.get_center() - skeleton.joints[sphere.get_parent_joint()])
        for sphere in manager.spheres
    ])


def template_payload(gender: str) -> dict:
    """The startup template of gender as the tool's JSON payload."""
    path = startup_spheremesh_template_path(gender)
    if path.suffix.lower() == ".txt":
        return convert_text_to_json_payload(path.read_text(encoding="utf-8"))
    return json.loads(path.read_text(encoding="utf-8"))


def repose(window) -> None:
    """Run FK at the current pose through the handler a theta change uses."""
    window._on_theta_changed(window.theta)


def main() -> int:
    set_non_interactive(True)
    application = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow(load_smpl_model(MALE), MALE)
    window.resize(1400, 900)
    window.show()
    application.processEvents()
    manager = window.sphere_manager
    theta_group = window.theta_slider_group
    lbs_checkbox = theta_group.lbs_binding_checkbox

    print("Startup pose")
    expected_theta = tc.zeros(72, dtype=tc.float64)
    expected_theta[3 * LEFT_HIP_JOINT + 2] = STAR_POSE_HIP_ABDUCTION
    expected_theta[3 * RIGHT_HIP_JOINT + 2] = -STAR_POSE_HIP_ABDUCTION
    check(bool(tc.equal(window.theta, expected_theta)), "theta: left hip Z +0.3, right hip Z -0.3, rest 0")
    check(bool(tc.equal(default_theta_tensor(), expected_theta)), "default_theta_tensor() is that pose")
    check(window.theta is theta_group.get_theta(), "MainWindow and Theta Controls share one theta tensor")
    displayed_vertices = np.array(pv.wrap(window.mesh_actor.GetMapper().GetInput()).points, dtype=np.float64)
    star_pose_vertices = loadObjMesh(STAR_POSE_OBJ_PATH)[0]
    body_gaps = cKDTree(displayed_vertices).query(star_pose_vertices)[0]
    check(
        float(body_gaps.max()) < BODY_MATCH_TOLERANCE,
        "body matches SMPL_starpose.obj (each OBJ vertex to the nearest shown vertex)",
        f"max {1000 * body_gaps.max():.4f} mm over {len(star_pose_vertices)} vertices",
    )
    check("edited" in theta_group.random_pose_slider.label.text(), "pose picker says edited, not rest pose",
          theta_group.random_pose_slider.label.text())
    check(manager._is_posed(), "manager sees a posed skeleton")

    print("\nStartup template")
    male_payload = template_payload(MALE)
    startup_template_path = startup_spheremesh_template_path(MALE)
    check(startup_template_path == REPOSITORY_ROOT / "data" / "template_male_1.3_starpose.txt",
          "male startup template is data/template_male_1.3_starpose.txt", str(startup_template_path))
    file_centres = np.array([sphere["center"] for sphere in male_payload["spheres"]], dtype=np.float64)
    file_radii = np.array([sphere["radius"] for sphere in male_payload["spheres"]], dtype=np.float64)
    check(manager.get_sphere_count() == len(file_centres) and len(manager.connections) == len(male_payload["connections"]),
          "sphere and connection counts match the file",
          f"{manager.get_sphere_count()} spheres, {len(manager.connections)} connections")
    loaded_centres = sphere_centres(manager)
    check(largest_move(loaded_centres, file_centres) < LOAD_ROUNDING_TOLERANCE,
          "spheres shown where the file puts them (after startup FK)",
          f"max {1000 * largest_move(loaded_centres, file_centres):.3f} mm")
    shown_radii = np.array([sphere.get_radius() for sphere in manager.spheres])
    check(float(np.abs(shown_radii - file_radii).max()) < 1e-12, "radii as in the file")
    check(not manager.has_unsaved_changes, "startup leaves no unsaved changes")

    print("\nBinding under the star pose")
    check(lbs_checkbox.isChecked(), "LBS binding on by default")
    repose(window)
    check(largest_move(sphere_centres(manager), loaded_centres) < STAY_PUT_TOLERANCE,
          "LBS: FK leaves every sphere where it was loaded",
          f"max {1000 * largest_move(sphere_centres(manager), loaded_centres):.2e} mm")
    lbs_checkbox.setChecked(False)
    window._load_spheremesh_file(startup_spheremesh_template_path(MALE))
    rigid_loaded_centres = sphere_centres(manager)
    repose(window)
    check(largest_move(sphere_centres(manager), rigid_loaded_centres) < STAY_PUT_TOLERANCE,
          "rigid: FK leaves every sphere where it was loaded",
          f"max {1000 * largest_move(sphere_centres(manager), rigid_loaded_centres):.2e} mm")
    window.reassign_all_parent_joints_clicked()
    repose(window)
    check(largest_move(sphere_centres(manager), rigid_loaded_centres) < STAY_PUT_TOLERANCE,
          "rigid: Reassign all parents, then FK, moves no sphere",
          f"max {1000 * largest_move(sphere_centres(manager), rigid_loaded_centres):.2e} mm")

    print("\nReset to the rest pose and back (rigid)")
    star_offsets = offsets_from_parent_joints(manager)
    left_leg_spheres = [index for index, sphere in enumerate(manager.spheres)
                        if sphere.get_center()[0] > 0 and sphere.get_center()[1] < -0.6]
    star_left_leg_x = sphere_centres(manager)[left_leg_spheres, 0]
    theta_group.reset()
    check(bool(tc.equal(window.theta, tc.zeros(72, dtype=tc.float64))), "Reset gives theta all zeros")
    check(not manager._is_posed(), "manager sees the rest skeleton")
    rest_offset_drift = float(np.abs(offsets_from_parent_joints(manager) - star_offsets).max())
    check(rest_offset_drift < STAY_PUT_TOLERANCE,
          "spheres follow their joints: offsets in the parent frame unchanged",
          f"max {1000 * rest_offset_drift:.2e} mm")
    rest_left_leg_x = sphere_centres(manager)[left_leg_spheres, 0]
    check(bool(len(left_leg_spheres) > 0 and np.all(rest_left_leg_x < star_left_leg_x)),
          "left leg spheres move in towards the body",
          f"{len(left_leg_spheres)} spheres, mean x {star_left_leg_x.mean():.3f} -> {rest_left_leg_x.mean():.3f} m")
    rest_centres = sphere_centres(manager)

    print("\nBinding at the rest pose (original code path)")
    probe_index = left_leg_spheres[0]
    probe_centre = manager.spheres[probe_index].get_center()
    expected_parent, _ = manager._closest_bone(probe_centre)
    expected_rest_offset = (
        manager.rest_skeleton.get_joint_rotation(expected_parent).T
        @ (probe_centre - manager.rest_skeleton.joints[expected_parent])
    )
    manager.assign_parent_joint(probe_index)
    check(manager.spheres[probe_index].get_parent_joint() == expected_parent
          and np.array_equal(manager.spheres[probe_index].get_rest_offset(), expected_rest_offset),
          "rest pose: parent and rest offset from the rest skeleton, bit for bit")
    check(manager.spheres[probe_index].get_lbs_joint_weights() is None,
          "rest pose: LBS weights left to be derived on the next pose")

    window._on_theta_changed(default_theta_tensor())
    check(largest_move(sphere_centres(manager), rigid_loaded_centres) < STAY_PUT_TOLERANCE,
          "back in the star pose every sphere is where it was loaded",
          f"max {1000 * largest_move(sphere_centres(manager), rigid_loaded_centres):.2e} mm")
    window._on_theta_changed(tc.zeros(72, dtype=tc.float64))
    check(largest_move(sphere_centres(manager), rest_centres) < STAY_PUT_TOLERANCE,
          "and back at rest where it was at rest")

    print("\nLBS: Reassign all parents under the star pose")
    lbs_checkbox.setChecked(True)
    window._on_theta_changed(default_theta_tensor())
    lbs_star_centres = sphere_centres(manager)
    window.reassign_all_parent_joints_clicked()
    repose(window)
    check(largest_move(sphere_centres(manager), lbs_star_centres) < STAY_PUT_TOLERANCE,
          "LBS: Reassign all parents, then FK, moves no sphere",
          f"max {1000 * largest_move(sphere_centres(manager), lbs_star_centres):.2e} mm")

    print("\nFemale model and back")
    manager.has_unsaved_changes = False
    window._on_gender_changed(FEMALE)
    female_payload = template_payload(FEMALE)
    check(manager.get_sphere_count() == len(female_payload["spheres"]),
          "female template loaded", f"{manager.get_sphere_count()} spheres")
    female_centres = sphere_centres(manager)
    repose(window)
    check(largest_move(sphere_centres(manager), female_centres) < STAY_PUT_TOLERANCE,
          "female: FK at the star pose is stable after the switch")
    window._on_gender_changed(MALE)
    check(largest_move(sphere_centres(manager), file_centres) < LOAD_ROUNDING_TOLERANCE,
          "male again: spheres where the startup template puts them",
          f"max {1000 * largest_move(sphere_centres(manager), file_centres):.3f} mm")

    print("\nGender switch away from the star pose (LBS): the template is bound in its own pose")
    theta_group.reset()
    window._on_gender_changed(FEMALE)
    window._on_gender_changed(MALE)
    check(bool(tc.equal(window.theta, tc.zeros(72, dtype=tc.float64))) and window.theta is theta_group.get_theta(),
          "the user's rest pose is kept, still shared with Theta Controls")
    left_leg_x_at_rest = sphere_centres(manager)[left_leg_spheres, 0]
    check(bool(np.all(left_leg_x_at_rest < star_left_leg_x - 0.05)),
          "at the rest pose the left leg spheres sit in with the legs, not spread",
          f"mean x {left_leg_x_at_rest.mean():.3f} m (star pose {star_left_leg_x.mean():.3f} m)")
    window._on_theta_changed(default_theta_tensor())
    check(largest_move(sphere_centres(manager), file_centres) < LOAD_ROUNDING_TOLERANCE,
          "back in the star pose: spheres where the file puts them (not posed twice)",
          f"max {1000 * largest_move(sphere_centres(manager), file_centres):.3f} mm")

    print("\nGender switch at beta[0] = 2 (rigid): the template follows the shape")
    lbs_checkbox.setChecked(False)
    beta_sliders = window.beta_slider_group.sliders_list
    beta_sliders[0].setValue(20)
    application.processEvents()
    window._on_gender_changed(FEMALE)
    window._on_gender_changed(MALE)
    shaped_move = largest_move(sphere_centres(manager), file_centres)
    check(abs(window.beta[0].item() - 2.0) < 1e-12 and shaped_move > 0.01,
          "betas kept and the spheres moved with the body", f"max {1000 * shaped_move:.1f} mm from the file")
    beta_sliders[0].setValue(0)
    application.processEvents()
    check(largest_move(sphere_centres(manager), file_centres) < LOAD_ROUNDING_TOLERANCE,
          "back at beta 0: spheres where the file puts them",
          f"max {1000 * largest_move(sphere_centres(manager), file_centres):.3f} mm")
    lbs_checkbox.setChecked(True)

    manager.has_unsaved_changes = False
    window.close()
    application.processEvents()

    print(f"\n{'All checks passed' if not failures else f'{len(failures)} check(s) FAILED'}")
    for label in failures:
        print(f"  - {label}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
