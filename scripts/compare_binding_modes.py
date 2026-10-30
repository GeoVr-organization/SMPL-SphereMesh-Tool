"""Rigid vs LBS sphere binding: how the posed collider sits against the posed SMPL body.

Drives the real MainWindow and SphereManager (no replica of the binding) for the male and
female startup templates over every pose of data/random_pose.csv, once with the Theta
Controls "LBS binding" box off and once with it on, and measures the exact signed distance
from every body vertex to the sphere-mesh (SphereManager.get_heatmap, negative inside).

Two directions are reported, because they fail differently:
  through skin  -d, the collider poking out through the skin (validate.py's "protrusion",
                the ~46 mm "rigid-binding floor" of .claude/rules/topology.md)
  outside       +d, skin standing off the collider, where cloth can sink into the body
Each is reported per region (the vertex's dominant SMPL joint) as the worst value over
poses and vertices, plus the animation term max_t d - d(rest) per vertex, which removes the
template's static fit (both modes coincide at the rest pose).

Run with: uv run python scripts/compare_binding_modes.py [--genders male female] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyvista as pv
import torch as tc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt6.QtWidgets import QApplication  # noqa: E402

from smpl_pytorch.smpl import JOINT_INDEX_TO_LABEL  # noqa: E402
from src.smpl_models import GENDERS, load_smpl_model  # noqa: E402
from src.ui_widgets import MainWindow, set_non_interactive  # noqa: E402

MODES = (("rigid", False), ("lbs", True))


def signed_distances(window) -> np.ndarray:
    """Exact signed distance (m) from every posed body vertex to the sphere-mesh, negative inside."""
    body_mesh = pv.wrap(window.mesh_actor.GetMapper().GetInput())
    return window.sphere_manager.get_heatmap(body_mesh)


def measure_gender(gender: str, app) -> dict:
    """Signed-distance statistics of one startup template in both binding modes.

    Each gender gets its own process (see main): measuring both in one process, with the
    first window closed before the second was built, made every VTK render fail
    (wglMakeCurrent: invalid handle) and the process exit with code 9.
    """
    window = MainWindow(load_smpl_model(gender), gender)
    window.resize(1200, 800)
    window.show()
    app.processEvents()
    manager = window.sphere_manager
    checkbox = window.theta_slider_group.lbs_binding_checkbox
    pose_rows = window.theta_slider_group.random_pose_rows
    dominant_joint = np.argmax(manager.skinning_weights, axis=1)

    window._on_theta_changed(tc.zeros(72, dtype=tc.float64))
    app.processEvents()
    rest_distances = signed_distances(window)

    results = {"gender": gender, "spheres": manager.get_sphere_count(), "poses": len(pose_rows), "modes": {}}
    for mode_name, enabled in MODES:
        checkbox.setChecked(enabled)
        app.processEvents()
        worst_through_skin = np.full(len(rest_distances), -np.inf)
        worst_outside = np.full(len(rest_distances), -np.inf)
        per_pose_through_skin = []
        for theta in pose_rows:
            window._on_theta_changed(theta.clone())
            app.processEvents()
            distances = signed_distances(window)
            np.maximum(worst_through_skin, -distances, out=worst_through_skin)
            np.maximum(worst_outside, distances, out=worst_outside)
            per_pose_through_skin.append(float(-distances.min()))
        animation_through_skin = worst_through_skin - (-rest_distances)
        animation_outside = worst_outside - rest_distances
        regions = {}
        for joint, label in JOINT_INDEX_TO_LABEL.items():
            region_vertices = dominant_joint == joint
            if not region_vertices.any():
                continue
            regions[label] = {
                "through_skin": float(max(worst_through_skin[region_vertices].max(), 0.0)),
                "outside": float(max(worst_outside[region_vertices].max(), 0.0)),
                "animation_through_skin": float(animation_through_skin[region_vertices].max()),
                "animation_outside": float(animation_outside[region_vertices].max()),
            }
        results["modes"][mode_name] = {
            "regions": regions,
            "through_skin_max": float(max(worst_through_skin.max(), 0.0)),
            "through_skin_mean_over_poses": float(np.mean(per_pose_through_skin)),
            "outside_max": float(max(worst_outside.max(), 0.0)),
            "outside_p95": float(np.percentile(np.maximum(worst_outside, 0.0), 95)),
            "outside_median": float(np.median(np.maximum(worst_outside, 0.0))),
            "animation_through_skin_max": float(animation_through_skin.max()),
            "animation_through_skin_median": float(np.median(animation_through_skin)),
            "animation_outside_max": float(animation_outside.max()),
            "animation_outside_median": float(np.median(animation_outside)),
        }
    checkbox.setChecked(True)
    window._on_theta_changed(tc.zeros(72, dtype=tc.float64))
    window.close()
    return results


def print_report(results: dict) -> None:
    """One table per gender, in millimetres, rigid vs LBS."""
    rigid, lbs = results["modes"]["rigid"], results["modes"]["lbs"]
    print(f"\n{results['gender']}: {results['spheres']} spheres, {results['poses']} CSV poses (mm)")
    print(f"  {'region':16s} {'through skin':>21s} {'outside':>21s} {'anim. through skin':>21s} {'anim. outside':>21s}")
    print(f"  {'':16s}" + f" {'rigid':>10s} {'LBS':>10s}" * 4)
    improved = {"through_skin": 0, "outside": 0}
    worsened = {"through_skin": 0, "outside": 0}
    for label, rigid_region in rigid["regions"].items():
        lbs_region = lbs["regions"][label]
        values = []
        for key in ("through_skin", "outside", "animation_through_skin", "animation_outside"):
            values += [rigid_region[key] * 1000, lbs_region[key] * 1000]
        print(f"  {label:16s}" + "".join(f" {value:10.1f}" for value in values))
        for key in ("through_skin", "outside"):
            difference = lbs_region[key] - rigid_region[key]
            if difference < -5e-4:
                improved[key] += 1
            elif difference > 5e-4:
                worsened[key] += 1
    region_count = len(rigid["regions"])
    for key, title in (("through_skin", "through skin"), ("outside", "outside")):
        print(f"  {title}: LBS better by > 0.5 mm in {improved[key]} of {region_count} regions, "
              f"worse in {worsened[key]}")
    for key, title in (
        ("through_skin_max", "through skin, worst"),
        ("through_skin_mean_over_poses", "through skin, mean over poses of the pose's worst"),
        ("outside_max", "outside, worst"),
        ("outside_p95", "outside, p95 over vertices"),
        ("outside_median", "outside, median over vertices"),
        ("animation_through_skin_max", "animation term through skin, max"),
        ("animation_through_skin_median", "animation term through skin, median"),
        ("animation_outside_max", "animation term outside, max"),
        ("animation_outside_median", "animation term outside, median"),
    ):
        print(f"  {title:52s} rigid {rigid[key] * 1000:7.1f}   LBS {lbs[key] * 1000:7.1f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--genders", nargs="+", default=list(GENDERS), choices=list(GENDERS))
    parser.add_argument("--json", type=Path, help="also write the numbers to this JSON file")
    parser.add_argument("--single-gender-output", type=Path, help=argparse.SUPPRESS)
    arguments = parser.parse_args()

    if arguments.single_gender_output is not None:
        set_non_interactive(True)
        app = QApplication.instance() or QApplication(sys.argv)
        results = measure_gender(arguments.genders[0], app)
        arguments.single_gender_output.write_text(json.dumps(results))
        app.quit()
        return 0

    all_results = []
    with tempfile.TemporaryDirectory() as temporary_directory:
        for gender in arguments.genders:
            single_gender_output = Path(temporary_directory) / f"{gender}.json"
            subprocess.run(
                [sys.executable, __file__, "--genders", gender, "--single-gender-output", str(single_gender_output)],
                check=True,
            )
            all_results.append(json.loads(single_gender_output.read_text()))
    for results in all_results:
        print_report(results)
    if arguments.json is not None:
        arguments.json.write_text(json.dumps(all_results, indent=2))
        print(f"\nWrote {arguments.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
