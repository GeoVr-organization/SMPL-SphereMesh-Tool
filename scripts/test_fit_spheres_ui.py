"""End-to-end check that the Fit Spheres button path works against a real MainWindow.

Builds the application, loads the male template, drives SphereManager.fit_spheres_to_mesh
with the same arguments the dialog produces, formats the report the button would show, and
finally clicks the button handler itself.

Runs with ui_widgets.set_non_interactive(True) so no modal waits for a human. Without it the
save-on-close confirmation blocks the event loop forever and the script only appears to hang.

Run with: python scripts/test_fit_spheres_ui.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.smpl_models import MALE, load_smpl_model  # noqa: E402
from src.ui_widgets import FitSpheresDialog, MainWindow, set_non_interactive  # noqa: E402


def main() -> int:
    set_non_interactive(True)
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow(load_smpl_model(MALE), MALE)
    manager = window.sphere_manager

    print(f"Loaded {manager.get_sphere_count()} spheres, {len(manager.connections)} connections")

    dialog = FitSpheresDialog(window)
    options = dialog.get_options()
    print(f"Dialog defaults: {options}")

    mesh = pv.wrap(window.mesh_actor.GetMapper().GetInput())
    before_centers = np.array([s.get_center() for s in manager.spheres])
    before_radii = np.array([s.get_radius() for s in manager.spheres])

    manager.spheres[0].set_locked(True)
    info = manager.fit_spheres_to_mesh(mesh, **options)
    assert info is not None, "fit returned None"

    after_centers = np.array([s.get_center() for s in manager.spheres])
    after_radii = np.array([s.get_radius() for s in manager.spheres])

    failures = []

    def check(condition: bool, label: str, detail: str = "") -> None:
        print(f"  [{'PASS' if condition else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))
        if not condition:
            failures.append(label)

    print("\nChecks")
    check(len(after_centers) == len(before_centers), "sphere count unchanged")
    check(
        np.array_equal(after_centers[0], before_centers[0])
        and after_radii[0] == before_radii[0],
        "locked sphere untouched",
    )
    check(not np.allclose(after_centers[1:], before_centers[1:]), "unlocked spheres moved")
    displacement = np.linalg.norm(after_centers - before_centers, axis=1)
    check(
        displacement.max() <= options["max_offset"] + 1e-12,
        "no centre exceeded the cap",
        f"max {displacement.max() * 1000:.3f} mm",
    )
    check(after_radii.min() >= options["min_radius"] - 1e-12, "no radius below the floor")
    check(manager.has_unsaved_changes, "unsaved-changes flag set")
    check(info["locked_skipped"] == 1, "one locked sphere reported")
    energies = [it["E"] for it in info["iterations"]]
    check(
        all(energies[i] <= energies[i - 1] for i in range(1, len(energies))),
        "E monotone",
        f"{len(energies)} iterations",
    )
    for sphere in manager.spheres:
        if sphere.get_parent_joint() is None:
            check(False, "parent joints preserved")
            break
    else:
        check(True, "parent joints preserved")

    print("\nReport the button would show:\n")
    print(window._format_fit_report(info))

    print("\nDriving the button handler itself (non-interactive mode)")
    window.fit_spheres_clicked()
    check(True, "fit_spheres_clicked returns without blocking on a modal")

    window.close()
    app.quit()

    print()
    if failures:
        print(f"{len(failures)} FAILURE(S): " + ", ".join(failures))
        return 1
    print("UI path OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
