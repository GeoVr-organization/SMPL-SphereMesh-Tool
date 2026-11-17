"""End-to-end check of the InfoPanel Delete button on a selected cone or triangle, on a real MainWindow.

Covers: Delete enabled while a connection is selected, the connection removed by object
(a duplicate with the same indices survives), its actors gone from the renderer, spheres and
other connections untouched, the selection and InfoPanel reset, the unsaved flag set, and a
selected sphere still deleted as before.

Runs with ui_widgets.set_non_interactive(True) so no modal waits for a human.

Run with: python scripts/test_delete_connection.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.smpl_models import MALE, load_smpl_model  # noqa: E402
from src.ui_widgets import MainWindow, set_non_interactive  # noqa: E402

failures: list[str] = []


def check(condition: bool, label: str, detail: str = "") -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))
    if not condition:
        failures.append(label)


def connection_actors(connection) -> list:
    """Every actor a connection owns (thin connector plus spheremesh actors)."""
    owned_actors = list(connection._actors)
    if connection._connector_actor is not None:
        owned_actors.append(connection._connector_actor)
    return owned_actors


def renderer_actor_ids(window) -> set[int]:
    """Identity of every actor currently in the plotter's renderer."""
    return {id(renderer_actor) for renderer_actor in window.plotter.renderer.actors.values()}


def delete_selected_connection_and_check(window, app, connection, label: str) -> None:
    """Select a connection, press Delete in the InfoPanel and check what changed."""
    manager = window.sphere_manager
    info_panel = window.info_panel
    description = manager.describe_connection(connection)

    check(manager.select_connection(connection), f"{label}: {description} selectable")
    app.processEvents()
    check(info_panel.delete_button.isEnabled(), f"{label}: Delete enabled with a connection selected")

    sphere_count_before = manager.get_sphere_count()
    sphere_centers_before = [sphere.get_center().copy() for sphere in manager.spheres]
    other_connections_before = [other for other in manager.connections if other is not connection]
    other_tuples_before = [tuple(other.sphere_indices) for other in other_connections_before]
    owned_actors = connection_actors(connection)
    check(
        bool(owned_actors) and all(id(actor) in renderer_actor_ids(window) for actor in owned_actors),
        f"{label}: its actors are in the renderer before",
        f"{len(owned_actors)} actors",
    )
    manager.has_unsaved_changes = False

    info_panel.delete_button.click()
    app.processEvents()

    check(connection not in manager.connections, f"{label}: connection removed")
    check(
        manager.connections == other_connections_before
        and [tuple(other.sphere_indices) for other in manager.connections] == other_tuples_before,
        f"{label}: every other connection kept, same order and indices",
    )
    check(manager.get_sphere_count() == sphere_count_before, f"{label}: no sphere deleted")
    check(
        all(
            (sphere.get_center() == center_before).all()
            for sphere, center_before in zip(manager.spheres, sphere_centers_before)
        ),
        f"{label}: no sphere moved",
    )
    remaining_actor_ids = renderer_actor_ids(window)
    check(
        not any(id(actor) in remaining_actor_ids for actor in owned_actors),
        f"{label}: its actors removed from the renderer",
    )
    check(
        manager.get_selected_connection() is None and manager.get_selected_sphere_index() is None,
        f"{label}: nothing selected afterwards",
    )
    check(info_panel.spheremesh_id_label.text() == "Spheremesh: None", f"{label}: InfoPanel reset")
    check(not info_panel.delete_button.isEnabled(), f"{label}: Delete disabled again")
    check(not window.divide_connection_button.isEnabled(), f"{label}: Divide disabled again")
    check(manager.has_unsaved_changes, f"{label}: marked unsaved")


def main() -> int:
    set_non_interactive(True)
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow(load_smpl_model(MALE), MALE)
    window.resize(1400, 900)
    window.show()
    app.processEvents()
    manager = window.sphere_manager
    info_panel = window.info_panel
    print(f"Loaded {manager.get_sphere_count()} spheres, {len(manager.connections)} connections")

    cones = [connection for connection in manager.connections if len(connection.sphere_indices) == 2]
    triangles = [connection for connection in manager.connections if len(connection.sphere_indices) == 3]
    check(len(cones) >= 2 and len(triangles) >= 1, "template has cones and triangles to delete")

    print("\nNothing selected")
    # The window starts with a sphere selected; clear_connection_selection refreshes the panel.
    manager.selected_sphere_index = None
    manager.clear_connection_selection()
    app.processEvents()
    check(not info_panel.delete_button.isEnabled(), "Delete disabled without a selection")
    sphere_count = manager.get_sphere_count()
    connection_count = len(manager.connections)
    info_panel._on_delete_clicked()
    check(
        manager.get_sphere_count() == sphere_count and len(manager.connections) == connection_count,
        "Delete handler without a selection does nothing",
    )

    print("\nCone")
    delete_selected_connection_and_check(window, app, cones[0], "cone")

    print("\nTriangle")
    delete_selected_connection_and_check(window, app, triangles[0], "triangle")

    print("\nDuplicate cone: the selected copy goes, the original stays")
    original_cone = cones[1]
    original_indices = tuple(original_cone.sphere_indices)
    check(manager.try_add_connection(original_indices), "duplicate cone added")
    duplicate_cone = manager.connections[-1]
    check(
        duplicate_cone is not original_cone and tuple(duplicate_cone.sphere_indices) == original_indices,
        "duplicate is a separate object with the same indices",
    )
    delete_selected_connection_and_check(window, app, duplicate_cone, "duplicate")
    check(original_cone in manager.connections, "duplicate: the original cone survives")

    print("\nSphere (unchanged behaviour)")
    manager.select_next_sphere()
    app.processEvents()
    selected_sphere = manager.spheres[manager.get_selected_sphere_index()]
    sphere_count = manager.get_sphere_count()
    check(info_panel.delete_button.isEnabled(), "Delete enabled with a sphere selected")
    info_panel.delete_button.click()
    app.processEvents()
    check(
        manager.get_sphere_count() == sphere_count - 1 and selected_sphere not in manager.spheres,
        "Delete on a sphere still deletes the sphere",
    )

    window.close()
    app.quit()

    print()
    if failures:
        print(f"{len(failures)} FAILURE(S): " + ", ".join(failures))
        return 1
    print("Delete connection OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
