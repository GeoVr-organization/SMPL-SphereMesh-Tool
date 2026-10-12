"""End-to-end check of the color schemes (Class Colors, Random Colors, Reset Colors).

Class Colors paints spheres red, capsuloids (cones) blue and prysmoids (triangles) violet,
in the colours of SMFitting's views, and is the scheme the tool starts in.
Checks that the toolbar handlers recolor sphere widgets, connectors and spheremesh actors;
that selection and lock colors still win; and that spheres and connections created later
(Add Sphere, Add Connection, Divide, loading a file) follow the active scheme.

Runs with ui_widgets.set_non_interactive(True) so no modal waits for a human.

Run with: python scripts/test_color_schemes.py
"""
from __future__ import annotations

import ast
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.smpl_models import MALE, load_smpl_model  # noqa: E402
from src.sphere_manager import (  # noqa: E402
    CAPSULOID_CLASS_COLOR,
    PRYSMOID_CLASS_COLOR,
    SPHERE_CLASS_COLOR,
)
from src.ui_widgets import MainWindow, set_non_interactive  # noqa: E402

failures: list[str] = []


def check(condition: bool, label: str, detail: str = "") -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))
    if not condition:
        failures.append(label)


def same_color(actual_rgb, expected_color) -> bool:
    return bool(np.allclose(actual_rgb, pv.Color(expected_color).float_rgb, atol=1e-6))


def sphere_color(sphere):
    return sphere.widget.GetSphereProperty().GetColor()


def connector_color(connection):
    return connection._connector_actor.GetProperty().GetColor()


def spheremesh_color(connection):
    return connection._actors[0].GetProperty().GetColor()


def plain_spheres(manager):
    """Spheres shown in their scheme color: neither selected nor locked."""
    return [
        sphere for index, sphere in enumerate(manager.spheres)
        if index != manager.get_selected_sphere_index() and not sphere.locked
    ]


def plain_connections(manager):
    """Connections whose spheremesh shows its scheme color: not selected, first sphere unlocked."""
    return [
        connection for connection in manager.connections
        if connection is not manager.get_selected_connection()
        and not manager.spheres[connection.sphere_indices[0]].locked
        and connection._connector_actor is not None
    ]


def smfitting_primitive_colors() -> dict:
    """primitiveColors from SMFitting/PlotlyView.py, read without importing plotly."""
    plotly_view_path = Path(__file__).resolve().parent.parent / "SMFitting" / "PlotlyView.py"
    for node in ast.parse(plotly_view_path.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "primitiveColors" for target in node.targets
        ):
            return ast.literal_eval(node.value)
    return {}


def expected_class_color(connection) -> str:
    return CAPSULOID_CLASS_COLOR if len(connection.sphere_indices) == 2 else PRYSMOID_CLASS_COLOR


def check_class_colors(manager, label: str) -> None:
    check(
        all(same_color(sphere_color(sphere), SPHERE_CLASS_COLOR) for sphere in plain_spheres(manager)),
        f"{label}: spheres are red",
    )
    connections = plain_connections(manager)
    check(
        all(same_color(connector_color(c), expected_class_color(c)) for c in connections),
        f"{label}: cone connectors blue, triangle connectors violet",
    )
    shown = [c for c in connections if c._actors and c._actors[0].GetVisibility()]
    check(
        bool(shown) and all(same_color(spheremesh_color(c), expected_class_color(c)) for c in shown),
        f"{label}: capsuloids blue, prysmoids violet",
        f"{len(shown)} spheremeshes shown",
    )


def main() -> int:
    set_non_interactive(True)
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow(load_smpl_model(MALE), MALE)
    manager = window.sphere_manager
    manager.set_spheremesh_visibility(True)
    cone_count = sum(1 for c in manager.connections if len(c.sphere_indices) == 2)
    print(f"Loaded {manager.get_sphere_count()} spheres, {cone_count} cones, "
          f"{len(manager.connections) - cone_count} triangles")

    print("\nStartup scheme")
    notebook_colors = smfitting_primitive_colors()
    tool_colors = (SPHERE_CLASS_COLOR, CAPSULOID_CLASS_COLOR, PRYSMOID_CLASS_COLOR)
    check(
        tool_colors == tuple(notebook_colors.get(name) for name in
                             ("Sphere Class", "Capsuloid Class", "Prysmoid Class")),
        "class colors are SMFitting's primitiveColors",
        ", ".join(tool_colors),
    )
    check(manager.color_scheme == "class", "starts on the class scheme")
    check_class_colors(manager, "at startup")

    print("\nReset Colors from startup")
    window.reset_colors_clicked()
    check(manager.color_scheme == "default", "scheme switched to default")
    check(
        all(same_color(sphere_color(s), manager.normal_color) for s in plain_spheres(manager)),
        "spheres light blue",
    )
    check(
        all(same_color(connector_color(c), "gray") for c in plain_connections(manager)),
        "connectors gray",
    )

    print("\nClass Colors")
    window.class_colors_clicked()
    check(manager.color_scheme == "class", "scheme switched to class")
    check_class_colors(manager, "after the button")
    selected_index = manager.get_selected_sphere_index()
    check(
        selected_index is not None
        and same_color(sphere_color(manager.spheres[selected_index]), manager.selected_color),
        "the selected sphere stays cyan",
    )

    print("\nSelection and lock still win")
    triangle = next(c for c in manager.connections if len(c.sphere_indices) == 3)
    manager.select_connection(triangle)
    check(same_color(connector_color(triangle), manager.selected_color), "a selected triangle turns cyan")
    manager.clear_connection_selection()
    check(same_color(connector_color(triangle), PRYSMOID_CLASS_COLOR), "and turns violet again")
    lock_index = triangle.sphere_indices[0]
    manager.spheres[lock_index].set_locked(True)
    check(same_color(sphere_color(manager.spheres[lock_index]), manager.locked_color), "a locked sphere is yellow")
    manager.spheres[lock_index].set_locked(False)
    check(same_color(sphere_color(manager.spheres[lock_index]), SPHERE_CLASS_COLOR), "and red once unlocked")

    print("\nNew spheres and connections follow the scheme")
    new_sphere_index = manager.add_sphere([0.0, 0.3, 0.6], 0.05)
    manager.select_next_sphere()  # deselect it so its own color shows
    check(same_color(sphere_color(manager.spheres[new_sphere_index]), SPHERE_CLASS_COLOR), "Add Sphere: red")
    cone_indices = next(tuple(c.sphere_indices) for c in manager.connections if len(c.sphere_indices) == 2)
    check(manager.try_add_connection(cone_indices[::-1]), "Add Connection (cone) succeeds")
    check(same_color(connector_color(manager.connections[-1]), CAPSULOID_CLASS_COLOR), "Add Connection: cone blue")
    check(manager.try_add_connection(tuple(triangle.sphere_indices)[::-1]), "Add Connection (triangle) succeeds")
    check(same_color(connector_color(manager.connections[-1]), PRYSMOID_CLASS_COLOR), "Add Connection: triangle violet")

    manager.select_connection(triangle)
    window.divide_connection_clicked()
    divided_index = manager.get_sphere_count() - 1
    manager.select_next_sphere()
    check(same_color(sphere_color(manager.spheres[divided_index]), SPHERE_CLASS_COLOR), "Divide: new sphere red")
    divided_pieces = [c for c in manager.connections if divided_index in c.sphere_indices]
    check(
        len(divided_pieces) >= 3
        and all(same_color(connector_color(c), expected_class_color(c)) for c in divided_pieces),
        "Divide: pieces keep their class color",
    )
    check_class_colors(manager, "after the edits")

    print("\nLoading keeps the scheme")
    with tempfile.TemporaryDirectory() as save_directory:
        save_path = Path(save_directory) / "colored.json"
        manager.save_to_json(str(save_path))
        manager.load_from_json(str(save_path))
    manager.set_spheremesh_visibility(True)
    check(manager.color_scheme == "class", "scheme survives a load")
    check_class_colors(manager, "after loading")

    print("\nRandom Colors")
    window.randomize_colors_clicked()
    check(manager.color_scheme == "random", "scheme switched to random")
    check(
        all(sphere.get_random_color() is not None for sphere in manager.spheres)
        and all(c.get_random_color() is not None for c in manager.connections),
        "every sphere and connection has a random color",
    )
    random_sphere_index = manager.add_sphere([0.0, 0.3, 0.7], 0.05)
    check(manager.spheres[random_sphere_index].get_random_color() is not None, "a new sphere gets one too")

    print("\nReset Colors")
    window.reset_colors_clicked()
    check(manager.color_scheme == "default", "scheme back to default")
    check(
        all(sphere.get_random_color() is None for sphere in manager.spheres)
        and all(c.get_random_color() is None for c in manager.connections),
        "no overrides left",
    )
    check(
        all(same_color(sphere_color(s), manager.normal_color) for s in plain_spheres(manager))
        and all(same_color(connector_color(c), "gray") for c in plain_connections(manager)),
        "spheres light blue and connectors gray again",
    )
    default_sphere_index = manager.add_sphere([0.0, 0.3, 0.8], 0.05)
    check(manager.spheres[default_sphere_index].get_random_color() is None, "a new sphere stays default")

    window.close()
    app.quit()

    print()
    if failures:
        print(f"{len(failures)} FAILURE(S): " + ", ".join(failures))
        return 1
    print("Color schemes OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
