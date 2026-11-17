"""End-to-end check of Ctrl+click connection creation on a real MainWindow.

Ctrl+click collects 2 or 3 spheres (a plain-clicked sphere seeds the set); releasing Ctrl
opens a dialog that creates the cone or triangle, optionally with its X mirror, or discards
the set. Covers the set rules (toggle, at most 3, Ctrl+click never selects a connection),
the redundancy rules (a duplicate or a cone on a triangle edge is refused; a new triangle
removes the cones on its edges, duplicates included, without changing the shape), the
mirror checkbox, the preview, Ctrl released before the mouse button, focus loss, the real
modal with Enter and Esc, the Add Connection… path, and clip planes.

Clicks and key releases are simulated (VTK event injection, QTest), not physical input.
Runs with ui_widgets.set_non_interactive(True) except for the one real-modal section.

Run with: python scripts/test_ctrl_select_connection.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from PyQt6.QtCore import QEvent, Qt, QTimer  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.smpl_models import MALE, load_smpl_model  # noqa: E402
from src.ui_widgets import CreateConnectionDialog, MainWindow, set_non_interactive  # noqa: E402
from test_divide_connection import (  # noqa: E402
    click,
    display_position,
    sample_points,
    snapshot_primitives,
    union_distance,
)

GEOMETRY_TOLERANCE = 1e-9
CYAN = (0.0, 1.0, 1.0)
failures: list[str] = []


def check(condition: bool, label: str, detail: str = "") -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))
    if not condition:
        failures.append(label)


# Spheres 0-5 and 9-10 are X twins, 6 and 8 sit on the midline, 7 has no twin.
SPHERE_ROWS = [
    ([-0.60, 0.00, 0.0], 0.04),   # 0
    ([0.60, 0.00, 0.0], 0.04),    # 1  twin of 0
    ([-0.30, 0.00, 0.0], 0.04),   # 2
    ([0.30, 0.00, 0.0], 0.04),    # 3  twin of 2
    ([-0.45, 0.30, 0.0], 0.04),   # 4
    ([0.45, 0.30, 0.0], 0.04),    # 5  twin of 4
    ([0.00, 0.50, 0.0], 0.04),    # 6  midline
    ([-0.45, -0.35, 0.0], 0.04),  # 7  no twin
    ([0.00, -0.30, 0.0], 0.04),   # 8  midline
    ([-0.15, -0.60, 0.0], 0.04),  # 9
    ([0.15, -0.60, 0.0], 0.04),   # 10 twin of 9
]


def payload() -> dict:
    """Cone (0, 2) twice, its mirror (1, 3), and triangle (8, 9, 10)."""
    return {
        "version": 2,
        "spheres": [{"center": center, "radius": radius} for center, radius in SPHERE_ROWS],
        "connections": [
            {"indices": [0, 2]},
            {"indices": [2, 0]},
            {"indices": [1, 3]},
            {"indices": [8, 9, 10]},
        ],
    }


def connection_tuples(manager) -> list[tuple[int, ...]]:
    """Every connection's sphere indices, in list order."""
    return [tuple(connection.sphere_indices) for connection in manager.connections]


def find_connection(manager, sphere_indices):
    """The first connection with exactly these indices in this order, or None."""
    return next(
        (connection for connection in manager.connections if tuple(connection.sphere_indices) == tuple(sphere_indices)),
        None,
    )


def actor_count(window) -> int:
    """Number of actors in the plotter's renderer."""
    return window.plotter.renderer.GetActors().GetNumberOfItems()


def sphere_is_cyan(manager, sphere_index) -> bool:
    """Whether the sphere is drawn in the selection colour."""
    return bool(np.allclose(manager.spheres[sphere_index].widget.GetSphereProperty().GetColor(), CYAN))


def actors_of(connection) -> int:
    """Number of actors a connection holds (connector, plus its surface once built)."""
    return (connection._connector_actor is not None) + len(connection._actors)


def live_create_connection_dialogs(window) -> int:
    """Create-connection dialogs still alive after deferred deletions ran."""
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    return len(window.findChildren(CreateConnectionDialog))


def reset_view(window, app) -> None:
    """Head-on XY view; resetting the camera also fixes the clipping range picks rely on."""
    window.resize(1400, 900)
    app.processEvents()
    window.plotter.view_xy()
    window.plotter.reset_camera(bounds=(-0.8, 0.8, -0.8, 0.8, -0.3, 0.3))
    window.plotter.render()
    app.processEvents()


class DialogAnswerer:
    """Stands in for the modal: records what the dialog showed, then answers Create or Discard."""

    def __init__(self, window):
        self.window = window
        self.records: list[dict] = []
        self.create = False
        self.mirror = None

    def answer(self, create: bool, mirror=None) -> None:
        """Set the next answer: Create or Discard, and optionally the mirror checkbox state."""
        self.create = create
        self.mirror = mirror

    def __call__(self, dialog) -> bool:
        manager = self.window.sphere_manager
        candidate = dialog.candidate
        preview_connection = candidate.connection
        mirror_preview_before = None
        if candidate.mirror_connection is not None:
            mirror_preview_before = bool(candidate.mirror_connection._actors[0].GetVisibility())
        mirror_checked_initially = None
        if dialog.mirror_checkbox is not None:
            mirror_checked_initially = dialog.mirror_checkbox.isChecked()
            if self.mirror is not None:
                dialog.mirror_checkbox.setChecked(self.mirror)
        record = {
            "question": dialog.question_label.text(),
            "blocking": dialog.blocking_label.text() if dialog.blocking_label else None,
            "removal": dialog.removal_label.text() if dialog.removal_label else None,
            "mirror_text": dialog.mirror_checkbox.text() if dialog.mirror_checkbox else None,
            "mirror_checked_initially": mirror_checked_initially,
            "mirror_note": dialog.mirror_note_label.text() if dialog.mirror_note_label else None,
            "create_enabled": dialog.create_button.isEnabled(),
            "create_text": dialog.create_button.text(),
            "connections_during": connection_tuples(manager),
            "preview_registered": preview_connection is not None and preview_connection in manager.connections,
            "preview_visible": (
                preview_connection is not None
                and bool(preview_connection._actors[0].GetVisibility())
                and bool(preview_connection._connector_actor.GetVisibility())
            ),
            "preview_cyan": (
                preview_connection is not None
                and np.allclose(preview_connection._actors[0].GetProperty().GetColor(), CYAN)
            ),
            "mirror_preview_before": mirror_preview_before,
            "mirror_preview_after_toggle": (
                None if candidate.mirror_connection is None
                else bool(candidate.mirror_connection._actors[0].GetVisibility())
            ),
            "pending_during": manager.get_pending_connection_sphere_indices(),
        }
        self.records.append(record)
        return self.create and dialog.create_button.isEnabled()


def release_control(app, receiver) -> None:
    """Send a Ctrl key release to a widget and run the deferred dialog it schedules."""
    QTest.keyRelease(receiver, Qt.Key.Key_Control)
    app.processEvents()
    app.processEvents()


def main() -> int:
    set_non_interactive(True)
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow(load_smpl_model(MALE), MALE)
    window.show()
    app.processEvents()
    manager = window.sphere_manager
    random_generator = np.random.default_rng(11)

    with tempfile.TemporaryDirectory() as payload_directory:
        payload_path = Path(payload_directory) / "ctrl_select.json"
        payload_path.write_text(json.dumps(payload()))
        manager.load_from_json(str(payload_path))
    reset_view(window, app)
    answerer = DialogAnswerer(window)
    window._exec_create_connection_dialog = answerer
    position_of = {i: display_position(window, center) for i, (center, _) in enumerate(SPHERE_ROWS)}
    check(manager.spheremesh_visible is False, "setup: spheremeshes hidden (preview must show anyway)")

    print("\nCollecting spheres")
    click(window, app, position_of[0])
    check(manager.get_selected_sphere_index() == 0, "plain click selects sphere 0")
    renderers_before = [s.widget.GetCurrentRenderer() for s in manager.spheres]
    click(window, app, position_of[2], control_key=True)
    check(manager.get_pending_connection_sphere_indices() == [0, 2], "Ctrl+click adds sphere 2 to the selected 0",
          str(manager.get_pending_connection_sphere_indices()))
    check(sphere_is_cyan(manager, 0) and sphere_is_cyan(manager, 2), "both spheres are cyan")
    check(not sphere_is_cyan(manager, 4), "other spheres are not")
    check(manager.get_selected_sphere_index() == 2 and manager.get_selected_connection() is None,
          "the last clicked sphere is the selected one, no connection selected")
    check(window.info_panel.spheremesh_id_label.text() == "Sphere #2", "InfoPanel shows sphere 2",
          window.info_panel.spheremesh_id_label.text())
    check([s.widget.GetCurrentRenderer() for s in manager.spheres] == renderers_before,
          "sphere widgets restored after Ctrl+click")
    check(manager.spheres[2].get_center()[0] == -0.30, "Ctrl+click does not move the sphere")

    click(window, app, display_position(window, [-0.45, 0.0, 0.0]), control_key=True)
    check(manager.get_selected_connection() is None, "Ctrl+click on cone (0, 2) does not select it")
    check(manager.get_pending_connection_sphere_indices() == [0, 2], "... and leaves the set alone")
    click(window, app, display_position(window, [0.0, 0.15, 0.0]), control_key=True)
    check(manager.get_pending_connection_sphere_indices() == [0, 2], "Ctrl+click on empty space changes nothing")

    click(window, app, position_of[4], control_key=True)
    check(manager.get_pending_connection_sphere_indices() == [0, 2, 4], "third sphere added")
    click(window, app, position_of[6], control_key=True)
    check(manager.get_pending_connection_sphere_indices() == [0, 2, 4], "a fourth sphere is ignored")
    check(not sphere_is_cyan(manager, 6), "... and not highlighted")
    click(window, app, position_of[2], control_key=True)
    check(manager.get_pending_connection_sphere_indices() == [0, 4], "Ctrl+click on a collected sphere removes it")
    check(not sphere_is_cyan(manager, 2), "... and un-highlights it")
    click(window, app, position_of[2], control_key=True)
    check(manager.get_pending_connection_sphere_indices() == [0, 4, 2], "and adds it back at the end")

    print("\nTriangle with mirror, replacing redundant cones")
    old_connections = list(manager.connections)
    actors_before = actor_count(window)
    # Loaded connections build their surface actor only once spheremeshes are shown.
    actors_per_connection_before = {id(connection): actors_of(connection) for connection in old_connections}
    manager.clear_unsaved_changes()  # the plain click on sphere 0 already counted as an edit
    answerer.answer(create=True)
    release_control(app, window.plotter)
    check(len(answerer.records) == 1, "Ctrl release opens the dialog once", f"{len(answerer.records)} dialog(s)")
    record = answerer.records[-1]
    check(record["question"] == "Create a triangle between spheres 0, 4 and 2?", "dialog asks about the triangle",
          record["question"])
    check(record["create_text"] == "Create triangle (0, 4, 2)" and record["create_enabled"],
          "Create button names the triangle")
    check(record["removal"] == "It replaces the redundant cone (0, 2), cone (2, 0) on its edges.",
          "dialog lists the redundant cones, the duplicate too", str(record["removal"]))
    check(record["mirror_text"] == "Also create the mirrored triangle (1, 5, 3) (replacing the redundant cone (1, 3))",
          "mirror checkbox names the mirrored triangle and its redundant cone", str(record["mirror_text"]))
    check(record["mirror_checked_initially"] is True, "mirror ticked because Mirror edits is on")
    check(record["connections_during"] == [(0, 2), (2, 0), (1, 3), (8, 9, 10)], "nothing registered while asking")
    check(not record["preview_registered"] and record["preview_visible"] and record["preview_cyan"],
          "preview drawn cyan, unregistered, with Show Spheremeshes off")
    check(record["mirror_preview_before"] is True, "mirror preview drawn too")

    tuples_after = connection_tuples(manager)
    check(tuples_after == [(8, 9, 10), (0, 4, 2), (1, 5, 3)], "triangle and mirror added, cones removed",
          str(tuples_after))
    new_triangle = find_connection(manager, (0, 4, 2))
    mirror_triangle = find_connection(manager, (1, 5, 3))
    check(manager.get_selected_connection() is new_triangle, "the new triangle is selected")
    check(not mirror_triangle._selected, "the mirror is not")
    check(manager.get_pending_connection_sphere_indices() == [], "the set is cleared")
    check(not sphere_is_cyan(manager, 0) and not sphere_is_cyan(manager, 4), "spheres lose the cyan")
    check(all(not c._spheremesh_visible for c in (new_triangle, mirror_triangle)),
          "new connections follow Show Spheremeshes (off)")
    check(manager.has_unsaved_changes, "marked unsaved")
    check(window.divide_connection_button.isEnabled(), "Divide is enabled for the new triangle")
    check(window.info_panel.spheremesh_id_label.text() == "Triangle (0, 4, 2)", "InfoPanel names the triangle",
          window.info_panel.spheremesh_id_label.text())
    removed_connections = [connection for connection in old_connections if connection not in manager.connections]
    check([tuple(connection.sphere_indices) for connection in removed_connections] == [(0, 2), (2, 0), (1, 3)],
          "exactly the three edge cones were removed")
    check(all(actors_of(connection) == 0 for connection in removed_connections), "their actors are gone")
    removed_actor_count = sum(actors_per_connection_before[id(connection)] for connection in removed_connections)
    added_actor_count = actors_of(new_triangle) + actors_of(mirror_triangle)
    check(actor_count(window) == actors_before - removed_actor_count + added_actor_count,
          "renderer actors: the removed cones' out, the new triangles' in, no preview left over",
          f"{actors_before} - {removed_actor_count} + {added_actor_count} = {actor_count(window)}")
    current_primitives = snapshot_primitives(manager, manager.connections)
    primitives_with_cones = current_primitives + [
        (
            np.array([manager.spheres[i].get_center() for i in cone]),
            np.array([manager.spheres[i].get_radius() for i in cone]),
        )
        for cone in [(0, 2), (1, 3)]
    ]
    points = sample_points(primitives_with_cones, random_generator)
    largest_difference = float(
        np.max(np.abs(union_distance(points, primitives_with_cones) - union_distance(points, current_primitives)))
    )
    check(largest_difference < GEOMETRY_TOLERANCE, "removing the edge cones leaves the union unchanged",
          f"max |d_with - d_without| = {largest_difference:.2e} over {len(points)} points")

    print("\nDiscard")
    click(window, app, position_of[7], control_key=True)
    check(manager.get_pending_connection_sphere_indices() == [7],
          "with a connection selected, Ctrl+click starts a fresh set", str(manager.get_pending_connection_sphere_indices()))
    check(manager.get_selected_connection() is None, "... and drops the connection selection")
    click(window, app, position_of[8], control_key=True)
    actors_before = actor_count(window)
    tuples_before = connection_tuples(manager)
    manager.clear_unsaved_changes()
    answerer.answer(create=False)
    release_control(app, window.plotter)
    record = answerer.records[-1]
    check(record["question"] == "Create a cone between spheres 7 and 8?", "dialog asks about cone (7, 8)", record["question"])
    check(record["mirror_text"] is None and record["mirror_note"] == "No mirror: sphere 7 has no X twin within 20 mm.",
          "no mirror checkbox; the note names the sphere without a twin", str(record["mirror_note"]))
    check(record["removal"] is None, "a cone removes nothing")
    check(connection_tuples(manager) == tuples_before, "Discard adds nothing")
    check(not manager.has_unsaved_changes, "Discard is not an edit")
    check(actor_count(window) == actors_before, "Discard removes the preview actors",
          f"{actors_before} -> {actor_count(window)}")
    check(manager.get_selected_sphere_index() is None and manager.get_selected_connection() is None,
          "Discard leaves nothing selected")
    check(manager.get_pending_connection_sphere_indices() == [] and not sphere_is_cyan(manager, 7), "set cleared")

    print("\nRefused: already covered")
    click(window, app, position_of[9], control_key=True)
    click(window, app, position_of[10], control_key=True)
    answerer.answer(create=True)
    release_control(app, window.plotter)
    record = answerer.records[-1]
    check(record["blocking"] == "Triangle (8, 9, 10) already contains this cone as an edge.",
          "a cone on a triangle edge is refused", str(record["blocking"]))
    check(not record["create_enabled"], "Create disabled")
    check(record["mirror_text"] is None, "no mirror offered for a refused connection")
    check(connection_tuples(manager) == tuples_before, "nothing added")
    for sphere_index in (8, 9, 10):
        click(window, app, position_of[sphere_index], control_key=True)
    release_control(app, window.plotter)
    record = answerer.records[-1]
    check(record["blocking"] == "It already exists: triangle (8, 9, 10).", "a duplicate triangle is refused",
          str(record["blocking"]))
    check(connection_tuples(manager) == tuples_before, "nothing added")
    check(not manager.has_unsaved_changes, "a refusal is not an edit")
    check(live_create_connection_dialogs(window) == 0, "no dialog outlives its question",
          f"{len(answerer.records)} dialogs shown so far")

    print("\nMirror unticked, then the mirror of an existing cone")
    click(window, app, position_of[4])
    click(window, app, position_of[6], control_key=True)
    answerer.answer(create=True, mirror=False)
    release_control(app, window.plotter)
    record = answerer.records[-1]
    check(record["mirror_text"] == "Also create the mirrored cone (5, 6)", "midline sphere 6 is its own twin",
          str(record["mirror_text"]))
    check(record["mirror_preview_after_toggle"] is False, "unticking hides the mirror preview")
    check(find_connection(manager, (4, 6)) is not None and find_connection(manager, (5, 6)) is None,
          "only cone (4, 6) added")
    click(window, app, position_of[5])
    click(window, app, position_of[6], control_key=True)
    answerer.answer(create=True)
    release_control(app, window.plotter)
    record = answerer.records[-1]
    check(record["mirror_text"] is None and record["mirror_note"] == "The mirrored cone (4, 6) already exists.",
          "the existing mirror is named, not offered", str(record["mirror_note"]))
    check(find_connection(manager, (5, 6)) is not None, "cone (5, 6) added")

    print("\nA single sphere")
    dialog_count = len(answerer.records)
    check(manager.get_selected_connection() is find_connection(manager, (5, 6)), "cone (5, 6) is selected")
    manager.clear_unsaved_changes()
    click(window, app, position_of[7], control_key=True)  # a selected cone doesn't seed the set
    release_control(app, window.plotter)
    check(len(answerer.records) == dialog_count, "no dialog for one sphere")
    check(not manager.has_unsaved_changes, "collecting one sphere is not an edit")
    check(manager.get_selected_sphere_index() == 7 and manager.get_pending_connection_sphere_indices() == [],
          "sphere 7 stays selected, set cleared")

    print("\nCtrl released before the mouse button")
    click(window, app, position_of[0])
    render_window_interactor = window.plotter.iren.interactor
    render_window_interactor.SetEventInformation(position_of[1][0], position_of[1][1], 1, 0)
    render_window_interactor.LeftButtonPressEvent()
    release_control(app, window.plotter)
    check(len(answerer.records) == dialog_count, "no dialog while the button is still down")
    answerer.answer(create=False)
    render_window_interactor.SetEventInformation(position_of[1][0], position_of[1][1], 0, 0)
    render_window_interactor.LeftButtonReleaseEvent()
    app.processEvents()
    app.processEvents()
    check(len(answerer.records) == dialog_count + 1, "the dialog opens after the mouse release")
    record = answerer.records[-1]
    check(record["question"] == "Create a cone between spheres 0 and 1?", "with the sphere of that click",
          record["question"])
    check(record["mirror_text"] is None and record["mirror_note"] is None,
          "twins 0 and 1: the cone is its own mirror")
    window.plotter.reset_camera(bounds=(-0.8, 0.8, -0.8, 0.8, -0.3, 0.3))
    camera_before = window.plotter.camera_position
    start = display_position(window, [0.0, 0.15, 0.0])
    click(window, app, start, release_position=(start[0] + 40, start[1] + 10))
    check(window.plotter.camera_position != camera_before, "the interactor still rotates the camera afterwards")
    reset_view(window, app)

    print("\nFocus lost while Ctrl is held")
    dialog_count = len(answerer.records)
    click(window, app, position_of[0])
    click(window, app, position_of[2], control_key=True)
    QApplication.sendEvent(window, QEvent(QEvent.Type.WindowDeactivate))
    app.processEvents()
    check(manager.get_pending_connection_sphere_indices() == [] and not sphere_is_cyan(manager, 0),
          "deactivating the window drops the set")
    release_control(app, window.plotter)
    check(len(answerer.records) == dialog_count, "a later Ctrl release opens no dialog")

    print("\nFocus lost after Ctrl went up, with the button still down")
    click(window, app, position_of[0])
    click(window, app, position_of[4], control_key=True)
    render_window_interactor.SetEventInformation(position_of[2][0], position_of[2][1], 1, 0)
    render_window_interactor.LeftButtonPressEvent()
    release_control(app, window.plotter)
    check(window._create_connection_dialog_waits_for_mouse_release, "setup: the dialog waits for the mouse release")
    QApplication.sendEvent(window, QEvent(QEvent.Type.WindowDeactivate))
    app.processEvents()
    check(manager.get_pending_connection_sphere_indices() == [] and not window._create_connection_dialog_waits_for_mouse_release,
          "deactivating drops the set and the wait")
    check(all(s.widget.GetCurrentRenderer() is not None for s in manager.spheres), "sphere widgets restored")
    # The style still holds the mouse focus from the lost press, so only it sees the next
    # press; its release must not count as the old Ctrl+click, even on the same pixel.
    click(window, app, position_of[2])
    app.processEvents()
    check(len(answerer.records) == dialog_count, "the next click opens no dialog",
          f"{[record['question'] for record in answerer.records[dialog_count:]]}")
    cyan_spheres = [i for i in range(len(SPHERE_ROWS)) if sphere_is_cyan(manager, i)]
    check(manager.get_pending_connection_sphere_indices() == [] and cyan_spheres == [4],
          "and collects nothing; the last collected sphere stays the selected one",
          f"set {manager.get_pending_connection_sphere_indices()}, cyan {cyan_spheres}")
    reset_view(window, app)

    print("\nThe real modal: Esc discards, Enter creates")
    del window._exec_create_connection_dialog
    set_non_interactive(False)
    watchdog_fired = []

    def watchdog():
        """Close any modal still open after 5 s, so a broken key press can't hang the test."""
        modal = QApplication.activeModalWidget()
        if modal is not None:
            watchdog_fired.append(True)
            modal.reject()

    for key, expect_created in ((Qt.Key.Key_Escape, False), (Qt.Key.Key_Return, True)):
        click(window, app, position_of[7])
        click(window, app, position_of[9], control_key=True)
        seen_modal = []

        def press_key(key=key, seen_modal=seen_modal):
            """Press the key on whatever modal is open, recording its title (it is deleted afterwards)."""
            modal = QApplication.activeModalWidget()
            seen_modal.append(None if modal is None else modal.windowTitle())
            if modal is not None:
                QTest.keyClick(modal, key)

        QTimer.singleShot(300, press_key)
        QTimer.singleShot(5000, watchdog)
        window.setFocus()
        release_control(app, window)
        for _ in range(30):
            if seen_modal:
                break
            QTest.qWait(50)
        key_name = "Esc" if key == Qt.Key.Key_Escape else "Enter"
        check(bool(seen_modal) and seen_modal[0] == "Create connection",
              f"{key_name}: the modal opened (release sent to MainWindow)", str(seen_modal))
        check((find_connection(manager, (7, 9)) is not None) == expect_created,
              f"{key_name}: {'creates' if expect_created else 'discards'} cone (7, 9)", str(connection_tuples(manager)))
    check(not watchdog_fired, "no modal left for the watchdog")
    set_non_interactive(True)
    window._exec_create_connection_dialog = answerer

    print("\nAdd Connection… path")
    tuples_before = connection_tuples(manager)
    manager.clear_unsaved_changes()
    window._add_connection_from_indices([9, 10])
    check(connection_tuples(manager) == tuples_before, "refuses a cone on a triangle edge")
    window._add_connection_from_indices([10, 8, 9])
    check(connection_tuples(manager) == tuples_before, "refuses a reordered duplicate")
    window._add_connection_from_indices([2, 0])
    check(connection_tuples(manager) == tuples_before, "refuses an edge of the new triangle (0, 4, 2)")
    check(not manager.has_unsaved_changes, "refusals are not edits")
    actors_before = actor_count(window)
    window._add_connection_from_indices([6, 8])
    check(find_connection(manager, (6, 8)) is not None, "adds a free cone")
    check(manager.has_unsaved_changes, "adding is an edit")
    check(actor_count(window) == actors_before + actors_of(find_connection(manager, (6, 8))),
          "only the new cone's actors were added (refusals left none behind)")
    selected_before = manager.get_selected_connection(), manager.get_selected_sphere_index()
    window._add_connection_from_indices([6, 8, 7])
    check(find_connection(manager, (6, 8, 7)) is not None and find_connection(manager, (6, 8)) is None,
          "a new triangle removes the cone on its edge")
    check((manager.get_selected_connection(), manager.get_selected_sphere_index()) == selected_before,
          "the selection is left alone")
    check(
        not any(
            connection._selected
            for connection in manager.connections
            if connection is not manager.get_selected_connection()
        ),
        "no stray cyan connection",
    )
    raw_count = len(manager.connections)
    check(manager.try_add_connection((10, 9)) and len(manager.connections) == raw_count + 1,
          "try_add_connection stays unchecked (tests build duplicates with it)")
    manager.delete_connection(manager.connections[-1])

    print("\nClip planes")
    window._add_clip_planes_axis("x")
    reset_view(window, app)
    # The clip-plane panel changes the plotter's size, so every display position moves.
    position_of = {i: display_position(window, center) for i, (center, _) in enumerate(SPHERE_ROWS)}
    lower_plane, upper_plane = window._clip_planes_vtk["x"]
    lower_clip_x, upper_clip_x = lower_plane.GetOrigin()[0], upper_plane.GetOrigin()[0]
    sphere_zero_x, sphere_zero_radius = SPHERE_ROWS[0][0][0], SPHERE_ROWS[0][1]
    check(sphere_zero_x + sphere_zero_radius < lower_clip_x and -0.30 > lower_clip_x and 0.30 < upper_clip_x,
          "setup: sphere 0 is clipped away, spheres 2, 3 and 6 are not",
          f"kept x in [{lower_clip_x:.3f}, {upper_clip_x:.3f}]")
    click(window, app, position_of[2])
    click(window, app, position_of[0], control_key=True)
    check(manager.get_pending_connection_sphere_indices() in ([], [2]),
          "Ctrl+click on a clipped-away sphere collects nothing", str(manager.get_pending_connection_sphere_indices()))
    click(window, app, position_of[6], control_key=True)
    answerer.answer(create=True, mirror=True)
    release_control(app, window.plotter)
    record = answerer.records[-1]
    check(record["question"] == "Create a cone between spheres 2 and 6?", "Ctrl+click works with clip planes on",
          record["question"])
    for sphere_indices in ((2, 6), (3, 6)):
        connection = find_connection(manager, sphere_indices)
        clipping_planes = None if connection is None else [
            actor.GetMapper().GetClippingPlanes() for actor in [connection._connector_actor] + connection._actors
        ]
        check(
            clipping_planes is not None
            and all(planes is not None and planes.GetNumberOfItems() > 0 for planes in clipping_planes),
            f"cone {sphere_indices} and its preview-built actors are clipped",
        )
    window._reset_clip_planes()

    window.close()
    app.quit()

    print()
    if failures:
        print(f"{len(failures)} FAILURE(S): " + ", ".join(failures))
        return 1
    print("Ctrl+click connections OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
