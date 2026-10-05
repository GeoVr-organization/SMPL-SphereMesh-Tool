---
paths:
  - "src/ui_widgets.py"
  - "src/managed_sphere.py"
  - "src/sphere_connection.py"
  - "scripts/test_*ui*.py"
  - "scripts/test_divide_connection.py"
---

# Mouse and keyboard in the 3D view (pyvista 0.48.4, pyvistaqt 0.12, VTK 9.6.2)

These were verified against `.venv` sources and in-memory experiments. Re-verify with the
**vtk-investigator** agent before relying on them after a library upgrade.

## Event dispatch
- **Sphere widgets claim left presses.**
  - `vtkSphereWidget` observes at priority **0.5**.
  - It aborts the left press and release whenever the pick lands on its sphere, even if
    another prop is in front.
  - The trackball style observes at 0.0.
  - A plain press and release on a sphere still fires the widget's end-interaction
    callback, which is how a sphere gets selected (`_on_sphere_interaction`).
- **Interactor observers:**
  - Add them with priorities through the raw interactor:
    `plotter.iren.interactor.AddObserver(event, callback, priority)`.
  - pyvista's `iren.add_observer` has no priority argument.
- **Releases:**
  - The style grabs focus on press, so a `LeftButtonReleaseEvent` observer on the
    interactor never fires.
  - Register releases with `plotter.iren.add_observer("LeftButtonReleaseEvent", ...)`,
    which attaches to the current **style** object (pyvista issue #4976).
  - If code ever replaces the interactor style, those observers must be re-added.
- **Pausing widgets:** `widget.SetCurrentRenderer(None)` makes a sphere widget ignore
  presses while it keeps rendering. Restore the saved renderer afterwards. This is how
  Ctrl+click reaches through spheres.
- **Off-screen:** `plotter.iren` is None, so guard against it.

## Connection picking (`MainWindow._install_connection_picking`)
- Use `vtkCellPicker` with `PickFromListOn()`, filled from
  `SphereManager.get_connection_pick_targets()` (visible connector and spheremesh actors
  only). Tolerance is `CONNECTION_PICK_TOLERANCE`.
- The picker respects mapper clip planes and skips invisible actors. The body mesh,
  skeleton and clip-plane outlines are not in the list, so they never steal clicks.
- A click is a press and release within `CONNECTION_CLICK_MAX_MOVEMENT_PIXELS`; anything
  further is a camera drag.
- Clicking empty space clears a selected connection and otherwise leaves the selection
  alone.
- `picker.GetActor() is actor` holds for pyvista `Actor` objects.

## Keys
- **Taken:**
  - Q/E: `keyPressEvent`, previous/next sphere.
  - X/Y/Z: `QShortcut`s for camera axes.
  - D: Divide.
  - Ctrl+O: load.
  - pyvista's own bindings: `q`, `b`, `v`, Shift+C, Up, Down, plus, minus.
- New shortcuts follow the X/Y/Z pattern: `QShortcut` with `ApplicationShortcut` context
  and `setAutoRepeat(False)`. Qt suppresses them while a modal dialog is open.
- After handling a click, call `self.setFocus()` so Q/E keep working.

## Layout side effects
- `_add_clip_planes_axis` and `_reset_clip_planes` call `adjustSize()`, which resizes the
  main window. Tests must resize again before computing display positions.
- Actors created after clip planes were enabled are not clipped until
  `_apply_clip_planes_to_all_actors()` runs.

## Driving it from a test
```python
renderer.SetWorldPoint(x, y, z, 1.0); renderer.WorldToDisplay(); display_x, display_y, _ = renderer.GetDisplayPoint()
interactor = window.plotter.iren.interactor
interactor.SetEventInformation(display_x, display_y, control_flag, shift_flag)
interactor.LeftButtonPressEvent()   # optionally MouseMoveEvent() steps for a drag
interactor.LeftButtonReleaseEvent(); app.processEvents()
```
Show and resize the window first (`window.show()`, `window.resize(...)`,
`app.processEvents()`) so the render window has real pixels.
