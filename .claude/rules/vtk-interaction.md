---
paths:
  - "src/ui_widgets.py"
  - "src/managed_sphere.py"
  - "src/sphere_connection.py"
  - "scripts/test_*ui*.py"
  - "scripts/test_divide_connection.py"
  - "scripts/test_sphere_click_picking.py"
---

# Mouse and keyboard in the 3D view (pyvista 0.48.4, pyvistaqt 0.12, VTK 9.6.2)

These were verified against `.venv` sources and in-memory experiments. Re-verify with the
**vtk-investigator** agent before relying on them after a library upgrade.

## Event dispatch
- **Sphere widgets claim left presses.**
  - `vtkSphereWidget` observes at priority **0.5**.
  - It aborts the left press and release whenever the pick lands on its sphere, even if
    another prop is in front. The right button does the same (a right drag scales).
  - The trackball style observes at 0.0.
  - A plain press and release on a sphere still fires the widget's end-interaction
    callback, which is how a sphere gets selected (`_on_sphere_interaction`).
  - **Stacked spheres:** each widget picks only its own sphere, and equal-priority
    observers run in the order added. Left alone, the lowest-index sphere anywhere on the
    ray takes the press, even one behind another. Its picker also has a few pixels of
    tolerance. Measured on a grid of 1573 clicks: 171 reached the wrong sphere, and a drag
    on a front sphere moved the one behind it.
    - The fix (`_leave_only_frontmost_sphere_widget`): the priority-1.0 left and right press
      observers pause every widget but the frontmost one under the cursor
      (`_sphere_index_at_display_position`, tolerance 0). On empty pixels they pause all of
      them, so a near miss goes to the camera. It costs about 1.6 ms per press on the
      63-sphere template.
    - Restoring: a widget that took the press also takes the release, so the release
      never reaches the style. Interactor release observers at priority 1.0 restore the
      widgets (`_on_release_claimed_by_sphere_widget`). Unclaimed presses restore
      themselves: left at the style release, right in the priority -1.0 press observer.
    - Never add a style observer for `RightButtonReleaseEvent`. A style observer replaces
      the style's own handler, and pyvista's capture mixin re-calls it only for the left
      button, so the zoom would never end.
    - Test: `scripts/test_sphere_click_picking.py`.
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
  - Until some release arrives, the style keeps that focus, so the **next press** also
    reaches only the style. The interactor-level press observers (priority 1.0 and -1.0)
    don't see it. Code that waits for a release must not count on a later press to reset
    it. `_abort_ctrl_sphere_selection` forgets the press position instead, so the late
    release does nothing.
- **Pausing widgets:** `widget.SetCurrentRenderer(None)` makes a sphere widget ignore
  presses while it keeps rendering. Restore the saved renderer afterwards. Shift+click and
  Ctrl+click both pause them, inside the priority-1.0 press observer (a pause set before
  the press is undone there by `_restore_sphere_widgets`).
  - Shift+click reaches connections hidden behind spheres.
  - Ctrl+click collects spheres for a new connection and never selects a connection.
  - Ctrl wins over Shift.
- **Modifier clicks and the style:** the trackball style maps left drags to rotate (plain),
  pan (Shift), spin (Ctrl) and dolly (Ctrl+Shift). A press and release without movement
  leaves the camera unchanged in all four.
- **Off-screen:** `plotter.iren` is None, so guard against it.

## Picking a sphere (`MainWindow._sphere_index_at_display_position`)
- `vtkSphereWidget` has no `GetSphereActor`. Its actor is the renderer actor whose property
  is the widget's `GetSphereProperty()`; match them by `GetAddressAsString("")`.
- Use `vtkCellPicker` with tolerance 0 and `PickFromListOn()`.
  - It honours the clip planes, which the widget actors carry like every other actor.
  - The vtk-investigator found it matches an analytic ray–sphere test on 1485 pixels, and
    the clip-aware analytic test on 13 of 13 pixels where clipping changes the answer.
- Use `interactor.GetEventPosition()` for picks, never a Qt position. pyvistaqt scales by
  the device pixel ratio and flips y. HiDPI is unverified (this machine has ratio 1.0).

## The Ctrl key release (`_ControlReleaseFilter`)
- Neither `MainWindow.keyReleaseEvent` nor the VTK `KeyReleaseEvent` alone sees it:
  - the focused `QtInteractor` consumes every key;
  - after a click, `self.setFocus()` moves focus to `MainWindow`, which VTK doesn't see.
- An event filter on `QApplication` sees it in both cases.
  - It sees one release several times, once per widget on the propagation chain.
  - Act on the first release and let a state flag ignore the rest.
  - Never swallow the event.
- **Focus loss:** switching to another program while Ctrl is held delivers no release at
  all. Catch `applicationStateChanged` and `WindowDeactivate`.
- **The modal:** open it with `QTimer.singleShot(0, ...)`. Run inside the filter, it holds
  the key release back from VTK until it closes.
- **Mouse still down:** if Ctrl comes up before the left button, a modal would take the
  mouse release and leave the style stuck mid-interaction. Wait for the release.
- **Tests:** `QTest.keyRelease(window.plotter or window, Qt.Key.Key_Control)` reaches the
  filter. A Qt virtual such as `keyReleaseEvent` can't be monkeypatched on an instance;
  plain Python methods (`_exec_create_connection_dialog`) can.

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
  - Ctrl release: ends a Ctrl+click sphere set (application event filter, not a shortcut).
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

Traps when driving clicks:
- **Reset the camera first.** Call `plotter.reset_camera(...)` before injecting clicks. With
  the default clipping range (0.01–1000) one enabled sphere widget claims every press, even
  on empty pixels.
- **Recompute display positions after showing clip planes.** The clip-plane panel changes
  the plotter's size, so earlier positions go stale.
- **New X clip planes clip.** They start a quarter of the scene span in from each side,
  about x = ±0.46 with the body loaded, so spheres outside that are clipped away.
