---
paths:
  - "scripts/test_*.py"
---

# Test scripts

- **Format:** tests are standalone scripts, not pytest.
  - Pattern: `check(condition, label, detail)` prints `[PASS]`/`[FAIL]`.
  - `main()` returns 1 when anything failed and `raise SystemExit(main())`.
- **GUI tests:**
  - Call `set_non_interactive(True)` before building `MainWindow`. Otherwise a
    save-on-close or report dialog blocks forever.
  - Build the window as `MainWindow(load_smpl_model(MALE), MALE)`.
  - Call `manager.update_sphere_positions()` first if a check compares positions with FK;
    loaded centres are rounded and may differ from FK by under 1 mm.
  - Drive features through the `MainWindow` handler (for example
    `window.divide_connection_clicked()`), not only the manager, so the UI wiring is
    tested too.
- **Topology tests:** pin behaviour against brute force. Take the reference distances from
  `src/topology/bumpers.py`: it is exact and independent of `src/sphere_mesh.py`, which
  the GUI and `SphereConnection.get_distance` use.
- **Isolation:**
  - Write files to `tempfile.TemporaryDirectory()`, never into `data/`.
  - Reset global state you changed: clip planes, Inflate, beta offsets, theta.
- **Synthetic topologies:** build them as a JSON payload and load it with
  `manager.load_from_json`. Picking tests need spheres placed so the clicks are
  unambiguous.
- **Reporting:** report numbers in check details (for example `max |d_old - d_new| = 4e-17`)
  so a pass shows its margin.
- **Running:** `uv run python scripts/<name>.py`. The output is noisy (`Interacted with
  sphere`, `Added sphere`, deprecation warnings); filter on `PASS|FAIL|Traceback|Error`.
- **Clicks are simulated:** a pass means the code works with simulated VTK events. Say so
  when reporting, and suggest a manual check for anything visual.
