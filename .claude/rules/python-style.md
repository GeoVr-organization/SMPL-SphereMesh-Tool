---
paths:
  - "**/*.py"
---

# Python style in this repo

- **Descriptive, full-word names** for locals, parameters and attributes.
  - Qualify a name with its kind: `endpoint_spheres`, `pickable_actor`,
    `replacement_connections`.
  - Not `s`, `c`, `tmp` or `conn`.
  - `i`, `j`, `k` are fine as plain loop counters; prefer a real name when the loop body is
    non-trivial.
  - Older code uses short names (`sp`, `si`, `c`). Don't rename them in passing; new code
    uses full words.
- **Docstrings:** every new function and method gets a one-line docstring saying what it
  does. Add the *why* in a second paragraph only when it is not obvious.
- **Comments:** explain a constraint or a trap (for example "the style grabs focus on
  press, so ..."), never what the next line does.
- **Fit in:** follow the module's existing idiom. That means `np.float64` arrays,
  `Optional[...]` typing in `sphere_manager.py`, `print(...)` for status lines, and
  `QMessageBox` or `MainWindow._report(...)` for user-facing messages. `_report` also works
  headless.
- **Constants:** module-level `UPPER_CASE` with a comment giving the unit or meaning (see
  `CONNECTION_PICK_TOLERANCE` in `ui_widgets.py`).
- **Scope:** keep refactors minimal and behaviour-preserving. Split a helper out only when a
  new feature needs it, as with `_build_connection` and `_closest_bone`.
