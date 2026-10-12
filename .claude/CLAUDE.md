# SMPL-SphereMesh-Tool

Interactive editor for **sphere-mesh collision templates** fitted to the SMPL body. A
template has spheres plus two kinds of connections: cones (2 spheres) and triangular slabs
(3 spheres). Stack: Python 3, PyQt6 UI, PyVista/VTK 3D through `pyvistaqt.QtInteractor`,
SMPL from `smpl_pytorch` on torch. Windows only (PowerShell; Git Bash also available).

The repo has two halves:

- **GUI editor**: `main.py` and `src/`.
- **Offline topology generator**: `src/topology/` and `scripts/build_topology.py`. It writes
  templates to `data/generated/` that you open in the GUI.

Detailed, file-specific knowledge lives in `.claude/rules/`. Those rules load automatically
when matching files are touched. Agents for recurring jobs live in `.claude/agents/`.

## Commands

| What | Command |
| --- | --- |
| Run the app | `uv run python main.py` |
| Packaging self-test | `uv run python main.py --selftest` |
| Windows build | `uv run python build_exe.py` (see README) |
| GUI tests (headless, real `MainWindow`) | `uv run python scripts/test_spheremesh_port.py`, `scripts/test_divide_connection.py`, `scripts/test_color_schemes.py`, `scripts/test_fit_spheres_ui.py` |
| SphereMesh class test | `scripts/test_sphere_mesh.py` |
| Fitter tests | `scripts/test_sphere_fit.py`, `scripts/test_sphere_fit_inflation.py` |
| Topology tests | `scripts/test_topology_{bumpers,sdf_torch,sections,validate,roundtrip,fit}.py` |
| Score templates side by side | `uv run python scripts/compare_topologies.py [--poses] a.json b.json` |
| Build a template | `uv run python scripts/build_topology.py --design data/generated/search/best_design.json` |
| Convert formats | `scripts/json2txt.py` (tool JSON → engine text), `scripts/txt2json.py` (engine text → tool JSON) |

About the tests:
- They are plain scripts, not pytest. Each prints `[PASS]`/`[FAIL]` lines and exits
  non-zero on failure.
- The GUI ones open real windows for a few seconds.
- Set `PYTHONDONTWRITEBYTECODE=1` for throwaway runs.
- The **test-runner** agent runs them and returns only the verdict.

## Code map

GUI modules:

| File | Contents |
| --- | --- |
| `src/ui_widgets.py` (~3.2k lines) | `MainWindow`: left button menu, plotter, `InfoPanel` overlay (top right), `ClipPlanesPanel`. Also `Skeleton`, the dialogs, and connection picking. |
| `src/sphere_manager.py` | `SphereManager`, the model/controller. Owns spheres and connections, selection, FK binding, Inflate preview, beta offsets, mirror edits, Divide, and JSON load/save. |
| `src/managed_sphere.py` | `ManagedSphere`: one sphere plus its `vtkSphereWidget`. It has no id; its identity is its index in `spheres`. |
| `src/sphere_connection.py` | `SphereConnection`: 2 or 3 sphere indices, a thin gray connector actor, and a spheremesh actor shown with "Show Spheremeshes". Its shape and distances come from a fresh `SphereMesh` primitive per update. |
| `src/sphere_mesh.py` | `SphereMesh` with `Sphere`, `Capsuloid` (cone), `Prysmoid` (slab): exact signed distances (scalar and vectorised), `closestSphere`, `buildGeometry`. The one implementation, shared with `SMFitting/` (camelCase, the user's style there). |
| `src/sphere_fit.py` | `optimize_spheres`, shared with the topology pipeline. Its `spheremesh_distance` measures with `src/sphere_mesh.py`. |
| `src/smpl_models.py`, `src/app_paths.py` | Model and template paths, frozen-build aware. |

Data:
- **Startup templates:** `data/template_{male,female}_1.0.json`.
- **GUI saves:** `data/save_DDMMYYYY_HHMM.json` plus a named `.txt`.
- **JSON format (version 2):**
  `{"spheres": [{center, radius, parent_joint, rest_offset, secondary_joint, secondary_rest_offset, lbs_weight, rest_radius, nearest_vertex_index, locked}], "connections": [{"indices": [a, b(, c)]}]}`

Topology docs:
- `TOPOLOGY_NOTES.md` has the results and how to reproduce them.
- `TOPOLOGY_HANDOFF.md` has the decisions, discoveries and dead ends.
- Read the relevant section before changing `src/topology/`.

## Facts that are easy to get wrong

- `+X` is the body's **left** side.
- Connections refer to spheres **by index**:
  - New spheres are appended, never inserted.
  - Duplicate connections exist (the female template has `(0, 1)` three times), so
    identify a connection by its object, not by its index tuple.
- Binding is rigid and single-joint: `center = J[p] + R[p] @ rest_offset`.
  - `secondary_joint` and `lbs_weight` are stored, but the blend that would use them is
    commented out.
  - Dragging a sphere does not update `rest_offset`. The next pose change snaps it back
    until "Reassign all parents" is pressed.
- Loading rounds centres to 3 decimals.
- `save_to_json` drops `quad_groups`.
- `rest_radius` is a ratio (radius ÷ distance to the nearest vertex), not a length.
- The Undo/Redo menu items do nothing; there is no undo.
- `SMFitting/SphereMesh.py` is a stand-in that hands back `src.sphere_mesh` itself, so the
  marimo notebooks keep `from SphereMesh import SphereMesh`. Edit `src/sphere_mesh.py`.

## Working conventions

- **Names:** descriptive, full-word variable names in Python too (the user's global naming
  rule; see `.claude/rules/python-style.md`). Match the surrounding docstrings and comment
  density.
- **Git:**
  - Never commit unless asked. The user commits with GitHub Desktop.
  - `stash@{0}` is obsolete; its content is already committed.
- **User files:** don't delete or overwrite `data/save_*.json`, `data/template*.json` or
  `data/generated/` outputs. Write experiments to the scratchpad or a temp dir.
- **Tests for GUI features:** every new GUI feature gets a headless test script modelled on
  `scripts/test_divide_connection.py`.
- **Planning:** the user likes a short plan first and clear decision questions (with a
  recommended option). Then they want measured results, never bare assertions.
- **Reporting:** say what was verified by a test or by simulated input, and what was not
  tried by hand in the running app.

## Agents (`.claude/agents/`)

| Agent | Use it |
| --- | --- |
| **test-runner** (haiku) | After any change in `src/` or `scripts/`. It picks the tests that match the changed files and returns PASS/FAIL only. |
| **spheremesh-reviewer** | After implementing an editing feature, before calling it done. Read-only invariant review. |
| **vtk-investigator** (sonnet) | Before designing interaction or rendering changes. It settles PyVista/VTK/Qt behaviour from `.venv` sources and experiments. |
| **topology-validator** (sonnet) | To score or compare templates against the user's acceptance criteria, with measured numbers. |

## Feature log

- **Divide (2026-10):**
  - What it does: click a cone or triangle in the 3D view (Ctrl+click reaches through
    spheres), then press Divide or the D key. It adds a sphere at the midpoint or centroid
    with the mean radius, splits the triangles that share a divided cone's edge, and binds
    the new sphere to one of its endpoints' joints.
  - Where to see it: the InfoPanel shows the selected connection's spheres, one box each.
  - Code: `SphereManager.divide_connection`, `MainWindow._install_connection_picking`.
  - Test: `scripts/test_divide_connection.py`.
- **Color schemes (2026-10):**
  - What it does: the Random Colors, Class Colors and Reset Colors buttons set
    `SphereManager.color_scheme` ("random", "class" or "default"). Class Colors paints
    spheres red `#ee380a`, capsuloids blue `#2f76b9` and prysmoids violet `#9545d6`
    (SMFitting's `primitiveColors`), and is the scheme the tool starts in.
  - New spheres and connections (Add, Divide, load) follow the active scheme.
  - Selection (cyan) and lock (yellow) colours still take priority.
  - Mechanism: the per-object override is `set_random_color`.
  - Test: `scripts/test_color_schemes.py`.
- **SphereMesh port (2026-10):**
  - What it does: the tool draws and measures with `src/sphere_mesh.py`, the class
    SMFitting uses, instead of the old approximate `src/spheremesh.py` (deleted).
    - Cones are open tangent cones.
    - Triangles have exact tangent planes, solved in closed form.
    - Shading is flat; spheres are 31 x 16 like `Sphere.buildGeometry(16, 32)`.
  - Heatmap: the exact signed union distance, styled like SMFitting's
    `DistanceHeatmapSurface` (5 mm bands over ±20 mm, cool inside / warm outside, grey
    above). It recomputes after edits and pose/shape changes while on.
  - Fixed on the way:
    - Delete sphere re-indexed connections against the old list.
    - The Inflate slider left the cones stale.
    - Actors created after clip planes were set were not clipped.
  - Code: `SphereConnection.build_primitive`, `SphereManager.get_heatmap`,
    `MainWindow._on_show_heatmap` / `_refresh_heatmap_if_stale`.
  - Tests: `scripts/test_sphere_mesh.py` (class, against `bumpers.py`),
    `scripts/test_spheremesh_port.py` (GUI).
