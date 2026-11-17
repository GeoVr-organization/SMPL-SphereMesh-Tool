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
| GUI tests (headless, real `MainWindow`) | `uv run python scripts/test_spheremesh_port.py`, `scripts/test_divide_connection.py`, `scripts/test_color_schemes.py`, `scripts/test_fit_spheres_ui.py`, `scripts/test_random_pose_shape.py`, `scripts/test_lbs_binding.py`, `scripts/test_star_pose_startup.py`, `scripts/test_ctrl_select_connection.py`, `scripts/test_delete_connection.py`, `scripts/test_sphere_click_picking.py` |
| Rigid vs LBS binding against the posed body | `uv run python scripts/compare_binding_modes.py [--json out.json]` |
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
- On the 1.3 startup template two tests fail; both pass on 1.2, so they depend on the
  startup template. Not investigated (2026-11-13).
  - `test_divide_connection`: 1 check, FK at the rest pose is off by 0.045 mm.
  - `test_ctrl_select_connection`: 4 checks on the Ctrl+click modal and cone (7, 9).

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
- **Startup templates** (`startup_spheremesh_template_path`):
  - male: `data/template_male_1.3_starpose.txt` (engine text, star pose, 63 spheres). It is
    the **control fit**: `SMFitting/mainbeta.py` seeded with
    `data/template_male_1.3_seed_starpose.txt`, run on the base mesh
    (`examples/SMPL_starpose_head.obj` as the beta mesh). It is also the base of the beta
    offsets (`extract_beta_offsets.py`, default `--template`).
  - **Fitting surface** (since 2026-11-13): `SMFitting/examples/SMPL_starpose_head.obj` for the
    base, and `SMFitting/examples/OBJ_head/beta0..9.obj` for β N = +1.
    - It holds the 3823 vertices of `SMPL_starpose.obj`, in their order, plus the 1194 SMPL
      vertices whose dominant joint is the head: 5017 vertices and 9973 SMPL triangles.
    - Hands and feet stay out.
    - There are no `vn` lines, so `loadObjMesh` area-weights the normals for base and beta
      meshes alike.
    - Built from SMPL by the scratchpad's `build_head_obj.py`. The user's full-body
      `SMFitting/starpose w_head.obj` is SMPL itself (same order, same faces).
    - Without the head, nothing constrained s59 (the head sphere): it came out of every fit
      unchanged, with a zero offset.
    - Both notebooks' `objPath` defaults still name `examples/SMPL_starpose.obj` and
      `examples/OBJ/betaN.obj`. Override them, as for the other paths.
  - The family 1.3 (2026-11-13, regenerated the same day with P = 1, then with the head) has
    three files:
    - Starting template: `SMFitting/output/template_male_1.1.txt`, hand-fixed by the user
      and saved from the GUI. It is identical to `data/template_male_1.1.txt`.
    - Seed: `data/template_male_1.3_seed_starpose.txt`, the `SMFitting/main.py` fit of the
      starting template, hernia loop included.
    - Control: `data/template_male_1.3_starpose.txt`.
  - The **seed** of every mainbeta run is `1.3_seed`, for the control and for each
    `SMFitting/output/betaN.txt` alike. Offsets only cancel the procedure's drift when the
    template and the betaN fits come from the same seed.
    - Drift of the 1.3 control from its seed (P = 1): median 0.9 mm, max 33 mm. Only 3
      spheres move more than 10 mm: s27, s50 and s22.
    - Family 1.2 (P = 0.25): up to 46 mm from its seed, and 79 mm when seeded with its own
      control.
  - **Regularization:** both notebooks use `P, P_r = 1, 1` (`main.py` and `mainbeta.py`, the
    regularization cell), since 2026-11-13. It was 0.25.
    - At 0.25, overlapping spheres trade places along directions the tangency rows barely
      see.
      - The `main.py` fit slid s55 118 mm away from the user's hand fix.
      - The β1 = +1 fit then swapped s50 for s55: s50 shrank 25.7 mm and moved 28 mm back.
      - The linear offset replayed that swap backwards at β1 = −1: s50 reached r 149.5 mm,
        swallowed s55, and gave 49 hernias. The real β1 = −1 fit keeps s50 at 126.9 mm,
        with 0 hernias.
    - At P = 1 the template, β = +1 and β = −1 fits are as tight as at 0.25 (mean |d|
      2.9 mm). s55 slides 58 mm, not 118, and the s27/s55 mirror error is 17 mm, against
      28 mm in the starting template.
    - P = 4 is too stiff: mean |d| 3.3–3.6 mm at β1 = ±1, and 3–5 hernias at +1.
    - The experiment is in the scratchpad: `regularization_setup.py`,
      `regularization_evaluate.py` and `regularization/results.json`.
  - Regenerating the family takes four steps:
    1. `main.py` on the starting template gives the seed.
    2. `mainbeta.py` on the base mesh gives the control.
    3. `mainbeta.py` gives the ten betaN fits.
    4. `extract_beta_offsets.py --all` writes the CSVs.
  - Run the notebooks headless with their paths cell overridden
    (`app.run(defs={sphereMeshPath, objPath, [betaObjPath,] herniaSavePath})`), because
    their on-disk defaults are stale:
    - `main.py` saves to `SMFitting/output/template_male_1.1.txt`, which is now the starting
      template, so running it as is overwrites the user's hand fix.
    - `mainbeta.py` is still seeded with `data/template_male_1.1_starpose.txt`.
  - None of the 33 test shapes of the beta offsets (below) produces a degenerate primitive
    with the P = 1 family 1.3.
  - The 60-sphere family 1.2 (`data/template_male_1.1_starpose.txt` seed,
    `data/template_male_1.2_starpose.txt` control) stays in `data/`, unused.
  - female: `data/template_female_1.0.json`.
  - Scripts that fit on the neutral body keep `spheremesh_template_path`
    (`data/template_*_1.0.json`).
- **GUI saves:** `data/save_DDMMYYYY_HHMM.json` plus a named `.txt`.
- **JSON format (version 2):**
  `{"spheres": [{center, radius, parent_joint, rest_offset, secondary_joint, secondary_rest_offset, lbs_weight, rest_radius, nearest_vertex_index, locked}], "connections": [{"indices": [a, b(, c)]}]}`

Topology docs:
- `TOPOLOGY_NOTES.md` has the results and how to reproduce them.
- `TOPOLOGY_HANDOFF.md` has the decisions, discoveries and dead ends.
- Read the relevant section before changing `src/topology/`.

## Facts that are easy to get wrong

- `+X` is the body's **left** side.
- The GUI starts in the **star pose**, not θ = 0: `default_theta_tensor()` turns
  left_hip Z +0.3 and right_hip Z −0.3. That is `SMFitting/examples/SMPL_starpose.obj`, the
  pose the SMFitting templates are fitted in. Reset and pose row 0 still give θ = 0.
- Connections refer to spheres **by index**:
  - New spheres are appended, never inserted.
  - Duplicate connections exist (the female template has `(0, 1)` three times), so
    identify a connection by its object, not by its index tuple.
- Binding has two modes, switched by "LBS binding" in Theta Controls. It is **on (LBS) by
  default** in the GUI; a bare `SphereManager`, as scripts build it, is rigid.
  - Rigid: `center = J[p] + R[p] @ rest_offset`, where `J` is regressed from the posed mesh.
  - LBS: `center = Σ w_k (R_k c_rest + t_k)` over up to 4 joints, built from SMPL's FK
    transforms (`Skeleton.world_transforms`). The FK joints differ from the regressed `J` by
    up to 4.6 mm once posed, so a one-joint LBS sphere is not the rigid one.
  - The weights live on the sphere (`ManagedSphere.lbs_joint_weights`) and are never saved.
    They are derived from the 20 nearest rest vertices and re-derived on rebinding and on
    every beta change.
  - `secondary_joint` and `lbs_weight` are stored, but neither mode uses them; the old blend
    is commented out.
  - Dragging a sphere does not update `rest_offset`. The next pose change snaps it back
    until "Reassign all parents" is pressed.
  - Code that turns a posed centre back into `rest_offset` must use
    `SphereManager._rest_offset_from_posed_center`, which inverts the active mode.
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
  - What it does: click a cone or triangle in the 3D view (Shift+click reaches through
    spheres; it was Ctrl+click until 2026-11), then press Divide or the D key. It adds a sphere at the midpoint or centroid
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
- **CSV pose and shape pickers (2026-10):**
  - What it does: "Random pose (CSV)" (Theta Controls) and "Random shape (CSV)" (Beta
    Controls, replacing the old uniform Randomize) pick a row of `data/random_pose.csv` /
    `data/random_shapes.csv` (10 betas per row, no header), never the one shown. A slider
    under each button shows the row and applies every row it passes while dragged.
    - Position 0 is the rest pose / all betas 0; Reset moves the slider back to 0.
    - A hand edit (joint or beta slider, Load betas) adds "(edited)" to the label.
  - The beta sliders span ±3.0 (`BETA_SLIDER_INT_MIN/MAX` = ±30), which covers every CSV
    shape.
  - The rows are read once at startup; the applied row is a copy, because the joint slider
    edits `theta` in place.
  - Code: `CsvRowSlider`, `ThetaSliderGroup._apply_random_pose_row`,
    `BetaSliderGroup._apply_random_shape_row`.
  - Test: `scripts/test_random_pose_shape.py`.
- **LBS binding (2026-10):**
  - What it does: with "LBS binding (blend joints)" checked (Theta Controls, on by default),
    each sphere centre follows a blend of up to 4 SMPL joints instead of only its parent joint.
    Unchecked gives exactly the old rigid FK. Toggling re-poses at once and is not an edit.
  - The weights average the SMPL skin weights of the sphere's 20 nearest rest vertices,
    inverse-distance weighted (`1/(d + 0.1 r)`, base radius). Only vertices whose dominant
    joint is the parent joint, its parent or a child count. The strongest 4 joints are kept
    and renormalised. These are the ClothSimulator26 engine's rules.
  - Posed edits (InfoPanel centre, To joint / To bone, Fit spheres) invert the active mode,
    so they stay put on the next pose. In LBS mode the weights are re-solved for the new rest
    centre, so a later toggle or beta tick moves them only as much as the body moves. The
    exception is the crotch spheres, whose weights flip between two sets (up to 19 mm,
    measured).
  - Known weakness, measured with `compare_binding_modes.py`: with the engine's gate, the
    collider pokes through the skin about 30 mm deeper at the hips than with rigid. The cause
    is the big pelvis spheres splitting their weight between both hips. See the brief's §13.3
    for the measured alternative gates.
  - A status line `[lbs] joint weights for N spheres: neighbour distance ...` is printed on
    toggle-on, load and Reassign all parents.
  - Design and measurements: `spheremesh_lbs_binding_brief.md`.
  - Code: `SphereManager._derive_lbs_joint_weights`, `_skinning_transforms`, `_lbs_blend`,
    `_rest_offset_from_posed_center`, `update_sphere_positions`;
    `MainWindow._on_lbs_binding_toggled`.
  - Tests: `scripts/test_lbs_binding.py`; measurement: `scripts/compare_binding_modes.py`.
- **Star-pose startup (2026-11):**
  - What it does: the body starts with its legs apart, as in `SMPL_starpose.obj` (every OBJ
    vertex within 0.001 mm). The male template is the SMFitting control fit,
    `data/template_male_1.3_starpose.txt`, loaded where the file puts it. Theta Controls
    shares the pose tensor; its pose picker shows "(edited)".
  - Binding: `assign_parent_joint` is pose-aware (see `.claude/rules/gui-editing.md`), so
    templates loaded under a pose, and Reassign all parents under a pose, keep the spheres
    where they are shown.
  - A gender switch binds the new template at the star pose and betas 0 (the pose and shape
    of the male .txt), then restores the user's pose and betas.
  - Known limits:
    - Spheres bound under the star pose reproduce the file only in the mode active at load.
      Toggling LBS off moves them up to 9.1 mm.
    - The first beta change re-derives the LBS weights, and the crotch-flip limitation then
      moves a crotch sphere at the next FK, even back at β = 0. With the old 1.1 it was s22,
      8.7 mm (reviewer's measurement). With 1.2 it was s50, 13.9 mm: its weights went from
      pelvis 0.90 to pelvis 0.23 / left hip 0.35 / right hip 0.41. Not re-measured on 1.3.
      With beta offsets loaded a beta change runs no FK, so the offsets path does not show
      it.
    - Load Topology / Import still run no FK, so a θ = 0 JSON shows unposed until the next
      pose tick.
  - The template sits in `data/`, so the build bundles it and seeds it next to the exe.
  - Code: `default_theta_tensor`, `startup_spheremesh_template_path`,
    `MainWindow._load_spheremesh_file`, `SphereManager._is_posed` / `assign_parent_joint`.
  - Test: `scripts/test_star_pose_startup.py`.
- **Beta offsets against the control fit (2026-11):**
  - The offsets in `data/offsets` are `betaN.txt − template 1.3`, 63 rows each, both from
    the 1.3 seed.
  - The GUI does not reject a CSV from another family. `_load_single_beta_csv` skips rows
    past the sphere count with a warning and applies the rest by index, so a mismatched
    family lands on the wrong spheres without an error.
  - Taken against the seed instead of the control, every offset carries the procedure's
    drift D, and the GUI adds it Σβ times. Measured on family 1.2: up to 46 mm on s27, s23,
    s51, s50 and s54.
  - Measured in the GUI on 33 shapes, d < −10 mm counted on the 3823 vertices of
    `SMPL_starpose.obj`. The last column counts the 5017 vertices of
    `SMPL_starpose_head.obj` instead:

    | Shapes | 1.2 offsets vs seed 1.1 | 1.2 offsets vs control | 1.3, P = 0.25 | 1.3, P = 1 | 1.3, P = 1, head (current) |
    | --- | --- | --- | --- | --- | --- |
    | template | 3 | 4 | 0 | 0 | 1 |
    | β = +1 | 1–3 | 1–4 (exactly betaN.txt) | 0–2 | 0–1 (exactly betaN.txt) | 0–1 (exactly betaN.txt, 0.79 mm rounding) |
    | β = −1 | 93–108 | 4–17 | 0–1, except β1 = −1: 49 | 0–3 | 1–4 |
    | combinations | 9–40 | 1–11 | 0–1 | 6 and 1 | 2 and 0 |
    | random shapes | 1–1120 | 1–77 | 0–35, except random01: 227 | 0–22 | 0–23 (worst min d −14.5 mm) |
  - On 1.3 with P = 0.25, s50 dominated 211 of the 227 hernias of random01 and 47 of the 49
    at β1 = −1 (see Regularization under Data).
  - With P = 1 and no head, mean |d| was 2.79–3.58 mm on all 33 shapes.
  - With the head, mean |d| is 5.1–5.9 mm and ~850 vertices sit more than 10 mm outside.
    The extra all comes from the head, which one sphere (s59) cannot follow: face and jaw
    stay up to 44 mm away.
    - The body part (the old 3823 vertices) is unchanged. On the control: mean |d| 3.04 mm,
      1 hernia, 163 vertices over +10 mm. Without the head it was 2.88 mm, 0 and 138.
    - On the head (1194 vertices), the head-fitted control has 0 hernias, mean |d| 12.4 mm
      and 695 vertices over +10 mm. The control fitted without the head had 36 hernias
      (s59 poking out, min −14.4 mm) and mean |d| 18.1 mm.
  - s59 now has offsets: 19 mm for β0, 0.2–6.3 mm for the others. Its fitted radius is
    75 mm, against 88 mm in the starting template.
  - Backups in the scratchpad: `backup_1_3_P025` (P = 0.25, no head) and
    `backup_1_3_P1_nohead` (P = 1, no head).
  - Experiment page: `docs/beta_offsets_experiment/index.html`. It shows family 1.2, so it is
    out of date since the switch to 1.3. It uses `images_1.2/`, `stats_1.2.json` and
    `drift_1.2.json`. The user asked for no comparison with 1.1.
    - The 1.1 capture (`images/`, `stats.json`, `drift_analysis.json`) stays in the folder,
      unused by the page. It was taken with the same camera, so the images overlay pixel for
      pixel.
- **Ctrl+click connections (2026-11):**
  - What it does: Ctrl+click 2 or 3 spheres in the 3D view and release Ctrl. A dialog
    previews the cone or triangle (cyan, even with Show Spheremeshes off) and offers Create
    (Enter) or Discard (Esc).
    - A plain-clicked sphere seeds the set. Ctrl+click on a collected sphere removes it; a
      4th is ignored. Ctrl+click never selects a cone or triangle.
    - Release Ctrl with one sphere: no dialog. Focus lost while Ctrl is held: the set is
      dropped. Ctrl up before the mouse button: the dialog waits for the mouse release.
    - After Create the new connection is selected; after Discard nothing is.
  - Mirror: a checkbox adds the X-mirrored connection on the twin spheres (same corner
    order, as 22 of 24 mirrored triangles in template 1.2 have it). It is ticked when
    Mirror edits is on. Twins are found by `_mirror_twin_for_connection`, where a midline
    sphere is its own twin, like `check_x_symmetry`. With template 1.2, 57 of 60 spheres
    have a twin (not 23, 27, 55).
  - Redundancy, also for the Add Connection… button (`add_connection_checked`): a
    duplicate (same spheres, any order) or a cone on an existing triangle's edge is refused.
    A new triangle deletes the cones on its edges, duplicates included, and prints each
    removal; the union is unchanged (a Prysmoid contains its edge capsuloids). Only
    connections touching the new one are checked; older redundancy elsewhere stays.
    `try_add_connection` stays the unchecked primitive (tests build duplicates with it).
  - Reach-through to hidden connections moved from Ctrl+click to Shift+click.
  - Code: `SphereManager.toggle_pending_connection_sphere`, `prepare_/commit_/discard_
    connection_candidate`, `find_connection_covering`, `find_connections_made_redundant_by`;
    `MainWindow._ControlReleaseFilter`, `_finish_ctrl_sphere_selection`,
    `CreateConnectionDialog`.
  - Test: `scripts/test_ctrl_select_connection.py`.
- **Frontmost sphere picking (2026-11-16):**
  - What it does: a click, drag (move) or right drag (scale) on stacked spheres takes the
    frontmost one under the cursor. Before, it took the lowest-index sphere anywhere on the
    ray, because every `vtkSphereWidget` picks its own sphere and the first one added wins.
    Measured on a 1573-click grid: 171 wrong before, 0 after.
  - A click a few pixels outside a sphere no longer grabs it (the widget's own picker had
    tolerance). It goes to the camera or the connection picker.
  - Mechanism and traps: `.claude/rules/vtk-interaction.md`, "Stacked spheres".
  - Code: `MainWindow._leave_only_frontmost_sphere_widget`, `_on_right_press_before_widgets`,
    `_on_release_claimed_by_sphere_widget`.
  - Test: `scripts/test_sphere_click_picking.py`.
