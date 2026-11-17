---
paths:
  - "src/sphere_manager.py"
  - "src/ui_widgets.py"
  - "src/sphere_connection.py"
  - "src/managed_sphere.py"
  - "src/sphere_mesh.py"
---

# Editing operations: invariants and checklist

Any code that adds, removes or changes spheres or connections must keep all of these true.

## Indices and identity
- `SphereManager.spheres` and `.connections` are flat lists, and connections hold sphere
  **indices**.
  - Append new spheres at the end; nothing then needs re-indexing.
  - Deleting sphere `i` (`_delete_sphere_at_index`) re-indexes every connection and remaps
    `_pre_uniform_radius_by_index`.
  - It does **not** re-index the beta-offset arrays. That is a known bug, so don't copy the
    pattern.
- Identify a connection by its **object**. Duplicates exist, so matching index tuples is
  ambiguous.
- Sphere-index-keyed state that must stay aligned with `spheres`:
  - `_pre_uniform_radius_by_index`
  - `offsets_pos[k]`, `offsets_r[k]`
  - `_beta_template_pos`, `_beta_template_r` (only when beta offsets are loaded)
- A triangle does not list its edges as separate cones; `SphereMesh.Prysmoid` draws and
  measures its rim (its three `edgeCapsuloids`).

## Selection
- `selected_sphere_index` and `selected_connection` exclude each other.
  - `select_connection` clears the sphere.
  - Selecting a sphere (`_on_sphere_interaction`, Q/E, `add_sphere`) clears the connection.
- Use `get_selected_connection()`. It returns None for a connection that has since been
  deleted, and `_highlight_selection()` drops such stale references.
- The Ctrl+click set (`_pending_connection_spheres`) is extra to the selection.
  - It stores `ManagedSphere` objects, so a deletion or a load empties it rather than
    shifting it.
  - `_highlight_selection()` paints its spheres cyan, and `selected_sphere_index` is the
    last one added.
  - Every path that sets a new single selection clears it: plain click, Q/E,
    `select_connection`, `add_sphere`.
- `get_selected_sphere_display_data()` returns 1 slot for a sphere, or 2–3 slots (one per
  sphere, in connection order) for a connection.
  - The InfoPanel slot handlers resolve spheres through `data["endpoint_indices"][slot]`:
    centre/radius spinboxes, Grow, Lock, To joint, To bone.
  - `selected_local_index` is None for a connection.

## After every edit
1. Call `update_mesh()` on the incident connections (`_incident_connections(i)`).
2. For per-sphere edits only (move, radius, lock), propagate to the X twin with
   `_apply_mirror_to_twin` or `_apply_lock_to_twin`. Divide is deliberately not mirrored.
3. Call `update_sphere_information_callback()`. It refreshes the InfoPanel, the
   "Selected → closest joint" button and the Divide button.
4. Call `mark_unsaved_changes()`, but only for real edits. Pose, shape and preview updates
   don't count.
5. In the `MainWindow` handler:
   - If `self._clip_planes_vtk` is set, call `_apply_clip_planes_to_all_actors()` when new
     actors were created.
   - Then `info_panel.update_selected_info()` and `plotter.render()`.

## Radii and Inflate
- While Inflate or the uniform-radius preview is on, the displayed radius is
  `base × factor` (scale mode) or a constant (absolute mode).
- Any computation must use `_base_radius(i)`. A new sphere's base radius goes into
  `_pre_uniform_radius_by_index`; `add_sphere(center, base_radius)` already does this.
- `set_center` and `set_radius` do nothing on a locked sphere. Use the `*_force` variants
  only for pose, preview or beta overrides.

## Binding (rigid or LBS)
- FK, in `update_sphere_positions`:
  - Rigid: `center = J[p] + R[p] @ rest_offset`.
  - LBS mode (`lbs_binding_enabled`, the Theta Controls checkbox, on by default in the GUI):
    the rigid centre is computed first and then overridden by `Σ w_k (R_k c_rest + t_k)`.
    With the box off, the loop runs exactly the old code.
  - Spheres with no binding are skipped.
- LBS joint weights live on the sphere (`ManagedSphere.lbs_joint_weights`); None means "not
  derived yet", and the next LBS pose derives them.
  - Any code that changes a sphere's `parent_joint`/`rest_offset` by *rebinding* must clear
    them. `assign_parent_joint` does; a new sphere starts without any.
  - `set_rest_mesh_points` (every rest-skeleton rebuild, i.e. every beta change) clears all
    of them.
  - `report_lbs_binding_status` only fills in missing weights; it never overwrites them.
- Turning a **posed** centre back into `rest_offset` must go through
  `_rest_offset_from_posed_center`, which inverts the active mode.
  - A rigid inverse in LBS mode makes the sphere jump on the next pose (measured: 24 mm for
    one sphere).
  - In LBS mode it alternates "solve the rest centre" and "re-derive the weights there", and
    stores the pair whose re-derivation would move the sphere least. A later beta tick or
    toggle then moves most spheres only as much as the body moves.
  - Crotch spheres can have two weight sets that derive each other, and they still jump
    (measured: up to 19 mm after a posed Fit and a 0.01 beta tick).
- `assign_parent_joint(i)` binds the sphere where it is **shown**.
  - At the rest pose (`_is_posed()` False) it picks the closest kintree bone
    (`_closest_bone`) on `rest_skeleton`, as it always did.
  - Under a pose it picks the closest bone on the posed `skeleton`, carries the centre back
    to the rest frame through it, and stores `_rest_offset_from_posed_center` (LBS weights
    included).
  - Treating a posed centre as a rest one poses it twice. Measured at the star-pose
    startup: up to 255 mm at the feet.
- When binding something derived from existing spheres, work in the rest frame, as
  `_bind_divided_sphere` does:
  - Average the endpoints' rest positions.
  - Restrict the parent to the endpoints' own joints, so no bumper spans a third joint.
- `rest_skeleton` is reset to None on a beta change. Spheres without a `secondary_joint`
  are then re-bound by `MainWindow._update_skeleton`.

## Geometry
- Shapes and distances come from `src/sphere_mesh.py` (`SphereMesh.Capsuloid` for a cone,
  `SphereMesh.Prysmoid` for a triangle), the one implementation shared with `SMFitting/`
  (`SMFitting/SphereMesh.py` only re-exports it). Edit it there, and run
  `scripts/test_sphere_mesh.py`.
- The primitives cache their axis and planes when constructed. `SphereConnection` builds a
  fresh one (`build_primitive()`) for every update and distance query; never keep one
  across sphere edits.
- The only geometry failure is coincident centres: `_build_connection` refuses them
  (`SphereConnection.has_coincident_centers`). Everything else is drawn and measured
  exactly, with one `warnings.warn` per primitive and no exception:
  - a pair where one sphere contains the other is that sphere (its cone has zero area);
  - a triangle with collinear centres or no tangent plane is its three edge capsuloids.
- Build connections through `_build_connection(indices)`: it returns None on a geometry
  failure. Validate before mutating, and roll back on failure.
- Adding a connection from the UI goes through the redundancy checks: the Ctrl+click
  dialog and Add Connection… (`add_connection_checked`).
  - `find_connection_covering` refuses a duplicate (same sphere set) or a cone on an
    existing triangle's edge.
  - `find_connections_made_redundant_by` lists the cones a new triangle deletes.
  - `try_add_connection` is the unchecked primitive. Tests rely on it to build duplicates.
- `prepare_connection_candidate` builds the connection unregistered, as a preview.
  `commit_connection_candidate` registers it and sets its visibility like the others.
  `discard_connection_candidate` deletes its actors. Never leave a preview unregistered
  but alive.
- The primitives interpolate centres and radii linearly. That makes a midpoint or centroid
  split with the mean radius exactly volume-preserving.
- `SphereConnection.get_distance` / `get_distances` are exact. `scripts/test_sphere_mesh.py`
  pins them to the closed forms in `src/topology/bumpers.py`.

## Persistence
- Loading rounds centres to 3 decimals.
- `save_to_json` writes only the known fields; it drops `quad_groups` and `generated_by`.
- Spheres missing `rest_offset` are bound on load.
