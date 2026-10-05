---
paths:
  - "src/sphere_manager.py"
  - "src/ui_widgets.py"
  - "src/sphere_connection.py"
  - "src/managed_sphere.py"
  - "src/spheremesh.py"
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
- A triangle does not list its edges as separate cones; `Trispheremesh` draws its rim.

## Selection
- `selected_sphere_index` and `selected_connection` exclude each other.
  - `select_connection` clears the sphere.
  - Selecting a sphere (`_on_sphere_interaction`, Q/E, `add_sphere`) clears the connection.
- Use `get_selected_connection()`. It returns None for a connection that has since been
  deleted, and `_highlight_selection()` drops such stale references.
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

## Binding (rigid, single-joint)
- FK, in `update_sphere_positions`: `center = J[p] + R[p] @ rest_offset`. Spheres with no
  binding are skipped.
- `assign_parent_joint(i)` picks the closest kintree bone (`_closest_bone`) on
  `rest_skeleton`, using the sphere's **current** centre. That is only correct at the rest
  pose.
- When binding something derived from existing spheres, work in the rest frame, as
  `_bind_divided_sphere` does:
  - Average the endpoints' rest positions.
  - Restrict the parent to the endpoints' own joints, so no bumper spans a third joint.
- `rest_skeleton` is reset to None on a beta change. Spheres without a `secondary_joint`
  are then re-bound by `MainWindow._update_skeleton`.

## Geometry
- `Spheremesh` raises `ValueError` on coincident centres. `Trispheremesh` never raises; on a
  degenerate triangle it silently drops the caps.
- Build connections through `_build_connection(indices)`: it returns None on a geometry
  failure. Validate before mutating, and roll back on failure.
- The primitives interpolate centres and radii linearly. That makes a midpoint or centroid
  split with the mean radius exactly volume-preserving.
- `SphereConnection.get_distance` is approximate (off by up to 3 cm). For exact distances
  use `round_cone_distances` and `prysmoid_distances` in `src/topology/bumpers.py`.

## Persistence
- Loading rounds centres to 3 decimals.
- `save_to_json` writes only the known fields; it drops `quad_groups` and `generated_by`.
- Spheres missing `rest_offset` are bound on load.
