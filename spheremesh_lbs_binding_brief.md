# Linear-Blend Skinning for Sphere Centres in the SphereMesh Tool — Implementation Brief

> Target: this repo's GUI editor (`main.py` → `src/ui_widgets.py` `MainWindow`, `src/sphere_manager.py`).
> Goal: when the pose (theta) changes, let each sphere centre follow a *blend* of up to four SMPL
> joints instead of rigidly following its one `parent_joint`. The blend weights are averaged from
> the body's own skin weights at the sphere's nearest rest-pose vertices. A GUI checkbox switches
> it: **off = today's rigid single-joint FK, unchanged**; on = LBS.
> Status: IMPLEMENTED 2026-10-30. §13 records where the implementation departs from this plan
> and what the tests measured; read it before the rest. The user took the decisions on
> 2026-10-30 (§11). Ported from the
> ClothSimulator26 brief of the same name (C++, implemented there 2026-10-26); its measured outcome
> is summarised in §12. Sections that only applied to the engine (cloth attachment, kinematic
> frames, bumper frames, `attachment_dump`, head snap, `AppSettings`) are dropped; §0 lists every
> change.
> `path:line` citations are against 2bf7c4b plus the uncommitted `src/ui_widgets.py` of 2026-10-30.

## 0. What changed from the C++ brief

- **Kept:**
  - N = 20 neighbours, with inverse-distance weights `1/(d + 0.1·r)`.
  - The limb gate on each vertex's dominant joint, a cap of 4 joints, and renormalising to 1.
  - Pose never touches the radius.
  - Toggling re-poses the body at once.
  - The rigid path stays untouched as the A/B reference.
  - A determinant floor of 0.05.
- **Changed:**
  - The skinning transforms come from SMPL's FK world transforms `T`. The tool computes them and
    then throws them away today (§2, §5.2).
  - "Rigid = weight-1 LBS" is **not** exact here. The rigid path uses joints regressed from the
    posed mesh, which sit up to 4.6 mm from the FK joints (§2, measured).
  - Weights are part of a sphere's binding and are stored on the `ManagedSphere`. They are derived
    when the sphere is bound and again on **every beta-slider tick**, not once at construction
    (decision 10). Measured: about 8 ms per template.
  - **New problem the engine never had:** the tool *edits* spheres while posed. Three things turn
    a posed centre back into a `rest_offset`: the InfoPanel spinboxes, To joint / To bone, and Fit
    spheres. Today they invert the rigid map, so in LBS mode they must invert the blend (§5.5).
  - The toggle is a Qt checkbox **inside Theta Controls**, checked at startup. It re-runs the
    theta-change path (§6).
- **Dropped:**
  - `KinematicFrame`, `Rinv`, the NaN path into the cloth, and `ClothProxyAttachment`. The tool
    moves points, not frames (§2.4).
  - The bumper representative sphere: connections rebuild from the sphere centres.
  - `attachment_dump`, which becomes a headless GUI test (§9).
  - The head-snap co-registration check: the template and the body share coordinates.
  - The sticky across "Apply scenario", `AppSettings`, and the OpenMP notes.

## 1. What exists already

| Piece | Where | State |
|---|---|---|
| SMPL skin weights, 6890 × 24 | `SphereManager.skinning_weights` (`src/sphere_manager.py:69`). Set at startup (`src/ui_widgets.py:2227-2231`) and on a gender switch (`:2738-2742`). | Ready. **Measured:** every vertex has exactly 4 non-zero weights, and the rows sum to 1 (to 2e-16). |
| Rest-pose vertices (theta = 0, current betas) | `SphereManager.rest_mesh_points` (`sphere_manager.py:70`), set in `_update_skeleton` (`ui_widgets.py:2527-2530`) | Ready. Stored for the commented-out skin-weight secondary joint and unused since. |
| Rest skeleton (theta = 0, current betas) | `_update_skeleton`, `ui_widgets.py:2510-2526` | Its rotations are the identity. It is reset to None on every beta change (`:2758`) and on a gender switch (`:2743`). |
| FK world transforms `T`, 24 × 4 × 4 | Returned by `my_function_forward` (`src/smpl_forward_compat.py:33-36`) | **Thrown away.** `update_smplmodel` (`ui_widgets.py:2786-2789`) passes on only `Rs`, so `Skeleton` (`:201-242`) never sees `T`. |
| Rigid bind | `assign_parent_joint`, `sphere_manager.py:1110-1185`: the parent end of the closest kintree bone (`_closest_bone`, `:1073-1095`), with `rest_offset` in the rest frame | The anchor for LBS. Unchanged. |
| Rigid FK | `update_sphere_positions`, `:1196-1235`: `center = J[p] + R[p] @ rest_offset` | The off mode and the A/B reference. Its code stays as it is. |
| Dormant two-joint blend | `secondary_joint` (child end of the closest bone) and `lbs_weight` (position along that bone). Set in `assign_parent_joint` (`:1127-1144`) and `_bind_divided_sphere` (`:627-635`), saved to JSON. The blend itself (`:1213-1224`) is commented out, and so is an older skin-weight variant (`:1146-1183`). | Left alone (decision 11). The new runtime array is named `lbs_joint_weights` so it can't be confused with `lbs_weight`. |
| Posed body | `smpl_pytorch` `SMPL.forward`, `pose_blend_shapes=False` by default; the tool never passes True | Pure LBS with no pose correctives (§2) |
| Topology pose sweep | `pose_spheres_like_tool` (`src/topology/anatomy.py:754-773`), used by `validate.py:1269-1324` and so by `compare_topologies.py --poses` | A rigid replica. **It doesn't import today:** 165 committed conflict-marker lines across `anatomy.py`, `validate.py`, `generate.py` and `export.py`. |

## 2. The key identity in this tool, and where it breaks

In `smpl_pytorch` (`smpl.py:205-233`) the bind rotations are the identity, so the inverse bind of
joint k is `translate(−J_rest_k)`. Each joint's skinning transform is then:

```
A_k = T_k · inverse_bind_k        R_k = T_k[:3,:3]        t_k = T_k[:3,3] − R_k · J_rest_k
v   = Σ_k w_vk · (R_k · v_rest + t_k)
```

**Measured** with a scratch script over the 24 CSV poses, both genders, betas 0: rebuilding the body
from this formula matches the model's vertices to 8.9e-16 m. With `pose_blend_shapes=False` the
rendered body is *pure* LBS. LBS sphere centres therefore follow exactly the deformation model of
the body they stand in for, with no pose-corrective residue.

The rigid FK, with `rest_offset = R_rest[p]ᵀ · (c_rest − J_rest[p])` and `R_rest = I`:

```
rigid (today):   c = J_reg[p]   + R_p · rest_offset
weight-1 LBS:    c = A_p · c_rest = T_p[:3,3] + R_p · rest_offset
```

The two expressions are the same **except for the joint position**. The tool's `Skeleton.joints` is
`J_regressor @ posed vertices` (`smpl_forward_compat.py:32`), not the FK translation `T_p[:3,3]`.
The two agree at theta = 0 and drift apart once the body is posed. **Measured**, 24 CSV poses:

| regressed vs FK joint, worst pose | male | female |
|---|---|---|
| max over joints | 4.6 mm (right shoulder) | 4.4 mm (left shoulder) |
| median over joints | 1.8 mm | 1.2 mm |
| pelvis | 1.1 mm | 1.7 mm |

Consequences:

1. **LBS mode builds `A_k` from `T`**, never from `Skeleton.joints` plus `Rs`. A regressed joint
   inside a blend term would shift that term up to 4.6 mm away from the transform the body's
   vertices use. So `T` has to reach the manager (§5.2).
2. **Off and on differ by up to about 4.6 mm, even for a sphere whose whole weight is on its parent
   joint.** That is LBS being the more faithful of the two, not a bug. It is also why the regression
   check is "off is bit-identical to today" and not "on with one joint equals off".
3. **At theta = 0 the two modes coincide**: every `A_k = I` and `J_reg = J_fk`. That makes it the
   cheapest test of the whole change (§9, test 2).
4. **Points, not frames.**
   - The tool moves only sphere centres. Radii don't depend on the pose, and connections rebuild
     from the centres (`SphereConnection.update_mesh`).
   - The LBS centre `Σ w_k · (A_k · c_rest)` is a convex combination of finite points, so it is
     always finite.
   - The determinant floor is kept for two reasons only: parity with the engine (§5.4), and the
     posed → rest inverse (§5.5), which genuinely needs it.

## 3. Decisions

| # | Decision | vs C++ |
|---|---|---|
| 1 | **N = 20** neighbours (`LBS_NEIGHBOUR_COUNT`). A starting value, not a measured optimum. | kept |
| 2 | **Inverse-distance** vertex weights `1 / (d + 0.1·r)`. `r` is the sphere's **base** radius, `_base_radius(i)` (`sphere_manager.py:522`), so Inflate and the uniform-radius preview never change the weights. | kept; `r` pinned down |
| 3 | **Limb gate on each vertex's dominant joint**, anchored on the sphere's `parent_joint`: the joint itself, its kintree parent and every child. This is option A in §3.1, the plan as written and the engine's rule. | kept, as the engine has it |
| 4 | **Cap at 4 joints** (`LBS_MAX_JOINTS = 4`), renormalised to sum 1. The gate can admit 5 joints (spine3: spine2, neck and both collars besides itself); the cap keeps the strongest 4. SMPL never blends more than 4 per vertex (measured). Stored as one dense `(24,)` float64 row per sphere, with at most 4 non-zero entries. | kept; dense storage |
| 5 | **Pose never touches the radius.** Already true: a theta change calls `update_smplmodel(update_radius=False, …)` (`ui_widgets.py:2765`), and `update_sphere_radius` is commented out (`:2815`). | kept trivially |
| 6 | **On by default** in the GUI: the checkbox starts checked and `MainWindow` syncs the manager from it. A bare `SphereManager`, as used by headless scripts, stays rigid until told otherwise (§5.3). | kept (the engine also defaults to LBS) |
| 7 | ~~Bumper representative sphere~~: the tool keeps no per-connection frames. | dropped |
| 8 | ~~attachment_dump~~ is replaced by the headless GUI test `scripts/test_lbs_binding.py` (§9). | replaced |
| 9 | ~~BodyConform~~ is replaced by a measurement through the real `SphereManager` (§10.3). `pose_spheres_like_tool` stays rigid. | replaced |
| 10 | **Weights belong to the binding, and every beta-slider tick recomputes them.** Three things clear them: `assign_parent_joint`, `_bind_divided_sphere`, and every rest-skeleton rebuild (each beta tick, a gender switch, startup). LBS-mode FK then derives any missing weights lazily, at the new shape. A drag never re-derives them. ~~A spinbox edit or Fit spheres never re-derives them~~: superseded by §13.1, where posed edits re-solve the weights for the new rest centre. Measured cost: about 8 ms per tick (§12.3). | new |
| 11 | **Not persisted.** The weights are derived at runtime and JSON stays version 2. `secondary_joint`/`lbs_weight` are still read and written, and neither mode uses them. The mode lasts for the session only, like Inflate. | new |
| 12 | **The mode applies wherever FK runs**, not only on a theta tick. A beta change in a posed body, a gender switch, a load and the toggle all go through `update_sphere_positions`. Otherwise one beta tick would flip the spheres from LBS back to rigid. | new |

### 3.1 The limb gate in SMPL terms

The C++ gate is one hop in the bone tree: a candidate vertex counts only if its dominant joint is
the bind joint, its kintree parent, or a child. Gate first, then take the nearest 20 survivors. The
reasoning is unchanged: nearest-then-gate leaves a crotch sphere with 5 usable neighbours out of 20.

| Option | Gate for a sphere bound to joint `p` | Note |
|---|---|---|
| **A: as planned (decided)** | `{p, parent(p)} ∪ every child of p` | What ClothSimulator26 runs, so the tool previews the engine. The pelvis admits both hips and spine1; spine3 admits the neck and both collars. The C++ run measured pelvis and hips getting worse under this gate (§12). |
| B: nearest child (C++ §12.2 fix) | `{p, parent(p), c}`, where `c` is the child end of the closest bone hanging off `p`: `_closest_bone(c_rest, allowed_parent_joints={p})[1]` | **Not implemented and not measured.** Kept on record as the documented alternative, should the pelvis and hips get worse here too. Caveat: a midline sphere near the pelvis would pick one hip by a near tie and follow that leg alone. |

```python
def _lbs_gate_joints(self, parent_joint: int) -> set:
    """Joints whose vertices may lend the sphere their skin weights: one hop from parent_joint."""
    kintree = self.rest_skeleton.kintree
    gate_joints = {int(parent_joint)}
    if kintree[parent_joint] >= 0:
        gate_joints.add(int(kintree[parent_joint]))
    gate_joints.update(int(child) for child in np.flatnonzero(kintree == parent_joint))
    return gate_joints
```

The gate comes from the kintree and the bound joint, never from the stored `secondary_joint`. Saves
written by the old skin-weight logic may hold a secondary joint that isn't a child, and
`_bind_divided_sphere` stores None for a leaf. A leaf `p` (hand, foot, head) has no children, so its
gate is `{p, parent(p)}`. SMPL's root has parent −1 in the kintree.

Engine parity covers the algorithm, not the bits. The engine picks its own follow bone and derives
the weights from its FBX mesh after a head snap. The C++ known limit still holds: an accepted vertex
can carry a minor influence on a far joint, and the top-4 truncation suppresses it.

## 4. Derivation: `SphereManager._derive_lbs_joint_weights`

- **Inputs:**
  - `skinning_weights` (6890 × 24), and the dominant joint per vertex (`argmax`, hoisted into
    `set_skinning_weights` because it doesn't depend on shape).
  - `rest_mesh_points` (theta 0, current betas).
  - The sphere's **rest** centre `c_rest = rest_skeleton.transform_point(rest_offset, parent_joint)`.
    This is the same rest point `_bind_divided_sphere` uses (`:601-603`). It is *not*
    `get_center()`, which is posed.
  - Its base radius, and the gate (§3.1).

```python
def _derive_lbs_joint_weights(self, sphere_index: int) -> Optional[np.ndarray]:
    """Joint weights (24,) averaged from the skin weights of the sphere's nearest rest vertices."""
    sphere = self.spheres[sphere_index]
    parent_joint = sphere.get_parent_joint()
    rest_center = self.rest_skeleton.transform_point(sphere.get_rest_offset(), parent_joint)
    gate_joints = self._lbs_gate_joints(parent_joint)
    # Gate first, then the nearest survivors (see §3.1).
    candidate_vertices = np.flatnonzero(np.isin(self._dominant_joint_per_vertex, list(gate_joints)))
    if candidate_vertices.size == 0:
        return None
    distances = np.linalg.norm(self.rest_mesh_points[candidate_vertices] - rest_center, axis=1)
    neighbour_count = min(LBS_NEIGHBOUR_COUNT, candidate_vertices.size)
    nearest = np.argpartition(distances, neighbour_count - 1)[:neighbour_count]
    vertex_weights = 1.0 / (distances[nearest] + LBS_DISTANCE_SOFTEN * self._base_radius(sphere_index))
    joint_weights = vertex_weights @ self.skinning_weights[candidate_vertices[nearest]]
    joint_weights[np.argsort(joint_weights)[:-LBS_MAX_JOINTS]] = 0.0
    weight_sum = joint_weights.sum()
    if weight_sum <= 1e-8:
        return None
    return joint_weights / weight_sum
```

- **None** means the sphere stays rigid in LBS mode and is counted in the status line.
- **One sphere at a time, not one matrix.** Measured on the real templates (60 and 61 spheres):
  this loop takes 7.6–8.3 ms (median) for a whole template. A fully vectorised `(spheres × 6890)`
  distance matrix takes 19.5–20.1 ms, because the broadcast costs more than the loop. So the
  per-beta-tick cost is about 8 ms (§12.3).
- **Status line:** a `print(...)` like the module's other status output. Example: `[lbs] joint
  weights for 60 spheres: neighbour distance mean 1.04 radii, worst 1.43 (sphere 38); 0 with no
  neighbour`.
  - It is printed when the box is switched on, after a load and after "Reassign all parents".
    **It is never printed on a slider tick**, which would print one line per tick. The derivation
    stores its statistics, and the status line reads them.
  - This replaces the C++ co-registration readout. Here it falsifies a *stale binding*, for
    example "Reassign all parents" pressed while posed, which is the documented
    only-correct-at-rest trap.
  - The readout is in radii, as the C++ implementation ended up using (§12).

## 5. Code changes

### 5.1 `src/managed_sphere.py`

`self.lbs_joint_weights = None` sits beside the binding attributes (`:29-35`), with
`set_lbs_joint_weights` / `get_lbs_joint_weights`. It is not part of save/load.

Because the weights live on the sphere object, nothing else needs bookkeeping: deleting and
re-indexing, Divide's append and duplicate connections all leave them on the right sphere. The
sphere-index-keyed beta arrays have exactly this alignment problem (`.claude/rules/gui-editing.md`).

### 5.2 Getting `T` to the manager: `src/ui_widgets.py`

- `Skeleton.__init__(..., world_transforms=None)` stores a `(J, 4, 4)` float64 array, or None.
- `update_smplmodel` passes `T` on: `_update_skeleton(..., world_transforms=T)` (`:2786`, `:2803`),
  which puts `T.detach().cpu().numpy()` into the current `Skeleton` (`:2508`). The rest skeleton
  doesn't need it.
- With the installed `smpl_pytorch`, `T` is never None. If it is (the legacy fallback,
  `smpl_forward_compat.py:37-40`), LBS mode stays rigid and prints the reason once.

### 5.3 `src/sphere_manager.py`: state

```python
LBS_NEIGHBOUR_COUNT = 20          # rest vertices averaged per sphere
LBS_DISTANCE_SOFTEN = 0.1         # inverse-distance softening, as a fraction of the sphere's base radius
LBS_MAX_JOINTS = 4                # joints kept per sphere, as SMPL keeps per vertex
LBS_MIN_BLEND_DETERMINANT = 0.05  # |det Σ w·R| floor; below it the sphere uses its rigid bind
```

- **`self.lbs_binding_enabled = False` in the manager.** The GUI default of *on* (decision 6) comes
  from the checkbox, synced at `MainWindow` init the way Mirror edits is (`ui_widgets.py:2236`).
  Headless scripts that build a bare `SphereManager` therefore keep rigid semantics; for example,
  `test_topology_roundtrip.py` §3 compares against the rigid `pose_spheres_like_tool`.
- `set_lbs_binding_enabled(enabled)` is the only setter.
- `invalidate_lbs_joint_weights()` clears every sphere's weights. `_update_skeleton` calls it when it
  rebuilds the rest skeleton (`ui_widgets.py:2511-2530`), which happens on each beta tick and on a
  gender switch.
- `_dominant_joint_per_vertex` is filled in `set_skinning_weights`.

### 5.4 `update_sphere_positions`: an override, not a replacement

The rigid computation stays exactly as it is (`p1`, `:1205-1207`); LBS overwrites `new_center` only
when the box is on and the blend is healthy. With the box off the same lines run as today, so the
result is bit-identical (§9, test 1).

```python
skin_rotations, skin_translations = self._skinning_transforms()   # both None unless LBS is usable
...
            new_center = p1
            if skin_rotations is not None:
                joint_weights = self._lbs_joint_weights_for(i)       # derives lazily, decision 10
                if joint_weights is not None:
                    blend_rotation = np.einsum("k,kij->ij", joint_weights, skin_rotations)
                    if abs(np.linalg.det(blend_rotation)) >= LBS_MIN_BLEND_DETERMINANT:
                        rest_center = self.rest_skeleton.transform_point(rest_offset, parent)
                        new_center = blend_rotation @ rest_center + joint_weights @ skin_translations
                    else:
                        singular_blend_count += 1
            sp.set_center_force(new_center)
```

`_skinning_transforms()` builds the 24 transforms once per call: `R_k = T[:, :3, :3]` and
`t_k = T[:, :3, 3] − einsum("kij,kj->ki", R_k, rest_skeleton.joints)`. The rest skeleton has the
same betas as the current one, because `_update_skeleton` rebuilds it before FK on a beta change.
Spheres with no binding are still skipped. Locked spheres still follow the pose (`set_center_force`),
as today.

Cost:
- **Theta tick:** the 24 transforms plus about 60 3×3 einsums. That should be noise next to 60
  `set_center_force` widget updates.
- **Beta tick:** the same, plus the full re-derivation, which measured about 8 ms.
- **Confirm both** with `measure_time`, which is already imported and wraps this function in a
  commented-out line (`ui_widgets.py:2811`). `test_random_pose_shape.py` also reports per-step slider
  times (§9, test 10).

### 5.5 Posed → rest inversions must use the active mode

`set_sphere_center_global` (`sphere_manager.py:878-884`) turns an edited **posed** centre back into a
`rest_offset` by inverting the rigid map. Three actions use it: the InfoPanel centre spinboxes
(`ui_widgets.py:1675`), To joint (`:3585`) and To bone (`:3630`). `fit_spheres_to_mesh` repeats the
same inversion at `:1506-1513`.

In LBS mode the forward map is the blend. A rigid inverse would make the sphere jump on the next pose
update by the LBS − rigid difference, which is centimetres near the joints. Both sites call one
helper instead:

```python
def _rest_offset_from_posed_center(self, sphere, posed_center) -> np.ndarray:
    """The rest_offset that the active binding mode maps to posed_center under the current pose."""
```

- **Rigid:** today's expression, unchanged: `np.dot(R_current.T, posed_center - J_current)`.
- **LBS:**
  - `rest_center = np.linalg.solve(Σ w·R, posed_center − Σ w·t)`, then
    `rest_offset = R_rest[p]ᵀ · (rest_center − J_rest[p])`.
  - It uses the sphere's **existing** weights (decision 10), so the forward map gives back
    `posed_center` exactly. The next beta tick re-derives the weights at the new shape, but that
    moves the whole body anyway.
  - Below the determinant floor, or with no weights, it falls back to rigid, which is the same
    choice the forward map made.
- **At the rest pose** both modes reduce to the identity, so edits made at theta 0 behave as today.

Same in both modes, with no change:
- A drag (`_on_sphere_interaction`, `:790`) and the mirrored twin (`_apply_mirror_to_twin`,
  `:253-259`) don't touch `rest_offset`, so they snap back on the next pose update. That is the
  documented behaviour.

## 6. GUI: the toggle

- **Widget:** a `QCheckBox("LBS binding (blend joints)")`, **checked**, inside `ThetaSliderGroup`
  (`ui_widgets.py:861`). `_setup_ui` adds it to the layout under the Random pose slider
  (`:941-943`).
  - `_toggle_visibility` hides and shows it with the group's other widgets.
  - The group starts collapsed (`:874-881`), so at startup LBS is on while the box is out of sight
    until Theta Controls is opened.
  - "Reset" (`:981-994`) resets theta only and leaves the box alone.
- **Tooltip:** "Each sphere follows a weighted blend of up to 4 joints, averaged from the SMPL skin
  weights at its 20 nearest body vertices, which is the body's own deformation model. Off = rigid
  single-joint binding (parent joint only)."
- **Signal:** the group exposes `lbsBindingToggled = pyqtSignal(bool)`, next to `valuesChanged`
  (`:864`).
- **Wiring:** `MainWindow` connects the signal where it connects `valuesChanged` (`:2154-2156`). At
  init it syncs the manager from the box, as it does for Mirror edits (`:2236`).
- **Handler:**

```python
def _on_lbs_binding_toggled(self, enabled: bool):
    """Switch sphere FK between the rigid single-joint bind and LBS, and re-pose at once."""
    self.sphere_manager.set_lbs_binding_enabled(enabled)
    self.update_smplmodel(update_radius=False, update_pose=True)
```

The handler makes the same call as `_on_theta_changed` (`:2762-2765`), so the toggle is a theta
change with the same theta. Beta offsets, the heatmap refresh (`:2817`, by fingerprint), the
InfoPanel refresh (via `update_sphere_information_callback`) and the render all come along. Toggling
is not an edit, so there is no `mark_unsaved_changes()`. It creates no actors, so clip planes need
nothing.

**No InfoPanel readout of joints or weights** (decided). The status line (§4) is the only
diagnostic.

## 7. Interaction with the rest of the editor

| Feature | Effect |
|---|---|
| Beta slider | Each tick rebuilds the rest skeleton, which clears every sphere's weights. They are re-derived at the new shape in the same `update_smplmodel` call: about 8 ms (decision 10). |
| Beta offsets loaded | `apply_beta_offsets` runs, then `update_sphere_positions` on pose updates. Both modes overwrite the beta positions from `rest_offset`; that already happens today. LBS takes `c_rest` from `rest_offset` like rigid does, so the two agree on what "rest" means. A beta change with offsets loaded runs no FK (`ui_widgets.py:2806-2809`), so its weights are derived on the next pose update instead. |
| Inflate / uniform radius | Radii are untouched. The weights use the base radius (decision 2). |
| Divide (`divide_connection`, `:664`) | `_bind_divided_sphere` clears the new sphere's weights, and they are derived at its rest point on the next LBS pose. In LBS mode while posed, the new sphere lands at `LBS(rest midpoint)`. That is not exactly the posed midpoint of its endpoints, because their weights differ (rigid with a shared joint is exact). §9 test 8 reports the gap. The `_bind_divided_sphere` docstring ("stays at their midpoint in every pose", `:580-582`) becomes a rigid-mode statement and needs saying so. |
| Delete | Nothing to do: the weights travel with the sphere object. |
| Mirror edits | No change. Twins get their weights independently from the near-symmetric SMPL mesh. The test reports the left/right weight asymmetry. |
| Reassign all parents | Clears and re-derives every sphere's weights at the current betas. |
| Gender switch | The template's new spheres have no weights and derive them lazily. `set_skinning_weights` refreshes the dominant joints. |
| Save / load | No new fields. Saving while posed already writes posed centres; the mode only changes *which* posed centres. |
| Heatmap | Recomputes after the toggle, because the fingerprint changes when the centres move. |
| Unbound spheres (`add_sphere` binds nothing, `:393`) | Skipped in both modes, as today. |
| Topology pipeline | Unchanged and rigid. With LBS on by default, the GUI's posed view **no longer matches** what `pose_spheres_like_tool` predicts, nor the "~46 mm rigid-binding floor" in `.claude/rules/topology.md`. Giving the validator an LBS option waits until the conflict markers in `src/topology/` are resolved, and is out of scope here. The topology rule needs a note that the floor is the rigid one. |

## 8. Implementation order

1. **Pass `T` through** (§5.2). Behaviour doesn't change; run the existing GUI tests.
2. **The core:** the `ManagedSphere` attribute, the gate, the derivation, the FK override and the
   setter. The manager stays rigid until told otherwise. Tests 1–5.
3. **The inverse helper** (§5.5). Test 6.
4. **Weight lifetime hooks** (decision 10), including invalidation on every rest-skeleton rebuild.
   Tests 8–9.
5. **The checkbox in Theta Controls**, its signal, the handler and the init sync, with the box on by
   default (§6). Tests 7, 10 and 11.
6. **Existing tests.** The GUI now starts in LBS mode, so every GUI test runs in LBS unless it says
   otherwise. One existing check asserts the rigid rule: `test_divide_connection.py:366-370`,
   "posed: shared-joint sphere stays at the endpoints' midpoint".
   - Run that check with the box off (it tests the rigid binding).
   - Add an LBS variant that reports the gap, as in test 8.
   - The other GUI tests check geometry, timing or rest-pose centres, so they are mode-independent.
   - `test_topology_roundtrip.py` uses a bare manager, so it stays rigid. It doesn't import today
     anyway (conflict markers).
7. **The measurement** (§10.3). Its numbers go into §12.3 of this brief.
8. **Review and docs:**
   - Run the **spheremesh-reviewer**, then the **test-runner** over all GUI tests.
   - Update the docs:
     - A CLAUDE.md feature-log entry.
     - The fact "Binding is rigid and single-joint" becomes "LBS by default in the GUI, rigid with
       the box off".
     - The Binding section of `.claude/rules/gui-editing.md`.
     - A note in `.claude/rules/topology.md` that the 46 mm floor is the rigid one.

## 9. Tests: `scripts/test_lbs_binding.py`

Headless with the real `MainWindow`, modelled on `scripts/test_divide_connection.py`. It loads
`data/template_male_1.0.json` and uses 3 rows of `data/random_pose.csv`.

1. **Off is today.** With the box switched off explicitly, every bound centre equals
   `J_reg[p] + R[p] @ rest_offset` exactly (`np.array_equal`) under every pose.
2. **Rest pose.** At theta 0, centres with the box on equal centres with it off, to 1e-12.
3. **Weights.**
   - Every bound sphere has at most 4 non-zero joints, summing to 1 (to 1e-12).
   - Each sphere's *strongest* joint is inside its gate. Not *all* of its joints: an admitted
     vertex's minor influences can sit outside the gate (§13).
   - The status line is printed and the neighbour distance in radii is reported.
4. **Agreement with the body.** A sphere placed at rest vertex v, with its weights forced to v's
   SMPL row, follows posed vertex v to 1e-9 under every pose. This end-to-end check catches a blend
   built from `J_reg` instead of `T`.
5. **A single joint.** With the weights forced one-hot on `p`, the posed centre equals the rigid
   centre plus `(T_p[:3,3] − J_reg[p])`, to 1e-12. This pins the §2 difference.
6. **The round trip.** In LBS mode while posed, call `set_sphere_center_global(i, c)` and re-apply
   the pose: the centre stays at c (to 1e-9). The same check in rigid mode pins today's behaviour.
   Repeat once through Fit spheres.
7. **The toggle**, driven through the Theta Controls checkbox, not only the manager.
   - It doesn't set `has_unsaved_changes`.
   - When posed it moves the centres; at rest it doesn't.
   - With the heatmap on, the heatmap recomputes.
8. **Weight lifetime.**
   - After deleting a sphere, the others keep their weight rows (by object identity).
   - Divide gives the new sphere weights on the next LBS pose and reports the gap from the posed
     midpoint.
   - Reassign all parents re-derives the weights.
   - A beta change re-derives them too: after a CSV shape, every sphere's weights equal a fresh
     `_derive_lbs_joint_weights` at that shape, to 1e-12.
9. **Save.** The JSON saved at the rest pose with the box on has the same fields as the one saved
   with it off, and every number agrees to 1e-12. Not byte-identical: the FK translations at rest
   can differ from `J_rest` in the last bit.

10. **Default.** A fresh `MainWindow` has the box checked and `lbs_binding_enabled` True; a bare
    `SphereManager` has it False.
11. **Slider cost.** Per-step times for the beta and theta sliders with the box on and off, reported
    (not gated), using `test_random_pose_shape.py`'s `timing_summary`. Expected beta-step overhead:
    about 8 ms.

Every existing GUI test must still pass. After the change they run in LBS mode, except the one rigid
check moved to box-off in §8 step 6.

## 10. Validation

### 10.1 Mode-off regression (do this first)

Test 1, plus the existing GUI tests. The override-don't-replace structure of §5.4 is what makes
"bit-identical" checkable rather than hoped for.

### 10.2 Binding sanity (do this second)

Read the status line on both templates. A worst-case neighbour distance far above one radius means
a stale binding (rest offsets taken while posed), not a weighting problem. Fix that first; no
tuning of N or the falloff helps.

Before implementation (scratch prototype, §12.3):
- **Male:** mean 1.04 radii, worst 1.43.
- **Female:** mean 1.11 radii, worst **4.26: sphere 32** (spine3, r = 2 cm, centre
  (−0.067, 0.084, 0.053)). That is about 8.5 cm to its 20 neighbours, and the one sphere to look at
  first: either a small sphere sitting deep in the chest, or a gate that starves it.

### 10.3 Does it reduce the divergence?

Write `scripts/compare_binding_modes.py`. It drives the **real** `SphereManager` headless, not a
replica. The lesson from C++ §12.1: the tool's replicated bind graded a different space, and its
check T0 failed on every asset while every other check passed.

- **Templates:** `data/template_male_1.0.json` and `data/template_female_1.0.json`.
- **Poses:** the 24 CSV poses.
- **Configurations:** rigid versus LBS with the decided gate (A).
- **Reported:**
  - Per region (each vertex's dominant joint): the maximum over poses of the body-vertex protrusion
    (`−get_heatmap` distance).
  - Overall max / p95 / median.
  - The animation term `max_t d − d(rest)`.
  - Per-pose time of `update_sphere_positions` in each mode.
- **Baseline to beat:** `.claude/rules/topology.md` records a posed pop-out of about 46 mm as "the
  GUI's rigid-binding floor".

**Expected (a prediction, not a target):** C++ §12 measured 13 of 16 regions improving, the
animation term halving and the pelvis and hips getting worse. That was with gate A on a mocap take
and the engine's FBX.

### 10.4 In-app A/B: by the user

This step was not run by the implementer. In the app: Random pose (CSV), heatmap on, toggling the
box. Watch the shoulders, elbows, hips and crotch, and the triangles across the torso.

## 11. Decisions taken (user, 2026-10-30)

| # | Question | Decision | Where it lands |
|---|---|---|---|
| Q1 | Limb gate | **As in the plan**: bind joint, its parent and every child (option A, the engine's rule), with at most 4 joints kept per sphere (`LBS_MAX_JOINTS = 4`) | Decisions 3–4, §3.1 |
| Q2 | Default state of the box | **On** | Decision 6, §5.3, §6, §8 step 6 |
| Q3 | Weight lifetime | **Recomputed on every beta-slider tick** (and when a sphere is bound) | Decision 10, §5.3, §7 |
| Q4 | Where the checkbox goes | **Inside Theta Controls**, under the Random pose slider | §6 |
| Q5 | InfoPanel readout of joints and weights | **None** | §6 |

Carried from C++ §11, and to be tuned only once §10.3 gives a number:
- N (try 8 and 40).
- The softening 0.1, and inverse versus inverse-squared falloff.
- A per-influence gate instead of the dominant-joint one.

## 12. Evidence from the C++ implementation (ClothSimulator26, 2026-10-26)

### 12.1 What carries over

- **Rigid stayed bit-identical** under the override-don't-replace structure (§5.4 here).
- **No refused blends:** zero below the determinant floor, on every pose of all three assets. The
  floor is a guard, not a regular path.
- **The replica trap:** a tool that re-implements the bind silently grades a different space.
  Hence §10.3 drives the real class.
- **The readout:** the co-registration readout ended up in sphere radii (§4 here).

### 12.2 Measured there: `body_conform`, 00122/take8, `template_male_1.0`, gate A

Values are the maximum over the take of the distance d per region, in cm.

| region | rigid | LBS |
|---|---|---|
| right / left shoulder | 2.92 / 2.50 | 2.44 / 2.07 |
| spine3 / spine1 | 2.90 / 1.75 | 2.44 / 1.37 |
| left / right knee | 2.60 / 2.30 | 2.06 / 1.82 |
| **pelvis** | **3.21** | **3.54** |
| **right / left hip** | **1.57 / 1.55** | **2.05 / 1.71** |
| overall max / p95 / median | 3.21 / 1.37 / 0.33 | 3.54 / 1.21 / 0.23 |
| animation term `max_t d − d(bind)`, max / median | 3.63 / 0.21 | 1.70 / 0.11 |

LBS improved 13 of 16 regions and halved the divergence caused by animation. It made exactly the
pelvis and both hips worse: a pelvis-bound sphere admits both hips at one hop and averages left
against right. Gate A keeps that behaviour (Q1); §10.3 shows whether it repeats here.

### 12.3 Measured here

**Before implementation (scratch prototypes, 2026-10-30; betas 0, gate A, N = 20, 0.1·r, top 4).**
These ran outside the tool, against the SMPL models and the two startup templates.

| | male (`template_male_1.0.json`) | female (`template_female_1.0.json`) |
|---|---|---|
| bound spheres | 60 of 60 | 61 of 61 |
| body rebuilt from `Σ w·(T_k · inverse_bind_k)` vs model vertices, 24 CSV poses | 8.9e-16 m | 8.9e-16 m |
| regressed vs FK joint, worst pose: max / median over joints | 4.6 / 1.8 mm | 4.4 / 1.2 mm |
| derivation for the whole template, per-sphere loop (median of 20) | 7.6 ms | 8.3 ms |
| the same as one `(spheres × 6890)` matrix | 19.5 ms | 20.1 ms |
| joints per sphere after the cap | 4 for every sphere | 4 for every sphere |
| weight left on the rigid `parent_joint`: median / min | 0.66 / 0.13 | 0.66 / 0.18 |
| spheres with less than half their weight on the rigid parent | 13 | 20 |
| neighbour distance in radii: mean / worst | 1.04 / 1.43 (sphere 38) | 1.11 / 4.26 (sphere 32) |

The two rows about the rigid parent show how far LBS will move things. About a quarter of the spheres follow some
other joint more than their rigid parent. That is expected: the tool binds to the *parent* end of
the closest bone, so a sphere near the knee is rigidly a hip sphere while its vertices follow the
knee.

**After implementation: `scripts/compare_binding_modes.py`** (2026-10-30). Startup templates,
betas 0, the 24 CSV poses, gate A. "Through skin" = the collider poking out through the skin (−d);
"outside" = skin standing off the collider (+d). Each value is the worst over poses and vertices,
in mm.

| | male rigid | male LBS | female rigid | female LBS |
|---|---|---|---|---|
| regions better by > 0.5 mm, through skin (of 24) | | 15 (worse 3) | | 9 (worse 9) |
| regions better by > 0.5 mm, outside (of 24) | | 15 (worse 5) | | 11 (worse 8) |
| **left / right hip, through skin** | **49.0 / 55.4** | **79.3 / 77.6** | **45.1 / 44.1** | **82.6 / 76.6** |
| pelvis, through skin | 56.1 | 45.4 | 44.8 | 46.3 |
| left / right knee, through skin | 25.3 / 30.5 | 15.9 / 22.3 | 35.8 / 28.5 | 20.1 / 31.6 |
| left / right collar, through skin | 29.8 / 25.6 | 24.5 / 19.2 | 31.7 / 20.1 | 29.1 / 17.4 |
| left / right shoulder, through skin | 58.3 / 53.9 | 52.3 / 55.1 | 55.2 / 52.0 | 54.8 / 50.9 |
| through skin, worst (all regions) | 58.3 | **79.3** | 55.2 | **82.6** |
| through skin, mean over poses of each pose's worst | 49.9 | 49.0 | 45.1 | 45.9 |
| outside, median over vertices | 10.9 | 9.2 | 10.6 | 10.6 |
| animation term, outside: max / median | 37.1 / 1.7 | **22.1 / 0.9** | 42.8 / 1.9 | **23.7 / 1.0** |
| animation term, through skin: median | 5.9 | 5.1 | 8.0 | 5.1 |

- **What LBS does well:** it roughly halves how far the skin moves away from the collider under
  animation, for both genders, and it shrinks the median motion-induced protrusion.
- **What it does badly:** at the hips the collider pokes through the skin up to about 30 mm
  deeper than rigid. That makes the hips the overall worst case: 79 / 83 mm against 58 / 55 mm.
  This is the C++ §12 pelvis/hip warning, larger here. §13.3 has the diagnosis.
- The feet, head and hands show large "outside" values in both modes: those regions are
  truncated by design, and the numbers don't depend on the mode.

## 13. Implementation outcome (2026-10-30)

### 13.1 Where the implementation departs from this plan

- **Invalidation.** It happens inside `set_rest_mesh_points` and `set_skinning_weights`, rather
  than as a separate call from `_update_skeleton`. Whatever changes the rest data drops the
  weights, headless managers included.
- **The derivation's return value.** `_derive_lbs_joint_weights` returns
  `(weights, mean neighbour distance in radii)`.
  - "No neighbour" is stored as an all-zero row, so the sphere isn't re-derived on every pose.
  - None still means "not derived yet".
- **Divide.** `_bind_divided_sphere` clears nothing; the new sphere is created without weights.
- **Status line.**
  - `report_lbs_binding_status()` re-derives every weight and prints the line. It runs on
    toggle-on, the startup load, `_post_load_refresh` (load, import, gender switch) and Reassign
    all parents.
  - `set_lbs_binding_enabled` prints the mode.
  - Singular blends are counted only when the status line prints, not on every pose. Every
    print in the tests and runs showed 0.
- **Test 3's original claim was wrong.** An admitted vertex brings its minor influences along,
  so 75 weights across the 60 male spheres sit on joints *outside* the gate. The largest is
  0.263. The C++ brief called this residue suppressed by the top-4 cut, and it isn't. §13.3
  shows it matters.
- **`compare_binding_modes.py` runs each gender in its own process.** In one process, building
  the second window after closing the first made every VTK render fail (wglMakeCurrent: invalid
  handle) and the process exit with code 9.
- **Review fix 1: the status line no longer overwrites stored weights.**
  - As first built, `report_lbs_binding_status` re-derived and stored every sphere's weights.
    After a posed edit or Fit, toggling off and on (or a load refresh) then moved the spheres.
    The spheremesh-reviewer measured up to 14.7 mm after a posed Fit, with 41 of 60 spheres
    moving more than 0.5 mm.
  - It now stores weights only for spheres that have none. Re-measured: 0.00 mm.
- **Review fix 2: the posed inverse iterates.** This replaces decision 10's "an edit never
  re-derives" and the single solve in §5.5.
  - Why: keeping the old weights made the *next* re-derivation (every beta tick) jump spheres
    edited while posed. After a posed Fit and `beta[0] += 0.01`, the jump was 18.9 mm max and
    1.10 mm median, of which the body's own change accounts for 0.49 / 0.13 mm.
  - `_rest_offset_from_posed_center` now alternates "solve the rest centre" and "re-derive the
    weights there", up to `LBS_INVERSE_ITERATIONS = 8` times. It stores the solved pair whose
    re-derivation would move the sphere least. Every candidate pair lands exactly on the edited
    centre, so the round trip stays exact.
  - Re-measured: median 0.16 mm, so most spheres now move only as much as the body. The max is
    unchanged at 18.9 mm. 16 of 60 spheres have no fixed point: the crotch sphere 23 flips
    between "pelvis 0.61–0.78" and "right_hip 0.43 / left_hip 0.40", each set's rest centre
    deriving the other. After a posed Fit it sits right on that discontinuity, so any later
    shape change flips it.
  - With the B+influence gate, the same probe's max falls to 8.3 mm (§13.3), smaller but not
    gone. The 20-nearest and top-4 cuts are discontinuous by construction.

### 13.2 Measured by the tests

These came from simulated input in headless runs. The app itself was not tried by hand.

- **`scripts/test_lbs_binding.py`, 30 checks, all pass** (after the review fixes).
  - Box off: bit-identical to `J[p] + R[p] @ rest_offset` on 3 CSV poses.
  - At theta 0: on and off agree to 2.2e-16 m.
  - A sphere carrying a vertex's skin weights follows that posed vertex to 2.4e-16 m.
  - One joint on right_shoulder: LBS equals rigid plus (FK − regressed joint), to 5.6e-17 m.
    That offset is 1.5 / 3.6 / 1.7 mm on the three poses.
  - Posed InfoPanel edit, LBS: round-trips to 5.8e-17 m, and toggling off and on afterwards
    leaves it there (5.8e-17 m). A rigid inverse would have landed the sphere 24.0 mm away.
    Rigid mode: 8.7e-19 m.
  - Fit spheres while posed (it moved spheres up to 10 mm): round-trips to 3.3e-16 m. A
    following `beta[0] += 0.01` tick moves the median sphere 0.16 mm and sphere 23 by 11.9 mm;
    4 spheres move more than 1 mm.
  - Toggling on at CSV pose 11 moves the spheres by up to 18.6 mm (median 3.3 mm). It leaves the
    unsaved flag alone and recomputes the heatmap.
  - Every sphere ends up with 4 joints, and the weights sum to 1 to 2.2e-16.
  - Divide in LBS mode: the new sphere sits 1.10–1.92 mm from its endpoints' posed midpoint.
  - A beta change re-derives the weights, by up to 0.456 for one sphere on CSV shape 0.
- **The existing suites:** all GUI and class suites pass. `test_divide_connection.py`'s posed
  midpoint check now runs with the box off, plus an LBS variant (within 1 cm).
  `test_topology_roundtrip.py` still fails at import (the conflict markers).
- **Cost.**
  - `update_sphere_positions`: 79.1 ms rigid against 80.2 ms LBS. The 79 ms are the connection
    meshes.
  - Re-derivation: 8.9 ms per beta tick.
  - GUI slider steps (median): beta 106 ms rigid against 122–126 ms LBS. Theta steps are
    within run-to-run noise.

### 13.3 The hip regression

**Diagnosis** (`compare_binding_modes.py` worst cases; scratch diagnostic, CSV pose 15):
- The deepest hip protrusion comes from the big pelvis-bound spheres at the crotch and buttocks
  (r ≈ 116–130 mm, |x| ≈ 0.02–0.07 m).
  - Gate A admits both hips for a pelvis-bound sphere, so their weights split about
    0.43 / 0.40 between right_hip and left_hip.
  - Male sphere 23 sits on the *left* (x = +0.053) and still carries right_hip 0.43, left_hip
    0.40, pelvis 0.17. Its connection [20, 23] is 79.3 mm inside the skin at vertex 1514 under
    LBS, against 10.6 mm rigid.
- The minor-influence leak adds to it. Male sphere 52 is bound to right_hip, and its gate
  excludes left_hip, yet it carries left_hip 0.26 through the pelvis-dominant crotch vertices.
- The female template behaves the same way: sphere 22 has right_hip 0.42 / left_hip 0.40, and
  connection [19, 22] is 82.6 mm inside, against 12.7 mm rigid.

This is the "averages left against right and follows neither" failure from §3.1, at the joint
where the two sides meet.

**Gate variants, measured** (scratch: each variant's derivation patched into `SphereManager` in
its own process, then measured exactly like `compare_binding_modes.py`; the tool is unchanged).
The patched A reproduces the implemented A's numbers exactly, which checks the patch.
- **A:** as implemented.
- **+influence:** only skin weights *on* gate joints are accumulated, which closes the
  minor-influence leak.
- **B:** only the child of the closest bone hanging off the parent is admitted (§3.1).

Through skin, in mm (the worst over 24 poses):

| | rigid | A | A+influence | B | **B+influence** |
|---|---|---|---|---|---|
| male left / right hip | 49.0 / 55.4 | 79.3 / 77.6 | 79.6 / 77.6 | 57.8 / 69.4 | **47.5 / 47.9** |
| male pelvis | 56.1 | 45.4 | 46.4 | 43.4 | **43.0** |
| male spine3 / right shoulder | 57.5 / 53.9 | 56.5 / 55.1 | 59.0 / 56.9 | 56.5 / 55.1 | 59.0 / 56.9 |
| male worst / mean over poses | 58.3 / 49.9 | 79.3 / 49.0 | 79.6 / 49.0 | 69.4 / 48.3 | **59.0 / 47.1** |
| male regions better / worse (of 24) | | 15 / 3 | 14 / 5 | 15 / 3 | **16 / 3** |
| male animation term outside, max | 37.1 | 22.1 | 21.6 | 19.2 | 24.8 |
| female left / right hip | 45.1 / 44.1 | 82.6 / 76.6 | 83.8 / 75.8 | 66.3 / 58.8 | **44.4 / 42.2** |
| female pelvis | 44.8 | 46.3 | 44.2 | 40.7 | **36.0** |
| female worst / mean over poses | 55.2 / 45.1 | 82.6 / 45.9 | 83.8 / 46.1 | 66.3 / 44.8 | **54.7 / 43.3** |
| female regions better / worse (of 24) | | 9 / 9 | 10 / 9 | 10 / 8 | **12 / 7** |
| female animation term outside, max | 42.8 | 23.7 | 23.0 | 19.2 | 20.6 |

What the variants show:
- **Neither fix works alone.** The influence gate on its own changes nothing at the hips. B on
  its own only halves the regression.
- **Together they remove it.** B+influence puts the hips back at or below rigid for both
  genders, keeps most of LBS's animation gain (outside max 37 → 25 mm and 43 → 21 mm), and is
  the only variant whose worst case is within a millimetre of rigid's (59.0 vs 58.3 male; 54.7
  vs 55.2 female).
- **Its cost:** spine3 and right_shoulder are 1.5–3 mm worse than A, because the influence gate
  drops weight that the neck and collar vertices lent there.
- **Parity:** B+influence departs from the engine's rule, so the engine would need the same
  change to stay in parity.
