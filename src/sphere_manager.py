"""
SphereManager: flat list of ManagedSphere + SphereConnection objects.

SphereManager: spheres, connections, selection, load/save, FK.
ManagedSphere: per-sphere widget and kinematic data.
SphereConnection: connectors + capsuloid / prysmoid actors (src/sphere_mesh.py).
"""
from __future__ import annotations

import colorsys
import csv
import glob
import itertools
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.managed_sphere import ManagedSphere
from src.sphere_connection import SphereConnection
from src.sphere_fit import optimize_spheres, select_band

from src.utils import measure_time

JSON_VERSION = 2

# "Class Colors" scheme: one color per primitive kind, the primitiveColors of SMFitting's
# views (PlotlyView.py, SphereMeshView.py). The tool starts in this scheme.
SPHERE_CLASS_COLOR = "#ee380a"      # red: spheres
CAPSULOID_CLASS_COLOR = "#2f76b9"   # blue: capsuloids (cones between 2 spheres)
PRYSMOID_CLASS_COLOR = "#9545d6"    # violet: prysmoids (slabs between 3 spheres)

# ---------------------------------------------------------------------------
# Beta-offset constants
# ---------------------------------------------------------------------------
_MIN_RADIUS_BETA: float = 1e-4
_BETA_CSV_REQUIRED: set = {"sphere_id", "dx", "dy", "dz", "dr"}


def _trim_center_xyz_3decimals(center) -> list[float]:
    """Round sphere center x,y,z to three digits after the decimal when loading saves."""
    c = center
    return [round(float(c[0]), 3), round(float(c[1]), 3), round(float(c[2]), 3)]


class SphereManager:
    def __init__(self, plotter, sphere_information_callback):
        self.plotter = plotter
        self.spheres: List[ManagedSphere] = []
        self.connections: List[SphereConnection] = []
        self.selected_sphere_index: Optional[int] = None
        # Cone or triangle picked in the 3D view. Exclusive with selected_sphere_index.
        # Stored as the object, not its indices: duplicate connections exist.
        self.selected_connection: Optional[SphereConnection] = None

        self.update_sphere_information_callback = sphere_information_callback
        self.spheremesh_visible = False
        self.sphere_widgets_visible = True
        self.spheremesh_opacity = 0.5
        self.spheremesh_toggle_callback = None
        # Called with every VTK actor a connection creates after load (lazy spheremesh
        # actors, re-indexed connectors), so MainWindow can give it the active clip planes.
        self.actor_created_callback = None
        self.skeleton = None
        self.rest_skeleton = None
        self.skinning_weights = None   # (N_verts, N_joints) float array from SMPL model
        self.rest_mesh_points = None   # (N_verts, 3) float array at rest pose
        self.lock_button_callback = None
        self.refocus_callback = None

        # True when spheres/connections were edited since the last save or load.
        # Pose/shape driven updates (FK, beta offsets, previews) do not set this.
        self.has_unsaved_changes = False

        self.normal_color = "lightblue"
        self.selected_color = "cyan"
        self.locked_color = "yellow"
        self.locked_selected_color = "orange"
        # "default", "random" or "class". Spheres and connections created later follow it.
        self.color_scheme = "class"

        self._uniform_radius_override_active = False
        self._uniform_radius_value = 1.0
        self._uniform_radius_mode = "scale"
        self._pre_uniform_radius_by_index: Dict[int, float] = {}
        self._mirror_edits_enabled = True
        self._is_mirroring_update = False

        # ---------------------------------------------------------------------------
        # Beta-offset state (populated by load_beta_offsets_from_dir)
        #
        # offsets_pos : Dict[int, np.ndarray]  beta_index -> (N, 3)
        # offsets_r   : Dict[int, np.ndarray]  beta_index -> (N,)
        #
        # The key is the numeric index parsed from the filename (e.g. beta3.csv -> 3),
        # so each CSV maps exactly to the SMPL beta slider with the same index.
        # Template is the sphere state snapshot taken at load time.
        # ---------------------------------------------------------------------------
        self.offsets_pos: Dict[int, np.ndarray] = {}   # beta_index -> (N, 3)
        self.offsets_r:   Dict[int, np.ndarray] = {}   # beta_index -> (N,)
        self._beta_template_pos: Optional[np.ndarray] = None  # (N, 3)
        self._beta_template_r:   Optional[np.ndarray] = None  # (N,)

    def _compute_touch_radius_and_nearest_vertex(self, sphere: ManagedSphere, mesh_points):
        """Compute touch radius from the globally closest SMPL vertex."""
        if mesh_points is None or len(mesh_points) == 0:
            return None, None
        center = sphere.get_center()
        deltas = mesh_points - center
        distances = np.linalg.norm(deltas, axis=1)
        nearest_idx = int(np.argmin(distances))
        radius = float(distances[nearest_idx])
        return radius, nearest_idx

    def initialize_radius_scaling_from_mesh(self, mesh):
        """Initialize missing rest_radius/nearest_vertex_index for loaded or new spheres."""
        if mesh is None:
            return
        mesh_points = np.asarray(mesh.points)
        if mesh_points.size == 0:
            return
        for sp in self.spheres:
            touch_radius, nearest_idx = self._compute_touch_radius_and_nearest_vertex(sp, mesh_points)
            if touch_radius is None:
                continue
            if sp.get_nearest_vertex_index() is None:
                sp.set_nearest_vertex_index(nearest_idx)
            if sp.get_rest_radius() is None and touch_radius > 1e-12:
                sp.set_rest_radius(float(sp.get_radius()) / touch_radius)

    def update_sphere_radius(self, mesh):
        """Update all sphere radii based on current SMPL mesh and each sphere rest_radius factor."""
        if mesh is None:
            return
        if self._uniform_radius_override_active:
            return
        mesh_points = np.asarray(mesh.points)
        if mesh_points.size == 0:
            return

        changed = False
        for sp in self.spheres:
            touch_radius, nearest_idx = self._compute_touch_radius_and_nearest_vertex(sp, mesh_points)
            if touch_radius is None or touch_radius <= 1e-12:
                continue
            sp.set_nearest_vertex_index(nearest_idx)
            rest_radius = sp.get_rest_radius()
            if rest_radius is None:
                rest_radius = float(sp.get_radius()) / touch_radius
                sp.set_rest_radius(rest_radius)
            sp.set_radius_force(rest_radius * touch_radius)
            changed = True

        if changed:
            for c in self.connections:
                c.update_mesh()
            self.update_sphere_information_callback()

    def mark_unsaved_changes(self) -> None:
        """Record that the sphere configuration diverged from the last saved/loaded file."""
        self.has_unsaved_changes = True

    def clear_unsaved_changes(self) -> None:
        """Record that the sphere configuration matches the last saved/loaded file."""
        self.has_unsaved_changes = False

    def set_spheremesh_toggle_callback(self, callback):
        self.spheremesh_toggle_callback = callback

    def notify_actor_created(self, actor) -> None:
        """Pass an actor a connection just added to the plotter on to actor_created_callback."""
        if self.actor_created_callback is not None:
            self.actor_created_callback(actor)

    def set_lock_button_callback(self, callback):
        self.lock_button_callback = callback

    def set_refocus_callback(self, callback):
        self.refocus_callback = callback

    def set_skinning_weights(self, weights) -> None:
        """Store SMPL LBS weights (N_verts × N_joints float64 array)."""
        self.skinning_weights = np.asarray(weights, dtype=np.float64)

    def set_rest_mesh_points(self, points) -> None:
        """Store SMPL rest-pose vertex positions (N_verts × 3 float64 array)."""
        self.rest_mesh_points = np.asarray(points, dtype=np.float64)

    def set_spheremesh_opacity(self, opacity):
        self.spheremesh_opacity = opacity
        for s in self.spheres:
            s.set_opacity(opacity)
        for c in self.connections:
            c.set_opacity(opacity)

    def set_spheremesh_visibility(self, visible):
        self.spheremesh_visible = visible
        for c in self.connections:
            c.set_visibility(visible)

    def set_sphere_widget_visibility(self, visible):
        self.sphere_widgets_visible = visible
        for s in self.spheres:
            if s.widget is not None:
                s.widget.On() if visible else s.widget.Off()
        for c in self.connections:
            c.set_connector_visibility(visible)

    def _incident_connections(self, sphere_index: int):
        return [c for c in self.connections if sphere_index in c.sphere_indices]

    def set_mirror_edits_enabled(self, enabled: bool):
        self._mirror_edits_enabled = bool(enabled)

    def _find_mirror_sphere_index(self, sphere_index: int, pos_tol: float = 0.02) -> Optional[int]:
        """Find closest sphere mirrored across X axis for the given index."""
        if sphere_index < 0 or sphere_index >= len(self.spheres):
            return None
        c = np.asarray(self.spheres[sphere_index].get_center(), dtype=np.float64)
        target = np.array([-c[0], c[1], c[2]], dtype=np.float64)
        best_j = None
        best_d = np.inf
        for j, sp in enumerate(self.spheres):
            if j == sphere_index:
                continue
            cj = np.asarray(sp.get_center(), dtype=np.float64)
            d = float(np.linalg.norm(cj - target))
            if d <= pos_tol and d < best_d:
                best_d = d
                best_j = j
        return best_j

    def get_mirror_sphere_index(self, sphere_index: int, pos_tol: float = 0.02) -> Optional[int]:
        """Public accessor for mirrored twin index across X axis."""
        return self._find_mirror_sphere_index(sphere_index, pos_tol=pos_tol)

    def _apply_mirror_to_twin(self, sphere_index: int, mirror_center: bool, mirror_radius: bool):
        """Propagate edits to mirrored twin (X axis), if found."""
        if not self._mirror_edits_enabled:
            return
        if self._is_mirroring_update:
            return
        twin = self._find_mirror_sphere_index(sphere_index)
        if twin is None:
            return
        self._is_mirroring_update = True
        try:
            src = self.spheres[sphere_index]
            dst = self.spheres[twin]
            if mirror_center:
                c = np.asarray(src.get_center(), dtype=np.float64)
                mirrored_center = np.array([-c[0], c[1], c[2]], dtype=np.float64)
                dst.set_center_force(mirrored_center)
                #self.assign_parent_joint(twin)
                for cobj in self._incident_connections(twin):
                    cobj.update_mesh()
            if mirror_radius:
                src_radius = float(src.get_radius())
                if self._uniform_radius_override_active:
                    # Keep the real/base radius in sync while override is active.
                    if self._uniform_radius_mode == "scale" and self._uniform_radius_value != 0:
                        base_radius = src_radius / float(self._uniform_radius_value)
                    else:
                        base_radius = src_radius
                    self._pre_uniform_radius_by_index[twin] = float(base_radius)
                    if self._uniform_radius_mode == "scale":
                        dst.set_radius_force(float(base_radius) * float(self._uniform_radius_value))
                    else:
                        dst.set_radius_force(float(self._uniform_radius_value))
                else:
                    dst.set_radius_force(src_radius)
                for cobj in self._incident_connections(twin):
                    cobj.update_mesh()
        finally:
            self._is_mirroring_update = False

    def _apply_lock_to_twin(self, sphere_index: int):
        """Propagate lock state to mirrored twin (X axis), if found."""
        if not self._mirror_edits_enabled:
            return
        twin = self._find_mirror_sphere_index(sphere_index)
        if twin is None:
            return
        state = bool(self.spheres[sphere_index].locked)
        self.spheres[twin].set_locked(state)

    def _refresh_all_connection_meshes(self):
        for c in self.connections:
            c.update_mesh()

    def is_uniform_radius_override_active(self) -> bool:
        return bool(self._uniform_radius_override_active)

    def set_uniform_radius_override(self, active: bool, radius: float = 0.2, mode: str = "absolute"):
        """Temporarily override sphere radii; restore originals when disabled.

        mode:
        - "absolute": force all radii to `radius`
        - "scale": set each radius to `base_radius * radius`
        """
        active = bool(active)
        value = float(radius)
        mode = str(mode)
        if mode not in ("absolute", "scale"):
            mode = "absolute"
        if active == self._uniform_radius_override_active and active:
            self._uniform_radius_mode = mode
            self._uniform_radius_value = value
            self._apply_uniform_radius_preview()
            self.update_sphere_information_callback()
            return
        if active:
            self._uniform_radius_override_active = True
            self._uniform_radius_mode = mode
            self._uniform_radius_value = value
            self._pre_uniform_radius_by_index.clear()
            for i, sp in enumerate(self.spheres):
                self._pre_uniform_radius_by_index[i] = float(sp.get_radius())
                if self._uniform_radius_mode == "scale":
                    sp.set_radius_force(self._pre_uniform_radius_by_index[i] * self._uniform_radius_value)
                else:
                    sp.set_radius_force(self._uniform_radius_value)
            self._refresh_all_connection_meshes()
        else:
            self._uniform_radius_override_active = False
            for i, sp in enumerate(self.spheres):
                stored = self._pre_uniform_radius_by_index.get(i)
                if stored is not None:
                    sp.set_radius_force(stored)
            self._pre_uniform_radius_by_index.clear()
            self._refresh_all_connection_meshes()
        self.update_sphere_information_callback()

    def _apply_uniform_radius_preview(self):
        if not self._uniform_radius_override_active:
            return
        for i, sp in enumerate(self.spheres):
            base = self._pre_uniform_radius_by_index.get(i)
            if base is None:
                base = float(sp.get_radius())
                self._pre_uniform_radius_by_index[i] = base
            if self._uniform_radius_mode == "scale":
                sp.set_radius_force(base * self._uniform_radius_value)
            else:
                sp.set_radius_force(self._uniform_radius_value)
        self._refresh_all_connection_meshes()

    def add_sphere(
        self,
        center=None,
        radius=None,
    ):
        c = np.array(center if center is not None else [1.0, -1, 0], dtype=np.float64)
        r = float(radius if radius is not None else 0.1)

        callback_ref = {"sphere": None}

        def _cb(pos):
            sphere_ref = callback_ref["sphere"]
            if sphere_ref is None:
                return
            try:
                i = self.spheres.index(sphere_ref)
            except ValueError:
                return
            self._on_sphere_interaction(i, pos)

        self._unhighlight_all()
        sphere = ManagedSphere(
            self.plotter,
            c,
            r,
            interaction_callback=_cb,
            normal_color=self.normal_color,
            selected_color=self.selected_color,
            locked_color=self.locked_color,
            locked_selected_color=self.locked_selected_color,
            opacity=self.spheremesh_opacity,
        )
        callback_ref["sphere"] = sphere
        self.spheres.append(sphere)
        sphere.set_random_color(self._scheme_color_for_sphere())
        new_i = len(self.spheres) - 1
        if self._uniform_radius_override_active:
            self._pre_uniform_radius_by_index[new_i] = float(r)
            if self._uniform_radius_mode == "scale":
                sphere.set_radius_force(float(r) * self._uniform_radius_value)
            else:
                sphere.set_radius_force(self._uniform_radius_value)
        #self.assign_parent_joint(new_i)
        self.selected_sphere_index = new_i
        self.selected_connection = None
        self._highlight_selection()

        if self.spheremesh_visible:
            for conn in self._incident_connections(self.selected_sphere_index):
                conn.set_visibility(True)
        sphere.set_opacity(self.spheremesh_opacity)
        self.set_sphere_widget_visibility(self.sphere_widgets_visible)

        if self.lock_button_callback:
            self._call_lock_button_callback()
        if self._uniform_radius_override_active:
            self._refresh_all_connection_meshes()
        self.mark_unsaved_changes()
        print(f"Added sphere #{self.selected_sphere_index}")
        return self.selected_sphere_index

    def try_add_connection(self, indices: Tuple[int, ...]) -> bool:
        """Return True if connection was created. indices length 2 or 3, distinct, in range."""
        idxs = tuple(int(i) for i in indices)
        if len(idxs) not in (2, 3) or len(set(idxs)) != len(idxs):
            print("Invalid connection: need 2 or 3 distinct sphere indices")
            return False
        n = len(self.spheres)
        for i in idxs:
            if i < 0 or i >= n:
                print("Invalid connection: index out of range")
                return False
        conn = self._build_connection(idxs)
        if conn is None:
            return False

        self.connections.append(conn)
        self._show_connection_like_others(conn)
        self._highlight_selection()
        self.update_sphere_information_callback()
        self.mark_unsaved_changes()
        print(f"Added connection {conn.sphere_indices}")
        return True

    def _build_connection(self, sphere_indices: Tuple[int, ...]) -> Optional[SphereConnection]:
        """Create a connection with its geometry, not yet registered; None on bad geometry."""
        try:
            connection = SphereConnection(
                self.plotter,
                sphere_indices,
                self,
                normal_color=self.normal_color,
                selected_color=self.selected_color,
                locked_color=self.locked_color,
                locked_selected_color=self.locked_selected_color,
                opacity=self.spheremesh_opacity,
            )
        except ValueError as e:
            print(f"Could not create connection: {e}")
            return None
        connection.set_random_color(self._scheme_color_for_connection(connection))
        connection._build_and_update_meshes()
        if connection.spheremesh is None:
            connection.delete()
            print("Could not build mesh for connection (geometry error)")
            return None
        return connection

    def _show_connection_like_others(self, connection: SphereConnection) -> None:
        """Apply the current spheremesh and connector visibility to a new connection."""
        if self.spheremesh_visible:
            connection.set_visibility(True)
        connection.set_connector_visibility(self.sphere_widgets_visible)

    def delete_selected_sphere(self) -> bool:
        si = self.selected_sphere_index
        if si is None or si < 0 or si >= len(self.spheres):
            print("No valid sphere selected for deletion")
            return False
        self._delete_sphere_at_index(si)
        self.update_sphere_information_callback()
        self.mark_unsaved_changes()
        print("Deleted sphere")
        return True

    def _delete_sphere_at_index(self, i: int):
        """Remove sphere i, drop every connection that includes i, reindex other connections."""
        surviving_connections: List[SphereConnection] = []
        for c in self.connections:
            if i in c.sphere_indices:
                c.delete()
                continue
            surviving_connections.append(c)
        self.connections = surviving_connections

        self.spheres[i].delete_widget()
        del self.spheres[i]

        # Only now: set_sphere_indices rebuilds the geometry from self.spheres, which must
        # already be the shortened list the new indices point into.
        for connection in surviving_connections:
            connection.set_sphere_indices(
                tuple(j if j < i else j - 1 for j in connection.sphere_indices)
            )

        if self._uniform_radius_override_active:
            remapped: Dict[int, float] = {}
            for old_idx, rad in self._pre_uniform_radius_by_index.items():
                if old_idx == i:
                    continue
                new_idx = old_idx if old_idx < i else old_idx - 1
                remapped[new_idx] = rad
            self._pre_uniform_radius_by_index = remapped

        if self.selected_sphere_index == i:
            self.selected_sphere_index = None
        elif self.selected_sphere_index is not None and self.selected_sphere_index > i:
            self.selected_sphere_index -= 1

        self._highlight_selection()

    def delete_connection(self, conn: SphereConnection) -> bool:
        if conn not in self.connections:
            return False
        conn.delete()
        self.connections.remove(conn)
        self._highlight_selection()
        self.update_sphere_information_callback()
        self.mark_unsaved_changes()
        return True

    def _base_radius(self, sphere_index: int) -> float:
        """Radius without the Inflate / uniform-radius preview applied."""
        if self._uniform_radius_override_active:
            stored_radius = self._pre_uniform_radius_by_index.get(sphere_index)
            if stored_radius is not None:
                return float(stored_radius)
        return float(self.spheres[sphere_index].get_radius())

    def _plan_division(
        self, connection: SphereConnection, new_index: int
    ) -> List[Tuple[SphereConnection, List[Tuple[int, ...]]]]:
        """Pair every connection the divide replaces with the index tuples replacing it.

        Cone (a, b): every connection holding both a and b (the cone, its duplicates and
        each triangle on edge a-b) becomes one copy with b -> m and one with a -> m, which
        keeps the cyclic order: (a, b, c) -> (a, m, c) + (m, b, c).
        Triangle (a, b, c): the triangle and its duplicates become (a, b, m), (b, c, m),
        (c, a, m) in the selected triangle's order.
        Each sphere set is created once, so duplicates collapse into one set of pieces.
        """
        endpoint_indices = connection.sphere_indices
        endpoint_set = set(endpoint_indices)
        replacement_plan = []
        planned_sphere_sets = set()
        for existing_connection in self.connections:
            existing_indices = existing_connection.sphere_indices
            if len(endpoint_indices) == 2:
                if not endpoint_set.issubset(existing_indices):
                    continue
                first_endpoint, second_endpoint = endpoint_indices
                candidate_tuples = [
                    tuple(new_index if i == second_endpoint else i for i in existing_indices),
                    tuple(new_index if i == first_endpoint else i for i in existing_indices),
                ]
            else:
                if set(existing_indices) != endpoint_set:
                    continue
                first_corner, second_corner, third_corner = endpoint_indices
                candidate_tuples = [
                    (first_corner, second_corner, new_index),
                    (second_corner, third_corner, new_index),
                    (third_corner, first_corner, new_index),
                ]
            replacement_tuples = []
            for candidate_tuple in candidate_tuples:
                sphere_set = frozenset(candidate_tuple)
                if sphere_set in planned_sphere_sets:
                    continue
                planned_sphere_sets.add(sphere_set)
                replacement_tuples.append(candidate_tuple)
            replacement_plan.append((existing_connection, replacement_tuples))
        return replacement_plan

    def _bind_divided_sphere(self, new_index: int, endpoint_indices: Tuple[int, ...]) -> None:
        """Bind the new sphere at the rest-pose mean of its endpoints, to one of their joints.

        Like assign_parent_joint, the binding lives in the rest skeleton, evaluated at the
        mean of the endpoints' rest positions. Restricting the parent to the endpoints' own
        joints keeps the rigid single-joint rule: a shared joint is reused, which makes the
        new rest offset the mean of the endpoints' offsets (so the sphere stays at their
        midpoint in every pose), and mixed endpoints never pull in a third joint.
        """
        if self.rest_skeleton is None:
            return
        new_sphere = self.spheres[new_index]
        endpoint_spheres = [self.spheres[i] for i in endpoint_indices]
        endpoint_parent_joints = {
            int(endpoint_sphere.get_parent_joint())
            for endpoint_sphere in endpoint_spheres
            if endpoint_sphere.get_parent_joint() is not None
        }

        endpoint_rest_positions = []
        for endpoint_sphere in endpoint_spheres:
            endpoint_parent = endpoint_sphere.get_parent_joint()
            endpoint_rest_offset = endpoint_sphere.get_rest_offset()
            if endpoint_parent is None or endpoint_rest_offset is None:
                endpoint_rest_positions = []
                break
            endpoint_rest_positions.append(
                self.rest_skeleton.transform_point(endpoint_rest_offset, endpoint_parent)
            )
        if endpoint_rest_positions:
            rest_point = np.mean(endpoint_rest_positions, axis=0)
        else:
            # Unbound endpoint: fall back to the current center, as assign_parent_joint does.
            rest_point = new_sphere.get_center()

        allowed_parent_joints = endpoint_parent_joints if endpoint_parent_joints else None
        parent_joint, child_joint = self._closest_bone(rest_point, allowed_parent_joints)
        if parent_joint is None:
            if allowed_parent_joints is None:
                return
            # Every endpoint joint is a leaf (hand, foot, head): bind to the nearest one.
            parent_joint = min(
                allowed_parent_joints,
                key=lambda joint: float(np.linalg.norm(self.rest_skeleton.joints[joint] - rest_point)),
            )

        parent_rest_rotation = self.rest_skeleton.get_joint_rotation(parent_joint)
        rest_offset = np.dot(
            parent_rest_rotation.T, rest_point - self.rest_skeleton.joints[parent_joint]
        )
        new_sphere.set_parent_joint(int(parent_joint), rest_offset)

        if child_joint is None:
            new_sphere.set_secondary_joint(None, None, 1.0)
            return
        lbs_weight = self._primary_lbs_weight_on_bone(rest_point, parent_joint, child_joint)
        child_rest_rotation = self.rest_skeleton.get_joint_rotation(child_joint)
        secondary_rest_offset = np.dot(
            child_rest_rotation.T, rest_point - self.rest_skeleton.joints[child_joint]
        )
        new_sphere.set_secondary_joint(int(child_joint), secondary_rest_offset, lbs_weight)

    def _extend_beta_offsets_for_divided_sphere(
        self, new_index: int, endpoint_indices: Tuple[int, ...]
    ) -> None:
        """Give the new sphere the mean of its endpoints' beta rows (exact: the model is linear)."""
        if self._beta_template_pos is None or self._beta_template_r is None:
            return
        if self._beta_template_pos.shape[0] != new_index:
            print(
                f"[warn] Divide: beta offsets cover {self._beta_template_pos.shape[0]} spheres, "
                f"not {new_index}; the new sphere will not follow the beta sliders"
            )
            return
        endpoint_rows = list(endpoint_indices)
        self._beta_template_pos = np.vstack(
            [self._beta_template_pos, self._beta_template_pos[endpoint_rows].mean(axis=0)]
        )
        self._beta_template_r = np.append(
            self._beta_template_r, self._beta_template_r[endpoint_rows].mean()
        )
        for beta_index in self.offsets_pos:
            self.offsets_pos[beta_index] = np.vstack(
                [self.offsets_pos[beta_index], self.offsets_pos[beta_index][endpoint_rows].mean(axis=0)]
            )
            self.offsets_r[beta_index] = np.append(
                self.offsets_r[beta_index], self.offsets_r[beta_index][endpoint_rows].mean()
            )

    def divide_connection(self, connection: SphereConnection, body_mesh=None) -> Optional[int]:
        """Split a cone at its midpoint or a triangle at its centroid with a new sphere.

        The new sphere m gets the mean center and the mean (base) radius of the endpoints
        and is appended, so no existing index changes. A cone (a, b) becomes (a, m) + (m, b),
        and every triangle on edge a-b is split along with it; a triangle (a, b, c) becomes
        (a, b, m) + (b, c, m) + (c, a, m). The union of the pieces equals the original
        volume exactly. Nothing changes if any new piece cannot be built.

        Returns the new sphere's index, or None on failure.
        """
        if connection not in self.connections:
            print("Divide: the connection is not part of the current topology")
            return None
        endpoint_indices = connection.sphere_indices
        new_index = len(self.spheres)
        new_center = np.mean([self.spheres[i].get_center() for i in endpoint_indices], axis=0)
        new_base_radius = float(np.mean([self._base_radius(i) for i in endpoint_indices]))

        replacement_plan = self._plan_division(connection, new_index)

        def _center_of(sphere_index: int):
            if sphere_index == new_index:
                return new_center
            return self.spheres[sphere_index].get_center()

        for _, replacement_tuples in replacement_plan:
            for replacement_tuple in replacement_tuples:
                for first_index, second_index in itertools.combinations(replacement_tuple, 2):
                    if np.linalg.norm(_center_of(first_index) - _center_of(second_index)) <= 1e-9:
                        print(f"Divide: spheres {first_index} and {second_index} coincide, nothing changed")
                        return None

        previous_selected_sphere_index = self.selected_sphere_index
        had_unsaved_changes = self.has_unsaved_changes
        self.add_sphere(new_center, new_base_radius)  # colored by the active scheme
        new_sphere = self.spheres[new_index]

        built_replacements: List[Tuple[SphereConnection, List[SphereConnection]]] = []
        built_connections: List[SphereConnection] = []
        for replaced_connection, replacement_tuples in replacement_plan:
            replacement_connections = []
            for replacement_tuple in replacement_tuples:
                replacement_connection = self._build_connection(replacement_tuple)
                if replacement_connection is None:
                    for built_connection in built_connections:
                        built_connection.delete()
                    self.spheres.pop().delete_widget()
                    self._pre_uniform_radius_by_index.pop(new_index, None)
                    self.selected_sphere_index = previous_selected_sphere_index
                    self.selected_connection = connection
                    self.has_unsaved_changes = had_unsaved_changes
                    self._highlight_selection()
                    self.update_sphere_information_callback()
                    print(f"Divide: could not build {replacement_tuple}, nothing changed")
                    return None
                replacement_connections.append(replacement_connection)
                built_connections.append(replacement_connection)
            built_replacements.append((replaced_connection, replacement_connections))

        # Swap in place so the saved connection order stays stable.
        replacements_by_connection_id = {
            id(replaced_connection): replacement_connections
            for replaced_connection, replacement_connections in built_replacements
        }
        updated_connections: List[SphereConnection] = []
        for existing_connection in self.connections:
            replacement_connections = replacements_by_connection_id.get(id(existing_connection))
            if replacement_connections is None:
                updated_connections.append(existing_connection)
                continue
            inherited_random_color = existing_connection.get_random_color()
            existing_connection.delete()
            for replacement_connection in replacement_connections:
                if inherited_random_color is not None:
                    replacement_connection.set_random_color(inherited_random_color)
                updated_connections.append(replacement_connection)
        self.connections = updated_connections
        for built_connection in built_connections:
            self._show_connection_like_others(built_connection)

        self._bind_divided_sphere(new_index, endpoint_indices)
        if body_mesh is not None:
            touch_radius, nearest_vertex_index = self._compute_touch_radius_and_nearest_vertex(
                new_sphere, np.asarray(body_mesh.points)
            )
            if touch_radius is not None:
                new_sphere.set_nearest_vertex_index(nearest_vertex_index)
                if touch_radius > 1e-12:
                    new_sphere.set_rest_radius(new_base_radius / touch_radius)
        self._extend_beta_offsets_for_divided_sphere(new_index, endpoint_indices)

        self.selected_connection = None
        self._highlight_selection()
        self.update_sphere_information_callback()
        self.mark_unsaved_changes()
        replaced_descriptions = ", ".join(
            str(list(replaced_connection.sphere_indices)) for replaced_connection, _ in built_replacements
        )
        new_descriptions = ", ".join(str(list(c.sphere_indices)) for c in built_connections)
        print(f"Divided {replaced_descriptions} with sphere #{new_index} into {new_descriptions}")
        return new_index

    def _on_sphere_interaction(self, sphere_index: int, center):
        if self._is_mirroring_update:
            return
        center = np.array(center, dtype=np.float64)
        print(f"Interacted with sphere {sphere_index}")

        if self.selected_sphere_index != sphere_index or self.selected_connection is not None:
            self._unhighlight_all()
            self.selected_sphere_index = sphere_index
            self.selected_connection = None
            self._highlight_selection()

        if self.lock_button_callback:
            self._call_lock_button_callback()
        if self.refocus_callback:
            self.refocus_callback()

        sp = self.spheres[sphere_index]
        if sp.locked:
            sp.sync_widget_to_stored()
            self.update_sphere_information_callback()
            return

        sp.set_center(center)
        if not self._uniform_radius_override_active:
            sp.sync_radius_from_widget()
        else:
            sp.sync_widget_to_stored()
        #self.assign_parent_joint(sphere_index)
        for c in self._incident_connections(sphere_index):
            c.update_mesh()
        self._apply_mirror_to_twin(sphere_index, mirror_center=True, mirror_radius=True)
        self.update_sphere_information_callback()
        self.mark_unsaved_changes()

    def toggle_lock_selected_pair(self):
        """Legacy: lock all spheres in first incident connection; if orphan, toggle that sphere."""
        if self.selected_sphere_index is None:
            return False
        conns = self._incident_connections(self.selected_sphere_index)
        if conns:
            idxs = list(conns[0].sphere_indices)
            all_locked = all(self.spheres[j].locked for j in idxs)
            new_state = not all_locked
            for j in idxs:
                self.spheres[j].set_locked(new_state)
                self._apply_lock_to_twin(j)
        else:
            s = self.spheres[self.selected_sphere_index]
            s.set_locked(not s.locked)
            self._apply_lock_to_twin(self.selected_sphere_index)
        self._highlight_selection()
        for c in self.connections:
            c.refresh_style()
        if self.lock_button_callback:
            self._call_lock_button_callback()
        self.mark_unsaved_changes()
        return True

    def toggle_lock_selected_sphere(self):
        if self.selected_sphere_index is None:
            return False
        s = self.spheres[self.selected_sphere_index]
        s.set_locked(not s.locked)
        self._apply_lock_to_twin(self.selected_sphere_index)
        self._highlight_selection()
        for c in self.connections:
            c.refresh_style()
        if self.lock_button_callback:
            self._call_lock_button_callback()
        self.mark_unsaved_changes()
        return True

    def toggle_lock_sphere_at_display_slot(self, local_slot: int) -> bool:
        data = self.get_selected_sphere_display_data()
        if data is None:
            return False
        ep = data["endpoint_indices"]
        if local_slot < 0 or local_slot >= len(ep):
            return False
        gidx = ep[local_slot]
        s = self.spheres[gidx]
        s.set_locked(not s.locked)
        self._apply_lock_to_twin(gidx)
        self._highlight_selection()
        for c in self.connections:
            c.refresh_style()
        if self.lock_button_callback:
            self._call_lock_button_callback()
        self.mark_unsaved_changes()
        return True

    def _call_lock_button_callback(self):
        if self.lock_button_callback is None:
            return
        data = self.get_selected_sphere_display_data()
        if data is None:
            return
        n = data["num_slots"]
        locked = [self.spheres[data["endpoint_indices"][k]].locked for k in range(n)]
        while len(locked) < 3:
            locked.append(False)
        self.lock_button_callback(locked, n)

    def set_sphere_center_global(self, global_index: int, new_center):
        if global_index < 0 or global_index >= len(self.spheres):
            return
        if self.spheres[global_index].locked:
            return
        sp = self.spheres[global_index]
        sp.set_center(np.asarray(new_center, dtype=np.float64))
        parent = sp.get_parent_joint()
        rest_offset = sp.get_rest_offset()
        if self.skeleton is not None and parent is not None and rest_offset is not None:
            current_joint_pos = self.skeleton.joints[parent]
            R_current = self.skeleton.get_joint_rotation(parent)
            new_rest_offset = np.dot(R_current.T, sp.get_center() - current_joint_pos)
            sp.set_parent_joint(parent, new_rest_offset)
        #self.assign_parent_joint(global_index)
        for c in self._incident_connections(global_index):
            c.update_mesh()
        self._apply_mirror_to_twin(global_index, mirror_center=True, mirror_radius=False)
        self.update_sphere_information_callback()
        self.mark_unsaved_changes()

    def set_sphere_radius_global(self, global_index: int, new_radius):
        if global_index < 0 or global_index >= len(self.spheres):
            return
        if self.spheres[global_index].locked:
            return
        if self._uniform_radius_override_active:
            self._pre_uniform_radius_by_index[global_index] = float(new_radius)
            if self._uniform_radius_mode == "scale":
                self.spheres[global_index].set_radius_force(
                    float(new_radius) * self._uniform_radius_value
                )
            else:
                self.spheres[global_index].set_radius_force(self._uniform_radius_value)
        else:
            self.spheres[global_index].set_radius(float(new_radius))
        for c in self._incident_connections(global_index):
            c.update_mesh()
        self._apply_mirror_to_twin(global_index, mirror_center=False, mirror_radius=True)
        self.update_sphere_information_callback()
        self.mark_unsaved_changes()

    def grow_sphere(self, mesh, sphere_index):
        """sphere_index is UI local slot; resolved via current display endpoints."""
        data = self.get_selected_sphere_display_data()
        if data is None:
            return
        ep = data["endpoint_indices"]
        if sphere_index < 0 or sphere_index >= len(ep):
            return
        gidx = ep[sphere_index]
        if self._uniform_radius_override_active:
            if mesh is None:
                return
            min_distance = np.inf
            for point in mesh.points:
                d = np.linalg.norm(point - self.spheres[gidx].center)
                if d < min_distance:
                    min_distance = d
            self._pre_uniform_radius_by_index[gidx] = float(min_distance)
            if self._uniform_radius_mode == "scale":
                self.spheres[gidx].set_radius_force(
                    float(min_distance) * self._uniform_radius_value
                )
            else:
                self.spheres[gidx].set_radius_force(self._uniform_radius_value)
        else:
            self.spheres[gidx].grow_from_mesh(mesh)
        for c in self._incident_connections(gidx):
            c.update_mesh()
        self._apply_mirror_to_twin(gidx, mirror_center=False, mirror_radius=True)
        self.update_sphere_information_callback()
        self.mark_unsaved_changes()

    def _unhighlight_all(self):
        for s in self.spheres:
            s.set_highlighted(False)
        for c in self.connections:
            c.set_selected(False)

    def _highlight_selection(self):
        si = self.selected_sphere_index
        highlighted = set()
        # Never auto-highlight connected spheres/connections: only the current sphere,
        # or the connection picked in the 3D view.
        selected_connection = self.get_selected_connection()
        self.selected_connection = selected_connection
        for c in self.connections:
            c.set_selected(c is selected_connection)
        if si is not None and 0 <= si < len(self.spheres):
            highlighted.add(si)
        for i, s in enumerate(self.spheres):
            s.set_highlighted(i in highlighted)
        if self.lock_button_callback and si is not None:
            self._call_lock_button_callback()

    def get_selected_connection(self) -> Optional[SphereConnection]:
        """The picked cone/triangle, or None (also when it has been deleted since)."""
        if self.selected_connection is None or self.selected_connection not in self.connections:
            return None
        return self.selected_connection

    def select_connection(self, connection: SphereConnection) -> bool:
        """Select a cone/triangle; this deselects the sphere so sphere edits can't hit it unseen."""
        if connection not in self.connections:
            return False
        self.selected_sphere_index = None
        self.selected_connection = connection
        self._highlight_selection()
        self.update_sphere_information_callback()
        return True

    def clear_connection_selection(self) -> None:
        self.selected_connection = None
        self._highlight_selection()
        self.update_sphere_information_callback()

    def get_connection_pick_targets(self) -> List[Tuple[object, SphereConnection]]:
        """(actor, connection) for every visible actor a click can select a connection by."""
        return [
            (pickable_actor, connection)
            for connection in self.connections
            for pickable_actor in connection.get_pickable_actors()
        ]

    @staticmethod
    def describe_connection(connection: SphereConnection) -> str:
        """Short label such as 'cone (3, 7)' or 'triangle (1, 3, 4)'."""
        sphere_indices = connection.sphere_indices
        kind = "cone" if len(sphere_indices) == 2 else "triangle"
        return f"{kind} ({', '.join(str(i) for i in sphere_indices)})"

    @staticmethod
    def _random_vivid_color():
        """Random, well-saturated RGB color that stays readable against the model."""
        hue = random.random()
        saturation = random.uniform(0.55, 0.95)
        value = random.uniform(0.65, 1.0)
        red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
        return (red, green, blue)

    def _scheme_color_for_sphere(self):
        """Override color a sphere gets under the active color scheme; None keeps the default."""
        if self.color_scheme == "class":
            return SPHERE_CLASS_COLOR
        if self.color_scheme == "random":
            return self._random_vivid_color()
        return None

    def _scheme_color_for_connection(self, connection: SphereConnection):
        """Override color a cone/triangle gets under the active color scheme; None keeps the default."""
        if self.color_scheme == "class":
            if len(connection.sphere_indices) == 2:
                return CAPSULOID_CLASS_COLOR
            return PRYSMOID_CLASS_COLOR
        if self.color_scheme == "random":
            return self._random_vivid_color()
        return None

    def _apply_color_scheme(self):
        """Recolor every sphere and connection from the active scheme."""
        for sphere in self.spheres:
            sphere.set_random_color(self._scheme_color_for_sphere())
        for connection in self.connections:
            connection.set_random_color(self._scheme_color_for_connection(connection))
        # Selection/lock colors take precedence, so re-apply them on top.
        self._highlight_selection()

    def randomize_colors(self):
        """Give every sphere, cone (2-sphere) and plane (3-sphere) its own random color."""
        self.color_scheme = "random"
        self._apply_color_scheme()
        return len(self.spheres) + len(self.connections)

    def apply_class_colors(self):
        """Color by primitive: spheres red, capsuloids (cones) blue, prysmoids (triangles) violet."""
        self.color_scheme = "class"
        self._apply_color_scheme()

    def reset_colors(self):
        """Drop color overrides and restore the default sphere/connection colors."""
        self.color_scheme = "default"
        self._apply_color_scheme()

    def toggle_spheremeshes(self, state):
        self.spheremesh_visible = bool(state)
        self.set_spheremesh_visibility(self.spheremesh_visible)

    def point_segment_distance(self, P, A, B):
        P = np.asarray(P)
        A = np.asarray(A)
        B = np.asarray(B)
        AB = B - A
        AP = P - A
        denom = np.dot(AB, AB)
        if denom < 1e-12:
            return np.linalg.norm(AP)
        t = np.dot(AP, AB) / denom
        t = np.clip(t, 0.0, 1.0)
        closest = A + t * AB
        return np.linalg.norm(P - closest)

    def _closest_bone(self, point, allowed_parent_joints=None) -> Tuple[Optional[int], Optional[int]]:
        """(parent, child) of the rest-skeleton bone closest to point; (None, None) if none.

        allowed_parent_joints, when given, restricts the search to bones whose parent end
        is one of those joints.
        """
        min_dist = np.inf
        best_parent = None
        best_child = None
        kintree = self.rest_skeleton.kintree
        for i in range(1, len(kintree)):
            p = kintree[i]
            c = i
            if allowed_parent_joints is not None and int(p) not in allowed_parent_joints:
                continue
            p_pos = self.rest_skeleton.joints[p]
            c_pos = self.rest_skeleton.joints[c]
            dist = self.point_segment_distance(point, p_pos, c_pos)
            if dist < min_dist:
                min_dist = dist
                best_parent = p
                best_child = c
        return best_parent, best_child

    def _primary_lbs_weight_on_bone(self, point, parent_joint: int, child_joint: int) -> float:
        """Weight of the parent end: 1 at the parent joint, 0 at the child joint (rest pose)."""
        p_pos = self.rest_skeleton.joints[parent_joint]
        c_pos = self.rest_skeleton.joints[child_joint]
        seg = c_pos - p_pos
        seg_len_sq = float(np.dot(seg, seg))
        if seg_len_sq > 1e-12:
            t = float(np.dot(point - p_pos, seg) / seg_len_sq)
            t = max(0.0, min(1.0, t))
        else:
            t = 0.0
        return 1.0 - t

    def assign_parent_joint(self, sphere_index: int):
        if self.rest_skeleton is None:
            return
        if sphere_index < 0 or sphere_index >= len(self.spheres):
            return
        sp = self.spheres[sphere_index]
        center = sp.get_center()
        best_parent, best_child = self._closest_bone(center)

        rest_offset = None
        if best_parent is not None:
            rest_joint_pos = self.rest_skeleton.joints[best_parent]
            R_parent_rest = self.rest_skeleton.get_joint_rotation(best_parent)
            world_offset = center - rest_joint_pos
            rest_offset = np.dot(R_parent_rest.T, world_offset)
        sp.set_parent_joint(best_parent, rest_offset)

        # --- secondary joint = child of the best parent-child segment ---
        # _closest_bone already picked the kintree segment (best_parent -> best_child)
        # closest to the sphere center.  Re-use its child end as the secondary joint
        # and derive the LBS blend weight from the projection of the sphere center
        # onto that segment: t = 0 at the parent end (all weight on primary),
        # t = 1 at the child end (all weight on secondary).
        secondary_joint = None
        secondary_rest_offset = None
        lbs_weight = 1.0

        if best_parent is not None and best_child is not None:
            secondary_joint = int(best_child)
            c_pos = self.rest_skeleton.joints[secondary_joint]
            lbs_weight = self._primary_lbs_weight_on_bone(center, best_parent, secondary_joint)

            R_secondary_rest = self.rest_skeleton.get_joint_rotation(secondary_joint)
            world_offset_2 = center - c_pos
            secondary_rest_offset = np.dot(R_secondary_rest.T, world_offset_2)

        # --- previous logic: secondary joint via SMPL skinning weights ---
        # Find the closest rest-pose vertex; read its per-joint skin weights.
        # The joint with the highest weight that differs from best_parent becomes
        # the secondary joint.  The two skin weights (primary, secondary), after
        # normalization, are used as LBS blend weights.
        # If skinning data is unavailable, the sphere keeps a single joint.
        # secondary_joint = None
        # secondary_rest_offset = None
        # lbs_weight = 1.0
        #
        # if (
        #     best_parent is not None
        #     and self.skinning_weights is not None
        #     and self.rest_mesh_points is not None
        # ):
        #     # Closest rest-pose vertex to the sphere center.
        #     deltas = self.rest_mesh_points - center
        #     vertex_idx = int(np.argmin(np.linalg.norm(deltas, axis=1)))
        #
        #     skin_w = self.skinning_weights[vertex_idx]  # shape: (N_joints,)
        #
        #     # Highest-weight joint that is NOT best_parent.
        #     sorted_joints = np.argsort(skin_w)[::-1]  # descending weight
        #     for j in sorted_joints:
        #         if int(j) != best_parent:
        #             secondary_joint = int(j)
        #             break
        #
        #     if secondary_joint is not None:
        #         w_primary   = float(skin_w[best_parent])
        #         w_secondary = float(skin_w[secondary_joint])
        #         total = w_primary + w_secondary
        #         lbs_weight = w_primary / total if total > 1e-12 else 1.0
        #
        #         rest_joint_pos_2  = self.rest_skeleton.joints[secondary_joint]
        #         R_secondary_rest  = self.rest_skeleton.get_joint_rotation(secondary_joint)
        #         world_offset_2    = center - rest_joint_pos_2
        #         secondary_rest_offset = np.dot(R_secondary_rest.T, world_offset_2)

        sp.set_secondary_joint(secondary_joint, secondary_rest_offset, lbs_weight)

    def reassign_parent_joints_all(self):
        """Recompute parent joint and rest offset for every sphere."""
        if self.rest_skeleton is None:
            return
        for i in range(len(self.spheres)):
            self.assign_parent_joint(i)
        self.mark_unsaved_changes()
        #self.update_sphere_positions()

    def update_sphere_positions(self):
        if self.skeleton is None or self.rest_skeleton is None:
            return
        for i, sp in enumerate(self.spheres):
            parent = sp.get_parent_joint()
            rest_offset = sp.get_rest_offset()
            if parent is None or rest_offset is None:
                continue

            current_joint_pos = self.skeleton.joints[parent]
            current_R = self.skeleton.get_joint_rotation(parent)
            p1 = current_joint_pos + np.dot(current_R, rest_offset)

            
            secondary = sp.get_secondary_joint()
            secondary_rest_offset = sp.get_secondary_rest_offset()
            lbs_weight = sp.get_lbs_weight()
            '''
            if (
                secondary is not None
                and secondary_rest_offset is not None
                and lbs_weight < 1.0
            ):
                current_joint_pos_2 = self.skeleton.joints[secondary]
                current_R_2 = self.skeleton.get_joint_rotation(secondary)
                p2 = current_joint_pos_2 + np.dot(current_R_2, secondary_rest_offset)
                new_center = lbs_weight * p1 + (1.0 - lbs_weight) * p2
            else:
            '''    
            new_center = p1

            sp.set_center_force(new_center)

        def _update_all_connection_meshes() -> None:
            for c in self.connections:
                c.update_mesh()

        #measure_time(_update_all_connection_meshes, label="update_mesh (all connections)")
        _update_all_connection_meshes()
        self.update_sphere_information_callback()

    def get_selected_sphere_index(self):
        return self.selected_sphere_index

    def get_selected_sphere_display_data(self):
        """UI: the selected sphere (one slot), or the spheres of the selected cone/triangle
        (one slot per sphere, in connection order)."""
        selected_connection = self.get_selected_connection()
        if selected_connection is not None:
            endpoint_indices = list(selected_connection.sphere_indices)
            endpoint_spheres = [self.spheres[i] for i in endpoint_indices]
            return {
                "sphere_index": None,
                "connection": selected_connection,
                "twin_index": None,
                "num_slots": len(endpoint_indices),
                "endpoint_indices": endpoint_indices,
                "centers": [endpoint_sphere.get_center().tolist() for endpoint_sphere in endpoint_spheres],
                "radii": [endpoint_sphere.get_radius() for endpoint_sphere in endpoint_spheres],
                "locked_per_sphere": [endpoint_sphere.locked for endpoint_sphere in endpoint_spheres],
                "selected_local_index": None,
                "incident_connections": [],
            }
        si = self.selected_sphere_index
        if si is None or not self.spheres:
            return None
        if si < 0 or si >= len(self.spheres):
            return None
        sp = self.spheres[si]
        incident = [list(c.sphere_indices) for c in self.connections if si in c.sphere_indices]
        return {
            "sphere_index": si,
            "twin_index": self.get_mirror_sphere_index(si, pos_tol=0.02),
            "num_slots": 1,
            "endpoint_indices": [si],
            "centers": [sp.get_center().tolist()],
            "radii": [sp.get_radius()],
            "locked_per_sphere": [sp.locked],
            "selected_local_index": 0,
            "incident_connections": incident,
        }

    def get_sphere_count(self):
        return len(self.spheres)

    def check_x_symmetry(self, pos_tol: float = 0.02, radius_tol: float = 1e-4):
        """Check mirror symmetry across X axis for spheres and connections.

        Returns:
            dict with:
              - ok: bool
              - mirror_of: dict[int, int]
              - issues: list[str]
        """
        n = len(self.spheres)
        issues: List[str] = []
        mirror_of: Dict[int, int] = {}
        used = set()

        centers = [np.asarray(s.get_center(), dtype=np.float64) for s in self.spheres]
        radii = [float(s.get_radius()) for s in self.spheres]

        for i in range(n):
            if i in mirror_of:
                continue
            ci = centers[i]
            target = np.array([-ci[0], ci[1], ci[2]], dtype=np.float64)

            best_j = None
            best_d = np.inf
            for j in range(n):
                if j in used and j != i:
                    continue
                cj = centers[j]
                d = float(np.linalg.norm(cj - target))
                if d <= pos_tol and d < best_d:
                    best_d = d
                    best_j = j

            if best_j is None:
                issues.append(
                    f"Sphere {i} at {ci.tolist()} has no mirrored twin near {target.tolist()} (tol={pos_tol})."
                )
                continue

            mirror_of[i] = best_j
            mirror_of[best_j] = i
            used.add(i)
            used.add(best_j)

            if abs(radii[i] - radii[best_j]) > radius_tol:
                issues.append(
                    f"Radius mismatch for pair ({i}, {best_j}): {radii[i]:.6f} vs {radii[best_j]:.6f} (tol={radius_tol})."
                )

        conn_set = {tuple(sorted(c.sphere_indices)) for c in self.connections}
        for conn in conn_set:
            mirrored = []
            missing_map = False
            for idx in conn:
                if idx not in mirror_of:
                    issues.append(
                        f"Connection {list(conn)} contains sphere {idx} with no mirrored twin."
                    )
                    missing_map = True
                    break
                mirrored.append(mirror_of[idx])
            if missing_map:
                continue
            mirrored_conn = tuple(sorted(mirrored))
            if mirrored_conn not in conn_set:
                issues.append(
                    f"Connection {list(conn)} is missing mirrored connection {list(mirrored_conn)}."
                )

        return {
            "ok": len(issues) == 0,
            "mirror_of": mirror_of,
            "issues": issues,
        }

    def select_previous_sphere(self):
        n = len(self.spheres)
        if n == 0:
            return
        si = self.selected_sphere_index
        if si is None:
            new_i = n - 1
        else:
            new_i = (si - 1) % n
        self._unhighlight_all()
        self.selected_sphere_index = new_i
        self.selected_connection = None
        self._highlight_selection()
        self.update_sphere_information_callback()

    def select_next_sphere(self):
        n = len(self.spheres)
        if n == 0:
            return
        si = self.selected_sphere_index
        if si is None:
            new_i = 0
        else:
            new_i = (si + 1) % n
        self._unhighlight_all()
        self.selected_sphere_index = new_i
        self.selected_connection = None
        self._highlight_selection()
        self.update_sphere_information_callback()

    def _indices_in_any_connection(self):
        s = set()
        for c in self.connections:
            for i in c.sphere_indices:
                s.add(i)
        return s

    def get_heatmap(self, mesh) -> np.ndarray:
        """Signed distance in metres from every mesh vertex to the sphere-mesh, negative inside.

        The plain minimum over every connection's capsuloid / prysmoid, the d that
        SMFitting's SphereMesh.closestSphere() reports. Spheres in no connection count as
        spheres, which SMFitting's meshes never have but a template edited here can.
        """
        points = np.asarray(mesh.points, dtype=np.float64)
        distances = np.full(len(points), np.inf)
        for connection in self.connections:
            np.minimum(distances, connection.get_distances(points), out=distances)
        in_any_connection = self._indices_in_any_connection()
        for sphere_index, sphere in enumerate(self.spheres):
            if sphere_index not in in_any_connection:
                lone_sphere_distances = np.linalg.norm(points - sphere.get_center(), axis=1) - sphere.get_radius()
                np.minimum(distances, lone_sphere_distances, out=distances)
        return distances

    def geometry_fingerprint(self) -> int:
        """A hash of every sphere's centre and radius and every connection's indices.

        Equal fingerprints mean nothing a distance query depends on has changed, which is
        how MainWindow decides whether the heatmap is stale.
        """
        sphere_state = np.array(
            [[*sphere.get_center(), sphere.get_radius()] for sphere in self.spheres], dtype=np.float64
        )
        connection_state = tuple(connection.sphere_indices for connection in self.connections)
        return hash((sphere_state.tobytes(), connection_state))

    # ---------------------------------------------------------------------------
    # Geometric sphere fitting
    # ---------------------------------------------------------------------------

    def fit_spheres_to_mesh(
        self,
        mesh,
        *,
        max_offset: float,
        band_max: float = 0.05,
        min_radius: float = 1e-3,
        max_outer: int = 60,
        protrusion_weight: float = 10.0,
        assign_radius_scale: float = 1.75,
        respect_locks: bool = True,
    ) -> Optional[dict]:
        """Least-squares fit every sphere to the mesh surface, capped by ``max_offset``.

        Delegates to ``src.sphere_fit.optimize_spheres``; see that module's docstring for
        the objective, the degenerate direction and what the two reported RMS figures mean.
        Topology is invariant: sphere count, order, connections and parent joints are
        untouched, and only ``center`` and ``radius`` are written back.

        ``protrusion_weight`` penalises a sphere poking through the surface harder than it
        penalises a gap. At 1 the fit is a plain surface fit, whose minimiser in radius is
        the mean assigned distance rather than the minimum, so every sphere inflates until
        it protrudes; larger values approach the inscribed sphere that is actually wanted.
        Expect the reported RMS to rise as it does, and judge the run by the
        negative-residual fraction instead.

        Locked spheres are frozen rather than dropped, so they keep covering their region
        of the surface during assignment while never being solved for.

        The band is computed from the sphere configuration and mesh as they stand when this
        is called. Comparing RMS figures across two calls is only meaningful when the body
        shape was the same for both, since a different shape gives a different band and
        therefore a different point population.

        Returns the optimiser ``info`` dict, or ``None`` when there is nothing to fit.
        """
        if mesh is None or not self.spheres:
            return None
        if self._uniform_radius_override_active:
            raise RuntimeError(
                "Turn off the uniform radius preview before fitting: it overrides the "
                "displayed radii, so the fitted values could not be seen or saved."
            )
        points = np.asarray(mesh.points, dtype=np.float64)
        if points.size == 0:
            return None

        centers = np.array([s.get_center() for s in self.spheres], dtype=np.float64)
        radii = np.array([s.get_radius() for s in self.spheres], dtype=np.float64)
        connections = [tuple(c.sphere_indices) for c in self.connections]
        frozen = (
            np.array([s.locked for s in self.spheres], dtype=bool)
            if respect_locks
            else np.zeros(len(self.spheres), dtype=bool)
        )

        band = select_band(points, centers, radii, connections, band_max=band_max)
        if band.size < 4:
            return None

        new_centers, new_radii, info = optimize_spheres(
            centers,
            radii,
            points[band],
            max_offset=float(max_offset),
            min_radius=float(min_radius),
            max_outer=int(max_outer),
            protrusion_weight=float(protrusion_weight),
            assign_radius_scale=float(assign_radius_scale),
            connections=connections,
            frozen=frozen,
        )

        for index, sphere in enumerate(self.spheres):
            if frozen[index]:
                continue
            sphere.set_center_force(new_centers[index])
            sphere.set_radius_force(float(new_radii[index]))
            parent = sphere.get_parent_joint()
            rest_offset = sphere.get_rest_offset()
            if self.skeleton is not None and parent is not None and rest_offset is not None:
                joint_position = self.skeleton.joints[parent]
                rotation = self.skeleton.get_joint_rotation(parent)
                sphere.set_parent_joint(
                    parent, np.dot(rotation.T, sphere.get_center() - joint_position)
                )

        self._refresh_all_connection_meshes()
        self.update_sphere_information_callback()
        self.mark_unsaved_changes()

        info["band_size"] = int(band.size)
        info["mesh_size"] = int(points.shape[0])
        info["locked_skipped"] = int(np.count_nonzero(frozen))
        return info

    # ---------------------------------------------------------------------------
    # Beta-offset API
    # ---------------------------------------------------------------------------

    def has_beta_offsets_loaded(self) -> bool:
        """True after ``load_beta_offsets_from_dir`` successfully loaded CSVs and a template."""
        return bool(self.offsets_pos) and self._beta_template_pos is not None

    @staticmethod
    def _parse_beta_index(path: str) -> Optional[int]:
        """Extract the numeric index from a filename like 'beta3.csv' -> 3.

        Returns None if the name does not match the pattern.
        """
        import re
        name = os.path.basename(path)
        m = re.fullmatch(r"beta(\d+)\.csv", name, re.IGNORECASE)
        return int(m.group(1)) if m else None

    def _load_single_beta_csv(self, path: str, num_spheres: int) -> tuple:
        """Return (dp, dr) from one beta*.csv.

        dp shape (N, 3), dr shape (N,).  Rows whose sphere_id is out of range
        are skipped with a warning.  Sphere ids absent from the file stay zero.
        """
        dp = np.zeros((num_spheres, 3), dtype=np.float64)
        dr = np.zeros(num_spheres, dtype=np.float64)
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or not _BETA_CSV_REQUIRED.issubset(reader.fieldnames):
                missing = _BETA_CSV_REQUIRED - set(reader.fieldnames or [])
                raise ValueError(f"missing columns {missing}")
            for row in reader:
                sid = int(row["sphere_id"])
                if 0 <= sid < num_spheres:
                    dp[sid] = [float(row["dx"]), float(row["dy"]), float(row["dz"])]
                    dr[sid] = float(row["dr"])
                else:
                    print(f"[warn] {path}: sphere_id {sid} out of range [0, {num_spheres}), skipped")
        return dp, dr

    def load_beta_offsets_from_dir(self, data_dir: str) -> None:
        """Scan *data_dir* for beta*.csv files and load them as offset modes.

        Each file's name determines which SMPL beta slider it corresponds to:
        ``beta0.csv`` maps to beta[0], ``beta3.csv`` to beta[3], etc.
        Missing indices simply contribute zero offset for that slider.

        Also snapshots the current sphere state as the rest template.
        Replaces any previously loaded offsets.
        """
        n = len(self.spheres)
        if n == 0:
            print("[warn] load_beta_offsets_from_dir: no spheres loaded yet")
            return

        pattern = os.path.join(data_dir, "beta*.csv")
        paths = sorted(glob.glob(pattern))
        if not paths:
            print(f"[warn] load_beta_offsets_from_dir: no beta*.csv found in {data_dir}")
            return

        new_offsets_pos: Dict[int, np.ndarray] = {}
        new_offsets_r:   Dict[int, np.ndarray] = {}
        for p in paths:
            beta_idx = self._parse_beta_index(p)
            if beta_idx is None:
                print(f"[warn] skipping {p}: cannot parse beta index from filename")
                continue
            try:
                dp, dr = self._load_single_beta_csv(p, n)
                new_offsets_pos[beta_idx] = dp
                new_offsets_r[beta_idx]   = dr
                print(f"  loaded beta{beta_idx} offset: {p}")
            except Exception as exc:
                print(f"[warn] skipping {p}: {exc}")

        if not new_offsets_pos:
            print("[warn] load_beta_offsets_from_dir: no valid CSV files loaded")
            return

        self.offsets_pos = new_offsets_pos
        self.offsets_r   = new_offsets_r

        # Snapshot current sphere state as the template (rest configuration).
        self._beta_template_pos = np.array(
            [s.get_center() for s in self.spheres], dtype=np.float64
        )
        self._beta_template_r = np.array(
            [float(s.get_radius()) for s in self.spheres], dtype=np.float64
        )
        indices = sorted(new_offsets_pos.keys())
        print(f"Beta offsets ready: indices {indices}, {n} spheres")

    def apply_beta_offsets(self, beta) -> None:
        """Apply beta-weighted sphere offsets on top of the template.

        beta : array-like of length >= 1  (torch.Tensor or numpy array).
               Each loaded beta{k}.csv is multiplied by beta[k].
               Indices with no CSV file contribute zero.

        Formula (per sphere i, summed only over loaded indices k):
            new_pos_i = template_pos_i + Σ_k  beta[k] * offsets_pos[k][i]
            new_r_i   = template_r_i   + Σ_k  beta[k] * offsets_r[k][i]

        No-op when no offsets have been loaded.

        # Extension point: topology scaling and safety clamping can be inserted
        # here as a multiplier on delta_pos / delta_r before set_*_force calls.
        """
        if not self.offsets_pos or self._beta_template_pos is None:
            return

        # Convert beta to a plain float64 numpy vector.
        if hasattr(beta, "detach"):
            betas_np = beta.detach().cpu().numpy().astype(np.float64).ravel()
        else:
            betas_np = np.asarray(beta, dtype=np.float64).ravel()

        n = self._beta_template_pos.shape[0]
        delta_pos = np.zeros((n, 3), dtype=np.float64)
        delta_r   = np.zeros(n,      dtype=np.float64)

        for k, dp in self.offsets_pos.items():
            if k < len(betas_np):
                delta_pos += betas_np[k] * dp
                delta_r   += betas_np[k] * self.offsets_r[k]

        for i in range(min(len(self.spheres), n)):
            new_center = self._beta_template_pos[i] + delta_pos[i]
            new_radius = max(float(self._beta_template_r[i] + delta_r[i]), _MIN_RADIUS_BETA)
            self.spheres[i].set_center_force(new_center)
            self.spheres[i].set_radius_force(new_radius)

        for c in self.connections:
            c.update_mesh()
        self.update_sphere_information_callback()

    def save_to_json(self, filepath):
        spheres_out = []
        for sp in self.spheres:
            pj = sp.get_parent_joint()
            ro = sp.get_rest_offset()
            sj = sp.get_secondary_joint()
            sro = sp.get_secondary_rest_offset()
            spheres_out.append(
                {
                    "center": sp.get_center().tolist(),
                    "radius": float(sp.get_radius()),
                    "parent_joint": int(pj) if pj is not None else None,
                    "rest_offset": ro.tolist() if ro is not None else None,
                    "secondary_joint": int(sj) if sj is not None else None,
                    "secondary_rest_offset": sro.tolist() if sro is not None else None,
                    "lbs_weight": float(sp.get_lbs_weight()),
                    "rest_radius": sp.get_rest_radius(),
                    "nearest_vertex_index": sp.get_nearest_vertex_index(),
                    "locked": bool(sp.locked),
                }
            )
        connections_out = [{"indices": list(c.sphere_indices)} for c in self.connections]
        payload = {
            "version": JSON_VERSION,
            "spheres": spheres_out,
            "connections": connections_out,
        }
        with open(filepath, "w") as f:
            json.dump(payload, f, indent=2)
        self.clear_unsaved_changes()
        print(f"Sphere configuration saved to {filepath}")

    def _read_spheremesh_payload(self, filepath):
        path = Path(filepath)
        if not path.is_file():
            print(f"No file at {filepath}, skipping load")
            return None
        with open(path, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "spheres" not in data or "connections" not in data:
            raise ValueError(
                f"{filepath} must be an object with 'spheres' and 'connections' (new format only)"
            )
        return data

    def _build_from_payload(self, data, clear_existing: bool = True, with_connections: bool = True):
        base_index = 0
        if clear_existing:
            self.plotter.clear_sphere_widgets()
            for c in self.connections:
                c.delete()
            self.connections.clear()
            for s in self.spheres:
                s.delete_widget()
            self.spheres.clear()
            self.selected_sphere_index = None
        else:
            base_index = len(self.spheres)
        self.selected_connection = None

        for entry in data["spheres"]:
            center = _trim_center_xyz_3decimals(entry["center"])
            radius = float(entry["radius"])
            locked = bool(entry.get("locked", False))
            pj = entry.get("parent_joint")
            ro = entry.get("rest_offset")
            sj = entry.get("secondary_joint")
            sro = entry.get("secondary_rest_offset")
            lbs_w = entry.get("lbs_weight")
            rr = entry.get("rest_radius")
            nvi = entry.get("nearest_vertex_index")

            callback_ref = {"sphere": None}

            def _cb(pos, _ref=callback_ref):
                sphere_ref = _ref["sphere"]
                if sphere_ref is None:
                    return
                try:
                    i = self.spheres.index(sphere_ref)
                except ValueError:
                    return
                self._on_sphere_interaction(i, pos)

            sp = ManagedSphere(
                self.plotter,
                center,
                radius,
                interaction_callback=_cb,
                normal_color=self.normal_color,
                selected_color=self.selected_color,
                locked_color=self.locked_color,
                locked_selected_color=self.locked_selected_color,
                opacity=self.spheremesh_opacity,
            )
            if pj is not None and ro is not None:
                sp.set_parent_joint(int(pj), np.array(ro, dtype=np.float64))
            elif pj is not None:
                sp.set_parent_joint(int(pj), None)
            if sj is not None and sro is not None:
                sp.set_secondary_joint(
                    int(sj),
                    np.array(sro, dtype=np.float64),
                    float(lbs_w) if lbs_w is not None else 1.0,
                )
            sp.set_rest_radius(rr)
            sp.set_nearest_vertex_index(nvi)
            sp.set_locked(locked)
            sp.set_random_color(self._scheme_color_for_sphere())
            callback_ref["sphere"] = sp
            self.spheres.append(sp)

        if with_connections:
            for conn_entry in data["connections"]:
                idxs = tuple(int(i) for i in conn_entry["indices"])
                if len(idxs) not in (2, 3) or len(set(idxs)) != len(idxs):
                    raise ValueError(f"Invalid connection indices: {idxs}")
                mapped = tuple(i + base_index for i in idxs)
                nsp = len(self.spheres)
                for i in mapped:
                    if i < 0 or i >= nsp:
                        raise ValueError(f"Connection index out of range: {i}")
                conn = SphereConnection(
                    self.plotter,
                    mapped,
                    self,
                    normal_color=self.normal_color,
                    selected_color=self.selected_color,
                    locked_color=self.locked_color,
                    locked_selected_color=self.locked_selected_color,
                    opacity=self.spheremesh_opacity,
                )
                #print(f"[DEBUG] Connection created (load/import): {mapped}")
                conn.set_random_color(self._scheme_color_for_connection(conn))
                if self.spheremesh_visible:
                    conn.set_visibility(True)
                conn.set_connector_visibility(self.sphere_widgets_visible)
                self.connections.append(conn)

        for i in range(base_index, len(self.spheres)):
            if self.spheres[i].get_rest_offset() is None:
                self.assign_parent_joint(i)

        if self.spheres:
            self.selected_sphere_index = len(self.spheres) - 1
        self._highlight_selection()
        self.set_sphere_widget_visibility(self.sphere_widgets_visible)
        self.update_sphere_information_callback()

    def load_from_json(self, filepath):
        data = self._read_spheremesh_payload(filepath)
        if data is None:
            return
        self._build_from_payload(data, clear_existing=True)
        self.clear_unsaved_changes()
        print(f"Sphere configuration loaded from {filepath}")

    def append_from_json(self, filepath):
        data = self._read_spheremesh_payload(filepath)
        if data is None:
            return
        self._build_from_payload(data, clear_existing=False)
        self.mark_unsaved_changes()
        print(f"Sphere configuration appended from {filepath}")

    def append_spheres_only_from_json(self, filepath):
        """Append spheres from file without importing any of its connections."""
        data = self._read_spheremesh_payload(filepath)
        if data is None:
            return
        self._build_from_payload(data, clear_existing=False, with_connections=False)
        self.mark_unsaved_changes()
        print(f"Spheres (no connections) appended from {filepath}")
