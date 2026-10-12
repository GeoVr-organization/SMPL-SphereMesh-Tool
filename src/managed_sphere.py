"""One managed collision proxy sphere: state + VTK sphere widget."""
import numpy as np
import pyvista as pv

# Widget tessellation matching SMFitting's SphereMesh.Sphere.buildGeometry(16, 32): 16 rings
# from pole to pole (VTK's phi), and 31 facets around (VTK's theta) because buildGeometry's
# 32 longitudes repeat the first one to close the seam.
SPHERE_WIDGET_THETA_RESOLUTION = 31
SPHERE_WIDGET_PHI_RESOLUTION = 16


class ManagedSphere:
    def __init__(
        self,
        plotter,
        center,
        radius,
        *,
        interaction_callback,
        normal_color="lightblue",
        selected_color="cyan",
        locked_color="yellow",
        locked_selected_color="orange",
        opacity=0.5,
    ):
        self._plotter = plotter
        self.center = np.array(center, dtype=np.float64)
        self.radius = float(radius)
        self.parent_joint = None
        self.rest_offset = None
        self.secondary_joint = None
        self.secondary_rest_offset = None
        self.lbs_weight = 1.0
        self.rest_radius = None
        self.nearest_vertex_index = None
        self.locked = False
        self._highlighted = False
        self._opacity = opacity

        self._normal_color = normal_color
        self._selected_color = selected_color
        self._locked_color = locked_color
        self._locked_selected_color = locked_selected_color
        # Optional per-sphere override replacing the normal color (random coloring).
        self._random_color = None

        def _cb(pos):
            interaction_callback(np.array(pos, dtype=np.float64))

        self.widget = plotter.add_sphere_widget(
            _cb,
            center=tuple(self.center),
            radius=self.radius,
            theta_resolution=SPHERE_WIDGET_THETA_RESOLUTION,
            phi_resolution=SPHERE_WIDGET_PHI_RESOLUTION,
            style="surface",
            test_callback=False,
        )
        # SMFitting's views shade every primitive flat; the drag highlight follows suit.
        self.widget.GetSphereProperty().SetInterpolationToFlat()
        self.widget.GetSelectedSphereProperty().SetInterpolationToFlat()
        self.apply_widget_color()

    def set_opacity(self, opacity):
        self._opacity = opacity
        if self.widget is not None:
            self.widget.GetSphereProperty().SetOpacity(opacity)

    def set_highlighted(self, highlighted: bool):
        self._highlighted = bool(highlighted)
        self.apply_widget_color()

    def set_random_color(self, color):
        """Override the normal (unselected, unlocked) color; None restores the default."""
        self._random_color = color
        self.apply_widget_color()

    def get_random_color(self):
        return self._random_color

    def apply_widget_color(self):
        if self.widget is None:
            return
        if self._highlighted:
            color = self._locked_selected_color if self.locked else self._selected_color
        elif self.locked:
            color = self._locked_color
        elif self._random_color is not None:
            color = self._random_color
        else:
            color = self._normal_color
        self.widget.GetSphereProperty().SetColor(pv.Color(color).float_rgb)
        self.widget.GetSphereProperty().SetOpacity(self._opacity)

    def sync_radius_from_widget(self):
        if self.widget is not None:
            self.radius = float(self.widget.GetRadius())

    def sync_widget_to_stored(self):
        if self.widget is not None:
            self.widget.SetCenter(tuple(self.center))
            self.widget.SetRadius(self.radius)

    def set_center(self, pos):
        if self.locked:
            return
        self.center = np.asarray(pos, dtype=np.float64)
        if self.widget is not None:
            self.widget.SetCenter(tuple(self.center))

    def set_center_force(self, pos):
        """Set center and widget even when locked (pose-follow override)."""
        self.center = np.asarray(pos, dtype=np.float64)
        if self.widget is not None:
            self.widget.SetCenter(tuple(self.center))

    def set_radius(self, r):
        if self.locked:
            return
        self.radius = float(r)
        if self.widget is not None:
            self.widget.SetRadius(self.radius)

    def set_radius_force(self, r):
        """Set radius and widget even when locked (UI preview override)."""
        self.radius = float(r)
        if self.widget is not None:
            self.widget.SetRadius(self.radius)

    def set_parent_joint(self, parent, rest_offset):
        self.parent_joint = parent
        self.rest_offset = (
            np.array(rest_offset, dtype=np.float64) if rest_offset is not None else None
        )

    def set_secondary_joint(self, secondary, secondary_rest_offset, lbs_weight: float = 1.0):
        """Set the secondary LBS joint and its rest-pose offset.

        lbs_weight is the weight assigned to the *primary* joint (0–1).
        The secondary joint receives weight (1 - lbs_weight).
        """
        self.secondary_joint = secondary
        self.secondary_rest_offset = (
            np.array(secondary_rest_offset, dtype=np.float64)
            if secondary_rest_offset is not None
            else None
        )
        self.lbs_weight = float(lbs_weight)

    def get_secondary_joint(self):
        return self.secondary_joint

    def get_secondary_rest_offset(self):
        if self.secondary_rest_offset is None:
            return None
        return self.secondary_rest_offset.copy()

    def get_lbs_weight(self) -> float:
        return self.lbs_weight

    def set_rest_radius(self, rest_radius):
        self.rest_radius = None if rest_radius is None else float(rest_radius)

    def get_rest_radius(self):
        return None if self.rest_radius is None else float(self.rest_radius)

    def set_nearest_vertex_index(self, vertex_index):
        self.nearest_vertex_index = None if vertex_index is None else int(vertex_index)

    def get_nearest_vertex_index(self):
        return None if self.nearest_vertex_index is None else int(self.nearest_vertex_index)

    def get_center(self):
        return self.center.copy()

    def get_radius(self):
        return self.radius

    def get_parent_joint(self):
        return self.parent_joint

    def get_rest_offset(self):
        if self.rest_offset is None:
            return None
        return self.rest_offset.copy()

    def set_locked(self, locked: bool):
        self.locked = bool(locked)
        self.apply_widget_color()

    def grow_from_mesh(self, mesh):
        if mesh is None:
            return
        min_distance = np.inf
        for point in mesh.points:
            d = np.linalg.norm(point - self.center)
            if d < min_distance:
                min_distance = d
        self.set_radius(min_distance)

    def get_distance(self, point):
        """Signed distance to sphere surface (analytic)."""
        return float(np.linalg.norm(np.asarray(point) - self.center) - self.radius)

    def delete_widget(self):
        if self.widget is None:
            return
        self.widget.Off()
        if hasattr(self._plotter, "sphere_widgets") and self.widget in self._plotter.sphere_widgets:
            self._plotter.sphere_widgets.remove(self.widget)
        self.widget = None
