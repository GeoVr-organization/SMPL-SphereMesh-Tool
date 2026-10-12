"""Connection between 2 or 3 spheres: thin connector plus its capsuloid / prysmoid surface.

The surface and the distances come from src/sphere_mesh.py, the SphereMesh class shared
with SMFitting: a cone is a SphereMesh.Capsuloid, a triangle a SphereMesh.Prysmoid. Both
cache their axis and planes when they are built, so a fresh one is built from the spheres'
current centres and radii for every update and every distance query.
"""
import numpy as np
import pyvista as pv

from src.sphere_mesh import SphereMesh

# buildGeometry() subdivisions. A cone's side is straight along its axis, so 2 rows give
# the very surface the SMFitting views draw with their default 16 at an eighth of the
# triangles; 32 around matches their default.
SPHEREMESH_NUM_THETA = 2
SPHEREMESH_NUM_PHI = 32

# Metres. Two sphere centres closer than this make a connection meaningless; Add
# Connection and Divide refuse to create one.
COINCIDENT_CENTER_TOLERANCE = 1e-9


def triangles_to_faces(triangles: np.ndarray) -> np.ndarray:
    """(T, 3) triangle indices as a flat VTK faces array: 3, a, b, c per triangle."""
    triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    return np.hstack([np.full((len(triangles), 1), 3, dtype=np.int64), triangles]).ravel()


class SphereConnection:
    def __init__(
        self,
        plotter,
        sphere_indices,
        manager,
        *,
        normal_color="lightblue",
        selected_color="cyan",
        locked_color="yellow",
        locked_selected_color="orange",
        opacity=0.5,
    ):
        sphere_indices = tuple(int(i) for i in sphere_indices)
        if len(sphere_indices) not in (2, 3):
            raise ValueError("SphereConnection requires 2 or 3 sphere indices")
        if len(sphere_indices) != len(set(sphere_indices)):
            raise ValueError("Duplicate sphere index in connection")
        self._plotter = plotter
        self._sphere_indices = sphere_indices
        self._manager = manager
        # The drawn surface (pv.PolyData); None until first built, and when the geometry
        # is invalid (coincident centres) - _build_connection relies on that.
        self.spheremesh = None

        self._opacity = opacity
        self._selected = False
        self._spheremesh_visible = False

        self._normal_color = normal_color
        self._selected_color = selected_color
        self._locked_color = locked_color
        self._locked_selected_color = locked_selected_color
        # Optional per-connection override replacing the normal color (random coloring).
        self._random_color = None

        self._connector_actor = None
        self._actors = []
        self._build_connector()

    @property
    def sphere_indices(self):
        return self._sphere_indices

    def set_sphere_indices(self, new_indices):
        """Update indices (after sphere list reindexing); rebuild connector and mesh."""
        new_indices = tuple(int(i) for i in new_indices)
        if len(new_indices) not in (2, 3):
            raise ValueError("SphereConnection requires 2 or 3 sphere indices")
        self._sphere_indices = new_indices
        # The rebuilt connector keeps the old one's visibility ("Show sphere widgets").
        connector_was_visible = True
        if self._connector_actor is not None:
            connector_was_visible = bool(self._connector_actor.GetVisibility())
            self._plotter.remove_actor(self._connector_actor)
            self._connector_actor = None
        self._build_connector()
        self.set_connector_visibility(connector_was_visible)
        self.update_mesh()

    def _spheres(self):
        return [self._manager.spheres[i] for i in self._sphere_indices]

    def build_primitive(self):
        """A fresh SphereMesh.Capsuloid / Prysmoid over the spheres' current centres and radii.

        Each SphereMesh.Sphere carries the tool's sphere index, so the primitive orders its
        spheres, and names them in its warnings, the way the tool numbers them.
        """
        primitive_spheres = [
            SphereMesh.Sphere(sphere_index, managed_sphere.get_center(), managed_sphere.get_radius())
            for sphere_index, managed_sphere in zip(self._sphere_indices, self._spheres())
        ]
        if len(primitive_spheres) == 2:
            return SphereMesh.Capsuloid(*primitive_spheres)
        return SphereMesh.Prysmoid(*primitive_spheres, 1.0)

    def has_coincident_centers(self) -> bool:
        """Whether two of the connection's sphere centres coincide."""
        endpoint_centers = [managed_sphere.get_center() for managed_sphere in self._spheres()]
        return any(
            np.linalg.norm(endpoint_centers[first] - endpoint_centers[second]) < COINCIDENT_CENTER_TOLERANCE
            for first in range(len(endpoint_centers))
            for second in range(first + 1, len(endpoint_centers))
        )

    def _build_tri_prismoid_mesh(self, c0, c1, c2, half_thickness=0.0025):
        """Centered triangular prism from triangle (c0,c1,c2)."""
        triangle = pv.Triangle([c0, c1, c2])
        normal = np.cross(c1 - c0, c2 - c0)
        nlen = np.linalg.norm(normal)
        if nlen <= 1e-10:
            return None
        normal = normal / nlen
        full_thickness = float(half_thickness) * 2.0

        # Build one closed prism and center it around the source triangle plane.
        prism = triangle.extrude(normal * full_thickness, capping=True)
        prism = prism.translate(-normal * float(half_thickness), inplace=False)
        return prism

    def _build_connector(self):
        S = self._spheres()
        c0, c1 = S[0].get_center(), S[1].get_center()
        if len(self._sphere_indices) == 2:
            line = pv.Line(c0, c1)
            tube = line.tube(radius=0.005)
            self._connector_actor = self._plotter.add_mesh(
                tube,
                color=self._current_connector_color(),
                opacity=self._opacity,
                smooth_shading=True,
            )
        else:
            c2 = S[2].get_center()
            prismoid = self._build_tri_prismoid_mesh(c0, c1, c2, half_thickness=0.0025)
            if prismoid is not None:
                self._connector_actor = self._plotter.add_mesh(
                    prismoid,
                    color=self._current_connector_color(),
                    opacity=self._opacity,
                    smooth_shading=True,
                )
        if self._connector_actor is not None:
            self._manager.notify_actor_created(self._connector_actor)

    def set_connector_visibility(self, visible):
        if self._connector_actor is not None:
            self._connector_actor.SetVisibility(1 if visible else 0)

    def set_random_color(self, color):
        """Override the normal (unselected, unlocked) color; None restores the default."""
        self._random_color = color
        self.refresh_style()

    def get_random_color(self):
        return self._random_color

    def _current_mesh_color(self):
        s0 = self._spheres()[0]
        if self._selected:
            return self._locked_selected_color if s0.locked else self._selected_color
        if s0.locked:
            return self._locked_color
        if self._random_color is not None:
            return self._random_color
        return self._normal_color

    def _current_connector_color(self):
        if self._selected:
            return self._selected_color
        return self._random_color if self._random_color is not None else "gray"

    def get_pickable_actors(self):
        """Visible actors a click can land on: the connector and, if shown, the spheremesh."""
        pickable_actors = []
        if self._connector_actor is not None and self._connector_actor.GetVisibility():
            pickable_actors.append(self._connector_actor)
        for spheremesh_actor in self._actors:
            if spheremesh_actor.GetVisibility():
                pickable_actors.append(spheremesh_actor)
        return pickable_actors

    def _update_connector(self):
        if self._connector_actor is None:
            return
        S = self._spheres()
        c0, c1 = S[0].get_center(), S[1].get_center()
        if len(self._sphere_indices) == 2:
            line = pv.Line(c0, c1)
            tube = line.tube(radius=0.005)
            mesh_vtk = getattr(tube, "_vtk_obj", getattr(tube, "VTKObject", tube))
            self._connector_actor.GetMapper().SetInputData(mesh_vtk)
        else:
            c2 = S[2].get_center()
            prismoid = self._build_tri_prismoid_mesh(c0, c1, c2, half_thickness=0.0025)
            if prismoid is not None:
                mesh_vtk = getattr(prismoid, "_vtk_obj", getattr(prismoid, "VTKObject", prismoid))
                self._connector_actor.GetMapper().SetInputData(mesh_vtk)

    def _build_and_update_meshes(self):
        """Rebuild the primitive from the current spheres and draw its buildGeometry() surface.

        The first build adds the actor; later ones overwrite the same PolyData in place, so
        the mapper, the clip planes and the pick list keep pointing at it. Coincident centres
        leave a connection that never had a surface without one.
        """
        if self.spheremesh is None and self.has_coincident_centers():
            print(f"Could not build spheremesh for {self._sphere_indices}: two sphere centres coincide")
            return
        vertices, triangles = self.build_primitive().buildGeometry(SPHEREMESH_NUM_THETA, SPHEREMESH_NUM_PHI)
        color_rgb = pv.Color(self._current_mesh_color()).float_rgb

        if self.spheremesh is None:
            self.spheremesh = pv.PolyData(vertices, triangles_to_faces(triangles))
            actor = self._plotter.add_mesh(
                self.spheremesh, color=color_rgb, opacity=self._opacity, smooth_shading=False
            )
            # SMFitting's views shade every primitive flat.
            actor.GetProperty().SetInterpolationToFlat()
            actor.SetVisibility(bool(self._spheremesh_visible))
            self._actors.append(actor)
            self._manager.notify_actor_created(actor)
            return

        self.spheremesh.points = vertices
        # A prysmoid that gains or loses its tangent faces changes its triangle count.
        if self.spheremesh.n_cells != len(triangles):
            self.spheremesh.faces = triangles_to_faces(triangles)
        self.spheremesh.Modified()
        actor = self._actors[0]
        actor.GetProperty().SetColor(color_rgb)
        actor.GetProperty().SetOpacity(self._opacity)
        if self._spheremesh_visible:
            actor.SetVisibility(True)

    def update_mesh(self):
        """Follow the spheres: the connector always, the surface while it is shown."""
        if self._spheremesh_visible:
            self._build_and_update_meshes()
        self._update_connector()

    def set_visibility(self, visible):
        self._spheremesh_visible = visible
        if visible:
            # Refresh once when becoming visible so hidden-time edits are applied.
            self.update_mesh()
        for a in self._actors:
            a.SetVisibility(visible)

    def set_opacity(self, opacity):
        self._opacity = opacity
        for a in self._actors:
            a.GetProperty().SetOpacity(opacity)
        if self._connector_actor is not None:
            self._connector_actor.GetProperty().SetOpacity(opacity)

    def set_selected(self, selected):
        self._selected = bool(selected)
        self.refresh_style()

    def refresh_style(self):
        """Update mesh actor colors from current sphere lock/selection state."""
        color = self._current_mesh_color()
        color_rgb = pv.Color(color).float_rgb
        for a in self._actors:
            a.GetProperty().SetColor(color_rgb)
        if self._connector_actor is not None:
            connector_rgb = pv.Color(self._current_connector_color()).float_rgb
            self._connector_actor.GetProperty().SetColor(connector_rgb)

    def delete(self):
        if self._connector_actor is not None:
            self._plotter.remove_actor(self._connector_actor)
            self._connector_actor = None
        for a in self._actors:
            self._plotter.remove_actor(a)
        self._actors.clear()
        self.spheremesh = None

    def get_distance(self, point):
        """Signed distance from point to the connection's capsuloid / prysmoid (negative inside)."""
        return self.build_primitive().signedDistance(np.asarray(point, dtype=np.float64))

    def get_distances(self, points):
        """get_distance() for every row of an (N, 3) array at once."""
        return self.build_primitive().signedDistances(np.asarray(points, dtype=np.float64))
