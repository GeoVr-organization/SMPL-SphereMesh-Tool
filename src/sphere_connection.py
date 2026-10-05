"""Connection between 2 or 3 spheres: connector geometry + Spheremesh / Trispheremesh actor."""
import numpy as np
import pyvista as pv
from src.spheremesh import Spheremesh, Trispheremesh

from src.utils import measure_time


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
        if self._connector_actor is not None:
            self._plotter.remove_actor(self._connector_actor)
            self._connector_actor = None
        self._build_connector()
        self.update_mesh()

    def _spheres(self):
        return [self._manager.spheres[i] for i in self._sphere_indices]

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

    def _set_actor_mesh(self, actor, mesh, color, smooth_shading=True):
        mapper = actor.GetMapper()
        data = getattr(mesh, "_vtk_obj", getattr(mesh, "VTKObject", mesh))
        mapper.SetInputData(data)
        actor.GetProperty().SetColor(pv.Color(color).float_rgb)
        actor.GetProperty().SetOpacity(self._opacity)

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

    def _build_pair_mesh(self, color):
        S = self._spheres()
        c1 = tuple(S[0].get_center())
        r1 = S[0].get_radius()
        c2 = tuple(S[1].get_center())
        r2 = S[1].get_radius()
        try:
            if self.spheremesh is not None and len(self._actors) > 0:
                self.spheremesh.update(c1, r1, c2, r2)
                actor = self._actors[0]
                actor.GetProperty().SetColor(pv.Color(color).float_rgb)
                actor.GetProperty().SetOpacity(self._opacity)
                if self._spheremesh_visible:
                    actor.SetVisibility(True)
            else:
                self.spheremesh = Spheremesh(
                    center1=c1, radius1=r1, center2=c2, radius2=r2, resolution=32
                )
                actor = self._plotter.add_mesh(
                    self.spheremesh, color=color, opacity=self._opacity, smooth_shading=True
                )
                self._actors.append(actor)
                self._actors[0].SetVisibility(bool(self._spheremesh_visible))
        except ValueError as e:
            print(f"Could not update Spheremesh for pair: {e}")

    def _build_triple_mesh(self, color):
        S = self._spheres()
        c1 = tuple(S[0].get_center())
        r1 = S[0].get_radius()
        c2 = tuple(S[1].get_center())
        r2 = S[1].get_radius()
        c3 = tuple(S[2].get_center())
        r3 = S[2].get_radius()
        try:
            if self.spheremesh is not None and len(self._actors) > 0:
                self.spheremesh.update(c1, r1, c2, r2, c3, r3)
                actor = self._actors[0]
                actor.GetProperty().SetColor(pv.Color(color).float_rgb)
                actor.GetProperty().SetOpacity(self._opacity)
                if self._spheremesh_visible:
                    actor.SetVisibility(True)
            else:
                self.spheremesh = Trispheremesh(
                    center1=c1, radius1=r1, center2=c2, radius2=r2, center3=c3, radius3=r3, resolution=32
                )
                actor = self._plotter.add_mesh(
                    self.spheremesh, color=color, opacity=self._opacity, smooth_shading=True
                )
                self._actors.append(actor)
                self._actors[0].SetVisibility(bool(self._spheremesh_visible))
        except ValueError as e:
            print(f"Could not update Trispheremesh for triple: {e}")

    def _build_and_update_meshes(self):
        color = self._current_mesh_color()
        if len(self._sphere_indices) == 2:
            self._build_pair_mesh(color)
        else:
            self._build_triple_mesh(color)

    def update_mesh(self):
        
        if self._spheremesh_visible:
            #measure_time(lambda:self._build_and_update_meshes(),label="build_and_update_meshes")
            self._build_and_update_meshes()
        else:
            #measure_time(lambda:self._update_connector(),label="update_connector")
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
        if self.spheremesh is None:
            self._build_and_update_meshes()
        if self.spheremesh is None:
            return float("inf")
        return self.spheremesh.get_distance(point)
