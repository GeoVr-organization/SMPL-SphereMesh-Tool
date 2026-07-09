import pyvista as pv
import numpy as np


class Spheremesh(pv.PolyData):
    """
    Mesh connecting two spheres: truncated cone when radii differ, or a cylinder
    plus the two clipped spheres when radii are equal.
    """

    @staticmethod
    def _build_combined(center1, radius1, center2, radius2, resolution):
        """Compute geometry and return (combined PolyData, c1, r1, c2, r2)."""
        if radius1 < radius2:
            center1, center2 = center2, center1
            radius1, radius2 = radius2, radius1

        center1 = np.array(center1, dtype=float)
        center2 = np.array(center2, dtype=float)

        direction = center2 - center1
        distance  = np.linalg.norm(direction)
        if distance < 1e-12:
            raise ValueError("Sphere centers coincide; cannot build segment")
        direction /= distance

        l         = np.sqrt(max(distance**2 - (radius1 - radius2)**2, 0.0))
        cos_alpha = l / distance
        d1 = (radius1 - radius2) * radius1 / distance
        d2 = (radius1 - radius2) * radius2 / distance

        base_large = center1 + direction * d1
        base_small = center2 + direction * d2
        R_large    = radius1 * cos_alpha

        '''
        sphere1 = pv.Sphere(radius=radius1, center=center1,
                            phi_resolution=resolution, theta_resolution=resolution)
        sphere2 = pv.Sphere(radius=radius2, center=center2,
                            phi_resolution=resolution, theta_resolution=resolution)
        sphere1 = sphere1.clip(normal= direction, origin=base_large)
        sphere2 = sphere2.clip(normal=-direction, origin=base_small)
        '''
        if abs(radius1 - radius2) < 1e-6:
            cyl_center = (base_large + base_small) / 2.0
            cyl_height = np.linalg.norm(base_small - base_large)
            cyl = pv.Cylinder(
                center=cyl_center, direction=direction,
                radius=R_large,
                height=cyl_height,
                resolution=resolution, capping=True,
            )
            combined = cyl
        else:
            apex        = center2 + direction * (radius2 * distance / (radius1 - radius2))
            cone_h      = float(np.linalg.norm(apex - base_large))
            cone_center = (apex + base_large) / 2.0
            cone = pv.Cone(
                center=cone_center,
                direction=direction,
                height=cone_h,
                radius=R_large,
                resolution=resolution,
                capping=True,
            )
            truncated = cone.clip(normal=direction, origin=base_small)
            combined  = truncated

        return combined, center1, radius1, center2, radius2

    def __init__(self, center1=None, radius1=None, center2=None, radius2=None, resolution=16):
        self.resolution = resolution
        # PyVista may instantiate subclasses with no args during internal copy/filter passes.
        # Support that path with an empty PolyData instance.
        if center1 is None or radius1 is None or center2 is None or radius2 is None:
            super().__init__()
            self.center1 = np.zeros(3, dtype=float)
            self.radius1 = 0.0
            self.center2 = np.zeros(3, dtype=float)
            self.radius2 = 0.0
            return
        combined, c1, r1, c2, r2 = Spheremesh._build_combined(center1, radius1, center2, radius2, resolution)
        super().__init__(combined.points, combined.faces)
        self.center1 = c1
        self.radius1 = r1
        self.center2 = c2
        self.radius2 = r2

    def update(self, center1, radius1, center2, radius2):
        """Recompute geometry in-place: updates the existing VTK object so the
        mapper does not need to be reconnected — just calls Modified() at the end."""
        combined, c1, r1, c2, r2 = Spheremesh._build_combined(
            center1, radius1, center2, radius2, self.resolution
        )
        self.copy_from(combined)
        self.center1 = c1
        self.radius1 = r1
        self.center2 = c2
        self.radius2 = r2
        self.Modified()

    def get_distance(self, point):
        """Get distance to the point."""
        distance = min(np.linalg.norm(point - self.center1)-self.radius1, np.linalg.norm(point - self.center2)-self.radius2)
       

        distance= min(distance, self.point_segment_distance_with_radius(point, self.center1, self.center2))
        
        return distance
    
    def point_segment_distance_with_radius(self, P, A, B):
        """Distance between point P and segment AB in 3D."""
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

        distance=np.linalg.norm(P-closest)-(self.radius1+t*(self.radius2-self.radius1))
        return distance
    

class Trispheremesh(pv.PolyData):
    """
    A mesh connecting three spheres: three truncated cones (Spheremesh segments)
    between each pair of spheres, plus two triangular caps. Same structure as
    Spheremesh (single PolyData with stored parameters).

    Parameters
    ----------
    center1, center2, center3 : tuple or array-like
        Centers of the three spheres (x, y, z)
    radius1, radius2, radius3 : float
        Radii of the three spheres
    resolution : int, optional
        Facet resolution for cones and spheres, default is 64
    """

    @staticmethod
    def _build_combined(center1, radius1, center2, radius2, center3, radius3, resolution):
        """Compute geometry and return combined PolyData."""
        center1 = np.array(center1)
        center2 = np.array(center2)
        center3 = np.array(center3)
        centers = [center1, center2, center3]
        radii = [float(radius1), float(radius2), float(radius3)]

        connections = [(0, 1), (1, 2), (0, 2)]
        meshes_to_merge = []
        for i, j in connections:
            sm = Spheremesh(
                centers[i], radii[i],
                centers[j], radii[j],
                resolution=resolution,
            )
            meshes_to_merge.append(sm)

        c0, c1, c2 = centers[0], centers[1], centers[2]
        r0, r1, r2 = radii[0], radii[1], radii[2]
        v1 = c1 - c0
        v2 = c2 - c0
        normal = np.cross(v1, v2)
        normal_length = np.linalg.norm(normal)
        if normal_length > 1e-6:
            normal = normal / normal_length
            points_top = np.array([c0 + normal * r0, c1 + normal * r1, c2 + normal * r2])
            meshes_to_merge.append(pv.PolyData(points_top, np.array([3, 0, 1, 2])))
            points_bottom = np.array([c0 - normal * r0, c1 - normal * r1, c2 - normal * r2])
            meshes_to_merge.append(pv.PolyData(points_bottom, np.array([3, 0, 1, 2])))

        if not meshes_to_merge:
            raise ValueError("Trispheremesh: no valid segments (all pairs may have equal radii)")

        combined = meshes_to_merge[0]
        for m in meshes_to_merge[1:]:
            combined = combined.merge([m])

        return combined, center1, radii[0], center2, radii[1], center3, radii[2]

    def __init__(
        self,
        center1=None,
        radius1=None,
        center2=None,
        radius2=None,
        center3=None,
        radius3=None,
        resolution=64,
    ):
        self.resolution = resolution
        # PyVista may instantiate subclasses with no args during internal copy/filter passes.
        if (
            center1 is None or radius1 is None or center2 is None
            or radius2 is None or center3 is None or radius3 is None
        ):
            super().__init__()
            self.center1 = np.zeros(3, dtype=float)
            self.radius1 = 0.0
            self.center2 = np.zeros(3, dtype=float)
            self.radius2 = 0.0
            self.center3 = np.zeros(3, dtype=float)
            self.radius3 = 0.0
            return
        combined, c1, r1, c2, r2, c3, r3 = Trispheremesh._build_combined(
            center1, radius1, center2, radius2, center3, radius3, resolution
        )
        super().__init__(combined.points, combined.faces)
        self.center1 = c1
        self.radius1 = r1
        self.center2 = c2
        self.radius2 = r2
        self.center3 = c3
        self.radius3 = r3

    def update(self, center1, radius1, center2, radius2, center3, radius3):
        """Recompute geometry in-place: updates the existing VTK object so the
        mapper does not need to be reconnected — just calls Modified() at the end."""
        combined, c1, r1, c2, r2, c3, r3 = Trispheremesh._build_combined(
            center1, radius1, center2, radius2, center3, radius3, self.resolution
        )
        self.copy_from(combined)
        self.center1 = c1
        self.radius1 = r1
        self.center2 = c2
        self.radius2 = r2
        self.center3 = c3
        self.radius3 = r3
        self.Modified()


    def get_distance(self, point):
        """Get distance to the point."""
        distance = min(np.linalg.norm(point - self.center1)-self.radius1, np.linalg.norm(point - self.center2)-self.radius2, np.linalg.norm(point - self.center3)-self.radius3)

        distance= min(distance, self.point_segment_distance_with_radius(point, self.center1, self.center2, self.radius1, self.radius2))
        distance= min(distance, self.point_segment_distance_with_radius(point, self.center2, self.center3, self.radius2, self.radius3))
        distance= min(distance, self.point_segment_distance_with_radius(point, self.center1, self.center3, self.radius1, self.radius3))

        return distance
    
    def point_segment_distance_with_radius(self, P, A, B, radius1, radius2):
        """Distance between point P and segment AB in 3D."""
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
        distance=np.linalg.norm(P-closest)-(radius1+t*(radius2-radius1))
        return distance



