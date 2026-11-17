"""Headless tests for the Sphere Plane Fit add-on.

Runs with the bpy module from PyPI (pip install bpy), no Blender UI:

    python blender_addon/tests/test_headless.py
"""

import math
import sys
from pathlib import Path

import bpy  # noqa: I001  (bpy before bmesh/mathutils outside Blender)
import bmesh
import numpy as np
from mathutils import Euler, Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sphere_plane_fit as spf  # noqa: E402

FAILURES = []


def check(condition, message):
    print(("ok   " if condition else "FAIL ") + message)
    if not condition:
        FAILURES.append(message)


def close(a, b, tol):
    return np.allclose(np.asarray(a, float), np.asarray(b, float), atol=tol, rtol=0)


# ---------------------------------------------------------------------------
# Solver alone
# ---------------------------------------------------------------------------


def sphere_patch(center, radius, n=200, max_polar=1.0, seed=0):
    rng = np.random.default_rng(seed)
    polar = rng.uniform(0, max_polar, n)
    azimuth = rng.uniform(0, 2 * np.pi, n)
    normals = np.column_stack(
        [np.sin(polar) * np.cos(azimuth), np.sin(polar) * np.sin(azimuth), np.cos(polar)]
    )
    return np.asarray(center) + radius * normals, normals


def test_solver():
    center, radius = np.array([1.0, 2.0, 3.0]), 0.5
    points, normals = sphere_patch(center, radius)
    start = np.array([0.0, 0.0, 0.0, 1.0])

    res = spf.solve_tangency(points, normals, start, [False] * 4)
    check(close(res.x, [*center, radius], 1e-10), "solver: recovers an exact sphere patch")
    check(res.rank == 4 and np.abs(res.residuals).max() < 1e-10, "solver: rank 4, zero residuals")

    # Lock y at a wrong value: y must stay, the rest must equal the reduced
    # system with the y column removed and n_y * y0 moved to the right.
    current = np.array([0.0, 2.3, 0.0, 1.0])
    res = spf.solve_tangency(points, normals, current, [False, True, False, False])
    A = np.column_stack([normals, np.ones(len(points))])
    b = np.einsum("ij,ij->i", points, normals)
    reduced = np.linalg.lstsq(A[:, [0, 2, 3]], b - A[:, 1] * 2.3, rcond=None)[0]
    check(res.x[1] == 2.3, "solver: locked y keeps its value exactly")
    check(close(res.x[[0, 2, 3]], reduced, 1e-12), "solver: lock == column removed, moved to the right-hand side")

    # Every unknown locked: nothing moves, residuals of the current sphere.
    res = spf.solve_tangency(points, normals, current, [True] * 4)
    check(close(res.x, current, 0) and res.n_free == 0, "solver: all locked leaves the sphere alone")

    # A flat patch only fixes cz + r; x and y keep their current value.
    flat = np.column_stack([np.random.default_rng(1).uniform(-1, 1, (50, 2)), np.zeros(50)])
    up = np.tile([0.0, 0.0, 1.0], (50, 1))
    res = spf.solve_tangency(flat, up, [0.3, -0.4, 0.2, 0.5], [False] * 4)
    check(res.rank == 1, "solver: flat patch has rank 1")
    check(close(res.x[:2], [0.3, -0.4], 1e-12), "solver: undetermined directions stay at their current value")
    check(abs(res.x[2] + res.x[3]) < 1e-12, "solver: flat patch satisfies cz + r = 0")

    c0, r0 = spf.initial_sphere(points)
    check(close(c0, points.mean(0), 1e-12), "initial sphere: center is the centroid")
    check(abs(r0 - np.linalg.norm(points - c0, axis=1).mean()) < 1e-12, "initial sphere: mean distance")


# ---------------------------------------------------------------------------
# Inside Blender
# ---------------------------------------------------------------------------


def new_mesh_object(name, build):
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    build(bm)
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def enter_edit(obj, select):
    """Edit Mode on obj with the vertices for which select(local co) is True."""
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    for o in bpy.context.selected_objects:
        o.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.context.tool_settings.mesh_select_mode = (True, False, False)
    bm = bmesh.from_edit_mesh(obj.data)
    for f in bm.faces:
        f.select = False
    for e in bm.edges:
        e.select = False
    for v in bm.verts:
        v.select = bool(select(v.co))
    bm.select_flush_mode()
    bmesh.update_edit_mesh(obj.data)
    return sum(v.select for v in bm.verts)


def reference_world_data(obj, group_name):
    """World positions and normals of a group, computed with mathutils."""
    gi = obj.vertex_groups[group_name].index
    mw = obj.matrix_world
    nm = mw.to_3x3().inverted().transposed()
    pts, nos = [], []
    for v in obj.data.vertices:
        if any(g.group == gi for g in v.groups):
            pts.append(mw @ v.co)
            nos.append((nm @ v.normal).normalized())
    return np.array(pts), np.array(nos)


def group_size(obj, group_name):
    gi = obj.vertex_groups[group_name].index
    return sum(any(g.group == gi for g in v.groups) for v in obj.data.vertices)


def test_blender():
    scene = bpy.context.scene

    # Icosphere of radius 0.5 under rotation, translation and uniform scale 2:
    # in world space a sphere of center `loc` and radius 1.
    body = new_mesh_object(
        "Body", lambda bm: bmesh.ops.create_icosphere(bm, subdivisions=4, radius=0.5)
    )
    loc = Vector((1.0, -2.0, 0.5))
    body.matrix_world = Matrix.LocRotScale(loc, Euler((0.3, 0.7, -1.1)), Vector((2, 2, 2)))
    bpy.context.view_layer.update()

    n_sel = enter_edit(body, lambda co: co.z > 0.2)
    bpy.ops.sphere_fit.create()
    sph = scene.sphere_fit_edit
    s = sph.sphere_fit
    check(spf.is_fit_sphere(sph), "create: the new object is a fit sphere")
    check(s.source == body and s.group_name in body.vertex_groups, "create: points to the mesh and its group")
    check(s.n_verts == n_sel, f"create: group holds the {n_sel} selected vertices")
    cx, cy, cz, r = spf.sphere_params(sph)
    check(close([cx, cy, cz], loc, 5e-3) and abs(r - 1.0) < 5e-3, f"create: fit near center {tuple(loc)}, r 1 -> {cx:.4f} {cy:.4f} {cz:.4f} {r:.4f}")
    check(s.status == "OK", f"create: status OK ({s.status} {s.message})")

    # Same numbers as an independent world-space computation.
    bpy.ops.object.mode_set(mode="OBJECT")
    pts, nos = reference_world_data(body, s.group_name)
    expected = spf.solve_tangency(pts, nos, [0, 0, 0, 1], [False] * 4).x
    check(not bpy.ops.sphere_fit.fit.poll(), "fit: no target when the mesh is active")
    bpy.context.view_layer.objects.active = sph
    bpy.ops.sphere_fit.fit()
    check(close(spf.sphere_params(sph), expected, 1e-6), "object mode: world transform matches mathutils reference")

    # Lock y at a moved value through the property: the update refits.
    sph.location.y = -1.7
    bpy.context.view_layer.update()
    s.lock_y = True
    p = spf.sphere_params(sph)
    check(abs(p[1] + 1.7) < 1e-6, f"lock y: y stays at -1.7 -> {p[1]:.6f}")
    A = np.column_stack([nos, np.ones(len(nos))])
    b = np.einsum("ij,ij->i", pts, nos)
    reduced = np.linalg.lstsq(A[:, [0, 2, 3]], b - A[:, 1] * -1.7, rcond=None)[0]
    check(close(p[[0, 2, 3]], reduced, 1e-6), "lock y: other unknowns solve the reduced system")
    s.lock_y = False
    check(close(spf.sphere_params(sph), expected, 1e-6), "unlock y: back to the free fit")

    # Lock r: radius stays.
    sph.scale = (0.8, 0.8, 0.8)
    bpy.context.view_layer.update()
    s.lock_r = True
    check(abs(spf.sphere_params(sph)[3] - 0.8) < 1e-6, "lock r: radius stays at 0.8")
    s.lock_r = False

    # Edit the vertex list from Object Mode, through Edit Mode.
    bpy.ops.sphere_fit.edit_vertices()
    check(bpy.context.mode == "EDIT_MESH" and bpy.context.edit_object == body, "edit vertices: Edit Mode on the mesh")
    bm = bmesh.from_edit_mesh(body.data)
    check(sum(v.select for v in bm.verts) == n_sel, "edit vertices: the sphere's vertices are selected")

    n_band = enter_edit(body, lambda co: -0.1 < co.z <= 0.2)
    scene.sphere_fit_edit = sph
    bpy.ops.sphere_fit.assign_vertices(mode="ADD")
    check(s.n_verts == n_sel + n_band, f"add: {n_sel} + {n_band} vertices")
    enter_edit(body, lambda co: co.z > 0.2)
    bpy.ops.sphere_fit.assign_vertices(mode="REMOVE")
    check(s.n_verts == n_band, "remove: only the band is left")
    enter_edit(body, lambda co: co.x > 0.25)
    n_side = sum(v.select for v in bmesh.from_edit_mesh(body.data).verts)
    bpy.ops.sphere_fit.assign_vertices(mode="REPLACE")
    check(s.n_verts == n_side, "replace: the group is the selection")
    edit_mode_fit = spf.sphere_params(sph)
    bpy.ops.sphere_fit.select_vertices()
    bpy.ops.sphere_fit.finish_edit()
    check(bpy.context.mode == "OBJECT" and bpy.context.active_object == sph, "finish: Object Mode, sphere active")
    check(group_size(body, s.group_name) == n_side, "the group survives leaving Edit Mode")
    bpy.ops.sphere_fit.fit()
    check(close(spf.sphere_params(sph), edit_mode_fit, 1e-6), "edit-mode and object-mode data give the same fit")
    cx, cy, cz, r = spf.sphere_params(sph)
    check(close([cx, cy, cz], loc, 1e-2) and abs(r - 1) < 1e-2, "side patch still recovers the sphere")

    # Reset to the initial sphere.
    bpy.ops.sphere_fit.reset_initial()
    pts, _ = reference_world_data(body, s.group_name)
    c0, r0 = spf.initial_sphere(pts)
    check(close(spf.sphere_params(sph), [*c0, r0], 1e-6), "reset: centroid and mean distance")

    # Group renamed: link broken, then repaired by setting the name.
    body.vertex_groups[s.group_name].name = "renamed"
    try:
        bpy.ops.sphere_fit.fit()  # reports an error, which raises outside the UI
    except RuntimeError:
        pass
    check(s.status == "ERROR", "renamed group: reported as missing")
    s.group_name = "renamed"
    check(s.status == "OK", "renamed group: setting the name relinks and refits")

    # A flat patch: degenerate, warned, x and y untouched.
    plane = new_mesh_object("Plane", lambda bm: bmesh.ops.create_grid(bm, x_segments=6, y_segments=6, size=1.0))
    enter_edit(plane, lambda co: True)
    bpy.ops.sphere_fit.create()
    flat = scene.sphere_fit_edit
    check(flat.sphere_fit.status == "WARNING" and "degenere" in flat.sphere_fit.message, f"flat patch: degenerate warning ({flat.sphere_fit.message})")

    # Inverted normals: negative radius, sphere left where it was.
    inside = new_mesh_object("Inverted", lambda bm: (bmesh.ops.create_icosphere(bm, subdivisions=3, radius=1.0), bmesh.ops.reverse_faces(bm, faces=bm.faces)))
    enter_edit(inside, lambda co: co.z > 0.3)
    try:
        bpy.ops.sphere_fit.create()
    except RuntimeError:
        pass
    inv = scene.sphere_fit_edit
    check(inv.sphere_fit.status == "ERROR" and "≤ 0" in inv.sphere_fit.message, "inverted normals: negative radius refused")
    check(spf.sphere_params(inv)[3] > 0, "inverted normals: sphere keeps the initial radius")

    # A loose vertex in the group is dropped, not turned into r = 0.
    def with_loose(bm):
        bmesh.ops.create_icosphere(bm, subdivisions=3, radius=1.0)
        bm.verts.new((0.0, 0.0, 5.0))

    loose = new_mesh_object("Loose", with_loose)
    enter_edit(loose, lambda co: co.z > 0.3)
    bpy.ops.sphere_fit.create()
    lsph = scene.sphere_fit_edit
    check("senza facce" in lsph.sphere_fit.message, f"loose vertex: dropped ({lsph.sphere_fit.message})")
    check(abs(spf.sphere_params(lsph)[3] - 1) < 2e-2, "loose vertex: fit unaffected")
    bpy.ops.object.mode_set(mode="OBJECT")

    # Delete sphere and group.
    group = s.group_name
    for o in bpy.context.selected_objects:
        o.select_set(False)
    bpy.context.view_layer.objects.active = sph
    bpy.ops.sphere_fit.delete("EXEC_DEFAULT")
    check("SPF_Sphere" not in [o.name for o in bpy.data.objects] and group not in body.vertex_groups, "delete: sphere and group gone")

    test_panel()


# ---------------------------------------------------------------------------
# Panel: a recording layout checks properties, operators and icons.
# ---------------------------------------------------------------------------

ICONS = set(bpy.types.UILayout.bl_rna.functions["prop"].parameters["icon"].enum_items.keys())


class Layout:
    errors = []

    def __getattr__(self, name):
        return lambda *a, **k: Layout()

    def _icon(self, k):
        if "icon" in k and k["icon"] not in ICONS:
            Layout.errors.append(f"icon {k['icon']}")

    def prop(self, data, name, **k):
        if name not in data.bl_rna.properties:
            Layout.errors.append(f"prop {name}")
        self._icon(k)
        return Layout()

    def prop_search(self, data, name, search_data, search_name, **k):
        if name not in data.bl_rna.properties or search_name not in search_data.bl_rna.properties:
            Layout.errors.append(f"prop_search {name}")
        return Layout()

    def operator(self, idname, **k):
        module, op = idname.split(".")
        if not hasattr(getattr(bpy.ops, module), op):
            Layout.errors.append(f"operator {idname}")
        self._icon(k)
        return Layout()

    def label(self, **k):
        self._icon(k)
        return Layout()


class FakePanel:
    layout = Layout()


def test_panel():
    scene = bpy.context.scene
    sph = next(o for o in bpy.data.objects if spf.is_fit_sphere(o) and o.sphere_fit.source.name == "Loose")
    bpy.context.view_layer.objects.active = sph
    spf.SPF_PT_panel.draw(FakePanel, bpy.context)  # object mode, sphere active
    bpy.context.view_layer.objects.active = sph.sphere_fit.source
    spf.SPF_PT_panel.draw(FakePanel, bpy.context)  # object mode, no sphere
    scene.sphere_fit_edit = sph
    enter_edit(sph.sphere_fit.source, lambda co: True)
    spf.SPF_PT_panel.draw(FakePanel, bpy.context)  # edit mode with a sphere
    bpy.ops.object.mode_set(mode="OBJECT")
    check(not Layout.errors, f"panel: properties, operators and icons exist {Layout.errors}")


# ---------------------------------------------------------------------------
# Colored groups
# ---------------------------------------------------------------------------


def attribute_colors(obj):
    attr = obj.data.color_attributes[spf.GROUP_COLOR_ATTRIBUTE]
    values = np.empty(len(obj.data.vertices) * 4, dtype=np.float32)
    attr.data.foreach_get("color", values)
    return values.reshape(-1, 4)


def members(obj, name):
    gi = obj.vertex_groups[name].index
    return {v.index for v in obj.data.vertices if any(g.group == gi for g in v.groups)}


def test_color_groups():
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    body = new_mesh_object("Groups", lambda bm: bmesh.ops.create_icosphere(bm, subdivisions=3, radius=1.0))
    # A sphere group on the same mesh must survive everything below.
    n_top = enter_edit(body, lambda co: co.z > 0.5)
    bpy.ops.sphere_fit.create()
    sphere_group = bpy.context.scene.sphere_fit_edit.sphere_fit.group_name

    n_a = enter_edit(body, lambda co: co.x > 0.3)
    bpy.ops.sphere_fit.color_group_new()
    enter_edit(body, lambda co: co.x < -0.3)
    bpy.ops.sphere_fit.color_group_new()
    items = body.spf_color_groups
    check(len(items) == 2 and items[0].label == "Gruppo 1" and items[1].label == "Gruppo 2", "colored: two groups created")
    check(items[0].count == n_a, f"colored: first group holds the {n_a} selected vertices")

    # Exclusive: give group 2 a slab overlapping group 1.
    n_slab = enter_edit(body, lambda co: co.x > 0.6)
    body.spf_color_groups_index = 1
    bpy.ops.sphere_fit.color_group_assign(mode="ADD")
    check(items[1].count > n_slab and items[0].count == n_a - n_slab, "colored: assigning moves vertices out of the other group")
    bpy.ops.object.mode_set(mode="OBJECT")
    g1, g2 = members(body, items[0].group), members(body, items[1].group)
    check(not (g1 & g2), "colored: no vertex in two colored groups")
    check(len(members(body, sphere_group)) == n_top, "colored: the sphere's vertex group is untouched")

    # Attribute colors: group colors where assigned, neutral elsewhere.
    colors = attribute_colors(body)
    c1, c2 = (*items[0].color, 1.0), (*items[1].color, 1.0)
    ok = all(np.allclose(colors[i], c1, atol=1e-6) for i in g1) and all(np.allclose(colors[i], c2, atol=1e-6) for i in g2)
    rest = set(range(len(body.data.vertices))) - g1 - g2
    ok = ok and all(np.allclose(colors[i], spf.UNGROUPED_COLOR, atol=1e-6) for i in rest)
    check(ok, "colored: attribute matches group colors, neutral elsewhere")
    check(body.data.color_attributes.active_color_name == spf.GROUP_COLOR_ATTRIBUTE, "colored: attribute is the active color")

    # Changing a color updates the attribute (object mode).
    items[0].color = (0.0, 0.0, 1.0)
    i = next(iter(g1))
    check(np.allclose(attribute_colors(body)[i], (0, 0, 1, 1), atol=1e-6), "colored: color change rewrites the attribute")

    # Rename through the list renames the vertex group; a taken name is made unique.
    items[0].label = "Braccio"
    check(items[0].group == "Braccio" and "Braccio" in body.vertex_groups, "colored: rename renames the vertex group")
    items[1].label = "Braccio"
    check(items[1].group == items[1].label != "Braccio", f"colored: duplicate name made unique ({items[1].label})")

    # Edit Mode: remove, select, deselect, color change.
    enter_edit(body, lambda co: False)
    body.spf_color_groups_index = 0
    bpy.ops.sphere_fit.color_group_select(select=True)
    bm = bmesh.from_edit_mesh(body.data)
    check({v.index for v in bm.verts if v.select} == g1, "colored: select picks the group's vertices")
    bpy.ops.sphere_fit.color_group_select(select=False)
    check(not any(v.select for v in bm.verts), "colored: deselect clears them")
    enter_edit(body, lambda co: co.x > 0.3 and co.z > 0)
    n_sel = sum(v.select for v in bmesh.from_edit_mesh(body.data).verts)
    before = items[0].count
    bpy.ops.sphere_fit.color_group_assign(mode="REMOVE")
    check(items[0].count < before, f"colored: remove takes selected vertices out ({before} -> {items[0].count})")
    items[0].color = (1.0, 0.0, 0.0)  # edit-mode recolor path
    bpy.ops.object.mode_set(mode="OBJECT")
    g1 = members(body, items[0].group)
    check(all(np.allclose(attribute_colors(body)[i], (1, 0, 0, 1), atol=1e-6) for i in g1), "colored: edit-mode colors survive leaving Edit Mode")

    # Remove a colored group: vertex group gone, vertices back to neutral.
    removed = set(g1)
    body.spf_color_groups_index = 0
    bpy.ops.sphere_fit.color_group_remove()
    check(len(items) == 1 and "Braccio" not in body.vertex_groups, "colored: removing deletes item and vertex group")
    check(all(np.allclose(attribute_colors(body)[i], spf.UNGROUPED_COLOR, atol=1e-6) for i in removed), "colored: its vertices turn neutral")

    # Panel draws in both modes.
    bpy.context.view_layer.objects.active = body
    spf.SPF_PT_color_groups.draw(FakePanel, bpy.context)
    enter_edit(body, lambda co: True)
    spf.SPF_PT_color_groups.draw(FakePanel, bpy.context)
    item = items[0]
    spf.SPF_UL_color_groups.draw_item(None, bpy.context, Layout(), body, item, 0, body, "spf_color_groups_index", 0)
    bpy.ops.object.mode_set(mode="OBJECT")
    check(not Layout.errors, f"colored panel: properties, operators and icons exist {Layout.errors}")


if __name__ == "__main__":
    bpy.ops.wm.read_factory_settings(use_empty=True)
    spf.register()
    test_solver()
    test_blender()
    test_color_groups()
    spf.unregister()
    print(f"\n{len(FAILURES)} failures")
    sys.exit(1 if FAILURES else 0)
