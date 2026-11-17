# SPDX-License-Identifier: GPL-3.0-or-later
"""Sphere Plane Fit.

Fits one sphere to a set of mesh vertices with the tangency condition of
SMFitting/main.py, section 8, reduced to a single sphere whose contributions
are all 1. Every vertex p_i with normal n_i gives one row

    c . n_i + r = p_i . n_i

so the least-squares solution minimizes the distance between the sphere and
the tangent plane of every vertex. Unknowns are cx, cy, cz, r, in Blender
world coordinates, from the rest pose of the mesh (its base vertices, no
modifiers).

Any unknown can be locked. A locked unknown keeps the sphere's current value:
its column leaves the system and its column times that value moves to the
right-hand side. There is no soft regularization.

The vertices a sphere uses are a vertex group on the source mesh. The sphere
object stores a pointer to the mesh and the name of the group, plus the locks,
so it can be refitted after its vertices or its locks change.

Colored groups: named vertex groups of a mesh, each with a color, to select
again later. A vertex belongs to at most one colored group; the colors are a
point color attribute, rebuilt whenever a group, its vertices or its color
change, and shown in Solid shading with Color: Attribute.
"""

bl_info = {
    "name": "Sphere Plane Fit",
    "author": "SMPL-SphereMesh-Tool",
    "version": (0, 2, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > Sphere Fit",
    "description": "Least-squares sphere tangent to the selected vertices (sphere-plane distance)",
    "category": "Mesh",
}

import colorsys
from dataclasses import dataclass

import bpy  # noqa: I001  (bpy first: bmesh and mathutils need it outside Blender)
import bmesh
import numpy as np
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from mathutils import Matrix

COMPONENTS = ("x", "y", "z", "r")
UNIT_SPHERE_MESH = "SPF_UnitSphere"
SPHERE_COLLECTION = "Sphere Fit"
DEFAULT_SPHERE_NAME = "SPF_Sphere"
# Singular values below RCOND times the largest count as zero.
RCOND = 1e-9
# Above this condition number the fit is reported as ill-conditioned.
ILL_CONDITIONED = 1e4
GROUP_COLOR_ATTRIBUTE = "SPF_GroupColor"
# Linear color of the vertices in no colored group.
UNGROUPED_COLOR = (0.35, 0.35, 0.35, 1.0)


# ---------------------------------------------------------------------------
# Solver: plain numpy, no bpy.
# ---------------------------------------------------------------------------


@dataclass
class FitResult:
    x: np.ndarray  # cx, cy, cz, r
    residuals: np.ndarray  # (p_i - c) . n_i - r, one per vertex
    n_free: int
    rank: int
    cond: float


def solve_tangency(points, normals, current, locks):
    """Least-squares sphere tangent to the planes (p_i, n_i).

    points, normals: (N, 3), normals of unit length.
    current: (4,) cx, cy, cz, r of the sphere now.
    locks: (4,) bools; a locked unknown keeps its value in `current`.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    normals = np.asarray(normals, dtype=float).reshape(-1, 3)
    current = np.asarray(current, dtype=float).reshape(4)
    free = ~np.asarray(locks, dtype=bool).reshape(4)

    A = np.column_stack([normals, np.ones(len(points))])
    b = np.einsum("ij,ij->i", points, normals)

    x = current.copy()
    n_free = int(free.sum())
    rank, cond = 0, 0.0
    if n_free and len(points):
        # Solve for the change of the free unknowns from their current value.
        # b - A @ current moves every locked column, times its value, to the
        # right-hand side; and lstsq's minimum-norm answer leaves any
        # direction the vertices do not constrain at its current value.
        delta, _, rank, sv = np.linalg.lstsq(A[:, free], b - A @ current, rcond=RCOND)
        x[free] += delta
        rank = int(rank)
        cond = float(sv[0] / sv[n_free - 1]) if rank == n_free else float("inf")
    return FitResult(x=x, residuals=b - A @ x, n_free=n_free, rank=rank, cond=cond)


def initial_sphere(points):
    """Center at the centroid, radius the mean distance from it."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    center = points.mean(axis=0)
    radius = float(np.linalg.norm(points - center, axis=1).mean())
    return center, radius


# ---------------------------------------------------------------------------
# Mesh data
# ---------------------------------------------------------------------------


def group_local_data(obj, group_index):
    """Indices, positions and normals, in object space, of a group's vertices.

    Also returns which of them belong to a face. In Edit Mode they come from
    the edit mesh, otherwise from the mesh data: the base mesh in both cases,
    without modifiers.
    """
    me = obj.data
    if obj.mode == "EDIT":
        bm = bmesh.from_edit_mesh(me)
        bm.verts.index_update()
        bm.normal_update()
        layer = bm.verts.layers.deform.active
        verts = [] if layer is None else [v for v in bm.verts if group_index in v[layer]]
        indices = np.array([v.index for v in verts], dtype=int)
        co = np.array([v.co[:] for v in verts], dtype=float).reshape(-1, 3)
        no = np.array([v.normal[:] for v in verts], dtype=float).reshape(-1, 3)
        on_face = np.array([len(v.link_faces) > 0 for v in verts], dtype=bool)
        return indices, co, no, on_face

    indices = np.array(
        [v.index for v in me.vertices if any(g.group == group_index for g in v.groups)],
        dtype=int,
    )
    n = len(me.vertices)
    co = np.empty(n * 3)
    no = np.empty(n * 3)
    me.vertices.foreach_get("co", co)
    me.vertices.foreach_get("normal", no)
    loop_vertices = np.empty(len(me.loops), dtype=np.int32)
    me.loops.foreach_get("vertex_index", loop_vertices)
    used = np.zeros(n, dtype=bool)
    used[loop_vertices] = True
    return indices, co.reshape(-1, 3)[indices], no.reshape(-1, 3)[indices], used[indices]


def group_world_data(obj, group_index):
    """Like group_local_data, in world space, for the vertices on a face.

    Returns indices, points, unit normals and how many vertices were dropped.
    A vertex on no face has no surface normal (Blender gives it its position,
    normalized), so its row would be meaningless. Normals go through the
    inverse transpose of the object's linear part, so non-uniform scale keeps
    them perpendicular to the surface.
    """
    indices, co, no, on_face = group_local_data(obj, group_index)
    mw = np.array(obj.matrix_world, dtype=float)
    linear, translation = mw[:3, :3], mw[:3, 3]
    points = co @ linear.T + translation
    # inv(L)^T n for each normal, written for row vectors: n^T inv(L).
    normals = no @ np.linalg.inv(linear)
    lengths = np.linalg.norm(normals, axis=1)
    keep = on_face & (lengths > 1e-12)
    dropped = int((~keep).sum())
    return indices[keep], points[keep], normals[keep] / lengths[keep, None], dropped


def edit_group_from_selection(obj, group_index, mode):
    """Set, extend or reduce a vertex group from the edit-mode selection.

    mode: 'REPLACE', 'ADD' or 'REMOVE'. Members get weight 1. Returns how many
    vertices joined and how many left the group.
    """
    me = obj.data
    bm = bmesh.from_edit_mesh(me)
    layer = bm.verts.layers.deform.verify()
    added = removed = 0
    for v in bm.verts:
        dv = v[layer]
        member = group_index in dv
        if mode == "REPLACE":
            wanted = v.select
        elif mode == "ADD":
            wanted = member or v.select
        else:
            wanted = member and not v.select
        if wanted and not member:
            dv[group_index] = 1.0
            added += 1
        elif member and not wanted:
            del dv[group_index]
            removed += 1
    bmesh.update_edit_mesh(me, loop_triangles=False, destructive=False)
    return added, removed


def select_group_vertices(obj, group_index, extend=False):
    me = obj.data
    bm = bmesh.from_edit_mesh(me)
    if not extend:
        for elements in (bm.verts, bm.edges, bm.faces):
            for element in elements:
                element.select = False
    layer = bm.verts.layers.deform.active
    count = 0
    if layer is not None:
        for v in bm.verts:
            if group_index in v[layer]:
                v.select = True
                count += 1
    bm.select_mode = {"VERT"}
    bm.select_flush_mode()
    bmesh.update_edit_mesh(me, loop_triangles=False, destructive=False)
    return count


# ---------------------------------------------------------------------------
# Sphere objects
# ---------------------------------------------------------------------------


def is_fit_sphere(obj):
    return obj is not None and obj.sphere_fit.is_sphere


def sphere_params(sph):
    """cx, cy, cz, r of a sphere object, in world space."""
    mw = sph.matrix_world
    radius = float(np.mean([abs(s) for s in mw.to_scale()]))
    return np.array([*mw.translation, radius], dtype=float)


def set_sphere_params(sph, center, radius):
    """Unit sphere mesh: world location is the center, uniform scale the radius."""
    r = float(radius)
    sph.matrix_world = Matrix.Translation(tuple(map(float, center))) @ Matrix.Diagonal((r, r, r, 1.0))


def unit_sphere_mesh():
    me = bpy.data.meshes.get(UNIT_SPHERE_MESH)
    if me is None:
        me = bpy.data.meshes.new(UNIT_SPHERE_MESH)
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=1.0)
        bm.to_mesh(me)
        bm.free()
    return me


def sphere_collection(scene):
    coll = bpy.data.collections.get(SPHERE_COLLECTION)
    if coll is None:
        coll = bpy.data.collections.new(SPHERE_COLLECTION)
    if coll not in scene.collection.children_recursive:
        scene.collection.children.link(coll)
    return coll


def _sphere_inputs(sph):
    """Source data of a sphere, or an error message."""
    s = sph.sphere_fit
    src = s.source
    if src is None or src.type != "MESH":
        return None, "Nessuna mesh sorgente"
    vg = src.vertex_groups.get(s.group_name)
    if vg is None:
        return None, f"Vertex group '{s.group_name}' non trovato su {src.name}"
    try:
        indices, points, normals, dropped = group_world_data(src, vg.index)
    except np.linalg.LinAlgError:
        return None, "Trasformazione della mesh non invertibile (scala 0?)"
    if len(indices) == 0:
        if dropped:
            return None, "Nessun vertice del gruppo appartiene a una faccia"
        return None, "Il vertex group è vuoto"
    return (points, normals, dropped), None


def _store(s, result, messages, status):
    s.n_verts = len(result.residuals)
    s.rms = float(np.sqrt(np.mean(result.residuals**2)))
    s.max_abs = float(np.abs(result.residuals).max())
    s.cond = result.cond
    s.has_stats = True
    s.status = status
    s.message = "; ".join(messages)


def _fail(s, message):
    s.has_stats = False
    s.status = "ERROR"
    s.message = message
    return False


def fit_sphere(sph):
    """Solve the tangency system for a sphere and move it. True on success."""
    s = sph.sphere_fit
    inputs, error = _sphere_inputs(sph)
    if error:
        return _fail(s, error)
    points, normals, dropped = inputs
    locks = [getattr(s, f"lock_{c}") for c in COMPONENTS]
    result = solve_tangency(points, normals, sphere_params(sph), locks)

    radius = result.x[3]
    if not radius > 0:
        return _fail(
            s,
            f"Raggio risultante {radius:.4g} ≤ 0 (patch concava o normali invertite): "
            "sfera non aggiornata",
        )
    set_sphere_params(sph, result.x[:3], radius)

    messages = []
    if dropped:
        messages.append(f"{dropped} vertici senza facce ignorati")
    if result.rank < result.n_free:
        messages.append(
            f"Sistema degenere: {result.n_free - result.rank} direzioni non determinate "
            "dai vertici, lasciate al valore attuale"
        )
    elif result.cond > ILL_CONDITIONED:
        messages.append(f"Sistema mal condizionato (cond {result.cond:.1e})")
    _store(s, result, messages, "WARNING" if messages else "OK")
    return True


def reset_to_initial(sph):
    """Centroid and mean distance of the sphere's vertices. True on success."""
    s = sph.sphere_fit
    inputs, error = _sphere_inputs(sph)
    if error:
        return _fail(s, error)
    points, normals, _ = inputs
    center, radius = initial_sphere(points)
    set_sphere_params(sph, center, radius)
    # Residuals of the initial sphere, nothing solved.
    result = solve_tangency(points, normals, sphere_params(sph), [True] * 4)
    _store(s, result, ["Sfera iniziale (non fittata)"], "OK")
    return True


def target_sphere(context):
    """The sphere the panel and the operators act on."""
    if context.mode == "EDIT_MESH":
        sph = context.scene.sphere_fit_edit
        if is_fit_sphere(sph) and sph.sphere_fit.source == context.edit_object:
            return sph
        return None
    obj = context.active_object
    return obj if is_fit_sphere(obj) else None


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def _refit(self, context):
    if self.is_sphere and self.auto_fit:
        fit_sphere(self.id_data)


def _poll_source(self, obj):
    return obj.type == "MESH" and not obj.sphere_fit.is_sphere


def _lock(axis):
    return BoolProperty(
        name=f"Blocca {axis.upper()}",
        description=f"Tiene {axis.upper()} al valore attuale della sfera: la sua colonna esce dal sistema",
        default=False,
        update=_refit,
    )


class SPF_SphereSettings(bpy.types.PropertyGroup):
    is_sphere: BoolProperty(default=False, options={"HIDDEN"})
    source: PointerProperty(
        name="Mesh",
        type=bpy.types.Object,
        poll=_poll_source,
        update=_refit,
        description="Mesh dei cui vertici la sfera è tangente",
    )
    group_name: StringProperty(
        name="Gruppo",
        update=_refit,
        description="Vertex group della mesh con i vertici della sfera",
    )
    lock_x: _lock("x")
    lock_y: _lock("y")
    lock_z: _lock("z")
    lock_r: _lock("r")
    auto_fit: BoolProperty(
        name="Rifitta automaticamente",
        description="Rifitta quando cambiano vertici, gruppo o blocchi",
        default=True,
    )

    has_stats: BoolProperty(default=False, options={"HIDDEN"})
    n_verts: IntProperty(name="Vertici")
    rms: FloatProperty(name="RMS", subtype="DISTANCE", precision=5)
    max_abs: FloatProperty(name="Max", subtype="DISTANCE", precision=5)
    cond: FloatProperty(name="Condizionamento")
    status: EnumProperty(
        items=[("NONE", "None", ""), ("OK", "OK", ""), ("WARNING", "Warning", ""), ("ERROR", "Error", "")],
        default="NONE",
    )
    message: StringProperty()


def _poll_edit_sphere(self, obj):
    if not is_fit_sphere(obj):
        return False
    edit_object = bpy.context.edit_object
    return edit_object is None or obj.sphere_fit.source == edit_object


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------


def _report_status(op, sph):
    s = sph.sphere_fit
    if s.status == "ERROR":
        op.report({"ERROR"}, s.message)
    elif s.status == "WARNING":
        op.report({"WARNING"}, s.message)


class SPF_OT_create_sphere(bpy.types.Operator):
    """Crea una sfera dai vertici selezionati: un vertex group sulla mesh e una sfera che lo usa"""

    bl_idname = "sphere_fit.create"
    bl_label = "Crea sfera dai vertici selezionati"
    bl_options = {"REGISTER", "UNDO"}

    fit: BoolProperty(
        name="Fitta subito",
        description="Risolve il sistema dopo aver creato la sfera iniziale",
        default=True,
    )

    @classmethod
    def poll(cls, context):
        obj = context.edit_object
        return context.mode == "EDIT_MESH" and obj is not None and not obj.sphere_fit.is_sphere

    def execute(self, context):
        src = context.edit_object
        bm = bmesh.from_edit_mesh(src.data)
        selected = [v.index for v in bm.verts if v.select]
        if not selected:
            self.report({"ERROR"}, "Nessun vertice selezionato")
            return {"CANCELLED"}

        sph = bpy.data.objects.new(DEFAULT_SPHERE_NAME, unit_sphere_mesh())
        sph.display_type = "WIRE"
        sphere_collection(context.scene).objects.link(sph)

        vg = src.vertex_groups.new(name=sph.name)
        bm = bmesh.from_edit_mesh(src.data)
        layer = bm.verts.layers.deform.verify()
        bm.verts.ensure_lookup_table()
        for i in selected:
            bm.verts[i][layer][vg.index] = 1.0
        bmesh.update_edit_mesh(src.data, loop_triangles=False, destructive=False)

        s = sph.sphere_fit
        s.source = src
        s.group_name = vg.name
        s.is_sphere = True  # last: the updates above must not refit yet
        reset_to_initial(sph)
        if self.fit:
            fit_sphere(sph)
        context.scene.sphere_fit_edit = sph

        _report_status(self, sph)
        self.report({"INFO"}, f"{sph.name}: {len(selected)} vertici nel gruppo {vg.name}")
        return {"FINISHED"}


class SPF_OT_fit(bpy.types.Operator):
    """Risolve il sistema sfera-piano con i vertici e i blocchi della sfera"""

    bl_idname = "sphere_fit.fit"
    bl_label = "Fit"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return target_sphere(context) is not None

    def execute(self, context):
        sph = target_sphere(context)
        fit_sphere(sph)
        _report_status(self, sph)
        return {"FINISHED"}


class SPF_OT_reset_initial(bpy.types.Operator):
    """Riporta la sfera al baricentro dei suoi vertici, con raggio la loro distanza media"""

    bl_idname = "sphere_fit.reset_initial"
    bl_label = "Sfera iniziale"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return target_sphere(context) is not None

    def execute(self, context):
        sph = target_sphere(context)
        reset_to_initial(sph)
        _report_status(self, sph)
        return {"FINISHED"}


class SPF_OT_assign_vertices(bpy.types.Operator):
    """Cambia i vertici della sfera con la selezione corrente"""

    bl_idname = "sphere_fit.assign_vertices"
    bl_label = "Vertici dalla selezione"
    bl_options = {"REGISTER", "UNDO"}

    mode: EnumProperty(
        name="Modo",
        items=[
            ("REPLACE", "Sostituisci", "I vertici della sfera diventano quelli selezionati"),
            ("ADD", "Aggiungi", "Aggiunge i vertici selezionati"),
            ("REMOVE", "Rimuovi", "Toglie i vertici selezionati"),
        ],
        default="REPLACE",
    )

    @classmethod
    def poll(cls, context):
        return context.mode == "EDIT_MESH" and target_sphere(context) is not None

    def execute(self, context):
        sph = target_sphere(context)
        s = sph.sphere_fit
        src = s.source
        vg = src.vertex_groups.get(s.group_name)
        if vg is None:
            vg = src.vertex_groups.new(name=s.group_name or sph.name)
            s.group_name = vg.name
        added, removed = edit_group_from_selection(src, vg.index, self.mode)
        if s.auto_fit:
            fit_sphere(sph)
        _report_status(self, sph)
        self.report({"INFO"}, f"{vg.name}: +{added} / -{removed} vertici")
        return {"FINISHED"}


class SPF_OT_select_vertices(bpy.types.Operator):
    """Seleziona i vertici della sfera sulla mesh"""

    bl_idname = "sphere_fit.select_vertices"
    bl_label = "Seleziona i suoi vertici"
    bl_options = {"REGISTER", "UNDO"}

    extend: BoolProperty(name="Estendi", description="Aggiunge alla selezione corrente", default=False)

    @classmethod
    def poll(cls, context):
        return context.mode == "EDIT_MESH" and target_sphere(context) is not None

    def execute(self, context):
        sph = target_sphere(context)
        s = sph.sphere_fit
        vg = s.source.vertex_groups.get(s.group_name)
        if vg is None:
            self.report({"ERROR"}, f"Vertex group '{s.group_name}' non trovato")
            return {"CANCELLED"}
        context.tool_settings.mesh_select_mode = (True, False, False)
        count = select_group_vertices(s.source, vg.index, self.extend)
        self.report({"INFO"}, f"{count} vertici selezionati")
        return {"FINISHED"}


class SPF_OT_edit_vertices(bpy.types.Operator):
    """Entra in Edit Mode sulla mesh della sfera con i suoi vertici selezionati"""

    bl_idname = "sphere_fit.edit_vertices"
    bl_label = "Modifica vertici"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        sph = target_sphere(context)
        return context.mode == "OBJECT" and sph is not None and sph.sphere_fit.source is not None

    def execute(self, context):
        sph = target_sphere(context)
        src = sph.sphere_fit.source
        if not src.visible_get():
            self.report({"ERROR"}, f"{src.name} non è visibile")
            return {"CANCELLED"}
        context.scene.sphere_fit_edit = sph
        for obj in context.selected_objects:
            obj.select_set(False)
        src.select_set(True)
        context.view_layer.objects.active = src
        bpy.ops.object.mode_set(mode="EDIT")
        context.tool_settings.mesh_select_mode = (True, False, False)
        vg = src.vertex_groups.get(sph.sphere_fit.group_name)
        if vg is not None:
            select_group_vertices(src, vg.index)
        return {"FINISHED"}


class SPF_OT_finish_edit(bpy.types.Operator):
    """Torna in Object Mode con la sfera attiva"""

    bl_idname = "sphere_fit.finish_edit"
    bl_label = "Fine"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.mode == "EDIT_MESH" and target_sphere(context) is not None

    def execute(self, context):
        sph = target_sphere(context)
        bpy.ops.object.mode_set(mode="OBJECT")
        for obj in context.selected_objects:
            obj.select_set(False)
        if sph.visible_get():
            sph.select_set(True)
            context.view_layer.objects.active = sph
        return {"FINISHED"}


class SPF_OT_delete_sphere(bpy.types.Operator):
    """Elimina la sfera e, se richiesto, il suo vertex group"""

    bl_idname = "sphere_fit.delete"
    bl_label = "Elimina sfera"
    bl_options = {"REGISTER", "UNDO"}

    delete_group: BoolProperty(name="Elimina anche il vertex group", default=True)

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT" and target_sphere(context) is not None

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        sph = target_sphere(context)
        s = sph.sphere_fit
        if self.delete_group and s.source is not None:
            vg = s.source.vertex_groups.get(s.group_name)
            if vg is not None:
                s.source.vertex_groups.remove(vg)
        bpy.data.objects.remove(sph, do_unlink=True)
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------


def draw_sphere_settings(layout, sph):
    s = sph.sphere_fit

    col = layout.column()
    col.prop(s, "source")
    if s.source is not None:
        col.prop_search(s, "group_name", s.source, "vertex_groups")
    else:
        col.prop(s, "group_name")

    box = layout.box()
    box.label(text="Blocca al valore attuale")
    row = box.row(align=True)
    for c in COMPONENTS:
        locked = getattr(s, f"lock_{c}")
        row.prop(s, f"lock_{c}", text=c.upper(), toggle=True, icon="LOCKED" if locked else "UNLOCKED")
    cx, cy, cz, r = sphere_params(sph)
    values = box.column(align=True)
    values.label(text=f"Centro  {cx:.4f}   {cy:.4f}   {cz:.4f}")
    values.label(text=f"Raggio  {r:.4f}")

    row = layout.row(align=True)
    row.scale_y = 1.3
    row.operator(SPF_OT_fit.bl_idname, icon="PLAY")
    row.operator(SPF_OT_reset_initial.bl_idname, icon="LOOP_BACK")
    layout.prop(s, "auto_fit")

    if s.has_stats or s.message:
        box = layout.box()
        if s.has_stats:
            stats = box.column(align=True)
            stats.label(text=f"Vertici: {s.n_verts}")
            stats.label(text=f"Distanza sfera-piano RMS: {s.rms:.5f}")
            stats.label(text=f"Distanza sfera-piano max: {s.max_abs:.5f}")
            if s.cond:
                stats.label(text=f"Condizionamento: {s.cond:.3g}")
        if s.message:
            icon = {"ERROR": "CANCEL", "WARNING": "ERROR"}.get(s.status, "INFO")
            for i, line in enumerate(s.message.split("; ")):
                box.label(text=line, icon=icon if i == 0 else "BLANK1")


class SPF_PT_panel(bpy.types.Panel):
    bl_label = "Sphere Plane Fit"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Sphere Fit"

    def draw(self, context):
        layout = self.layout

        if context.mode == "EDIT_MESH":
            layout.operator(SPF_OT_create_sphere.bl_idname, icon="MESH_UVSPHERE")
            layout.separator()
            layout.prop(context.scene, "sphere_fit_edit", text="Sfera")
            sph = target_sphere(context)
            if sph is None:
                if context.scene.sphere_fit_edit is not None:
                    layout.label(text="La sfera appartiene a un'altra mesh", icon="INFO")
                return
            box = layout.box()
            box.label(text="Vertici dalla selezione")
            row = box.row(align=True)
            for mode, text, icon in (
                ("REPLACE", "Sostituisci", "FILE_REFRESH"),
                ("ADD", "Aggiungi", "ADD"),
                ("REMOVE", "Rimuovi", "REMOVE"),
            ):
                row.operator(SPF_OT_assign_vertices.bl_idname, text=text, icon=icon).mode = mode
            row = box.row(align=True)
            row.operator(SPF_OT_select_vertices.bl_idname, icon="RESTRICT_SELECT_OFF")
            row.operator(SPF_OT_finish_edit.bl_idname, icon="CHECKMARK")
            draw_sphere_settings(layout, sph)
            return

        sph = target_sphere(context)
        if sph is None:
            layout.label(text="Seleziona una sfera, oppure entra in")
            layout.label(text="Edit Mode su una mesh e seleziona i vertici")
            return
        row = layout.row(align=True)
        row.operator(SPF_OT_edit_vertices.bl_idname, icon="EDITMODE_HLT")
        row.operator(SPF_OT_delete_sphere.bl_idname, text="", icon="TRASH")
        draw_sphere_settings(layout, sph)



# ---------------------------------------------------------------------------
# Colored groups
# ---------------------------------------------------------------------------


def distinct_color(index):
    """Well separated hues: the golden ratio walks around the color wheel."""
    hue = (0.08 + index * 0.618033988749895) % 1.0
    return colorsys.hsv_to_rgb(hue, 0.7, 0.85)


def color_group_indices(obj):
    """Vertex group index of every colored group that still has its group."""
    indices = []
    for item in obj.spf_color_groups:
        vg = obj.vertex_groups.get(item.group)
        indices.append(None if vg is None else vg.index)
    return indices


def update_group_colors(obj):
    """Rebuild the color attribute from the colored groups; refresh counts.

    The first group in the list wins if a vertex is in more than one.
    """
    me = obj.data
    items = obj.spf_color_groups
    indices = color_group_indices(obj)
    colors = [tuple(item.color) + (1.0,) for item in items]
    counts = [0] * len(items)

    def color_of(groups):
        """groups: the vertex group indices of one vertex."""
        for k, gi in enumerate(indices):
            if gi is not None and gi in groups:
                counts[k] += 1
                return colors[k]
        return UNGROUPED_COLOR

    if obj.mode == "EDIT":
        bm = bmesh.from_edit_mesh(me)
        deform = bm.verts.layers.deform.active
        layer = bm.verts.layers.float_color.get(GROUP_COLOR_ATTRIBUTE)
        if layer is None:
            layer = bm.verts.layers.float_color.new(GROUP_COLOR_ATTRIBUTE)
        for v in bm.verts:
            v[layer] = color_of(v[deform].keys() if deform is not None else ())
        bmesh.update_edit_mesh(me, loop_triangles=False, destructive=False)
    else:
        attr = me.color_attributes.get(GROUP_COLOR_ATTRIBUTE)
        if attr is None or attr.domain != "POINT" or attr.data_type != "FLOAT_COLOR":
            if attr is not None:
                me.color_attributes.remove(attr)
            attr = me.color_attributes.new(GROUP_COLOR_ATTRIBUTE, "FLOAT_COLOR", "POINT")
        values = np.array([color_of({g.group for g in v.groups}) for v in me.vertices], dtype=np.float32)
        attr.data.foreach_set("color", values.ravel())
        me.update()

    try:
        me.color_attributes.active_color_name = GROUP_COLOR_ATTRIBUTE
    except (AttributeError, TypeError):
        pass
    for item, count in zip(items, counts):
        if item.count != count:
            item.count = count


def assign_to_color_group(obj, item_index, mode):
    """Edit Mode: put the selected vertices into a colored group, or take them out.

    mode 'ADD' also takes them out of every other colored group, so each vertex
    keeps one color; 'REMOVE' takes them out of this group. Other vertex groups
    (the spheres' ones, for instance) are never touched. Returns how many
    vertices changed.
    """
    indices = color_group_indices(obj)
    target = indices[item_index]
    others = [gi for k, gi in enumerate(indices) if k != item_index and gi is not None]
    me = obj.data
    bm = bmesh.from_edit_mesh(me)
    deform = bm.verts.layers.deform.verify()
    changed = 0
    for v in bm.verts:
        if not v.select:
            continue
        dv = v[deform]
        if mode == "ADD":
            for gi in others:
                if gi in dv:
                    del dv[gi]
            if target not in dv or dv[target] != 1.0:
                dv[target] = 1.0
                changed += 1
        elif target in dv:
            del dv[target]
            changed += 1
    bmesh.update_edit_mesh(me, loop_triangles=False, destructive=False)
    return changed


def show_group_colors(context, show=True):
    """Solid shading with Color: Attribute in every 3D view, or back to Material."""
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type != "VIEW_3D":
                continue
            shading = area.spaces.active.shading
            if show:
                shading.type = "SOLID"
                shading.color_type = "VERTEX"
            else:
                shading.color_type = "MATERIAL"
            area.tag_redraw()


def _recolor(self, context):
    update_group_colors(self.id_data)


def _rename_group(self, context):
    obj = self.id_data
    vg = obj.vertex_groups.get(self.group)
    if vg is None or self.label == self.group:
        return
    vg.name = self.label
    self.group = vg.name
    if self.label != vg.name:  # the name was taken: Blender made it unique
        self.label = vg.name


class SPF_ColorGroup(bpy.types.PropertyGroup):
    label: StringProperty(name="Nome", update=_rename_group)
    group: StringProperty(options={"HIDDEN"})
    color: FloatVectorProperty(
        name="Colore", subtype="COLOR", size=3, min=0.0, max=1.0, default=(0.8, 0.8, 0.8), update=_recolor
    )
    count: IntProperty(name="Vertici", options={"HIDDEN"})


def _color_mesh(context):
    obj = context.active_object
    if obj is None or obj.type != "MESH" or obj.sphere_fit.is_sphere:
        return None
    return obj


def _active_color_group(context):
    obj = _color_mesh(context)
    if obj is None or not (0 <= obj.spf_color_groups_index < len(obj.spf_color_groups)):
        return None, None
    return obj, obj.spf_color_groups[obj.spf_color_groups_index]


class SPF_OT_color_group_new(bpy.types.Operator):
    """Nuovo gruppo colorato; in Edit Mode contiene i vertici selezionati"""

    bl_idname = "sphere_fit.color_group_new"
    bl_label = "Nuovo gruppo colorato"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return _color_mesh(context) is not None

    def execute(self, context):
        obj = _color_mesh(context)
        items = obj.spf_color_groups
        vg = obj.vertex_groups.new(name=f"Gruppo {len(items) + 1}")
        item = items.add()
        item.group = vg.name
        item.label = vg.name
        item.color = distinct_color(len(items) - 1)
        obj.spf_color_groups_index = len(items) - 1
        if obj.mode == "EDIT":
            assign_to_color_group(obj, len(items) - 1, "ADD")
        update_group_colors(obj)
        if len(items) == 1:
            show_group_colors(context)
        self.report({"INFO"}, f"{vg.name}: {item.count} vertici")
        return {"FINISHED"}


class SPF_OT_color_group_remove(bpy.types.Operator):
    """Elimina il gruppo colorato attivo e il suo vertex group"""

    bl_idname = "sphere_fit.color_group_remove"
    bl_label = "Elimina gruppo colorato"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return _active_color_group(context)[1] is not None

    def execute(self, context):
        obj, item = _active_color_group(context)
        vg = obj.vertex_groups.get(item.group)
        if vg is not None:
            obj.vertex_groups.remove(vg)
        obj.spf_color_groups.remove(obj.spf_color_groups_index)
        obj.spf_color_groups_index = min(obj.spf_color_groups_index, len(obj.spf_color_groups) - 1)
        update_group_colors(obj)
        return {"FINISHED"}


class SPF_OT_color_group_assign(bpy.types.Operator):
    """Aggiunge o toglie i vertici selezionati dal gruppo colorato attivo"""

    bl_idname = "sphere_fit.color_group_assign"
    bl_label = "Assegna al gruppo colorato"
    bl_options = {"REGISTER", "UNDO"}

    mode: EnumProperty(
        items=[
            ("ADD", "Assegna", "Mette i vertici selezionati nel gruppo e li toglie dagli altri gruppi colorati"),
            ("REMOVE", "Togli", "Toglie i vertici selezionati dal gruppo"),
        ],
        default="ADD",
    )

    @classmethod
    def poll(cls, context):
        return context.mode == "EDIT_MESH" and _active_color_group(context)[1] is not None

    def execute(self, context):
        obj, item = _active_color_group(context)
        if obj.vertex_groups.get(item.group) is None:
            vg = obj.vertex_groups.new(name=item.label or "Gruppo")
            item.group = vg.name
            item.label = vg.name
        changed = assign_to_color_group(obj, obj.spf_color_groups_index, self.mode)
        update_group_colors(obj)
        self.report({"INFO"}, f"{item.label}: {changed} vertici cambiati, {item.count} nel gruppo")
        return {"FINISHED"}


class SPF_OT_color_group_select(bpy.types.Operator):
    """Seleziona o deseleziona i vertici del gruppo colorato attivo"""

    bl_idname = "sphere_fit.color_group_select"
    bl_label = "Seleziona gruppo colorato"
    bl_options = {"REGISTER", "UNDO"}

    select: BoolProperty(name="Seleziona", default=True)

    @classmethod
    def poll(cls, context):
        return context.mode == "EDIT_MESH" and _active_color_group(context)[1] is not None

    def execute(self, context):
        obj, item = _active_color_group(context)
        vg = obj.vertex_groups.get(item.group)
        if vg is None:
            self.report({"ERROR"}, f"Vertex group '{item.group}' non trovato")
            return {"CANCELLED"}
        me = obj.data
        bm = bmesh.from_edit_mesh(me)
        deform = bm.verts.layers.deform.active
        if deform is not None:
            for v in bm.verts:
                if vg.index in v[deform]:
                    v.select = self.select
        bm.select_flush(self.select)
        bmesh.update_edit_mesh(me, loop_triangles=False, destructive=False)
        return {"FINISHED"}


class SPF_OT_color_group_show(bpy.types.Operator):
    """Mostra o nasconde i colori dei gruppi nel viewport (Solid, Color: Attribute)"""

    bl_idname = "sphere_fit.color_group_show"
    bl_label = "Colori nel viewport"

    show: BoolProperty(default=True)

    def execute(self, context):
        obj = _color_mesh(context)
        if obj is not None and self.show:
            update_group_colors(obj)
        show_group_colors(context, self.show)
        return {"FINISHED"}


class SPF_UL_color_groups(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        obj = data
        row = layout.row(align=True)
        sub = row.row()
        sub.scale_x = 0.35
        sub.prop(item, "color", text="")
        row.prop(item, "label", text="", emboss=False)
        if obj.vertex_groups.get(item.group) is None:
            row.label(text="", icon="ERROR")
        else:
            row.label(text=str(item.count))


class SPF_PT_color_groups(bpy.types.Panel):
    bl_label = "Gruppi colorati"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Sphere Fit"

    @classmethod
    def poll(cls, context):
        return _color_mesh(context) is not None

    def draw(self, context):
        layout = self.layout
        obj = _color_mesh(context)

        row = layout.row()
        row.template_list("SPF_UL_color_groups", "", obj, "spf_color_groups", obj, "spf_color_groups_index", rows=4)
        col = row.column(align=True)
        col.operator(SPF_OT_color_group_new.bl_idname, text="", icon="ADD")
        col.operator(SPF_OT_color_group_remove.bl_idname, text="", icon="REMOVE")

        if context.mode == "EDIT_MESH":
            row = layout.row(align=True)
            row.operator(SPF_OT_color_group_assign.bl_idname, text="Assegna", icon="ADD").mode = "ADD"
            row.operator(SPF_OT_color_group_assign.bl_idname, text="Togli", icon="REMOVE").mode = "REMOVE"
            row = layout.row(align=True)
            row.operator(SPF_OT_color_group_select.bl_idname, text="Seleziona", icon="RESTRICT_SELECT_OFF").select = True
            row.operator(SPF_OT_color_group_select.bl_idname, text="Deseleziona", icon="RESTRICT_SELECT_ON").select = False
        else:
            layout.label(text="Edit Mode per assegnare e selezionare", icon="INFO")

        row = layout.row(align=True)
        row.operator(SPF_OT_color_group_show.bl_idname, text="Mostra colori", icon="HIDE_OFF").show = True
        row.operator(SPF_OT_color_group_show.bl_idname, text="Nascondi", icon="HIDE_ON").show = False


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

classes = (
    SPF_SphereSettings,
    SPF_OT_create_sphere,
    SPF_OT_fit,
    SPF_OT_reset_initial,
    SPF_OT_assign_vertices,
    SPF_OT_select_vertices,
    SPF_OT_edit_vertices,
    SPF_OT_finish_edit,
    SPF_OT_delete_sphere,
    SPF_PT_panel,
    SPF_ColorGroup,
    SPF_OT_color_group_new,
    SPF_OT_color_group_remove,
    SPF_OT_color_group_assign,
    SPF_OT_color_group_select,
    SPF_OT_color_group_show,
    SPF_UL_color_groups,
    SPF_PT_color_groups,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Object.sphere_fit = PointerProperty(type=SPF_SphereSettings)
    bpy.types.Scene.sphere_fit_edit = PointerProperty(
        name="Sfera in modifica",
        type=bpy.types.Object,
        poll=_poll_edit_sphere,
        description="Sfera i cui vertici si modificano in Edit Mode",
    )
    bpy.types.Object.spf_color_groups = CollectionProperty(type=SPF_ColorGroup)
    bpy.types.Object.spf_color_groups_index = IntProperty(name="Gruppo colorato attivo", default=0)


def unregister():
    del bpy.types.Object.spf_color_groups_index
    del bpy.types.Object.spf_color_groups
    del bpy.types.Scene.sphere_fit_edit
    del bpy.types.Object.sphere_fit
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
