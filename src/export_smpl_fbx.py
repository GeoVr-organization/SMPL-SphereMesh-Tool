"""
Export current SMPL mesh to FBX with skeleton and vertex weights.

Uses Blender in background to create mesh, armature, and skin weights, then export FBX.
Requires Blender 3.x or 4.x installed and available on PATH (or set BLENDER_EXECUTABLE).
"""

import os
import shutil
import subprocess
import sys
import tempfile
from typing import Optional, Tuple

import numpy as np


def _find_blender():
    """Return path to Blender executable, or None if not found."""
    env_path = os.environ.get("BLENDER_EXECUTABLE")
    if env_path and os.path.isfile(env_path):
        return env_path
    exe = shutil.which("blender")
    if exe:
        return exe
    # Common install locations on Windows
    if sys.platform == "win32":
        for base in [
            os.path.expandvars(r"%ProgramFiles%\Blender Foundation"),
            os.path.expandvars(r"%ProgramFiles(x86)%\Blender Foundation"),
        ]:
            if os.path.isdir(base):
                for name in os.listdir(base):
                    path = os.path.join(base, name, "blender.exe")
                    if os.path.isfile(path):
                        return path
    return None


def _bone_tails(joints, parents):
    """Compute tail position for each bone from joint positions and hierarchy."""
    n = len(joints)
    tails = np.zeros_like(joints)
    # Build list of children per joint
    children = [[] for _ in range(n)]
    for i in range(n):
        p = parents[i]
        if 0 <= p < n:
            children[p].append(i)
    for i in range(n):
        if children[i]:
            # Use first child joint as tail
            tails[i] = joints[children[i][0]]
        else:
            # Leaf bone: small offset so bone has non-zero length
            tails[i] = joints[i] + np.array([0.0, 0.0, 0.05], dtype=joints.dtype)
    return tails


def _blender_script_content(data_npz_path: str, fbx_path: str) -> str:
    """Return the Blender Python script that builds mesh + armature + skin and exports FBX."""
    return r'''
import sys
import os

# Blender may strip argv; paths are in the npz as metadata or we pass via env
data_npz_path = r''' + repr(data_npz_path) + r'''
fbx_path = r''' + repr(fbx_path) + r'''

import bpy
import numpy as np

# Clear default scene
bpy.ops.wm.read_factory_settings(use_empty=True)

data = np.load(data_npz_path)
verts = np.asarray(data["verts"], dtype=np.float64)
faces = np.asarray(data["faces"], dtype=np.int32)
joints = np.asarray(data["joints"], dtype=np.float64)
parents = np.asarray(data["parents"], dtype=np.int32)
joint_names = [str(x) for x in data["joint_names"]]
weights = np.asarray(data["weights"], dtype=np.float64)

n_verts = verts.shape[0]
n_joints = len(joint_names)

# Build bone tails from hierarchy
children = [[] for _ in range(n_joints)]
for i in range(n_joints):
    p = int(parents[i])
    if 0 <= p < n_joints:
        children[p].append(i)
tails = np.zeros_like(joints)
for i in range(n_joints):
    if children[i]:
        tails[i] = joints[children[i][0]]
    else:
        tails[i] = joints[i] + np.array([0.0, 0.0, 0.05])

# --- Mesh ---
mesh = bpy.data.meshes.new("SMPL_Mesh")
mesh.from_pydata(verts.tolist(), [], faces.tolist())
mesh.update()
obj = bpy.data.objects.new("SMPL_Mesh", mesh)
bpy.context.collection.objects.link(obj)
bpy.context.view_layer.objects.active = obj

# --- Armature ---
arm_data = bpy.data.armatures.new("SMPL_Armature")
arm_obj = bpy.data.objects.new("SMPL_Armature", arm_data)
bpy.context.collection.objects.link(arm_obj)
bpy.context.view_layer.objects.active = arm_obj
bpy.ops.object.mode_set(mode="EDIT")

edit_bones = arm_data.edit_bones
for i in range(n_joints):
    bone = edit_bones.new(joint_names[i])
    bone.head = joints[i].tolist()
    bone.tail = tails[i].tolist()
    if 0 <= parents[i] < n_joints:
        bone.parent = edit_bones[joint_names[int(parents[i])]]

bpy.ops.object.mode_set(mode="OBJECT")

# --- Vertex groups (skin weights) ---
for j in range(n_joints):
    obj.vertex_groups.new(name=joint_names[j])

for v_idx in range(n_verts):
    for j in range(n_joints):
        w = float(weights[v_idx, j])
        if w > 1e-7:
            obj.vertex_groups[j].add([v_idx], w, "REPLACE")

# --- Armature modifier ---
mod = obj.modifiers.new(name="Armature", type="ARMATURE")
mod.object = arm_obj

# --- Export FBX ---
# Select only our objects for export
bpy.ops.object.select_all(action="DESELECT")
obj.select_set(True)
arm_obj.select_set(True)
bpy.context.view_layer.objects.active = obj

fbx_dir = os.path.dirname(fbx_path)
if fbx_dir and not os.path.isdir(fbx_dir):
    os.makedirs(fbx_dir, exist_ok=True)

bpy.ops.export_scene.fbx(
    filepath=fbx_path,
    use_selection=True,
    mesh_smooth_type="FACE",
    use_mesh_edges=False,
    use_armature_deform_only=True,
    add_leaf_bones=False,
)
'''


def export_smpl_to_fbx(
    filepath: str,
    verts: np.ndarray,
    faces: np.ndarray,
    joints: np.ndarray,
    parents: np.ndarray,
    joint_names: list,
    weights: np.ndarray,
    blender_executable: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Export SMPL mesh, skeleton, and skin weights to an FBX file using Blender.

    Args:
        filepath: Output FBX path.
        verts: Vertex positions (N, 3), float.
        faces: Triangle indices (F, 3), int.
        joints: Joint positions (24, 3), float.
        parents: Parent joint index per joint (24,), int. Root usually has parent -1 or 0.
        joint_names: List of 24 joint name strings in index order.
        weights: Vertex weights (N, 24), float.
        blender_executable: Path to Blender; if None, auto-detect.

    Returns:
        (success, message). On failure, message describes the error.
    """
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    joints = np.asarray(joints, dtype=np.float64)
    parents = np.asarray(parents, dtype=np.int32)
    weights = np.asarray(weights, dtype=np.float64)

    if verts.ndim != 2 or verts.shape[1] != 3:
        return False, "verts must be (N, 3)"
    if faces.ndim != 2 or faces.shape[1] != 3:
        return False, "faces must be (F, 3)"
    if joints.shape != (24, 3):
        return False, "joints must be (24, 3)"
    if parents.shape != (24,):
        return False, "parents must be (24,)"
    if len(joint_names) != 24:
        return False, "joint_names must have 24 elements"
    if weights.shape != (verts.shape[0], 24):
        return False, "weights must be (N, 24)"

    blender_exe = blender_executable or _find_blender()
    if not blender_exe:
        return False, (
            "Blender not found. Install Blender 3.x or 4.x and add it to PATH, "
            "or set BLENDER_EXECUTABLE to the blender executable path."
        )

    filepath = os.path.abspath(os.path.normpath(filepath))
    if not filepath.lower().endswith(".fbx"):
        filepath += ".fbx"

    tmpdir = tempfile.mkdtemp(prefix="smpl_fbx_")
    try:
        data_path = os.path.join(tmpdir, "smpl_data.npz")
        script_path = os.path.join(tmpdir, "blender_export_fbx.py")

        joint_names_arr = np.array(joint_names, dtype="U64")
        np.savez_compressed(
            data_path,
            verts=verts,
            faces=faces,
            joints=joints,
            parents=parents,
            joint_names=joint_names_arr,
            weights=weights,
        )

        script_content = _blender_script_content(data_path, filepath)
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)

        result = subprocess.run(
            [blender_exe, "--background", "--python", script_path],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=tmpdir,
        )

        if result.returncode != 0:
            err = (result.stderr or "").strip() or (result.stdout or "").strip()
            return False, f"Blender export failed: {err or 'unknown error'}"

        if not os.path.isfile(filepath):
            return False, "Blender ran but FBX file was not created."

        return True, f"Exported to {filepath}"
    except subprocess.TimeoutExpired:
        return False, "Blender export timed out (120s)."
    except Exception as e:
        return False, str(e)
    finally:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass
