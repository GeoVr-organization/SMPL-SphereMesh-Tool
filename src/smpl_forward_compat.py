"""Compat layer for smpl_pytorch.SMPL.forward() returning _SmplSkinnedModel."""

import torch as tc


def _squeeze_leading_batch(t: tc.Tensor) -> tc.Tensor:
    """Drop a leading batch dimension of size 1 (typical B=1 SMPL)."""
    if t.ndim >= 1 and t.shape[0] == 1:
        return t[0]
    return t


def my_function_forward(
    model: tc.nn.Module, beta: tc.Tensor, theta: tc.Tensor, **_kwargs
) -> tuple:
    """
    Run ``model.forward(beta, theta)`` and return the legacy 4-tuple
    ``(verts, joints, Rs, _)`` expected by SphereMeshTool.

    Extra keyword arguments (e.g. ``theta_in_rodrigues``) are ignored; pose is axis-angle per joint
    as in current smpl_pytorch.

    Returns:
        verts: posed mesh vertices.
        joints: world joint positions, ``matmul(J_regressor, verts)`` (matches older SMPL scripts).
        Rs: **global** joint rotation matrices ``(J, 3, 3)`` from each joint's world transform
            (not per-bone locals), so they match ``joints`` for attachment / FK.
        Fourth element: full ``joints_world_transforms`` ``(J, 4, 4)`` (batch squeezed).
    """
    out = model.forward(beta, theta)
    verts = _squeeze_leading_batch(out.vertices)
    joints = tc.matmul(model.skinning_joint_regressor, verts)
    T = getattr(out, "joints_world_transforms", None)
    if T is not None:
        T = _squeeze_leading_batch(T)
        Rs = T[..., :3, :3]
    else:
        # Legacy / fallback: locals + kin-tree composition happens in ``Skeleton``.
        Rs = _squeeze_leading_batch(out.joints_local_transforms[..., :3, :3])
        T = None
    return verts, joints, Rs, T
