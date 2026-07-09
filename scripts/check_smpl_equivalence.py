from __future__ import annotations

from pathlib import Path
import argparse
import sys
import numpy as np
import torch as tc


from smpl_pytorch.smpl import load_numpy_params, SMPL as NewSMPL
from smpl import load_pickle_smpl

SPHEREMESH_ROOT = Path(__file__).resolve().parent.parent
SMPL_MODELS_DIR = SPHEREMESH_ROOT / "SMPL_models"
if str(SPHEREMESH_ROOT) not in sys.path:
    sys.path.insert(0, str(SPHEREMESH_ROOT))

from src.smpl_forward_compat import my_function_forward



def _to_tensor(x) -> tc.Tensor:
    if isinstance(x, tc.Tensor):
        return x
    return tc.as_tensor(x)


def _normalize_rs(rs: tc.Tensor) -> tc.Tensor:
    rs = _to_tensor(rs)
    # New path may return 4x4 transforms; old path returns 3x3 rotations.
    if rs.ndim >= 2 and rs.shape[-2:] == (4, 4):
        rs = rs[..., :3, :3]
    return rs


def _forward_new(model, beta: tc.Tensor, theta: tc.Tensor):
    verts, joints, rs, _ = my_function_forward(model, beta, theta)
    rs = _normalize_rs(rs)
    return verts, joints, rs


def _forward_old(model, beta: tc.Tensor, theta: tc.Tensor):
    verts, joints, rs = model.forward(beta, theta, theta_in_rodrigues=True)
    verts = _to_tensor(verts)
    joints = _to_tensor(joints)
    
    return verts, joints, rs


def _ensure_batch(t: tc.Tensor) -> tc.Tensor:
    if t.ndim == 1:
        return t.unsqueeze(0)
    return t


def _stats(name: str, a: tc.Tensor, b: tc.Tensor, atol: float, rtol: float):
    a = _ensure_batch(_to_tensor(a).detach().cpu().to(tc.float64))
    b = _ensure_batch(_to_tensor(b).detach().cpu().to(tc.float64))
    if a.shape != b.shape:
        print(f"[FAIL] {name}: shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
        return False
    diff = (a - b).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    allclose = tc.allclose(a, b, atol=atol, rtol=rtol)
    status = "OK" if allclose else "FAIL"
    print(
        f"[{status}] {name}: shape={tuple(a.shape)} max_abs={max_abs:.6e} mean_abs={mean_abs:.6e}"
    )
    return bool(allclose)


def _run_case(case_name: str, new_model, old_model, beta: tc.Tensor, theta: tc.Tensor, atol: float, rtol: float):
    print(f"\n=== {case_name} ===")
    new_verts, new_joints, new_rs = _forward_new(new_model, beta, theta)
    old_verts, old_joints, old_rs = _forward_old(old_model, beta, theta)

    print(new_joints.shape, old_joints.shape)
    print(new_joints[10])
    print(old_joints[10])

    ok_v = _stats("verts", new_verts, old_verts, atol, rtol)
    ok_j = _stats("joints", new_joints, old_joints, atol, rtol)
    ok_r = _stats("Rs", new_rs, old_rs, atol, rtol)
    return ok_v and ok_j and ok_r


def main():
    parser = argparse.ArgumentParser(
        description="Compare SMPL outputs between smpl_pytorch(npz) and legacy smpl(pkl)."
    )
    parser.add_argument(
        "--npz",
        type=Path,
        default=SMPL_MODELS_DIR / "basicmodel_m_lbs_10_207_0_v1.0.0.npz",
        help="Path to NPZ model for smpl_pytorch loader.",
    )
    parser.add_argument(
        "--pkl",
        type=Path,
        default=SMPL_MODELS_DIR / "basicModel_m_lbs_10_207_0_v1.0.0.pkl",
        help="Path to PKL model for legacy loader.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance.")
    parser.add_argument("--rtol", type=float, default=1e-5, help="Relative tolerance.")
    args = parser.parse_args()

    if not args.npz.exists():
        raise FileNotFoundError(f"NPZ model not found: {args.npz}")
    if not args.pkl.exists():
        raise FileNotFoundError(f"PKL model not found: {args.pkl}")

    tc.set_default_dtype(tc.float64)
    tc.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("Loading models...")
    print(f"- new (npz): {args.npz}")
    print(f"- old (pkl): {args.pkl}")

    # New loader style used in spheremeshtool_main.py
    new_model = NewSMPL(load_numpy_params(args.npz))
    # Old loader style from smpl.py
    old_model = load_pickle_smpl(args.pkl)

    cases = []
    cases.append(("zero pose/shape", tc.zeros(10, dtype=tc.float64), tc.zeros(72, dtype=tc.float64)))
    cases.append(("random small", tc.randn(10, dtype=tc.float64) * 0.03, tc.randn(72, dtype=tc.float64) * 0.2))
    cases.append(("random medium", tc.randn(10, dtype=tc.float64) * 0.1, tc.randn(72, dtype=tc.float64) * 0.6))

    overall_ok = True
    for name, beta, theta in cases:
        case_ok = _run_case(name, new_model, old_model, beta, theta, args.atol, args.rtol)
        overall_ok = overall_ok and case_ok

    print("\n=== RESULT ===")
    if overall_ok:
        print("Equivalent within tolerance for tested cases.")
        return 0
    print("Differences detected above tolerance in at least one tested case.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

