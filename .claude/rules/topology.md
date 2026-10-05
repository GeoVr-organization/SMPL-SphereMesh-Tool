---
paths:
  - "src/topology/**"
  - "scripts/build_topology.py"
  - "scripts/search_topology.py"
  - "scripts/compare_topologies.py"
  - "scripts/test_topology_*.py"
  - "data/generated/**"
---

# Automatic topology pipeline

Read `TOPOLOGY_NOTES.md` ("how the pipeline works now", "Reproducing") and the matching
section of `TOPOLOGY_HANDOFF.md` before changing anything here.

## The user's acceptance criteria (every candidate is judged by these)
- Fewer than **100 spheres**, and at most **~20 triangles**.
- Quads exist only engine-side (`.txt`). The JSON carries the triangle split plus
  `quad_groups`.
- Bumpers overlap no more than needed.
- The bumper graph is a **connected cover** that the solver's watchdog can walk: links are
  topological, never proximity.
- No bumper spans two independently rotating joints. The joint rule is
  `anatomy.joints_compatible`: same joint, kintree parent/child, or trunk joints at most
  two bones apart.
- Hands, feet and head are truncated.
- Validate posed behaviour on `data/random_pose.csv` and `00163_take2_fps30.pkl`.
- The user likes the arms of `data/template.json` and dislikes its torso.
- Report **measured numbers**: `scripts/compare_topologies.py`, the validator reports in
  `data/generated/reports/`. Never bare assertions. Keep `Thresholds` in `validate.py`
  aligned with the criteria above.

## Pipeline
**place** (`generate.py`, `sections.py`, `anatomy.py`) → **fit** (`fit.py`,
`sdf_torch.py`; exact SDF, anchored to placement, multi-start) →
`repair.link_walk_breaks` → `repair.fill_gaps` → **fit** → **validate** → **render**.

## Traps already paid for
- A `--design` JSON that snapshots every `Design` field silently freezes defaults changed
  later. Keep design files minimal (only the varied keys).
- Fit offset caps must be anchored to the original placement, or refits compound.
- A posed pop-out of about 46 mm is the GUI's rigid-binding floor, not a placement bug.
- The GUI's `spheremesh_distance` / `get_distance` is wrong; use `bumpers.py`.
  `sdf_torch.py` is its autograd port.
- A sphere centred exactly on a joint binds non-deterministically.
- `spine1` and `neck` are nearly dead joints.
- Use mesh-plane sections (`sections.py`), not vertex bands.

## Validation
- Run all six `scripts/test_topology_*.py` before trusting a change; they all pass today.
- Show intermediate outputs early in `data/generated/` and note decisions and numbers in
  `TOPOLOGY_NOTES.md`.
- Nothing generated is proven until it has been opened in the GUI.
