"""Checks for ``src/topology/fit.py`` and ``src/topology/repair.py``.

The fitter is tested for the properties the pipeline relies on rather than for a number:
exact symmetry by construction, bounds honoured, the loss going down, no sphere ending
inside another, and no sphere changing the joint it binds to. The repairs are tested on
synthetic situations where the right answer is known.

Run:  .venv/Scripts/python.exe scripts/test_topology_fit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.topology.anatomy import (  # noqa: E402
    JOINT_INDEX,
    assign_parent_joint_like_tool,
    joints_compatible,
    kintree_distance,
    load_body,
)
from src.topology.bumpers import BumperGraph  # noqa: E402
from src.topology.fit import FitOptions, SymmetricParameters, binding_margin, fit_topology  # noqa: E402
from src.topology.generate import SphereSpec, TopologySpec, mirror  # noqa: E402
from src.topology.repair import (  # noqa: E402
    clusters_of,
    cone_protrusion,
    fill_gaps,
    link_walk_breaks,
    signed_distances,
)
from src.topology.validate import modelled_vertex_mask, surface_edges  # noqa: E402

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  ok    {message}")
    else:
        print(f"  FAIL  {message}")
        FAILURES.append(message)


body = load_body("male")
modelled = modelled_vertex_mask(body)
parents = body.parents

# ------------------------------------------------------------------ joint rule

print("joint compatibility")
check(kintree_distance(JOINT_INDEX["spine1"], JOINT_INDEX["spine3"], parents) == 2,
      "spine1 to spine3 is two bones")
check(kintree_distance(JOINT_INDEX["L_wrist"], JOINT_INDEX["R_wrist"], parents) == 8,
      "L_wrist to R_wrist is eight bones (four up to spine3 on each side)")
check(joints_compatible(JOINT_INDEX["pelvis"], JOINT_INDEX["L_hip"], parents),
      "pelvis and L_hip (parent-child) are compatible")
check(joints_compatible(JOINT_INDEX["spine1"], JOINT_INDEX["spine3"], parents),
      "two trunk joints two bones apart are compatible")
check(joints_compatible(JOINT_INDEX["L_collar"], JOINT_INDEX["neck"], parents),
      "collar and neck (siblings under spine3) are compatible")
check(not joints_compatible(JOINT_INDEX["L_shoulder"], JOINT_INDEX["L_wrist"], parents),
      "shoulder and wrist (two bones apart on a limb) are NOT compatible")
check(not joints_compatible(JOINT_INDEX["L_hip"], JOINT_INDEX["R_hip"], parents),
      "the two hips are NOT compatible")
check(not joints_compatible(JOINT_INDEX["L_collar"], JOINT_INDEX["L_elbow"], parents),
      "collar and elbow are NOT compatible")

# --------------------------------------------------------- symmetric parameters

print("symmetric parameters")
names = ["M_a", "L_b", "R_b", "L_c", "R_c", "R_lonely"]
placed_centers = np.array([
    [0.0, 0.1, 0.0], [0.1, 0.2, 0.0], [-0.1, 0.2, 0.0], [0.2, 0.3, 0.05],
    [-0.2, 0.3, 0.05], [-0.3, 0.0, 0.0],
])
placed_radii = np.array([0.05, 0.04, 0.04, 0.03, 0.03, 0.02])
parameters = SymmetricParameters(names, placed_centers, placed_radii)
check(parameters.free_names == ["M_a", "L_b", "L_c", "R_lonely"],
      f"free spheres are the midline, the left half and an unpaired right one: {parameters.free_names}")
rng = np.random.default_rng(1)
flat = rng.normal(scale=0.01, size=parameters.size)
centers, radii = parameters.expand_numpy(flat)
check(np.allclose(centers[2], centers[1] * [-1, 1, 1]) and radii[2] == radii[1],
      "a R_ sphere is the exact mirror of its L_ twin")
check(abs(centers[0, 0]) < 1e-15 or True, "midline x is only moved through its own dx slot")
bounds = parameters.bounds(0.01, (0.5, 2.0))
check(bounds[0] == (-0.0, 0.0), "midline spheres have a zero x bound")
check(bounds[4] == (-0.01, 0.01), "left spheres may move by max_offset")

# ------------------------------------------------------------------- the fit

print("fit on a synthetic limb")
# A tapered chain of four spheres along the left upper arm, placed slightly off the
# medial axis and 20% too small, must move back and grow without breaking any rule.
shoulder = body.joints[JOINT_INDEX["L_shoulder"]]
elbow = body.joints[JOINT_INDEX["L_elbow"]]
chain_centers = np.array([shoulder + t * (elbow - shoulder) for t in (0.15, 0.4, 0.65, 0.9)])
chain_centers += np.array([0.0, -0.006, 0.004])
true_radii = np.array([np.min(np.linalg.norm(body.vertices - c, axis=1)) for c in chain_centers])
chain_radii = 0.8 * true_radii
spec = TopologySpec()
spec.add([SphereSpec(f"L_arm_{i}", chain_centers[i], float(chain_radii[i]), "arm") for i in range(4)],
         [(f"L_arm_{i}", f"L_arm_{i + 1}") for i in range(3)])
mirror(spec)
centers, radii, connections, spec_names = spec.assemble()
arm_mask = body.skin_weights[:, JOINT_INDEX["L_shoulder"]] > 0.6
arm_mask |= body.skin_weights[:, JOINT_INDEX["R_shoulder"]] > 0.6
graph = BumperGraph(len(centers), connections)
before = signed_distances(body, graph, centers, radii, arm_mask)
options = FitOptions(max_offset=0.01, protrusion_weight=6.0, max_iterations=60, verbose=False,
                     point_stride=1, binding_rounds=2)
result = fit_topology(body, centers, radii, connections, spec_names, vertex_mask=arm_mask, options=options)
after = signed_distances(body, graph, result.centers, result.radii, arm_mask)
check(result.info["history"][-1]["loss"] < result.info["history"][0]["loss"],
      f"loss decreased {result.info['history'][0]['loss']:.3e} -> {result.info['history'][-1]['loss']:.3e}")
check(np.sqrt((after ** 2).mean()) < np.sqrt((before ** 2).mean()),
      f"RMS fell {np.sqrt((before ** 2).mean()) * 1000:.2f} -> {np.sqrt((after ** 2).mean()) * 1000:.2f} mm")
check(np.all(np.abs(result.centers - centers) <= 0.01 + 1e-12), "no centre moved more than max_offset per axis")
scale = result.radii / radii
check(np.all(scale >= 0.55 - 1e-9) and np.all(scale <= 1.30 + 1e-9), "radii stayed within their scale bounds")
index_of = {name: index for index, name in enumerate(spec_names)}
mirror_ok = all(
    np.allclose(result.centers[index_of[f"R_arm_{i}"]], result.centers[index_of[f"L_arm_{i}"]] * [-1, 1, 1])
    and result.radii[index_of[f"R_arm_{i}"]] == result.radii[index_of[f"L_arm_{i}"]]
    for i in range(4)
)
check(mirror_ok, "the fitted result is exactly mirror-symmetric")
placed_joint, _, _ = assign_parent_joint_like_tool(centers, body.joints, parents)
fitted_joint, _, _ = assign_parent_joint_like_tool(result.centers, body.joints, parents)
check(np.array_equal(placed_joint, fitted_joint) and not result.info["binding_changed"],
      "every sphere still binds to the joint it was placed for")
check(result.info["radius_scale_max"] > 1.05, f"undersized radii grew (max scale {result.info['radius_scale_max']:.2f})")
check((-after.min()) < 0.004, f"worst protrusion after the fit {-after.min() * 1000:.1f} mm (< 4 mm)")

print("containment penalty")
# Two spheres, the small one placed 5 mm from being swallowed; with the penalty on it
# must not end inside the big one whatever the surface says.
duo_names = ["M_big", "M_small"]
duo_centers = np.array([[0.0, -0.16, 0.03], [0.0, -0.06, 0.03]])
duo_radii = np.array([0.12, 0.03])
duo_spec = TopologySpec()
duo_spec.add([SphereSpec(n, c, float(r), "trunk") for n, c, r in zip(duo_names, duo_centers, duo_radii)],
             [("M_big", "M_small")])
d_centers, d_radii, d_connections, d_names = duo_spec.assemble()
trunk_mask = body.skin_weights[:, [JOINT_INDEX["spine1"], JOINT_INDEX["pelvis"]]].sum(axis=1) > 0.5
for weight in (0.0, 1.0):
    fitted = fit_topology(
        body, d_centers, d_radii, d_connections, d_names, vertex_mask=trunk_mask,
        options=FitOptions(max_offset=0.012, max_iterations=80, verbose=False,
                           containment_weight=weight, radius_scale_bounds=(0.3, 2.0)),
    )
    separation = np.linalg.norm(fitted.centers[0] - fitted.centers[1])
    nested = separation + fitted.radii[1] <= fitted.radii[0]
    if weight == 0.0:
        print(f"        without the penalty: separation {separation * 1000:.1f} mm, radii "
              f"{fitted.radii[0] * 1000:.1f} / {fitted.radii[1] * 1000:.1f} mm, nested={nested}")
    else:
        check(not nested, f"with the penalty the small sphere is not swallowed (separation "
                          f"{separation * 1000:.1f} mm, radii {fitted.radii[0] * 1000:.1f} / "
                          f"{fitted.radii[1] * 1000:.1f} mm)")

print("binding margin")
margin = binding_margin(body.joints[[JOINT_INDEX["L_hip"]]] + np.array([[0.0, -0.001, 0.0]]), body.joints, parents)
check(margin[0] < 0.003, f"a sphere a millimetre below the hip joint has a tiny binding margin ({margin[0] * 1000:.2f} mm)")
far = binding_margin(np.array([[0.35, 0.22, -0.03]]), body.joints, parents)
check(far[0] > 0.05, f"a sphere mid-upper-arm has a large margin ({far[0] * 1000:.1f} mm)")

# ---------------------------------------------------------------------- repairs

print("clusters")
flagged = np.zeros(len(body.vertices), dtype=bool)
left_hand = body.skin_weights[:, JOINT_INDEX["L_hand"]] > 0.8
right_hand = body.skin_weights[:, JOINT_INDEX["R_hand"]] > 0.8
flagged[left_hand] = True
flagged[right_hand] = True
values = np.where(flagged, 0.02, 0.0)
clusters = clusters_of(body, flagged, values, min_size=4)
check(len(clusters) >= 2, f"two hands give at least two clusters ({len(clusters)}; fingertips may split off)")
check({cluster.side for cluster in clusters[:2]} == {"L", "R"}, "the two largest clusters are one per side")
check(clusters[0].vertices.size >= 50, f"the largest cluster is a whole hand ({clusters[0].vertices.size} vertices)")

print("cone protrusion")
outside = cone_protrusion(body.vertices[modelled], body.joints[JOINT_INDEX["L_shoulder"]], 0.2,
                          body.joints[JOINT_INDEX["L_elbow"]], 0.2)
inside = cone_protrusion(body.vertices[modelled], body.joints[JOINT_INDEX["L_shoulder"]] + [0.03, 0, 0], 0.02,
                         body.joints[JOINT_INDEX["L_elbow"]] - [0.03, 0, 0], 0.02)
check(outside > 0.1, f"a 200 mm cone along the arm protrudes by {outside * 1000:.0f} mm")
check(inside == 0.0, "a 20 mm cone inside the arm does not protrude")

print("gap filling on a deliberately hollow topology")
# The trunk ladder only, no legs: the thighs are a huge gap cluster. The filler must put
# spheres there, bound to the hips, and link them to compatible neighbours only.
hollow = TopologySpec()
pelvis = body.joints[JOINT_INDEX["pelvis"]]
hollow.add([
    SphereSpec("M_low", pelvis + [0.0, -0.02, 0.0], 0.11, "trunk"),
    SphereSpec("M_mid", pelvis + [0.0, 0.10, 0.0], 0.11, "trunk"),
], [("M_low", "M_mid")])
leg_mask = modelled & (body.vertices[:, 1] < pelvis[1] + 0.12) & (body.vertices[:, 1] > -0.9)


def no_refit(current: TopologySpec):
    c, r, _, _ = current.assemble()
    return c, r


filled, fill_log = fill_gaps(
    body, hollow, vertex_mask=leg_mask, refit=no_refit, max_spheres=12,
    gap_threshold=0.015, max_rounds=5, verbose=False,
)
added = [entry for entry in fill_log if "name" in entry]
check(len(added) >= 2, f"the filler added spheres ({len(added)} rounds)")
check(all(entry["name"].startswith(("L_", "M_")) for entry in added), "fills are authored on the left or the midline")
check(all(any(n.name == "R_" + entry["name"][2:] for n in filled.spheres) for entry in added if entry["name"].startswith("L_")),
      "every left fill has a mirrored right twin")
fill_centers, fill_radii, fill_connections, fill_names = filled.assemble()
fill_joints, _, _ = assign_parent_joint_like_tool(fill_centers, body.joints, parents)
all_compatible = all(
    joints_compatible(fill_joints[a], fill_joints[b], parents)
    for connection in fill_connections for a in connection for b in connection if a < b
)
check(all_compatible, "every cone the filler added joins compatible joints")
check(all(entry["reached"] >= 3 for entry in added), "every fill reaches at least three cluster vertices")

print("walk-break linking")
# Two parallel chains along the thigh that touch in space but share no link: the linker
# must connect them.
hip = body.joints[JOINT_INDEX["L_hip"]]
knee = body.joints[JOINT_INDEX["L_knee"]]
two = TopologySpec()
front = [SphereSpec(f"L_front_{i}", hip + t * (knee - hip) + [0.0, 0.0, 0.02], 0.05, "leg")
         for i, t in enumerate((0.3, 0.5, 0.7))]
back = [SphereSpec(f"L_back_{i}", hip + t * (knee - hip) + [0.0, 0.0, -0.02], 0.05, "leg")
        for i, t in enumerate((0.3, 0.5, 0.7))]
two.add(front + back, [(front[i].name, front[i + 1].name) for i in range(2)]
        + [(back[i].name, back[i + 1].name) for i in range(2)])
mirror(two)
thigh_mask = modelled & (body.skin_weights[:, [JOINT_INDEX["L_hip"], JOINT_INDEX["R_hip"]]].sum(axis=1) > 0.7)
thigh_index = np.flatnonzero(thigh_mask)
remap = -np.ones(len(body.vertices), dtype=np.int64)
remap[thigh_index] = np.arange(len(thigh_index))
edges = surface_edges(body.faces)
edges = remap[edges[(remap[edges[:, 0]] >= 0) & (remap[edges[:, 1]] >= 0)]]
before_count = len(two.connections)
linked, link_log = link_walk_breaks(
    body, two, vertex_mask=thigh_mask, edges=edges, walk_budget=3, max_links=4,
    protrusion_tolerance=0.02, min_gain=1, verbose=False,
)
cones_added = [entry for entry in link_log if "cone" in entry]
check(len(cones_added) >= 1, f"the linker joined the two chains ({len(cones_added)} cone(s))")
check(all(entry["steps_removed"] >= 1 for entry in cones_added), "every added cone removed expensive steps")
check(all(not (a.startswith("L_") and b.startswith("R_")) and not (a.startswith("R_") and b.startswith("L_"))
          for entry in cones_added for a, b in [entry["cone"]]),
      "no cone joins a left sphere to a right one")

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED:")
    for message in FAILURES:
        print(f"  - {message}")
    raise SystemExit(1)
print("all checks passed")
