"""Checks for ``src/topology/validate.py``.

The structural checks are exercised on tiny hand-built inputs whose right answer is
obvious by construction -- a chain whose ends bind to opposite wrists, a triangle strip
with two separated patches of bad vertices, a straight chain against a pair of
interpenetrating unlinked chains -- and the whole suite is run on ``data/template.json``
to pin the pass/fail pattern of the pre-existing checks, which must not drift when the
report gains columns.

Run:  .venv/Scripts/python.exe scripts/test_topology_validate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.topology.anatomy import JOINT_INDEX, JOINT_NAMES, load_body  # noqa: E402
from src.topology.bumpers import CONE, BumperGraph  # noqa: E402
from src.topology.validate import (  # noqa: E402
    Thresholds,
    bumper_label,
    bumper_surface_samples,
    coverage_clusters,
    format_report,
    harmful_overlap,
    joint_crossing_cones,
    validate,
)

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  ok    {message}")
    else:
        print(f"  FAIL  {message}")
        FAILURES.append(message)


def check_named(report, name: str):
    matches = [entry for entry in report.checks if entry.name == name]
    check(len(matches) == 1, f"report has exactly one check named {name}")
    return matches[0] if matches else None


def load_design(path: Path):
    """Centres, radii, engine-side connections and names from a tool JSON.

    The tool stores each quad as its two split triangles ``(a, b, c), (a, c, d)`` next
    to a ``quad_groups`` entry; the engine sees the quad, so it is rebuilt here.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    centers = np.array([sphere["center"] for sphere in data["spheres"]], dtype=np.float64)
    radii = np.array([sphere["radius"] for sphere in data["spheres"]], dtype=np.float64)
    connections = [tuple(connection["indices"]) for connection in data["connections"]]
    quads = [tuple(quad) for quad in data.get("quad_groups", [])]
    if quads:
        split_triangles = set()
        for first, second, third, fourth in quads:
            split_triangles.add((first, second, third))
            split_triangles.add((first, third, fourth))
        connections = [c for c in connections if c not in split_triangles] + quads
    names = (data.get("generated_by") or {}).get("sphere_names")
    return centers, radii, connections, names


# --------------------------------------------------------------------------- labels


def test_bumper_label() -> None:
    print("bumper_label")
    graph = BumperGraph(3, [(0, 1), (1, 2), (0, 1, 2)])
    names = ["L_glute_0", "L_thigh_0", "L_thigh_1"]
    cone_index = graph.index_of(CONE, (0, 1))
    check(bumper_label(graph, cone_index) == "cone[0, 1]", "no names: indices")
    check(
        bumper_label(graph, cone_index, names) == "cone[L_glute_0, L_thigh_0]",
        "with names: cone[L_glute_0, L_thigh_0]",
    )
    check(
        bumper_label(graph, graph.index_of("prysmoid", (0, 1, 2)), names)
        == "prysmoid[L_glute_0, L_thigh_0, L_thigh_1]",
        "slab label lists all three names in connection order",
    )
    check(str(graph.bumpers[cone_index]) == "cone[0, 1]", "Bumper.__str__ unchanged")


# ---------------------------------------------------------------- binding checks


def test_one_bone_per_cone(body) -> None:
    print("one-bone-per-cone")
    left_wrist = body.joints[JOINT_INDEX["L_wrist"]]
    right_wrist = body.joints[JOINT_INDEX["R_wrist"]]
    left_elbow = body.joints[JOINT_INDEX["L_elbow"]]
    right_elbow = body.joints[JOINT_INDEX["R_elbow"]]

    # A chain running from one forearm straight across to the other. The end spheres sit
    # just short of each wrist, so the tool binds them to L_elbow and R_elbow, which are
    # not parent and child of each other.
    crossing_centers = np.array([
        left_wrist + 0.02 * (left_elbow - left_wrist) / np.linalg.norm(left_elbow - left_wrist),
        left_wrist + 0.35 * (right_wrist - left_wrist),
        left_wrist + 0.65 * (right_wrist - left_wrist),
        right_wrist + 0.02 * (right_elbow - right_wrist) / np.linalg.norm(right_elbow - right_wrist),
    ])
    crossing_radii = np.full(4, 0.03)
    chain = [(0, 1), (1, 2), (2, 3)]
    from src.topology.anatomy import assign_parent_joint_like_tool

    parent_joints, _, _ = assign_parent_joint_like_tool(
        crossing_centers, body.joints, body.parents
    )
    check(
        JOINT_NAMES[parent_joints[0]] == "L_elbow" and JOINT_NAMES[parent_joints[3]] == "R_elbow",
        f"end spheres bind to L_elbow / R_elbow (got {JOINT_NAMES[parent_joints[0]]} / "
        f"{JOINT_NAMES[parent_joints[3]]})",
    )
    graph = BumperGraph(4, chain)
    crossings = joint_crossing_cones(graph, parent_joints, body.parents)
    unrelated = [entry for entry in crossings if not entry["seam"]]
    check(len(unrelated) >= 1, f"{len(unrelated)} cone(s) span unrelated joints")

    report = validate(
        crossing_centers, crossing_radii, chain, body=body, thresholds=Thresholds(),
        poses=None, check_symmetry=False, names=["L_end", "mid_a", "mid_b", "R_end"],
    )
    result = check_named(report, "animation/one-bone-per-cone")
    check(result is not None and not result.passed, "crossing chain fails the check")
    recorded = report.worst["joint_crossing_cones"]
    violators = [entry["label"] for entry in recorded if not entry["seam"]]
    check(
        result is not None and "spans" in result.detail
        and all(violator in result.detail for violator in violators)
        and all(violator.startswith("cone[") and "[0" not in violator for violator in violators),
        f"violators are listed by name in the check text: {result.detail if result else ''}",
    )
    check(
        len(recorded) == len(crossings) and all("label" in entry for entry in recorded),
        "report.worst['joint_crossing_cones'] carries every crossing with a label",
    )
    check("chain seam" in result.detail and "unrelated" in result.detail,
          "check text separates seams from unrelated joints")

    # The same four spheres strung along the inside of one bone: L_elbow -> L_wrist.
    along = left_wrist - left_elbow
    single_bone_centers = np.array([
        left_elbow + parameter * along for parameter in (0.2, 0.4, 0.6, 0.8)
    ])
    parent_joints, _, _ = assign_parent_joint_like_tool(
        single_bone_centers, body.joints, body.parents
    )
    check(
        all(JOINT_NAMES[joint] == "L_elbow" for joint in parent_joints),
        "single-bone chain binds every sphere to L_elbow",
    )
    crossings = joint_crossing_cones(graph, parent_joints, body.parents)
    check(not crossings, "single-bone chain has no joint-crossing cone")
    report = validate(
        single_bone_centers, crossing_radii, chain, body=body, thresholds=Thresholds(),
        poses=None, check_symmetry=False,
    )
    result = check_named(report, "animation/one-bone-per-cone")
    check(result is not None and result.passed, "single-bone chain passes the check")

    # A chain across one joint only: the last sphere past the elbow binds to L_shoulder.
    shoulder_side = np.array([
        left_elbow + 0.3 * along, left_elbow + 0.1 * along,
        left_elbow - 0.1 * along, left_elbow - 0.3 * along,
    ])
    parent_joints, _, _ = assign_parent_joint_like_tool(
        shoulder_side, body.joints, body.parents
    )
    crossings = joint_crossing_cones(graph, parent_joints, body.parents)
    check(
        len(crossings) == 1 and crossings[0]["seam"],
        f"chain over the elbow: one crossing, a parent-child seam ({crossings})",
    )


def test_binding_as_intended(body) -> None:
    print("binding-as-intended")
    left_wrist = body.joints[JOINT_INDEX["L_wrist"]]
    left_elbow = body.joints[JOINT_INDEX["L_elbow"]]
    centers = np.array([
        left_elbow + parameter * (left_wrist - left_elbow) for parameter in (0.2, 0.4, 0.6, 0.8)
    ])
    radii = np.full(4, 0.03)
    chain = [(0, 1), (1, 2), (2, 3)]
    names = ["fa_0", "fa_1", "fa_2", "fa_3"]

    report = validate(
        centers, radii, chain, body=body, poses=None, check_symmetry=False,
    )
    check(
        not any(entry.name == "animation/binding-as-intended" for entry in report.checks),
        "no expected joints: the binding-as-intended check is not added",
    )

    intended = np.full(4, JOINT_INDEX["L_elbow"])
    report = validate(
        centers, radii, chain, body=body, poses=None, check_symmetry=False,
        names=names, expected_parent_joints=intended,
    )
    result = check_named(report, "animation/binding-as-intended")
    check(result is not None and result.passed, "correct intention passes")

    intended[2] = JOINT_INDEX["L_wrist"]
    report = validate(
        centers, radii, chain, body=body, poses=None, check_symmetry=False,
        names=names, expected_parent_joints=intended,
    )
    result = check_named(report, "animation/binding-as-intended")
    check(result is not None and not result.passed, "one wrong intention fails")
    check(
        result is not None and "fa_2" in result.detail and "L_wrist" in result.detail
        and "L_elbow" in result.detail,
        f"mismatch names the sphere and both joints: {result.detail if result else ''}",
    )


# ---------------------------------------------------------------------- clusters


def strip_mesh(columns: int) -> tuple[np.ndarray, np.ndarray]:
    """A two-row triangle strip in the xy plane, ``columns`` vertices per row."""
    xs = np.arange(columns, dtype=np.float64) * 0.01
    bottom = np.stack([xs, np.zeros(columns), np.zeros(columns)], axis=1)
    top = np.stack([xs, np.full(columns, 0.01), np.zeros(columns)], axis=1)
    vertices = np.concatenate([bottom, top], axis=0)
    faces = []
    for column in range(columns - 1):
        lower_left, lower_right = column, column + 1
        upper_left, upper_right = columns + column, columns + column + 1
        faces.append((lower_left, lower_right, upper_right))
        faces.append((lower_left, upper_right, upper_left))
    return vertices, np.array(faces)


def test_coverage_clusters() -> None:
    print("coverage_clusters")
    from src.topology.validate import surface_edges

    columns = 12
    vertices, faces = strip_mesh(columns)
    edges = surface_edges(faces)
    count = len(vertices)

    signed = np.zeros(count)
    # Patch A: columns 0-2, both rows (6 vertices); patch B: columns 6-9, both rows
    # (8 vertices); patch C: column 11, both rows (2 vertices -- below the minimum).
    for column in (0, 1, 2):
        signed[column] = 0.020
        signed[columns + column] = 0.030
    for column in (6, 7, 8, 9):
        signed[column] = 0.018
        signed[columns + column] = 0.025
    signed[11] = 0.040
    signed[columns + 11] = 0.040

    normals = np.tile(np.array([0.0, 0.0, 1.0]), (count, 1))
    normals[columns:] = np.array([0.0, 0.0, 1.0])
    dominant = np.zeros(count, dtype=np.int64)
    dominant[6:] = JOINT_INDEX["L_hip"]
    dominant[columns + 6:] = JOINT_INDEX["L_hip"]
    owner = np.zeros(count, dtype=np.int64)
    owner[6:10] = 3
    owner[columns + 6: columns + 10] = 3

    clusters = coverage_clusters(
        signed, edges, signed > 0.015, vertices, normals, dominant, owner,
        sign=1.0, min_vertices=5,
    )
    check(len(clusters) == 2, f"two clusters of >= 5 vertices (got {len(clusters)})")
    check(
        [cluster["vertices"] for cluster in clusters] == [8, 6],
        f"largest first: {[cluster['vertices'] for cluster in clusters]}",
    )
    check(clusters[0]["worst_mm"] == 25.0, f"worst gap of the 8-patch is 25 mm ({clusters[0]['worst_mm']})")
    check(clusters[0]["joint"] == "L_hip" and clusters[1]["joint"] == "pelvis",
          "dominant joint per cluster")
    check(clusters[0]["bumper"] == 3 and clusters[1]["bumper"] == 0, "owning bumper per cluster")
    check(all(cluster["facing"] == "front" for cluster in clusters), "+z normals face front")
    check(
        abs(clusters[1]["centroid"][0] - 0.01) < 1e-9 and clusters[1]["bbox_max"][0] == 0.02,
        "centroid and bbox of the 6-patch",
    )

    # Every vertex offending, but all disconnected: min_vertices excludes them one by one.
    clusters = coverage_clusters(
        signed, edges[:0], signed > 0.015, vertices, normals, dominant, owner,
        sign=1.0, min_vertices=2,
    )
    check(len(clusters) == 0, "without edges no cluster reaches two vertices")

    # Protrusion uses the negative side and reports a negative worst.
    clusters = coverage_clusters(
        -signed, edges, -signed < -0.015, vertices, -normals, dominant, owner,
        sign=-1.0, min_vertices=5,
    )
    check(len(clusters) == 2 and clusters[0]["worst_mm"] == -25.0,
          "protrusion clusters carry the signed worst value")
    check(all(cluster["facing"] == "back" for cluster in clusters), "-z normals face back")

    # Facing labels for the other axes.
    for axis_vector, expected in (
        ([1.0, 0.0, 0.0], "left"), ([-1.0, 0.0, 0.0], "right"),
        ([0.0, 1.0, 0.0], "up"), ([0.0, -1.0, 0.0], "down"),
    ):
        clusters = coverage_clusters(
            signed, edges, signed > 0.015, vertices, np.tile(axis_vector, (count, 1)),
            dominant, owner, sign=1.0,
        )
        check(clusters[0]["facing"] == expected, f"normal {axis_vector} faces {expected}")


# ---------------------------------------------------------------- harmful overlap


def straight_chain(count: int, spacing: float, radius: float, offset=(0.0, 0.0, 0.0)):
    centers = np.array([
        [index * spacing + offset[0], offset[1], offset[2]] for index in range(count)
    ])
    radii = np.full(count, radius)
    connections = [(index, index + 1) for index in range(count - 1)]
    return centers, radii, connections


def test_surface_samples() -> None:
    print("bumper_surface_samples")
    centers = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.1, 0.15, 0.0]])
    radii = np.array([0.05, 0.03, 0.04])
    graph = BumperGraph(3, [(0, 1, 2)])
    samples = bumper_surface_samples(graph, centers, radii, samples_per_bumper=64, seed=1)
    distance_matrix = graph.distances(np.concatenate(samples), centers, radii)
    start = 0
    for index, bumper in enumerate(graph.bumpers):
        own = distance_matrix[start:start + len(samples[index]), index]
        start += len(samples[index])
        check(
            len(own) > 0 and np.abs(own).max() < 1e-9,
            f"{bumper}: {len(own)} samples all on its own surface (|d| max {np.abs(own).max() if len(own) else float('nan'):.1e})",
        )


def test_harmful_overlap() -> None:
    print("harmful_overlap")
    centers, radii, connections = straight_chain(5, 0.08, 0.05)
    graph = BumperGraph(5, connections)
    result = harmful_overlap(graph, centers, radii, walk_radius=3)
    check(
        result["global_fraction"] < 0.01,
        f"straight 5-sphere chain: {result['global_fraction'] * 100:.2f}% harmful",
    )
    check(result["exposed_total"] > 0, "the chain presents exposed surface")
    check(
        all(entry["dead_fraction"] < 0.5 for entry in result["buried"]),
        "no cone of a straight chain is mostly buried",
    )

    # Two identical chains, one shifted by half a radius, with no connection between them.
    first_centers, first_radii, first_connections = straight_chain(5, 0.08, 0.05)
    second_centers, second_radii, _ = straight_chain(5, 0.08, 0.05, offset=(0.0, 0.025, 0.0))
    centers = np.concatenate([first_centers, second_centers])
    radii = np.concatenate([first_radii, second_radii])
    connections = first_connections + [(index + 5, index + 6) for index in range(4)]
    graph = BumperGraph(10, connections)
    check(len(graph.components()) == 2, "two unlinked chains are two components")
    result = harmful_overlap(graph, centers, radii, walk_radius=3)
    check(
        result["global_fraction"] > 0.3,
        f"interpenetrating unlinked chains: {result['global_fraction'] * 100:.1f}% harmful",
    )
    worst = result["per_bumper"][0]
    check(
        worst["fraction"] > 0.3 and worst["into"] is not None,
        f"worst bumper {graph.bumpers[worst['bumper']]} {worst['fraction'] * 100:.0f}% into "
        f"{graph.bumpers[worst['into']]}",
    )
    own_chain = set(range(5)) if worst["bumper"] < 5 else set(range(5, 10))
    check(
        not (set(graph.bumpers[worst["into"]].spheres) & own_chain)
        if worst["into"] is not None else False,
        "the swallowing bumper belongs to the other chain",
    )


# ------------------------------------------------------------------- the template


#: Pass/fail of the 14 checks ``validate`` produced before this change, recorded on
#: data/template.json with poses=None. Only the overlap check was allowed to change.
TEMPLATE_BEFORE = {
    "budget/spheres": True,
    "budget/triangles": True,
    "budget/quads": True,
    "graph/spheres-connected": True,
    "graph/bumpers-connected": True,
    "graph/no-degenerate-primitives": True,
    "geometry/mirror-symmetric": False,
    "fidelity/max-gap": False,
    "fidelity/gap-fraction": True,
    "fidelity/max-protrusion": False,
    "fidelity/protrusion-fraction": False,
    "watchdog/walk-break-rate": False,
    "watchdog/no-wasteful-connections": True,
    "watchdog/non-adjacent-overlap": False,
}


def test_template(body) -> None:
    print("data/template.json")
    centers, radii, connections, names = load_design(REPO_ROOT / "data" / "template.json")
    report = validate(
        centers, radii, connections, body=body, thresholds=Thresholds(), poses=None,
    )
    outcome = {entry.name: entry.passed for entry in report.checks}
    for name, expected in TEMPLATE_BEFORE.items():
        if name == "watchdog/non-adjacent-overlap":
            continue
        check(
            outcome.get(name) == expected,
            f"{name}: {'PASS' if expected else 'FAIL'} as before",
        )
    check("animation/one-bone-per-cone" in outcome, "one-bone-per-cone check present")
    check(
        outcome.get("watchdog/non-adjacent-overlap") is True,
        "the recalibrated overlap check passes the user's template",
    )
    overlap = report.stats["overlap"]
    check(
        overlap["global_fraction"] <= Thresholds().max_non_adjacent_overlap / 1.8,
        f"template harmful overlap {overlap['global_fraction'] * 100:.2f}% sits under the "
        f"limit with margin ({Thresholds().max_non_adjacent_overlap * 100:.0f}%)",
    )
    check(
        isinstance(report.stats.get("gap_clusters"), list)
        and isinstance(report.stats.get("protrusion_clusters"), list),
        "cluster lists recorded",
    )
    counts = [cluster["vertices"] for cluster in report.stats["protrusion_clusters"]]
    check(counts == sorted(counts, reverse=True) and all(count >= 5 for count in counts),
          f"protrusion clusters sorted by size, >= 5 vertices: {counts[:5]}")
    check(
        all(cluster["facing"] in ("front", "back", "left", "right", "up", "down")
            for cluster in report.stats["gap_clusters"] + report.stats["protrusion_clusters"]),
        "every cluster carries a facing label",
    )
    check(len(report.worst["buried_bumpers"]) == 5, "five most buried connections listed")
    check(
        all("a" in entry and "b" in entry and "fraction" in entry
            for entry in report.worst["non_adjacent_overlap"]),
        "overlap worst entries keep the a/b/fraction keys build_topology.py reads",
    )
    check(
        all("pair" in entry and "edges" in entry for entry in report.worst["walk_breaks"]),
        "walk-break worst entries keep the pair/edges keys build_topology.py reads",
    )

    text = format_report(report)
    for heading in (
        "checks passed", "walk hop histogram", "region", "gap clusters",
        "protrusion clusters", "worst walk breaks", "worst harmful overlaps",
        "most buried connections", "joint-crossing cones",
    ):
        check(heading in text, f"format_report contains '{heading}'")

    # Naming: the template has no names, so make some up and check they show everywhere.
    made_up = [f"s{index:02d}" for index in range(len(centers))]
    named = validate(
        centers, radii, connections, body=body, thresholds=Thresholds(), poses=None,
        names=made_up,
    )
    named_text = format_report(named)
    check("cone[s" in named_text and "cone[0," not in named_text.split("joint-crossing")[0],
          "names replace indices throughout the formatted report")
    walk_check = check_named(named, "watchdog/walk-break-rate")
    check(walk_check is not None and "s23" in walk_check.detail,
          "check details use names too")
    relabelled = format_report(report, names=made_up)
    check("cone[s" in relabelled, "format_report(names=...) relabels an unnamed report")


def main() -> int:
    body = load_body("male")
    test_bumper_label()
    test_one_bone_per_cone(body)
    test_binding_as_intended(body)
    test_coverage_clusters()
    test_surface_samples()
    test_harmful_overlap()
    test_template(body)
    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED")
        for message in FAILURES:
            print(f"  - {message}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
