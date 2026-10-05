"""Checks for ``src/topology/sections.py`` and the ``method="cut"`` path in ``anatomy``.

The cut is checked against an analytic circle on a triangulated sphere, then on the SMPL
male body against the things the handoff said the vertex band gets wrong: too few points,
a shallower trunk than the mesh actually has, and both thighs merged into one section.

Run:  .venv/Scripts/python.exe scripts/test_topology_sections.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.topology import anatomy  # noqa: E402
from src.topology.anatomy import joint_region_mask  # noqa: E402
from src.topology.sections import (  # noqa: E402
    interpolate_weights,
    loop_perimeter,
    plane_cut,
    select_loop,
)

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  ok    {message}")
    else:
        print(f"  FAIL  {message}")
        FAILURES.append(message)


# ------------------------------------------------------------- analytic sphere cut

print("analytic sphere")
import pyvista as pv  # noqa: E402

sphereMesh = pv.Sphere(radius=0.1, theta_resolution=60, phi_resolution=60).triangulate()
sphereVertices = np.asarray(sphereMesh.points, dtype=np.float64)
sphereFaces = np.asarray(sphereMesh.regular_faces, dtype=np.int64)
check(sphereMesh.n_open_edges == 0, "test sphere is watertight")

planeOffset = 0.03
sphereLoops = plane_cut(sphereVertices, sphereFaces, [0.0, 0.0, planeOffset], [0.0, 0.0, 1.0])
check(len(sphereLoops) == 1, f"one loop through the sphere (got {len(sphereLoops)})")
sphereLoop = sphereLoops[0]
check(sphereLoop.closed, "the sphere loop is closed")
analyticPerimeter = 2.0 * np.pi * np.sqrt(0.1 ** 2 - planeOffset ** 2)
measuredPerimeter = loop_perimeter(sphereLoop)
print(f"        perimeter {measuredPerimeter:.6f} m, analytic {analyticPerimeter:.6f} m")
check(
    abs(measuredPerimeter - analyticPerimeter) <= 0.01 * analyticPerimeter,
    "sphere loop perimeter within 1% of the analytic circle",
)
check(
    np.allclose(sphereLoop.points[:, 2], planeOffset, atol=1e-12),
    "every cut point lies on the plane",
)
check(
    np.all(sphereLoop.edge_parameters >= 0.0) and np.all(sphereLoop.edge_parameters <= 1.0),
    "edge parameters are within [0, 1]",
)

generator = np.random.default_rng(7)
madeUpWeights = generator.uniform(0.0, 1.0, size=(len(sphereVertices), 24))
madeUpWeights /= madeUpWeights.sum(axis=1, keepdims=True)
interpolated = interpolate_weights(sphereLoop, madeUpWeights)
check(interpolated.shape == (len(sphereLoop), 24), "interpolated weights are (K, 24)")
check(
    np.allclose(interpolated.sum(axis=1), 1.0, atol=1e-12),
    "interpolated weights sum to 1 per point",
)

# --------------------------------------------------------------------- male body

print("male body")
body = anatomy.load_body("male")
up = np.array([0.0, 1.0, 0.0])

thighLoops = plane_cut(body.vertices, body.faces, [0.0, -0.55, 0.0], up)
check(len(thighLoops) == 2, f"y = -0.55 cuts exactly two loops (got {len(thighLoops)})")
check(all(loop.closed for loop in thighLoops), "both thigh loops are closed")
# An independent cut of the same mesh (vtkCutter) must produce the same number of points
# and the same number of connected pieces. The SMPL mesh has exactly 34 edge crossings
# around each thigh at this height, so that -- not a round number -- is the bar.
bodyMesh = pv.PolyData(
    body.vertices, np.hstack([np.full((len(body.faces), 1), 3), body.faces]).ravel()
)
vtkSlice = bodyMesh.slice(normal=(0.0, 1.0, 0.0), origin=(0.0, -0.55, 0.0))
vtkRegions = int(vtkSlice.connectivity()["RegionId"].max()) + 1
print(f"        vtkCutter at y=-0.55: {vtkSlice.n_points} points in {vtkRegions} pieces")
check(vtkRegions == 2, "vtkCutter agrees on two pieces")
check(
    sum(len(loop) for loop in thighLoops) == vtkSlice.n_points,
    "plane_cut and vtkCutter agree on the number of cut points",
)
for index, loop in enumerate(thighLoops):
    print(f"        thigh loop {index}: {len(loop)} points, mean x {loop.points[:, 0].mean():+.3f}")
    check(len(loop) > 30, f"thigh loop {index} has more than 30 points ({len(loop)})")
leftThigh = select_loop(thighLoops, side="left")
check(leftThigh.points[:, 0].mean() > 0.0, "select_loop(side='left') has mean x > 0")
rightThigh = select_loop(thighLoops, side="right")
check(rightThigh.points[:, 0].mean() < 0.0, "select_loop(side='right') has mean x < 0")
leftHipMask = joint_region_mask(body, ("L_hip",))
check(
    select_loop(thighLoops, prefer_mask=leftHipMask) is leftThigh,
    "prefer_mask on L_hip selects the left thigh",
)
check(
    select_loop(thighLoops, near=body.joints[anatomy.JOINT_INDEX["R_hip"]] + [0.0, -0.3, 0.0])
    is rightThigh,
    "near a point under R_hip selects the right thigh",
)

# Trunk depth: the vertex band versus the cut at the same origin. The handoff measured the
# band the way the generator uses it -- dominant-joint filtered, +-10 mm -- so both the
# plain band and that one are reported.
trunkOrigin = np.array([0.0, -0.05, 0.0])
spineJoints = [anatomy.JOINT_INDEX[name] for name in ("pelvis", "spine1", "spine2", "spine3")]
dominantTrunk = np.isin(np.argmax(body.skin_weights, axis=1), spineJoints)
bandSection = anatomy.section_at(body, trunkOrigin, up)
filteredBandSection = anatomy.section_at(
    body, trunkOrigin, up, vertex_mask=dominantTrunk, half_thickness=0.010
)
cutSection = anatomy.section_at(body, trunkOrigin, up, method="cut")
bandWidth, bandDepth = bandSection.extent()
filteredWidth, filteredDepth = filteredBandSection.extent()
cutWidth, cutDepth = cutSection.extent()
print(
    f"        trunk y=-0.05: band +-12 mm      {len(bandSection.planar):3d} points, "
    f"width {bandWidth * 1000:.1f} mm, depth {bandDepth * 1000:.1f} mm"
)
print(
    f"        trunk y=-0.05: band dominant +-10 {len(filteredBandSection.planar):3d} points, "
    f"width {filteredWidth * 1000:.1f} mm, depth {filteredDepth * 1000:.1f} mm"
)
print(
    f"        trunk y=-0.05: cut               {len(cutSection.planar):3d} points, "
    f"width {cutWidth * 1000:.1f} mm, depth {cutDepth * 1000:.1f} mm"
)
print(
    f"        cut is deeper than the plain band by {(cutDepth - bandDepth) * 1000:.1f} mm "
    f"and than the dominant-joint band by {(cutDepth - filteredDepth) * 1000:.1f} mm"
)
check(cutDepth > bandDepth, "the mesh cut is deeper than the plain vertex band")
check(
    cutDepth - filteredDepth > 0.020,
    "the mesh cut is deeper than the dominant-joint band by tens of mm",
)

# The lateral pelvis at y = -0.35, which the handoff says the band misses entirely
# (0.287 m against a true 0.371 m).
pelvisOrigin = np.array([0.0, -0.35, 0.0])
pelvisBand = anatomy.section_at(
    body, pelvisOrigin, up, vertex_mask=dominantTrunk, half_thickness=0.010
)
pelvisCut = anatomy.section_at(body, pelvisOrigin, up, method="cut", vertex_mask=dominantTrunk)
print(
    f"        pelvis y=-0.35: band width {pelvisBand.extent()[0] * 1000:.1f} mm, "
    f"cut width {pelvisCut.extent()[0] * 1000:.1f} mm ({len(pelvisCut.planar)} points)"
)
# With a mask the cut is trimmed to the mask's population on purpose (so a plane through
# the trunk and the T-posed arms does not return a 1.3 m loop); the full pelvis width is
# what the *unmasked* cut must find.
pelvisCutFull = anatomy.section_at(body, pelvisOrigin, up, method="cut")
print(f"        pelvis y=-0.35: unmasked cut width {pelvisCutFull.extent()[0] * 1000:.1f} mm")
check(pelvisCutFull.extent()[0] > 0.36, "the unmasked cut finds the full pelvis width at y = -0.35")
check(
    pelvisCut.extent()[0] <= pelvisCutFull.extent()[0] + 1e-9,
    "the masked cut is trimmed to the mask's population",
)
check(len(cutSection.planar) > len(bandSection.planar), "the cut has more points than the band")
check(
    np.allclose(cutSection.first_axis, bandSection.first_axis)
    and np.allclose(cutSection.second_axis, bandSection.second_axis),
    "cut and band sections share the in-plane frame",
)
check(
    np.all(cutSection.vertex_indices >= 0) and np.all(cutSection.vertex_indices < len(body.vertices)),
    "cut section vertex_indices index the body",
)
cutWorld = cutSection.to_world(cutSection.planar)
check(
    np.abs(cutWorld[:, 1] - trunkOrigin[1]).max() < 1e-12,
    "cut section round-trips to the cutting plane",
)

# Symmetrised loop: the hull's half-planes come in exact mirror pairs.
symmetricSection = anatomy.section_at(body, trunkOrigin, up, method="cut", symmetrize=True)
normals, offsets = symmetricSection.half_planes()
mirrorFound = 0
for normal, offset in zip(normals, offsets):
    mirrored = np.array([-normal[0], normal[1]])
    match = np.abs(normals - mirrored).max(axis=1) < 1e-12
    if match.any() and np.abs(offsets[match] - offset).min() < 1e-12:
        mirrorFound += 1
print(f"        {mirrorFound} of {len(normals)} half-planes have an exact mirror twin")
check(mirrorFound == len(normals), "symmetrized hull half-planes come in exact mirror pairs")

# inscribed_radius through the KD-tree equals brute force.
interiorPoints = body.joints[generator.integers(0, 24, size=20)] + generator.uniform(
    -0.01, 0.01, size=(20, 3)
)
treeRadii = anatomy.inscribed_radius(body, interiorPoints)
bruteRadii = np.array(
    [np.min(np.linalg.norm(body.vertices - point, axis=1)) for point in interiorPoints]
)
check(
    np.abs(treeRadii - bruteRadii).max() < 1e-12,
    f"KD-tree inscribed_radius matches brute force (max diff {np.abs(treeRadii - bruteRadii).max():.2e})",
)
check(
    anatomy.inscribed_radius(body, interiorPoints[0]).shape == (1,),
    "inscribed_radius accepts a single centre",
)

# bone_stations, band versus cut.
stationParameters = (0.2, 0.5, 0.8)
for parentName, childName in (("L_hip", "L_knee"), ("L_shoulder", "L_elbow")):
    print(f"bone {parentName}->{childName}")
    limbMask = joint_region_mask(body, (parentName,))
    bandStations = anatomy.bone_stations(
        body, parentName, childName, parameters=stationParameters, vertex_mask=limbMask,
    )
    cutStations = anatomy.bone_stations(
        body, parentName, childName, parameters=stationParameters, vertex_mask=limbMask,
        method="cut",
    )
    check(len(cutStations) == 3, f"cut bone_stations returns all three stations ({len(cutStations)})")
    bandByParameter = {round(station.parameter, 4): station for station in bandStations}
    for cutStation in cutStations:
        bandStation = bandByParameter.get(round(cutStation.parameter, 4))
        if bandStation is None:
            print(f"        t={cutStation.parameter:.1f}: band has no station")
            continue
        medialShift = np.linalg.norm(cutStation.medial_point - bandStation.medial_point) * 1000
        print(
            f"        t={cutStation.parameter:.1f}: medial shift {medialShift:6.1f} mm, "
            f"inscribed band {bandStation.inscribed_radius * 1000:6.1f} mm / "
            f"cut {cutStation.inscribed_radius * 1000:6.1f} mm, "
            f"points band {bandStation.vertex_count} / cut {cutStation.vertex_count}"
        )
        check(cutStation.inscribed_radius > 0.0, f"t={cutStation.parameter:.1f} cut inscribed radius is positive")

# ------------------------------------------------------------------------ result

print()
if FAILURES:
    print(f"{len(FAILURES)} check(s) FAILED:")
    for message in FAILURES:
        print(f"  - {message}")
    sys.exit(1)
print("all checks passed")
