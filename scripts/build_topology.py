<<<<<<< Updated upstream
"""Generate a sphere-mesh template for an SMPL body from placement rules, then fit,
repair, validate and render it.

    .venv/Scripts/python.exe scripts/build_topology.py
    .venv/Scripts/python.exe scripts/build_topology.py --gender female --name auto_female
    .venv/Scripts/python.exe scripts/build_topology.py --quick          # no poses, short fit
    .venv/Scripts/python.exe scripts/build_topology.py --design my.json  # searched parameters

Writes ``data/generated/<name>.json`` (opens in the tool), ``<name>.txt`` (engine, quads
intact), ``reports/<name>.txt``, ``renders/<name>_*.png`` and ``<name>_log.json``.

The pipeline
------------
1. **Place.** Rules, not coordinates (``src/topology/generate.py``): chains on the medial
   point of each limb cross-section, a two-column ladder on the lower trunk and a
   three-column sheet on the upper trunk seamed with triangles, inscribed spheres in the
   buttock, stubs where the model stops. Every radius starts as the skin-touching one.
   Only the left half and midline are placed; the right half is the mirror.
2. **Fit.** Centres and radii are optimised against the *exact* sphere-mesh distance
   (``src/topology/fit.py``) with a one-sided loss that charges protrusion more than gaps,
   symmetric by construction, bounded so no sphere changes the joint it binds to.
3. **Repair.** The spare sphere budget is spent by measurement (``src/topology/repair.py``):
   cones are added where the watchdog's walk is measurably too long, spheres are added
   where the skin is measurably far from the collider, each addition re-fitted and
   re-measured.
4. **Fit again**, this time with a handful of real poses in the objective so that a bumper
   which pops out of the body under the tool's rigid binding is penalised where the user
   would see it.
5. **Validate and render** (``src/topology/validate.py``, ``src/topology/render.py``).

Every structural constant lives in ``Design`` so that ``scripts/search_topology.py`` can
vary it; nothing in this file is a coordinate.
=======
"""Generate a sphere-mesh template for an SMPL body from placement rules, then validate it.

    .venv/Scripts/python.exe scripts/build_topology.py
    .venv/Scripts/python.exe scripts/build_topology.py --gender female --name v1_female
    .venv/Scripts/python.exe scripts/build_topology.py --no-refine --no-poses

Writes ``data/generated/<name>.json`` (openable in the tool) and ``<name>.txt`` (engine,
quads intact), plus ``data/generated/reports/<name>.txt``.

The design
----------
Four structures, each chosen because the region it covers has a different *shape*, and one
rule that decides where they may be split.

**The lower trunk is a ladder, not a sheet.** From the pelvis to about y = +0.10 the trunk
is a fat, near-circular tube -- width/depth aspect 1.21-1.45, largest inscribed circle
95-100% of the half-depth -- so its medial axis is short and a wide multi-column sheet
there is spent on nothing. Two columns roughly 45 mm either side of the midline already
reach the full half-width; the quads between the rungs are big enough to earn their place.

**The upper trunk is a sheet.** Above y = +0.13 the section really does flatten, aspect
rising to 2.29 at the clavicle, and three columns with slabs between them pay off.

**Limbs are chains** on the *medial* point of each cross-section, never on the bone axis --
the two differ by 22 mm on the shin and 10 mm on the upper arm, and snapping to the
skeleton costs 17.5 mm RMS there against 2.6 mm. Four stations per segment, which is where
the accuracy curve flattens.

**Bulges get their own inscribed spheres.** The buttock is behind the pelvis while the
pelvis's own medial locus is 120 mm in front of it, so no medial construction can reach it;
``blobs`` searches a box for the largest spheres that fit inside the body instead.

The splitting rule is the joints. Rigid single-joint binding is almost free while a bumper
stays inside one bone -- 0.3 to 14 mm -- and catastrophic across one: 260 mm RMS across the
elbow, whose median local rotation is 132 degrees. So no bumper crosses a joint; chains stop
short on both sides and meet through a single short cone, and every chain stops short of the
joint *centre* as well, because a sphere sitting exactly on one binds to whichever of the two
bones wins a floating-point tie.

Triangles are rationed, and not for the reason the budget suggests. A slab's gain over its
own three edges is ``sqrt(rho^2 + r^2) - r`` with ``rho`` the medial inradius, so a slab
must be *large relative to its spheres* to be worth anything: between 120 mm torso spheres a
40x55 mm cell buys 1.7 mm. Hence eight big quads rather than thirty small ones.
>>>>>>> Stashed changes
"""
from __future__ import annotations

import argparse
<<<<<<< Updated upstream
import dataclasses
import json
import sys
import time
from dataclasses import dataclass, field
=======
import json
import sys
>>>>>>> Stashed changes
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.topology.anatomy import (  # noqa: E402
    JOINT_INDEX,
<<<<<<< Updated upstream
    JOINT_NAMES,
    TRUNK_JOINTS,
    Body,
    assign_parent_joint_like_tool,
    joints_compatible,
=======
>>>>>>> Stashed changes
    load_body,
    mocap_poses,
    random_poses,
)
from src.topology.export import build_payload, write_engine_text, write_tool_json  # noqa: E402
<<<<<<< Updated upstream
from src.topology.fit import FitOptions, fit_topology  # noqa: E402
=======
>>>>>>> Stashed changes
from src.topology.generate import (  # noqa: E402
    TopologySpec,
    blobs,
    chain,
    mirror,
<<<<<<< Updated upstream
=======
    refine,
    relieve_protrusion,
    ring,
>>>>>>> Stashed changes
    sheet,
    stub,
    symmetrize,
)
<<<<<<< Updated upstream
from src.topology.repair import (  # noqa: E402
    PoseSet,
    fill_gaps,
    link_walk_breaks,
    replace_degenerate_slabs,
)
from src.topology.validate import (  # noqa: E402
    Thresholds,
    modelled_vertex_mask,
    surface_edges,
    validate,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "generated"


# --------------------------------------------------------------------------- the design


@dataclass
class Design:
    """Every structural and fitting choice, in one searchable place.

    The defaults are the previous session's hand-tuned values where those were measured
    to be right (see TOPOLOGY_NOTES.md) and the new pipeline's defaults elsewhere.
    """

    # The trunk is two shapes and gets two structures: a fat near-circular tube below
    # about y = +0.10 (aspect 1.21-1.45, inscribed circle 95-100% of the half-depth) which
    # a two-column ladder carries, and a genuinely flat sheet above y = +0.13 (aspect
    # rising to 2.29 at the clavicle) which a three-column sheet carries. A midline column
    # is mandatory on the lower trunk: with two symmetric columns the slab depth at x = 0
    # is exactly 2r whatever the spacing, 18 mm short of the belly.
    lower_rows: tuple[float, ...] = (0.075, -0.005, -0.085, -0.165, -0.240)
    upper_rows: tuple[float, ...] = (0.140, 0.200, 0.260)
    lower_columns: tuple[float, ...] = (0.0, 0.70)
    upper_columns: tuple[float, ...] = (0.0, 0.75)
    #: A midline pubic node, or None. Off by default: any midline sphere below the pelvis
    #: joint is closer to a hip bone than to the spine, so the tool binds it to *one* hip
    #: and its cone to the other thigh spans two joints that rotate independently. The
    #: pubic surface is covered instead by the gap filler, whose sphere sits high enough
    #: to bind to the pelvis.
    crotch_row: float | None = None

    # Positions along each bone, 0 at the parent joint and 1 at the child; never exactly
    # 0 or 1 (a sphere on a joint centre binds by floating-point tie). Four per segment is
    # where the accuracy curve flattens; the floor is the section's departure from a
    # circle, not the taper.
    upper_arm: tuple[float, ...] = (0.06, 0.38, 0.68, 0.93)
    forearm: tuple[float, ...] = (0.10, 0.40, 0.70, 0.95)
    thigh: tuple[float, ...] = (0.07, 0.19, 0.36, 0.62, 0.90)
    shin: tuple[float, ...] = (0.08, 0.36, 0.64, 0.92)
    hip: tuple[float, ...] = (-0.26, -0.10)
    deltoid: float = 0.70
    clavicle: float = 0.20
    neck: tuple[float, ...] = (0.02, 0.42)

    # The buttock, as a box to search for the largest inscribed spheres: it is a bulge
    # behind the pelvis, whose medial locus is 120 mm in front of it.
    glute_count: int = 3
    glute_box_low: tuple[float, float, float] = (0.020, -0.060, -0.170)
    glute_box_high: tuple[float, float, float] = (0.170, 0.115, -0.045)
    glute_separation: float = 0.055

    #: "cut" measures every cross-section by cutting the mesh with the plane; "band"
    #: is the previous vertex-slab method, kept for comparison.
    section_method: str = "cut"

    # Fit. The centre cap is measured from the original placement (fits are anchored):
    # 12 mm left the shoulder spheres short of where the skin is, 20 mm reaches it (search
    # candidate #14, then confirmed by the full runs: worst gap 24.2 -> 21.4 mm).
    protrusion_weight: float = 6.0
    max_offset: float = 0.020
    radius_scale_bounds: tuple[float, float] = (0.70, 1.30)
    #: The pose term only penalises pop-outs beyond ``pose_tolerance``. Measured on the
    #: searched design with the 54-pose sweep (worst posed protrusion / rest RMS / rest
    #: vertices >15 mm off): weight 0 -> 46.1 mm / 5.3 mm / 0.69%; 0.1 at 12 mm -> 43.2 /
    #: 5.9 / 1.30%; 0.25 at 10 mm -> 39.6 / 7.3 / 5.78%; 0.5 at 6 mm -> 30.8 / 9.4 / 12.5%.
    #: Every millimetre of posed protrusion costs far more at rest, because the pop-out is
    #: the floor of rigid single-joint binding at the shoulder, not a placement error. Off
    #: by default; the machinery stays for when the tool's secondary-joint blend is enabled.
    pose_weight: float = 0.0
    pose_tolerance: float = 0.012

    # Repair.
    max_spheres: int = 96
    gap_threshold: float = 0.012
    fill_rounds: int = 12
    fill_protrusion_tolerance: float = 0.008
    link_max: int = 8
    link_protrusion_tolerance: float = 0.010
    walk_budget: int = 3
    #: A cone the repairs add between spheres bound to different joints may also be
    #: required to stay within this of the skin under the fit poses (rigidly posed ends);
    #: None disables the check. Off by default: measured on the searched design, the check
    #: at 25 mm did not lower the posed pop-out (49.1 mm against 46.0 without it, because
    #: the pop-out comes from the design's own deltoid seam, not from added cones) and
    #: cost walk breaks (2.99 % against 1.92 %).
    repair_posed_tolerance: float | None = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Design":
        known = {name.name for name in dataclasses.fields(cls)}
        values = {}
        for key, value in data.items():
            if key not in known:
                raise ValueError(f"unknown design parameter {key!r}")
            values[key] = tuple(value) if isinstance(value, list) else value
        if "crotch_row" in values and values["crotch_row"] is not None:
            values["crotch_row"] = float(values["crotch_row"])
        if "repair_posed_tolerance" in values and values["repair_posed_tolerance"] is not None:
            values["repair_posed_tolerance"] = float(values["repair_posed_tolerance"])
        return cls(**values)


# ----------------------------------------------------------------------------- masks


def trunk_mask(body: Body) -> np.ndarray:
    """Trunk vertices, with the arms kept out."""
=======
from src.topology.validate import Thresholds, modelled_vertex_mask, validate  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "generated"

# --------------------------------------------------------------------------- the design

#: The trunk is two different shapes and gets two different structures.
#:
#: Below about y = +0.10 it is a fat, near-circular tube: width/depth aspect 1.21-1.45 and a
#: largest inscribed circle that is 95-100% of the half-depth, so its medial axis is short
#: (36-57% of the half-width) and a wide sheet there is spent on nothing. Two columns close
#: to the midline carry it. Above y = +0.13 the section really does flatten -- aspect 1.49
#: rising to 2.29 at the clavicle, medial axis running the full half-width -- and a proper
#: three-column sheet earns its keep.
LOWER_ROWS = (0.075, -0.005, -0.085, -0.165, -0.240)
UPPER_ROWS = (0.140, 0.200, 0.260)

#: The lower trunk is a RING, not a row. Solving the minimax placement of K disks against
#: the true trunk outline, a collinear row -- what a coronal medial sheet degenerates to in
#: the section plane -- and a free ring score very differently: at y = -0.160 the row cannot
#: get below 10.9 mm however many disks it is given, while a ring of four reaches 4.8 mm.
#: A row can only ever be wrong in one direction, having no disk anywhere near the front or
#: back wall; a ring puts one against each.
#:
#: Angles are measured in the section plane from +x towards the second in-plane axis, and
#: come in mirror pairs so the ring is bilaterally symmetric by construction.
LOWER_ANGLES = (40.0, 140.0, 220.0, 320.0)
LOWER_RING_NAMES = ("L_back", "R_back", "R_front", "L_front")

#: The alternative: a two-column row per rung, each column's depth solved for separately.
#: Cheaper by ten spheres and fourteen quads. Which one wins is a measurement, not a
#: preference, so both are built from the same code and selected by --lower-structure.
LOWER_COLUMNS = (0.0, 0.70)
LOWER_LADDER_NAMES = ("M", "L")
UPPER_COLUMNS = (0.0, 0.75)
UPPER_COLUMN_NAMES = ("M", "L")

#: Positions along each bone, 0 at the parent joint and 1 at the child. Never exactly 0 or
#: 1: a sphere on a joint centre binds to whichever of the two bones wins a floating-point
#: tie, so every chain stops short of the joint at both ends.
#:
#: Four stations per segment, because that is where the curve flattens: chain RMS against
#: the true surface goes 4.8 / 3.0 / 2.9 / 2.6 mm for 2 / 3 / 4 / 5 spheres on the upper
#: arm and 8.6 / 3.3 / 2.8 / 2.6 on the shin. Past four, extra spheres are measurably
#: wasted -- the limit is not the taper but the cross-section's departure from a circle,
#: which no chain of any length can follow.
UPPER_ARM = (0.06, 0.38, 0.68, 0.93)
FOREARM = (0.10, 0.40, 0.70, 0.95)
THIGH = (0.07, 0.19, 0.36, 0.62, 0.90)
SHIN = (0.08, 0.36, 0.64, 0.92)

#: The hip and the buttock, in the thigh bone's parameter but above the hip joint. No thigh
#: chain can own this band: the cross-section is inseparable from the pelvis below t = 0.167
#: of the femur, and the skin it covers is 12.6% of the whole body.
HIP = (-0.26, -0.10)
#: The buttock, given as a box to find the largest inscribed spheres in rather than a locus
#: to walk along. It is a bulge *behind* the pelvis and the largest circle inscribed in a
#: pelvic cross-section sits 120 mm in front of it, so every medial construction puts the
#: sphere in the wrong place -- a dense 900-sphere medial cover still leaves a 48 mm hole
#: across the seat, and narrowing the section to posterior vertices to force the issue
#: yields 15 mm spheres hugging the skin. The box is relative to the hip joint in y, and
#: capped at z = -0.045 so the search cannot wander forward into the middle of the pelvis
#: where the genuinely biggest spheres are.
GLUTE_COUNT = 3
GLUTE_BOX_LOW = (0.020, -0.060, -0.170)
GLUTE_BOX_HIGH = (0.170, 0.115, -0.045)

#: One pubic/perineal sphere, placed low. It has to clear the bottom rung's own 135 mm
#: midline sphere, which reaches to y = -0.375: a candidate at y = -0.285 sits entirely
#: inside it and is a bumper that can never be nearest to anything. But deleting it outright
#: costs 10 mm of worst gap, because the coverage was never coming from the sphere -- it was
#: coming from its cones down to the hips and thighs. So: keep the node, put it where it is
#: not swallowed.
CROTCH_ROW = (-0.345,)


def trunk_mask(body):
    """Trunk vertices, with the arms kept out.

    Skinning weight on the spine alone is not enough near the armpit, where a vertex can be
    half spine and half collar; anything the shoulder chain owns a meaningful share of is
    excluded so the cross-sections do not bulge out along the arms.
    """
>>>>>>> Stashed changes
    spine = [JOINT_INDEX[name] for name in ("pelvis", "spine1", "spine2", "spine3")]
    arm = [JOINT_INDEX[name] for name in
           ("L_collar", "R_collar", "L_shoulder", "R_shoulder", "neck", "head")]
    return (
        (body.skin_weights[:, spine].sum(axis=1) >= 0.6)
        & (body.skin_weights[:, arm].sum(axis=1) <= 0.15)
    )


<<<<<<< Updated upstream
def upper_trunk_mask(body: Body) -> np.ndarray:
=======
def upper_trunk_mask(body):
    """The shoulder girdle: spine, collars and neck, with the arms proper kept out."""
>>>>>>> Stashed changes
    trunk = [JOINT_INDEX[name] for name in ("spine3", "L_collar", "R_collar", "neck")]
    arm = [JOINT_INDEX[name] for name in ("L_shoulder", "R_shoulder", "head")]
    return (
        (body.skin_weights[:, trunk].sum(axis=1) >= 0.5)
        & (body.skin_weights[:, arm].sum(axis=1) <= 0.20)
    )


<<<<<<< Updated upstream
def crotch_mask(body: Body) -> np.ndarray:
=======
def glute_mask(body):
    """Left buttock: pelvis and hip vertices behind the body's mid-depth."""
    columns = [JOINT_INDEX[name] for name in ("pelvis", "L_hip")]
    return (
        (body.skin_weights[:, columns].sum(axis=1) >= 0.5)
        & (body.vertices[:, 0] > 0.015)
        & (body.vertices[:, 2] < 0.010)
    )


def crotch_mask(body):
    """Pelvis vertices near the midline, below the sheet's last full row.

    The |x| bound is what makes a horizontal cut here meaningful: without it the section
    spans both thighs and its convex hull bridges the gap between the legs.
    """
>>>>>>> Stashed changes
    columns = [JOINT_INDEX[name] for name in ("pelvis", "L_hip", "R_hip")]
    return (
        (body.skin_weights[:, columns].sum(axis=1) >= 0.5)
        & (np.abs(body.vertices[:, 0]) < 0.075)
    )


<<<<<<< Updated upstream
def pelvis_side_mask(body: Body) -> np.ndarray:
=======
def pelvis_side_mask(body):
    """Left-hand pelvis and hip vertices.

    Restricted to one side on purpose: a horizontal cut through both hips has a convex hull
    that bridges the crotch, so its inscribed circle sits in mid-air between the legs.
    """
>>>>>>> Stashed changes
    columns = [JOINT_INDEX[name] for name in ("pelvis", "L_hip")]
    return (body.skin_weights[:, columns].sum(axis=1) >= 0.5) & (body.vertices[:, 0] > 0.015)


<<<<<<< Updated upstream
def limb_mask(body: Body, joints, weight_min: float = 0.5) -> np.ndarray:
=======
def limb_mask(body, joints, weight_min=0.5):
>>>>>>> Stashed changes
    columns = [JOINT_INDEX[name] for name in joints]
    return body.skin_weights[:, columns].sum(axis=1) >= weight_min


<<<<<<< Updated upstream
# ------------------------------------------------------------------------------ spec


def build_spec(body: Body, design: Design) -> TopologySpec:
    """The structural topology: left half and midline, mirrored at the end."""
    spec = TopologySpec()
    method = design.section_method

    # ------------------------------------------------------------------- lower trunk
    half, measured = sheet(
        body, design.lower_rows, design.lower_columns,
        name="lower", column_names=("M", "L"), group="trunk",
        vertex_mask=trunk_mask(body), method=method,
    )
    spec.add(half)
    lower_rows = len(design.lower_rows)
    bottom = lower_rows - 1
    lower_middle = {row: measured[(row, 0)] for row in range(lower_rows)}
    lower_left = {row: measured[(row, 1)] for row in range(lower_rows)}
    # A ladder of quads. Slab gain over its own cones is sqrt(rho^2 + r^2) - r with rho the
    # medial inradius, so cells must be large relative to their spheres to pay for
    # themselves: 80 x 80 mm between 100 mm spheres buys ~8 mm, 40 x 55 mm buys under 2.
    for row in range(lower_rows - 1):
        spec.connect(
            lower_middle[row], lower_left[row], lower_left[row + 1], lower_middle[row + 1]
        )

    # ------------------------------------------------- upper trunk: three-column sheet
    upper_spheres, upper_grid = sheet(
        body, design.upper_rows, design.upper_columns,
        name="upper", column_names=("M", "L"), group="trunk",
        vertex_mask=upper_trunk_mask(body), method=method,
    )
    spec.add(upper_spheres)
    upper_rows = len(design.upper_rows)
=======
def build_spec(body, lower_structure: str = "ladder") -> TopologySpec:
    spec = TopologySpec()

    # ------------------------------------------------------------------- lower trunk
    if lower_structure == "ring":
        lower_spheres, lower_grid = ring(
            body, LOWER_ROWS, LOWER_ANGLES,
            name="lower", column_names=LOWER_RING_NAMES,
            group="trunk", vertex_mask=trunk_mask(body),
        )
        corners = len(LOWER_ANGLES)
    elif lower_structure == "ladder":
        half, lower_grid = sheet(
            body, LOWER_ROWS, LOWER_COLUMNS,
            name="lower", column_names=LOWER_LADDER_NAMES,
            group="trunk", vertex_mask=trunk_mask(body),
        )
        lower_spheres = half
        # The mirrored column is not measured; it is generated, so the two halves cannot
        # drift apart. Referenced here by name before mirror() creates it, which is fine:
        # connections are resolved to indices only after the whole spec is built.
        measured = dict(lower_grid)
        columns_here = len(LOWER_COLUMNS)
        lower_grid = {}
        for row in range(len(LOWER_ROWS)):
            for column in range(columns_here):
                lower_grid[(row, column)] = measured[(row, column)]
            for column in range(1, columns_here):
                name = measured[(row, column)]
                lower_grid[(row, -column)] = "R_" + name[2:]
        corners = columns_here
    else:
        raise ValueError(f"unknown lower_structure {lower_structure!r}")
    spec.add(lower_spheres)
    lower_rows = len(LOWER_ROWS)
    bottom = lower_rows - 1

    ladder = lower_structure == "ladder"

    def rung(row, corner):
        """Ring corner by index, wrapping.

        For the ladder the same indices name its columns: 3 is the outermost left, 0 the
        midline, 1 the outermost right. That keeps every seam and limb attachment below
        written once for both families.
        """
        if ladder:
            return lower_grid[(row, {3: corners - 1, 0: 0, 1: -(corners - 1),
                                     2: -(corners - 1)}[corner % 4])]
        return lower_grid[(row, corner % corners)]

    # A tube of quads: one per ring edge per rung gap. These are the slabs that earn their
    # place -- gain over the three edge cones is sqrt(rho^2 + r^2) - r, so a cell 80 mm tall
    # by 80 mm around between 100 mm spheres buys about 8 mm, while the 40 x 55 mm cells of
    # a fine sheet buy under 2 mm and are pure cost.
    for row in range(lower_rows - 1):
        if ladder:
            for column in range(corners - 1):
                spec.connect(
                    lower_grid[(row, column)], lower_grid[(row, column + 1)],
                    lower_grid[(row + 1, column + 1)], lower_grid[(row + 1, column)],
                )
            continue
        for corner in range(corners):
            spec.connect(
                rung(row, corner), rung(row, corner + 1),
                rung(row + 1, corner + 1), rung(row + 1, corner),
            )

    # ------------------------------------------------- upper trunk: a three-column sheet
    upper_spheres, upper_grid = sheet(
        body, UPPER_ROWS, UPPER_COLUMNS,
        name="upper", column_names=UPPER_COLUMN_NAMES,
        group="trunk", vertex_mask=upper_trunk_mask(body),
    )
    spec.add(upper_spheres)
    upper_rows = len(UPPER_ROWS)
>>>>>>> Stashed changes
    upper_middle = {row: upper_grid[(row, 0)] for row in range(upper_rows)}
    upper_left = {row: upper_grid[(row, 1)] for row in range(upper_rows)}
    for row in range(upper_rows - 1):
        spec.connect(
            upper_middle[row], upper_left[row], upper_left[row + 1], upper_middle[row + 1]
        )
<<<<<<< Updated upstream
    # Seam between the two structures: a flank triangle and a front/back triangle.
    spec.connect(lower_left[0], upper_middle[0], upper_left[0])
    spec.connect(lower_left[0], lower_middle[0], upper_middle[0])

    # ------------------------------------------------------------------ lower pelvis
    crotch = None
    if design.crotch_row is not None:
        crotch_spheres, crotch_grid = sheet(
            body, (design.crotch_row,), (0.0,), name="crotch", column_names=("M",),
            group="trunk", vertex_mask=crotch_mask(body), half_thickness=0.024,
            method=method,
        )
        spec.add(crotch_spheres)
        crotch = crotch_grid[(0, 0)]
        spec.connect(lower_middle[bottom], crotch)

    # ------------------------------------------------------------------------ neck
    neck_spheres, neck_links = chain(
        body, "neck", "head", design.neck, name="neck", group="neck",
        vertex_mask=limb_mask(body, ("neck", "head"), weight_min=0.4),
        half_thickness=0.022, method=method,
    )
    spec.add(neck_spheres, neck_links)
    top = upper_rows - 1
    spec.connect("neck_0", upper_middle[top], upper_left[top])

    # ------------------------------------------------------------- left arm (mirrored)
    # No bumper crosses the elbow: chains stop short on both sides and meet through one
    # short cone. Across the elbow rigid binding is 260 mm RMS; inside a bone it is free.
    upper_arm, upper_arm_links = chain(
        body, "L_shoulder", "L_elbow", design.upper_arm,
        name="L_upperarm", group="L_arm",
        vertex_mask=limb_mask(body, ("L_shoulder", "L_elbow")), method=method,
    )
    fore, fore_links = chain(
        body, "L_elbow", "L_wrist", design.forearm,
        name="L_forearm", group="L_arm",
        vertex_mask=limb_mask(body, ("L_elbow", "L_wrist")), method=method,
    )
    spec.add(upper_arm, upper_arm_links)
    spec.add(fore, fore_links)
    spec.connect(f"L_upperarm_{len(design.upper_arm) - 1}", "L_forearm_0")
    wrist = stub(body, "L_elbow", "L_wrist", name="L_wrist_stub", group="L_arm",
                 beyond=0.008, vertex_mask=limb_mask(body, ("L_wrist", "L_elbow")))
    spec.add([wrist], [(f"L_forearm_{len(design.forearm) - 1}", "L_wrist_stub")])

    deltoid, _ = chain(
        body, "L_collar", "L_shoulder", (design.deltoid,),
        name="L_deltoid", group="L_arm",
        vertex_mask=limb_mask(body, ("L_collar", "L_shoulder")), method=method,
    )
    spec.add(deltoid, [("L_deltoid_0", "L_upperarm_0")])
    clavicle, _ = chain(
        body, "L_collar", "L_shoulder", (design.clavicle,),
        name="L_clavicle", group="trunk",
        vertex_mask=limb_mask(body, ("L_collar", "spine3", "L_shoulder")), method=method,
    )
    # No explicit clavicle-deltoid double: that edge is the rim of the triangle below and
    # listing it again would only duplicate the cone in the engine file.
    spec.add(clavicle)
=======

    # The two structures meet across a change of cross-section -- a 4-ring below, a 3-wide
    # row above -- so the seam is two quads on the flanks and two triangles closing the
    # front and the back.
    if ladder:
        spec.connect(rung(0, 3), upper_middle[0], upper_left[0])
        spec.connect(rung(0, 3), rung(0, 0), upper_middle[0])
    else:
        spec.connect(rung(0, 3), rung(0, 0), upper_left[0], upper_middle[0])
        spec.connect(rung(0, 3), rung(0, 2), upper_middle[0])
        spec.connect(rung(0, 0), rung(0, 1), upper_middle[0])

    # ------------------------------------------------------------------ lower pelvis
    crotch_spheres, crotch_grid = sheet(
        body, CROTCH_ROW, (0.0,), name="crotch", column_names=("M",),
        group="trunk", vertex_mask=crotch_mask(body), half_thickness=0.024,
    )
    spec.add(crotch_spheres)
    crotch = crotch_grid[(0, 0)]
    spec.connect(rung(bottom, 0), crotch)

    # ------------------------------------------------------------------------ neck
    # The head is truncated: two spheres carry the neck and the chain stops. The neck+head
    # skinning does not reach below y = 0.281, so the stations sit at and just above the
    # neck joint -- placing them lower puts the section inside the upper chest, where the
    # nearest surface is 100 mm away and the sphere bursts out of the throat. 31 mm of
    # protrusion over 94% of the neck, in the first attempt.
    #
    # Note the neck can never be tracked exactly: no SMPL vertex assigns the neck joint more
    # than 0.71 of its skinning, so a rigidly bound sphere there is always approximate.
    neck_vertices = limb_mask(body, ("neck", "head"), weight_min=0.4)
    neck_spheres, neck_links = chain(
        body, "neck", "head", (0.02, 0.42), name="neck", group="neck",
        vertex_mask=neck_vertices, half_thickness=0.022,
    )
    spec.add(neck_spheres, neck_links)
    top = upper_rows - 1
    spec.connect(upper_middle[top], "neck_0")
    spec.connect("neck_0", upper_middle[top], upper_left[top])

    # ------------------------------------------------------------- left arm (mirrored)
    # The chains stop short of the elbow on both sides and meet through a single cone. No
    # bumper crosses the elbow: it is the worst hinge on the body, median local rotation
    # 132 degrees, and a parent-bound bumper is already 77 mm out (p95) two centimetres past
    # the joint, 483 mm out at the wrist.
    upper_arm, upper_arm_links = chain(
        body, "L_shoulder", "L_elbow", UPPER_ARM,
        name="L_upperarm", group="L_arm",
        vertex_mask=limb_mask(body, ("L_shoulder", "L_elbow")),
    )
    fore, fore_links = chain(
        body, "L_elbow", "L_wrist", FOREARM,
        name="L_forearm", group="L_arm",
        vertex_mask=limb_mask(body, ("L_elbow", "L_wrist")),
    )
    spec.add(upper_arm, upper_arm_links)
    spec.add(fore, fore_links)
    spec.connect("L_upperarm_3", "L_forearm_0")
    wrist = stub(body, "L_elbow", "L_wrist", name="L_wrist_stub", group="L_arm",
                 beyond=0.008, vertex_mask=limb_mask(body, ("L_wrist", "L_elbow")))
    spec.add([wrist], [("L_forearm_3", "L_wrist_stub")])

    # The deltoid gets its own sphere and cannot be folded into the shaft chain: the
    # section flares 1.8x from the shoulder joint to t = 0.92, and its minimum-width annulus
    # is 15.5 mm against 2.6-5.3 mm along the shaft.
    deltoid, _ = chain(
        body, "L_collar", "L_shoulder", (0.70,),
        name="L_deltoid", group="L_arm",
        vertex_mask=limb_mask(body, ("L_collar", "L_shoulder")),
    )
    spec.add(deltoid, [("L_deltoid_0", "L_upperarm_0")])

    # One sphere on the inner clavicle. Without it the only bridge from the chest to the arm
    # is a single long cone over the hollow of the armpit, 27 mm off the skin at worst.
    clavicle, _ = chain(
        body, "L_collar", "L_shoulder", (0.20,),
        name="L_clavicle", group="trunk",
        vertex_mask=limb_mask(body, ("L_collar", "spine3", "L_shoulder")),
    )
    spec.add(clavicle, [("L_clavicle_0", "L_deltoid_0")])
>>>>>>> Stashed changes
    spec.connect("L_clavicle_0", upper_left[0])
    spec.connect("L_clavicle_0", upper_left[1], upper_left[2])
    spec.connect("L_deltoid_0", "L_clavicle_0", upper_left[1])
    spec.connect("L_deltoid_0", upper_left[0], upper_left[1])

    # ------------------------------------------------------------- left leg (mirrored)
    hip, hip_links = chain(
<<<<<<< Updated upstream
        body, "L_hip", "L_knee", design.hip, name="L_hip", group="L_leg",
        vertex_mask=pelvis_side_mask(body), half_thickness=0.022, method=method,
=======
        body, "L_hip", "L_knee", HIP, name="L_hip", group="L_leg",
        vertex_mask=pelvis_side_mask(body), half_thickness=0.022,
>>>>>>> Stashed changes
    )
    spec.add(hip, hip_links)
    hip_joint = body.joints[JOINT_INDEX["L_hip"]]
    glute, glute_links = blobs(
        body,
<<<<<<< Updated upstream
        (design.glute_box_low[0], hip_joint[1] + design.glute_box_low[1], design.glute_box_low[2]),
        (design.glute_box_high[0], hip_joint[1] + design.glute_box_high[1], design.glute_box_high[2]),
        design.glute_count, name="L_glute", group="L_leg",
        min_separation=design.glute_separation,
    )
    spec.add(glute, glute_links)
    thigh, thigh_links = chain(
        body, "L_hip", "L_knee", design.thigh,
        name="L_thigh", group="L_leg", vertex_mask=limb_mask(body, ("L_hip", "L_knee")),
        method=method,
    )
    shin, shin_links = chain(
        body, "L_knee", "L_ankle", design.shin,
        name="L_shin", group="L_leg", vertex_mask=limb_mask(body, ("L_knee", "L_ankle")),
        method=method,
    )
    spec.add(thigh, thigh_links)
    spec.add(shin, shin_links)
    last_thigh = f"L_thigh_{len(design.thigh) - 1}"
    last_shin = f"L_shin_{len(design.shin) - 1}"
    spec.connect(last_thigh, "L_shin_0")
    ankle = stub(body, "L_knee", "L_ankle", name="L_ankle_stub", group="L_leg",
                 beyond=0.008, vertex_mask=limb_mask(body, ("L_ankle", "L_knee")))
    spec.add([ankle], [(last_shin, "L_ankle_stub")])

    # The gluteal band belongs to no chain (12.6% of the body): the pelvis cluster is
    # linked explicitly to the top of the thigh, or the watchdog loses every node that
    # slides off the buttock.
    top_glute = f"L_glute_{design.glute_count - 1}"
    last_hip = f"L_hip_{len(design.hip) - 1}"
    spec.connect("L_glute_0", "L_thigh_0")
    spec.connect(top_glute, "L_hip_0")
    spec.connect("L_glute_1", last_hip)
    spec.connect(last_hip, "L_thigh_0")
    spec.connect("L_hip_0", lower_left[bottom], lower_left[bottom - 1])
    spec.connect("L_hip_0", lower_middle[bottom])
    if crotch is not None:
        spec.connect("L_thigh_0", crotch)
        spec.connect(last_hip, crotch)
    else:
        spec.connect("L_thigh_0", lower_middle[bottom])
    spec.connect(top_glute, lower_middle[bottom])
    spec.connect(last_hip, lower_left[bottom])
=======
        (GLUTE_BOX_LOW[0], hip_joint[1] + GLUTE_BOX_LOW[1], GLUTE_BOX_LOW[2]),
        (GLUTE_BOX_HIGH[0], hip_joint[1] + GLUTE_BOX_HIGH[1], GLUTE_BOX_HIGH[2]),
        GLUTE_COUNT, name="L_glute", group="L_leg", min_separation=0.055,
    )
    spec.add(glute, glute_links)

    thigh, thigh_links = chain(
        body, "L_hip", "L_knee", THIGH,
        name="L_thigh", group="L_leg", vertex_mask=limb_mask(body, ("L_hip", "L_knee")),
    )
    shin, shin_links = chain(
        body, "L_knee", "L_ankle", SHIN,
        name="L_shin", group="L_leg", vertex_mask=limb_mask(body, ("L_knee", "L_ankle")),
    )
    spec.add(thigh, thigh_links)
    spec.add(shin, shin_links)
    spec.connect(f"L_thigh_{len(THIGH) - 1}", "L_shin_0")
    ankle = stub(body, "L_knee", "L_ankle", name="L_ankle_stub", group="L_leg",
                 beyond=0.008, vertex_mask=limb_mask(body, ("L_ankle", "L_knee")))
    spec.add([ankle], [(f"L_shin_{len(SHIN) - 1}", "L_ankle_stub")])

    # The pelvis cluster has to be explicitly connected to the top thigh sphere, or the
    # watchdog loses every node that slides off the buttock onto the thigh -- and that band
    # of skin is an eighth of the whole body.
    top_glute = f"L_glute_{GLUTE_COUNT - 1}"
    spec.connect("L_glute_0", "L_thigh_0")
    spec.connect(top_glute, "L_hip_0")
    spec.connect("L_glute_1", f"L_hip_{len(HIP) - 1}")
    spec.connect(f"L_hip_{len(HIP) - 1}", "L_thigh_0")
    spec.connect("L_hip_0", rung(bottom, 3), rung(bottom - 1, 3))
    # A cone, not a triangle: radii of 85 / 126 / 135 mm over this span admit no plane
    # tangent to all three, so the slab would have no flat face and the engine would
    # silently drop both caps. The cone itself is well conditioned (100 mm apart, 50 mm of
    # radius difference) and gives the walk the same short-cut.
    spec.connect("L_hip_0", rung(bottom, 0))
    # Likewise: 77 / 85 / 126 mm radii over this spacing admit no tangent plane. The two
    # cones are already there, and cone-sphere-cone is the same two hops the slab would
    # have given, so nothing is lost from the walk either.
    spec.connect("L_thigh_0", crotch)
    spec.connect(f"L_hip_{len(HIP) - 1}", crotch)
    # Two plain cones that exist purely to shorten the walk. Measured before adding them:
    # a node sliding off the buttock onto the pelvis was 5 hops from where it came, and one
    # crossing from the clavicle to the chest seam was 5 as well.
    spec.connect(top_glute, rung(bottom, 0))
    spec.connect(f"L_hip_{len(HIP) - 1}", rung(bottom, 3))
>>>>>>> Stashed changes

    return mirror(spec)


<<<<<<< Updated upstream
# --------------------------------------------------------------------------- pipeline


@dataclass
class BuildResult:
    spec: TopologySpec
    centers: np.ndarray
    radii: np.ndarray
    connections: list[tuple[int, ...]]
    names: list[str]
    report: object
    log: dict = field(default_factory=dict)


def fit_poses(all_poses: np.ndarray) -> np.ndarray:
    """A small, fixed subset of the validation poses for the fitting objective.

    Two of the random draws and two mocap frames, chosen as the worst offenders of the
    session-1 sweep. Every pose in the objective multiplies the cost of an evaluation, so
    the set is kept small and the tolerance does the discriminating.
    """
    picks = [index for index in (12, 19, 27, 45) if index < len(all_poses)]
    return all_poses[picks]


def build(
    body: Body,
    design: Design,
    *,
    poses: np.ndarray | None,
    quick: bool = False,
    verbose: bool = True,
    thresholds: Thresholds | None = None,
) -> BuildResult:
    """Run the whole pipeline and return everything the caller may want to write."""
    log: dict = {"design": design.to_dict(), "timings": {}}
    clock = time.perf_counter()

    def lap(label: str) -> None:
        nonlocal clock
        now = time.perf_counter()
        log["timings"][label] = round(now - clock, 1)
        clock = now

    modelled = modelled_vertex_mask(body)
    modelled_index = np.flatnonzero(modelled)
    remap = -np.ones(len(body.vertices), dtype=np.int64)
    remap[modelled_index] = np.arange(len(modelled_index))
    edges = surface_edges(body.faces)
    edges = edges[(remap[edges[:, 0]] >= 0) & (remap[edges[:, 1]] >= 0)]
    edges = remap[edges]

    # ------------------------------------------------------------------------ place
    spec = build_spec(body, design)
    centers, radii, connections, names = spec.assemble()
    centers, radii = symmetrize(centers, radii, names)
    for sphere, center, radius in zip(spec.spheres, centers, radii):
        sphere.center, sphere.radius = center, float(radius)
    placed_parent, _, _ = assign_parent_joint_like_tool(centers, body.joints, body.parents)
    intended_joint = {name: int(joint) for name, joint in zip(names, placed_parent)}
    # Every fit measures its bounds from the original placement (fills anchor at their
    # proposal), so the quick fit, the fill refits and the full fit cannot compound into
    # a drift of several times max_offset. Measured before this: a deltoid sphere 20 mm
    # from its placement after nine refits, and a shoulder region that then collapsed.
    anchor = {name: (center.copy(), float(radius)) for name, center, radius in zip(names, centers, radii)}

    def anchors_for(current: TopologySpec, current_names):
        for sphere in current.spheres:
            if sphere.name not in anchor:
                anchor[sphere.name] = (sphere.center.copy(), float(sphere.radius))
        anchor_centers = np.array([anchor[name][0] for name in current_names])
        anchor_radii = np.array([anchor[name][1] for name in current_names])
        return anchor_centers, anchor_radii
    log["placed"] = {
        "spheres": len(centers), "summary": spec.summary(),
        "bindings": {name: JOINT_NAMES[joint] for name, joint in intended_joint.items()},
    }
    if verbose:
        print(f"place     {spec.summary()}")
    lap("place")

    # ------------------------------------------------------------------- quick fit
    quick_options = FitOptions(
        max_offset=design.max_offset, radius_scale_bounds=design.radius_scale_bounds,
        protrusion_weight=design.protrusion_weight, point_stride=3,
        max_iterations=25 if quick else 40, verbose=False, binding_rounds=2,
    )

    def quick_refit(current: TopologySpec) -> tuple[np.ndarray, np.ndarray]:
        current_centers, current_radii, current_connections, current_names = current.assemble()
        anchor_centers, anchor_radii = anchors_for(current, current_names)
        result = fit_topology(
            body, current_centers, current_radii, current_connections, current_names,
            vertex_mask=modelled, options=quick_options,
            anchor_centers=anchor_centers, anchor_radii=anchor_radii,
        )
        return result.centers, result.radii

    fitted = fit_topology(
        body, centers, radii, connections, names, vertex_mask=modelled, options=quick_options,
    )
    for sphere, center, radius in zip(spec.spheres, fitted.centers, fitted.radii):
        sphere.center, sphere.radius = center, float(radius)
    log["quick_fit"] = {key: value for key, value in fitted.info.items() if key != "history"}
    if verbose:
        last = fitted.info["history"][-1]
        print(
            f"quickfit  loss {last['loss']:.3e}, worst gap {last['max_gap'] * 1000:.1f} mm, "
            f"worst protrusion {last['max_protrusion'] * 1000:.1f} mm "
            f"({fitted.info['seconds']:.0f} s)"
        )
    lap("quick_fit")

    # ------------------------------------------------------------------ repair
    pose_set = None
    if poses is not None and len(poses) and design.repair_posed_tolerance:
        pose_set = PoseSet.from_thetas(
            body, fit_poses(poses), modelled, tolerance=design.repair_posed_tolerance,
        )
    spec, link_log = link_walk_breaks(
        body, spec, vertex_mask=modelled, edges=edges, walk_budget=design.walk_budget,
        max_links=design.link_max, protrusion_tolerance=design.link_protrusion_tolerance,
        verbose=verbose, poses=pose_set,
    )
    log["links"] = link_log
    lap("link")

    spec, fill_log = fill_gaps(
        body, spec, vertex_mask=modelled, refit=quick_refit,
        max_spheres=design.max_spheres, gap_threshold=design.gap_threshold,
        max_rounds=min(4, design.fill_rounds) if quick else design.fill_rounds,
        protrusion_tolerance=design.fill_protrusion_tolerance, verbose=verbose,
        poses=pose_set,
    )
    log["fills"] = fill_log
    lap("fill")

    # Links again: the new spheres changed ownership around them.
    spec, link_log_2 = link_walk_breaks(
        body, spec, vertex_mask=modelled, edges=edges, walk_budget=design.walk_budget,
        max_links=max(2, design.link_max // 2),
        protrusion_tolerance=design.link_protrusion_tolerance, verbose=verbose,
        poses=pose_set,
    )
    log["links_after_fill"] = link_log_2
    lap("link_2")

    # ---------------------------------------------------------------- full fit
    centers, radii, connections, names = spec.assemble()
    anchor_centers, anchor_radii = anchors_for(spec, names)
    full_options = FitOptions(
        max_offset=design.max_offset, radius_scale_bounds=design.radius_scale_bounds,
        protrusion_weight=design.protrusion_weight,
        poses=None if (quick or poses is None) else fit_poses(poses),
        pose_weight=design.pose_weight, pose_tolerance=design.pose_tolerance,
        pose_vertex_stride=4,
        point_stride=2 if quick else 1, max_iterations=60 if quick else 150,
        verbose=verbose, binding_rounds=3,
    )
    # Multi-start: the one-sided loss has more than one basin at the shoulder, and which
    # one L-BFGS-B lands in depends on the state the fill loop hands over (measured: the
    # same design reached RMS 5.3 mm from one start and 9.2 mm from another). Fit from the
    # filled state and from the original placement, keep the lower loss.
    starts = [("continued", centers, radii), ("from_placement", anchor_centers, anchor_radii)]
    fitted = None
    log["full_fit_starts"] = {}
    for label, start_centers, start_radii in starts:
        candidate = fit_topology(
            body, start_centers, start_radii, connections, names, vertex_mask=modelled,
            options=dataclasses.replace(full_options, verbose=False),
            anchor_centers=anchor_centers, anchor_radii=anchor_radii,
        )
        candidate_loss = candidate.info["rounds"][-1]["loss"]
        log["full_fit_starts"][label] = {
            "loss": candidate_loss,
            "max_gap_mm": round(candidate.info["history"][-1]["max_gap"] * 1000, 1),
            "max_protrusion_mm": round(candidate.info["history"][-1]["max_protrusion"] * 1000, 1),
        }
        if verbose:
            print(
                f"fit       start {label:<15} loss {candidate_loss:.4e}, worst gap "
                f"{candidate.info['history'][-1]['max_gap'] * 1000:.1f} mm, worst protrusion "
                f"{candidate.info['history'][-1]['max_protrusion'] * 1000:.1f} mm"
                + ("" if not candidate.info["binding_changed"]
                   else f", unresolved binding changes {candidate.info['binding_changed']}")
            )
        if fitted is None or candidate_loss < fitted.info["rounds"][-1]["loss"]:
            fitted = candidate
            log["full_fit_start"] = label
    centers, radii = fitted.centers, fitted.radii
    for sphere, center, radius in zip(spec.spheres, centers, radii):
        sphere.center, sphere.radius = center, float(radius)
    log["full_fit"] = {key: value for key, value in fitted.info.items() if key != "history"}
    log["full_fit"]["history_tail"] = fitted.info["history"][-3:]
    lap("full_fit")

    # The fit moved ownership boundaries; one last, stricter linking pass and a short
    # refit that lets the radii absorb whatever the new cones poke out.
    spec, link_log_3 = link_walk_breaks(
        body, spec, vertex_mask=modelled, edges=edges, walk_budget=design.walk_budget,
        max_links=max(2, design.link_max // 2),
        protrusion_tolerance=design.link_protrusion_tolerance * 0.5, verbose=verbose,
        poses=pose_set,
    )
    log["links_after_fit"] = link_log_3
    if any("cone" in entry for entry in link_log_3):
        centers, radii, connections, names = spec.assemble()
        polish_options = dataclasses.replace(
            full_options, max_iterations=30 if quick else 60, verbose=False,
        )
        anchor_centers, anchor_radii = anchors_for(spec, names)
        fitted = fit_topology(
            body, centers, radii, connections, names, vertex_mask=modelled,
            options=polish_options, anchor_centers=anchor_centers, anchor_radii=anchor_radii,
        )
        centers, radii = fitted.centers, fitted.radii
        for sphere, center, radius in zip(spec.spheres, centers, radii):
            sphere.center, sphere.radius = center, float(radius)
        log["polish_fit"] = {key: value for key, value in fitted.info.items() if key != "history"}
    lap("link_3")

    # ------------------------------------------------------------------ validate
    # Spheres added by the repairs were bound when they were proposed; everything else
    # must still bind to the joint the placement rules chose for it.
    for entry in log.get("fills", []):
        if "name" in entry:
            intended_joint[entry["name"]] = JOINT_INDEX[entry["binding"]]
            if entry["name"].startswith("L_"):
                # The twin binds by the tool's rule at its own (mirrored) centre; SMPL's
                # joints are not exactly symmetric, so near a tie it may differ from the
                # mirror of the left binding, and that is what the tool will do too.
                twin_center = np.asarray(entry["center"], dtype=np.float64) * np.array([-1.0, 1.0, 1.0])
                intended_joint["R_" + entry["name"][2:]] = int(assign_parent_joint_like_tool(
                    twin_center[None], body.joints, body.parents
                )[0][0])
    # A triangle whose radii the fit moved apart faster than its spacing has no flat
    # face any more; the engine would carry a slab that covers nothing.
    log["degenerate_slabs_replaced"] = replace_degenerate_slabs(spec, centers, radii, verbose=verbose)
    centers, radii, connections, names = spec.assemble()
    lap("slabs")

    # A sphere the fit let slide between two compatible trunk joints is bound where the
    # tool will bind it; only crossings the fit refused count as unintended.
    final_parent, _, _ = assign_parent_joint_like_tool(centers, body.joints, body.parents)
    for index, name in enumerate(names):
        old_joint, new_joint = intended_joint[name], int(final_parent[index])
        if old_joint != new_joint and old_joint in TRUNK_JOINTS and new_joint in TRUNK_JOINTS \
                and joints_compatible(old_joint, new_joint, body.parents):
            intended_joint[name] = new_joint
    expected_parent = np.array([intended_joint[name] for name in names])
    report = validate(
        centers, radii, connections, body=body, thresholds=thresholds or Thresholds(),
        poses=None if quick else poses, names=names, expected_parent_joints=expected_parent,
        walk_budget=design.walk_budget,
    )
    lap("validate")
    return BuildResult(spec, centers, radii, connections, list(names), report, log)


# ----------------------------------------------------------------------------- output


def write_outputs(result: BuildResult, body: Body, name: str, gender: str, *, render: bool) -> dict:
    from src.topology.validate import format_report

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = build_payload(body, result.centers, result.radii, result.connections)
    payload["generated_by"] = {
        "script": "scripts/build_topology.py",
        "gender": gender,
        "sphere_names": result.names,
        "groups": [sphere.group for sphere in result.spec.spheres],
        "design": result.log.get("design", {}),
    }
    written = {
        "json": str(write_tool_json(OUTPUT_DIR / f"{name}.json", payload)),
        "txt": str(write_engine_text(
            OUTPUT_DIR / f"{name}.txt", result.centers, result.radii, result.connections
        )),
    }
    text = [
        f"topology  {name}  ({gender})",
        f"spec      {result.spec.summary()}",
        "",
        format_report(result.report, names=result.names),
    ]
    if result.log.get("links"):
        text += ["", "walk links added:"]
        for entry in (result.log["links"] + result.log.get("links_after_fill", [])
                      + result.log.get("links_after_fit", [])):
            if "cone" in entry:
                text.append(
                    f"  cone[{entry['cone'][0]}, {entry['cone'][1]}]  removed "
                    f"{entry['steps_removed']} expensive steps, protrusion {entry['protrusion_mm']} mm"
                )
    if result.log.get("fills"):
        text += ["", "gap fills added:"]
        for entry in result.log["fills"]:
            if "name" in entry:
                text.append(
                    f"  {entry['name']:<12} r={entry['radius_mm']:5.1f} mm  {entry['binding']:<10} "
                    f"cluster {entry['cluster_joint']} x{entry['cluster_size']} "
                    f"(worst {entry['cluster_worst_mm']} mm)  links {entry['links']}"
                )
    text += ["", "timings (s): " + json.dumps(result.log.get("timings", {}))]
    report_path = OUTPUT_DIR / "reports" / f"{name}.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(text) + "\n", encoding="utf-8")
    written["report"] = str(report_path)

    log_path = OUTPUT_DIR / f"{name}_log.json"
    log_path.write_text(json.dumps(result.log, indent=1, default=_json_default), encoding="utf-8")
    written["log"] = str(log_path)

    if render:
        try:
            from src.topology.render import render_topology

            paths = render_topology(
                body, result.centers, result.radii, result.connections,
                OUTPUT_DIR / "renders", name, names=result.names,
                groups=[sphere.group for sphere in result.spec.spheres],
            )
            written["renders"] = [str(path) for path in paths]
        except Exception as error:  # noqa: BLE001 - rendering must never block the output
            written["renders_error"] = repr(error)
    return written


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return str(value)


# -------------------------------------------------------------------------------- main


=======
>>>>>>> Stashed changes
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gender", default="male", choices=("male", "female"))
    parser.add_argument("--name", default=None, help="Output stem; defaults to auto_<gender>")
<<<<<<< Updated upstream
    parser.add_argument("--design", type=Path, default=None,
                        help="JSON file of Design parameters overriding the defaults")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="Override one design parameter, e.g. --set protrusion_weight=8")
    parser.add_argument("--quick", action="store_true",
                        help="No poses, short fits, no gap filling: a few seconds")
    parser.add_argument("--no-poses", action="store_true", help="Skip the pose sweep")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--mocap-stride", type=int, default=5)
    args = parser.parse_args()

    design_values = {}
    if args.design is not None:
        design_values.update(json.loads(args.design.read_text(encoding="utf-8")))
    for override in args.set:
        key, _, raw = override.partition("=")
        design_values[key.strip()] = json.loads(raw)
    design = Design.from_dict(design_values)

=======
    parser.add_argument("--max-offset", type=float, default=0.010,
                        help="Centre displacement cap for the least-squares pass, metres")
    parser.add_argument("--protrusion-target", type=float, default=0.008,
                        help="Shrink radii until no bumper pokes further than this, metres")
    parser.add_argument("--min-radius-fraction", type=float, default=0.60,
                        help="Never let a sphere end below this fraction of its placed radius")
    parser.add_argument("--no-refine", action="store_true", help="Skip optimize_spheres")
    parser.add_argument("--no-relieve", action="store_true", help="Skip the shrink pass")
    parser.add_argument(
        "--lower-structure", default="ladder", choices=("ring", "ladder"),
        help="Cross-section family for the lower trunk. Measured on the male body, the "
             "ladder wins on every count -- 4.62%% of vertices beyond 15 mm against 4.87, "
             "walk-break 1.62%% against 1.71, protrusion 14.7 mm against 14.8 -- with ten "
             "fewer spheres and fourteen fewer quads. The ring's per-section advantage does "
             "not survive into the 3D union, because the ladder's left-to-right slabs "
             "already reach the front and back walls that the ring buys separately",
    )
    parser.add_argument("--no-poses", action="store_true", help="Skip the pose sweep")
    parser.add_argument("--mocap-stride", type=int, default=5)
    args = parser.parse_args()

>>>>>>> Stashed changes
    name = args.name or f"auto_{args.gender}"
    body = load_body(args.gender)
    print(f"body      {args.gender}, neutral shape, T-pose, {len(body.vertices)} vertices")

<<<<<<< Updated upstream
    poses = None
    if not (args.no_poses or args.quick):
        poses = np.concatenate([random_poses(), mocap_poses(stride=args.mocap_stride)], axis=0)

    result = build(body, design, poses=poses, quick=args.quick)
    written = write_outputs(result, body, name, args.gender, render=not args.no_render)

    print()
    print(result.report.text())
    print()
    for key, value in written.items():
        print(f"wrote     {key}: {value}")
    return 0 if result.report.passed else 1
=======
    spec = build_spec(body, lower_structure=args.lower_structure)
    centers, radii, connections, names = spec.assemble()
    print(f"spec      {spec.summary()}")

    modelled = modelled_vertex_mask(body)
    centers, radii = symmetrize(centers, radii, names)
    placed_radii = radii.copy()

    if not args.no_refine:
        centers, radii, info = refine(
            body, centers, radii, connections,
            max_offset=args.max_offset, vertex_mask=modelled,
        )
        print(
            f"refine    {len(info['iterations'])} outer iterations over "
            f"{info['band_size']} band vertices; centres moved "
            f"{info['mean_displacement'] * 1000:.1f} mm mean / "
            f"{info['max_displacement'] * 1000:.1f} mm max"
        )
        centers, radii = symmetrize(centers, radii, names)
        # Floor every radius against the one placement gave it. The least-squares pass fits
        # each sphere to the vertices nearest its own centre, so a sphere covering a small
        # or awkward patch -- the pubic node, whose partition is a handful of vertices in a
        # concavity -- gets fitted to that handful and collapses. Measured: 68 mm placed,
        # 45 mm after the fit, 27 mm after the shrink pass floored against the fitted value.
        # It is then swallowed by its neighbour and stops being a bumper at all.
        shrunk = radii < placed_radii * args.min_radius_fraction
        if shrunk.any():
            print(
                f"clamp     {int(shrunk.sum())} sphere(s) refitted below "
                f"{args.min_radius_fraction:.0%} of their placed radius, restored"
            )
        radii = np.maximum(radii, placed_radii * args.min_radius_fraction)

    if not args.no_relieve:
        radii, relief = relieve_protrusion(
            body, centers, radii, connections,
            vertex_mask=modelled, target=args.protrusion_target,
            floor=placed_radii * args.min_radius_fraction,
        )
        print(
            f"relieve   {relief['iterations']} shrink steps, worst protrusion now "
            f"{relief['final_protrusion'] * 1000:.1f} mm"
            + ("" if relief["converged"] else "  (DID NOT CONVERGE)")
        )
        centers, radii = symmetrize(centers, radii, names)

    poses = None
    if not args.no_poses:
        poses = np.concatenate(
            [random_poses(), mocap_poses(stride=args.mocap_stride)], axis=0
        )

    report = validate(
        centers, radii, connections, body=body,
        thresholds=Thresholds(), poses=poses,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = build_payload(body, centers, radii, connections)
    payload["generated_by"] = {
        "script": "scripts/build_topology.py",
        "gender": args.gender,
        "sphere_names": names,
        "groups": [sphere.group for sphere in spec.spheres],
    }
    json_path = write_tool_json(OUTPUT_DIR / f"{name}.json", payload)
    text_path = write_engine_text(OUTPUT_DIR / f"{name}.txt", centers, radii, connections)

    lines = [
        f"topology  {name}  ({args.gender})",
        f"spec      {spec.summary()}",
        "",
        report.text(),
        "",
        "walk hop histogram   " + json.dumps(report.stats["walk"]["histogram"]),
        "",
        f"  {'region':<12} {'verts':>6} {'rms':>7} {'gap':>8} {'protr':>8} {'walkbrk':>8}",
    ]
    for entry in report.stats["by_region"]:
        lines.append(
            f"  {entry['joint']:<12} {entry['vertices']:6d} {entry['rms_mm']:7.1f} "
            f"{entry['max_gap_mm']:8.1f} {entry['max_protrusion_mm']:8.1f} "
            f"{entry['walk_break_pct']:7.2f}%"
        )
    if report.worst.get("walk_breaks"):
        lines += ["", "worst walk breaks:"]
        for entry in report.worst["walk_breaks"][:10]:
            lines.append(f"  {entry['pair'][0]} <-> {entry['pair'][1]}  x{entry['edges']}")
    if report.worst.get("non_adjacent_overlap"):
        lines += ["", "worst unlinked overlaps:"]
        for entry in report.worst["non_adjacent_overlap"][:10]:
            lines.append(f"  {entry['a']} into {entry['b']}: {entry['fraction'] * 100:.1f}%")
    if report.worst.get("poses"):
        lines += ["", "worst poses:"]
        for entry in report.worst["poses"][:6]:
            lines.append(
                f"  pose {entry['pose']:3d}  protrusion {entry['max_protrusion'] * 1000:6.1f} mm "
                f"({entry['protrusion_fraction'] * 100:5.2f}% of verts), "
                f"gap {entry['max_gap'] * 1000:6.1f} mm"
            )

    report_path = OUTPUT_DIR / "reports" / f"{name}.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print(report.text())
    print()
    print(f"wrote     {json_path}")
    print(f"          {text_path}")
    print(f"          {report_path}")
    return 0 if report.passed else 1
>>>>>>> Stashed changes


if __name__ == "__main__":
    raise SystemExit(main())
