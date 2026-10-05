# Automated topology rebuild — working notes

> **Session 2 (2026-09-03, in progress).** The session-1 code was recovered from the git
> stash (it had never been committed) and is being rebuilt on. What changed, in order of
> importance: (1) a fitter that optimises centres *and* radii against the **exact**
> sphere-mesh distance with a one-sided protrusion loss, symmetric by construction,
> replacing `optimize_spheres` + the shrink pass + the radius floors; (2) automatic repair
> loops that spend the spare sphere budget where the validator finds gaps and add cones
> where the watchdog walk is too long; (3) mesh-plane cross-sections instead of vertex
> bands; (4) a parameter search over the design constants; (5) off-screen renders so the
> result is *looked at*; (6) validator: binding audit, one-bone-per-cone, error clusters
> with location and facing, a recalibrated overlap metric, named bumpers. Numbers below
> are updated as they come in; everything above the session-1 line is the new state.

## Session 2 — first measurements (old placement, new fitter and repairs)

Same 3725 modelled vertices, exact metric, rest pose. Starting from the session-1
placement (79 spheres) and changing only what happens *after* placement:

| step | spheres | RMS | worst gap | >15 mm off | worst protrusion | inside >5 mm | walk-break |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hand-made template | 60 | 6.12 | 35.5 | 0.83 % | 44.9 | 8.63 % | 3.68 % |
| session 1 (`optimize_spheres` + shrink) | 79 | 6.64 | 32.2 | 3.54 % | 15.9 | 4.42 % | 1.83 % |
| exact-SDF fit only (κ = 6, δ = 12 mm) | 79 | 7.01 | 35.9 | 3.70 % | 9.0 | 0.17 % | — |
| + 2 walk links, 9 gap fills, 4 more links, refit | **96** | **5.29** | **23.8** | **0.53 %** | **6.9** | **0.03 %** | 2.12 % |

The fitter alone trades gaps for protrusion (that is what a one-sided loss does); the
gap fills buy the gaps back and more. This already beats the hand-made template on every
rest metric. Timing: one exact forward+backward over all 3725 vertices is 0.24 s, a full
150-iteration fit 19 s, the whole repair sequence about a minute.

Where the filler put its spheres, in order: pubic front (74 mm, pelvis), front of the
thigh twice (the quadriceps bulge is not round), kneecap, lateral chest under the armpit
(43 mm, collar), inner thigh top, elbow tip (29 mm), lower chest. Walk links it added:
glute-to-thigh, crotch-to-glute, deltoid-to-upper-arm, chest-to-armpit fill.

## Session 2 — the new pipeline, quick mode (no poses, 4 fill rounds, short fits)

`scripts/build_topology.py --quick`: place, exact-SDF fit, link walk breaks, fill gaps
(each fill re-fitted), link again, fit again, link once more with a stricter protrusion
limit and polish. About one minute end to end.

| sections | spheres | RMS | worst gap | >15 mm off | worst protrusion | inside >5 mm | walk-break | degenerate | bindings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| vertex band (session 1 method) | 87 | 5.8 | 22.6 | 1.22 % | 7.8 | 0.11 % | 2.05 % | 2 (crotch swallowed) | all as placed |
| **mesh-plane cut, trimmed by mask** | 86 | 5.7 | 25.4 | 1.24 % | 7.2 | 0.11 % | **1.57 %** | none | all as placed |

The cut wins on walk breaks and protrusion and is now the default. Two things learned
on the way:

- **A plane cut must be trimmed by the mask, not just selected by it.** A horizontal plane
  at shoulder height crosses the trunk and both T-posed arms as one 1.3 m loop; the first
  version chose that loop whole and put the chest columns inside the arms (chest spheres
  bound to `L_elbow`). Keeping only the cut points whose nearer mesh vertex the mask
  accepts recovers exactly the population the band used, with 2-3x the points.
- **A midline sphere below the pelvis joint always binds to one hip.** The tool binds to
  the closest bone; below the pelvis the hip bones are closer than the spine, so the
  session-1 pubic node bound to `L_hip` and its cone to the right thigh spanned two joints
  that rotate independently. The node is gone from the design; the gap filler covers the
  pubis with a sphere high enough (y = -0.318) to bind to the pelvis.

Also new in the fitter since the first table: a containment penalty (no sphere may end
inside another -- the session-1 crotch sphere was being shrunk to 31 mm inside a 130 mm
neighbour), and near-midline fills are pinned to x = 0 instead of being mirrored into two
overlapping twins.

## Session 2 — the design search and the pose term, measured

**Search** (`scripts/search_topology.py`, 32 candidates in two rounds, quick pipeline,
6 workers, 13 minutes): the session-1 constants were already near a local optimum for
this pipeline. The best neighbour scores 11.13 against 11.56 for the defaults (lower is
better), by moving the clavicle sphere outward (0.20 -> 0.29 of the collar bone) and the
shin stations 6 mm up the bone. A larger centre-offset cap (20 mm) buys fidelity (worst
gap 26.9 -> 21.4 mm) but costs walk breaks (0.98 -> 1.36 %). Every evaluation is in
`data/generated/search/evaluations.jsonl`; the winner is `search/best_design.json` and is
what `build_topology.py --design` was run with below.

**The pose term** (protrusion under 4 real poses inside the fit objective), on the
searched design, full pipeline, 54-pose sweep:

| pose weight / tolerance | RMS | worst gap | >15 mm off | worst protrusion | walk-break | posed worst | posed mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **0 (off)** | **5.26** | **24.2** | **0.69 %** | 4.0 | **1.51 %** | 46.1 | 36.4 |
| 0.1 at 12 mm | 5.89 | 22.2 | 1.30 % | 3.9 | 1.83 % | 43.2 | 33.1 |
| 0.25 at 10 mm | 7.31 | 26.9 | 5.78 % | 4.1 | 2.06 % | 39.6 | 30.1 |
| 0.5 at 6 mm (band sections) | 9.42 | 31.6 | 12.50 % | 5.4 | 2.15 % | 30.8 | 22.5 |

Each millimetre of posed protrusion costs several millimetres of rest fidelity, because
the pop-out is not a placement error the fit can move away from: it is the floor of rigid
single-joint binding at the shoulder. The term is off by default and the machinery stays
for the day the tool's secondary-joint blend (commented out at
`src/sphere_manager.py:826`) is enabled.

**Where the posed pop-out comes from**, attributed bumper by bumper over the eight worst
poses (vertices more than 10 mm inside the collider): 329 of them belong to one cone,
`cone[L_upperarm_1, L_deltoid_0]`, added by the walk-break linker (it removed 38
over-budget steps at rest). Its two spheres bind to `L_shoulder` and `L_collar`, and the
skin it owns is 74 % shoulder-weighted, so when the arm rotates the collar-bound end stays
and the cone sweeps 51 mm out of the arm. The deltoid sphere itself is bound correctly --
the skin *it* owns is 60-82 % collar-weighted. Consequence: **the repairs now check every
joint-spanning cone under the fit poses as well** (`repair.PoseSet`, 25 mm tolerance)
before admitting it; the design's own seam cones (deltoid to upper arm, upper arm to
forearm, thigh to shin) are inherent and short.

## Session 2 — what the shoulder taught about binding

The skin under and behind the shoulder is 74 % shoulder-weighted in SMPL, yet in T-pose
it hangs *medial* to the shoulder joint (x = 0.15-0.21 against the joint at x = 0.199).
The tool binds a sphere to its closest bone, so any sphere large enough to cover that
skin from where it sits binds to the **collar**, and when the arm rotates the sphere and
every cone on it stay behind: that is the 45-51 mm pop-out in every pose sweep, and it is
a *binding* disagreement, not a placement error. Three consequences were built in:

1. `repair.fill_gaps` now prefers candidate spheres the tool will bind to the joint that
   dominates the cluster's skinning weights, and only falls back to pure geometry when no
   such candidate exists (`_propose_fill`).
2. Both repairs check a cone whose two spheres bind to different joints under the fit
   poses too (`repair.PoseSet`), because at rest such a cone can look perfect.
3. `fit_topology` no longer freezes a sphere whose binding moved during the fit; it halves
   that sphere's centre box and re-solves, pinning only as a last resort. Freezing three
   spheres at once had left the shoulder top with a 30 mm gap.

Also learned the hard way: a fill proposal must be rejected when most of its surface is
already inside the collider. A 91 mm sphere two centimetres from a 113 mm one, placed for a
six-vertex cluster, made the fitter shrink both to the radius floor and opened the very
gaps the fill was meant to close.

## Session 2 — result: `data/generated/auto_male.json` / `.txt`

```
.venv/Scripts/python.exe scripts/build_topology.py --design data/generated/search/best_design.json
```

**97 spheres · 123 doubles · 14 triples · 12 quads** (engine `.txt`; the `.json` carries
the quads split for the GUI and `quad_groups` to undo it). 3725 modelled vertices, exact
metric, 54-pose sweep, all three files scored by the same `scripts/compare_topologies.py`.

| | hand-made | session 1 | **session 2** |
| --- | ---: | ---: | ---: |
| spheres | 60 | 79 | 97 |
| RMS | 6.12 mm | 6.64 mm | **5.11 mm** |
| worst gap | 35.5 mm | 32.2 mm | **21.7 mm** |
| vertices > 15 mm off | 0.83 % | 3.54 % | **0.66 %** |
| worst protrusion | 44.9 mm | 15.9 mm | **5.0 mm** |
| vertices inside > 5 mm | 8.63 % | 4.42 % | **0.00 %** |
| walk steps needing > 3 hops | 3.68 % | 1.83 % | **1.98 %** |
| unreachable steps | 0 | 0 | 0 |
| harmful overlap (exposed surface under a far bumper) | 10.7 % | 2.1 % | 3.8 % |
| cones spanning unrelated joints | 1 | 0 | **0** |
| posed worst protrusion (54 poses) | 59.8 mm | 46.7 mm | **46.1 mm** |
| exactly mirror-symmetric | no | yes | yes |
| every sphere bound as intended | -- | -- | yes |
| checks passed | 9/17 | 10/17 | **13/17** |

Better than the hand-made template on every row, and better than session 1 on every row
but walk breaks (1.98 % against 1.83 %; both beat the 3.68 % of the hand-made file). The
four failing checks are the aspirational thresholds: worst gap 21.7 mm against 15, walk
breaks 1.98 % against 1 %, and the two posed-protrusion checks, whose floor is the tool's
rigid binding (see the pose-term table above).

Where the 97 spheres went: the session-1 structure (78 after dropping the pubic node), 8
gap fills per side plus one midline pubic sphere (pubis, front of the thigh twice, kneecap,
lower chest side, lower belly, elbow tip, inner thigh top), 8 walk links per side. Every
one of those is a measured decision logged in `data/generated/auto_male_log.json`.

The female body runs through the same rules with no numbers changed
(`--gender female`): 85 spheres, RMS 4.6 mm, 0.60 % of vertices > 15 mm off, worst
protrusion 4.8 mm, walk breaks 1.49 %, posed worst 46.9 mm. One design triangle per side
(clavicle / chest corner) loses its tangent plane after the fit on that body and is
replaced by its three cones automatically (`repair.replace_degenerate_slabs`); the
validator then reports one of those cones as owning no surface, which is the honest
residue of a slab that had none either.

### Female, same rules, no numbers changed

| | hand-made `template_female_1.0` | session 1 | **session 2** |
| --- | ---: | ---: | ---: |
| spheres | 61 | 79 | 85 |
| RMS | 6.17 mm | 7.27 mm | **4.60 mm** |
| worst gap | 28.8 mm | 32.9 mm | 29.3 mm |
| vertices > 15 mm off | 0.93 % | 3.93 % | **0.60 %** |
| worst protrusion | 38.1 mm | 41.5 mm | **4.8 mm** |
| vertices inside > 5 mm | 14.87 % | 6.62 % | **0.00 %** |
| walk steps needing > 3 hops | 2.59 % | 2.47 % | **1.49 %** |
| harmful overlap | 8.8 % | 3.2 % | **1.3 %** |
| cones spanning unrelated joints | 0 | 1 | **0** |
| posed worst protrusion | 51.0 mm | 46.8 mm | 46.9 mm |
| checks passed | 8/17 | 8/17 | **12/17** |

## Session 2 — where it is still short, honestly

- **Posed protrusion 46 mm** (target 20) is the floor of rigid single-joint binding at
  the shoulder, measured three ways: the pose term in the fit buys millimetres of it at a
  cost of centimetres at rest; rejecting joint-spanning cones under pose does not lower
  it (the pop-out is the design's own deltoid seam); and the offending skin is
  shoulder-weighted while lying on the collar side of the binding boundary. The fix is
  the tool's secondary-joint blend (`src/sphere_manager.py:826`), which is the user's
  code, not a topology change. The machinery to exploit it (pose term, posed-cone check,
  binding-aware fills) is in place and off by default.
- **Walk breaks 1.98 %** against a 1 % target. The linker stops when no admissible cone
  removes at least three steps; what remains is the neck seam, the crotch (pubic fill to
  the opposite thigh is a left-to-right cone the mirroring forbids), and the inner-thigh
  contact, where the two legs' cones own each other's skin. Only the first is a topology
  problem; the other two are two surfaces touching, which no graph can make adjacent
  without a cone across the midline.
- **Worst gap 21.7 mm** (target 15): the armpit hollow and the kneecap, each a concavity
  a sphere cannot enter without crossing the skin next to it. Two more fills would take
  it to 99 spheres; the search suggests the return is small.
- **The 3-hop walk budget is still a guess.** Ask the solver.
- **The quad count in a GUI round trip** still drops to triangles when the file is saved
  from the tool (`SphereManager.save_to_json` does not write `quad_groups`); the two-line
  fix is described in `scripts/test_topology_roundtrip.py`.
- **Timing.** A full male build is about six minutes (fill loop 40 s, two full fits
  150 s each, pose sweep 10 s); `--quick` is one minute.

## Reproducing

```
.venv/Scripts/python.exe scripts/build_topology.py --design data/generated/search/best_design.json
.venv/Scripts/python.exe scripts/build_topology.py --gender female --design data/generated/search/best_design.json --name auto_female
.venv/Scripts/python.exe scripts/compare_topologies.py --poses data/template.json data/generated/session1_auto_male.json data/generated/auto_male.json
.venv/Scripts/python.exe scripts/search_topology.py --rounds 2 --candidates 16 --workers 6
for t in bumpers sdf_torch sections validate roundtrip fit; do .venv/Scripts/python.exe scripts/test_topology_$t.py; done
```

`data/generated/reports/variants/` keeps the reports of every full-run variant quoted in
this file (pose weights, posed-cone check on/off, band sections).

## Session 2 — a lesson about design files

`scripts/search_topology.py` originally saved the winning design as a *full* snapshot of
every `Design` field. When the pose-term weight was later switched off by default, every
run made from that file silently kept `pose_weight = 0.5`, and a whole afternoon of
"why is the shoulder collapsing" experiments (radius floors, anchored offsets, multi-start
fits) chased an effect that was really the stale pose term. Three of those changes were
worth keeping anyway -- fits are now anchored to the original placement so refits cannot
compound their drift, the radius floor is per sphere, and the full fit is multi-start --
but the diagnosis came from comparing a good and a bad output *sphere by sphere against
the placement* and reading the run log's `design` block, not from the metrics. The design
file now holds only the parameters the search actually varied (`shin`, `clavicle`), so it
composes with the current defaults; the full snapshot is kept as
`search/best_design_full_snapshot.json`.

## Session 2 — how the pipeline works now

```
scripts/build_topology.py [--quick] [--design x.json] [--set key=value] [--gender female]
```

| stage | module | what it does |
| --- | --- | --- |
| place | `generate.py` (+ `sections.py`, `anatomy.py`) | the session-1 rules: chains on medial points, ladder + sheet trunk, inscribed buttock spheres, stubs; cross-sections now cut from the mesh and trimmed by the vertex mask; every radius starts skin-touching |
| fit | `fit.py` (+ `sdf_torch.py`) | L-BFGS-B over left-half + midline centres (box ±δ) and log-radius scales, objective = area-weighted one-sided loss on the **exact** sphere-mesh distance, autograd through the torch port of `bumpers.py`; containment penalty; binding audit with pin-and-refit; optional pose term |
| link | `repair.link_walk_breaks` | for the worst walk-break pair, the cone between their spheres that removes the most over-budget surface steps, subject to the joint rule and a protrusion cap |
| fill | `repair.fill_gaps` | for the largest cluster of vertices > 12 mm off, the largest interior sphere that reaches it, linked to the two nearest compatible spheres; refit after every fill |
| validate | `validate.py` | budgets, connectivity, degeneracy, symmetry, fidelity, walk-break rate, wasteful connections, harmful overlap (recalibrated), binding-as-intended, one-bone-per-cone, error clusters with location and facing, 54-pose sweep |
| render | `render.py` | front / back / left views and signed-distance heat maps, off-screen |

Supporting scripts: `scripts/search_topology.py` (parallel search over `Design`),
`scripts/compare_topologies.py` (one table for any set of templates),
`scripts/test_topology_{bumpers,sdf_torch,sections,validate,roundtrip,fit}.py` (all pass).

**The joint rule** used by both repairs and the validator (`anatomy.joints_compatible`):
a cone may join two spheres bound to the same joint, to a kintree parent and child, or
to two *trunk* joints (pelvis, spine1-3, neck, collars) at most two bones apart. The
trunk allowance is measured, not assumed: `spine1` and `neck` are nearly dead joints (no
vertex weights them above 0.73) and the ladder rows necessarily alternate between spine
joints. Limb joints are hinges and get no allowance.

**Two facts about the tool that shaped the design.** The GUI round trip through
`SphereManager.save_to_json` drops `quad_groups` (it only writes the fields it knows), so a
file saved from the GUI converts to triangles-only text; `scripts/json2txt.py` and
`txt2json.py` are now lossless in both directions when the key is present, and the two-line
fix in `sphere_manager.py` is described in `scripts/test_topology_roundtrip.py`. And
`data/template.json` carries 14 bindings the tool's own rule no longer reproduces (6 by more
than the loader's rounding; sphere 44 is bound to the wrong joint).


Live progress log. Everything I produce lands in the repo so you can open it:

| Where | What |
| --- | --- |
| `src/topology/` | the generator/validator library being built |
| `scripts/build_topology.py` | the CLI that produces a template end to end |
| `data/generated/` | every candidate template, as `.json` (open in the tool) and `.txt` (engine). `auto_male` is the one to look at; `auto_male_ring`/`_ladder` are the two lower-trunk families compared |
| `data/generated/reports/` | validator output per candidate |
| this file | decisions, measurements, and what is done vs. pending |

---

## Result — `data/generated/auto_male.json`

```
.venv/Scripts/python.exe scripts/build_topology.py
```

**79 spheres · 65 doubles · 14 triples · 12 quads** (engine `.txt`; the `.json` carries the
quads split into triangles so the GUI opens it).

| | hand-made | **generated** |
| --- | ---: | ---: |
| spheres | 60 | 79 |
| **rest: worst protrusion** | 44.9 mm | **15.9 mm** |
| **rest: vertices inside collider >5 mm** | 8.63 % | **4.42 %** |
| **watchdog: steps needing >3 hops** | 3.68 % | **1.84 %** |
| watchdog: unreachable steps | 0 | 0 |
| **posed: worst protrusion (54 poses)** | 59.8 mm | **46.7 mm** |
| **posed: mean worst-per-pose** | 44.8 mm | **35.7 mm** |
| **rest: worst gap** | 35.5 mm | **32.2 mm** |
| rest: RMS | **6.12 mm** | 6.64 mm |
| rest: vertices >15 mm off | **0.83 %** | 3.54 % |
| exactly mirror-symmetric | no (38 of 60 unpaired) | **yes** |
| degenerate primitives | none | none |

Protrusion cut by 2.8x, watchdog walk-breaks by 2x, posed artifacts by 20%, worst gap
slightly better, RMS level. The one clear regression is the *fraction* of vertices more
than 15 mm off — 3.54% against 0.83%.

## The brief, as I understood it

Rebuild the sphere-mesh template for the SMPL **male** body automatically: sphere count,
positions, radii and **connectivity**, replacing hand placement.

Hard constraints:

- **< 100 spheres**, **≤ ~20 three-sphere connections**.
- Quads (4-sphere) are allowed **engine-side in the `.txt`**; the `.json` carries the
  triangle split so the GUI can still open it. *(your call, answered 2026-09-02)*
- The **bumper graph must be a connected cover** of the body surface — the watchdog walks
  explicit topological links, never spatial proximity, so overlapping-but-unlinked bumpers
  lose nodes and force brute-force reacquisition.
- Bumpers must not overlap each other more than necessary.
- **No animation artifacts**: sphere binding is rigid and single-joint, so no bumper may
  span two joints that rotate relative to each other.
- Arms: keep the character of the existing hand-made template (you like them).
  Torso: rebuild (you don't).
- Hands, feet, head: truncated or radically simplified, not modelled.

## Ground truth I established from the code (not assumed)

- **Binding is rigid and single-joint.** `SphereManager.assign_parent_joint`
  (`src/sphere_manager.py:697`) picks the closest kintree *bone* `(p → c)` and binds the
  sphere to **`p`**, storing `rest_offset = R_pᵀ (centre − joint_p)`. The blend onto the
  secondary joint is commented out at `src/sphere_manager.py:826`, so posing is exactly
  `centre' = joints'[p] + R'[p] @ rest_offset`. Every artifact claim in this work is
  measured against *that* rule, not against SMPL's LBS.
- **A 3-connection is a slab, not three cones.** `Trispheremesh` really builds two
  triangular faces at `c_i ± n·r_i` plus the three pairwise cones — so a triangle of
  spheres covers a *sheet* of surface on both sides. This is the medial-axis view and it
  is what makes a cheap torso possible.
- **`spheremesh_distance` under-reports coverage on triangles.** `_segment_pairs`
  (`src/sphere_fit.py:357`) decomposes a 3-connection into its three *edges* and never
  evaluates the slab. A point resting on the middle of a triangle is scored as far
  outside. It is left untouched (it is the external metric, and `plan.md` forbids
  changing it) — `src/topology/bumpers.py` provides the exact metric for validation.
- **A fitter already exists.** `optimize_spheres` (`src/sphere_fit.py:665`) — asymmetric
  protrusion loss, frozen Euclidean partition. Topology generation feeds it; it is not
  re-implemented.
- SMPL male: 6890 verts, 24 joints, `weights` (6890, 24). **+X is the left side**, +Y up,
  +Z anterior. T-pose, arms along X.

## Plan

1. **Measure** the body — trunk cross-sections and medial extent, limb taper profiles,
   joint-dominance zones, real bend statistics from `random_pose.csv` + the mocap take,
   and a full diagnosis of the existing template. *(done — 5 of 6 probes; the sixth died
   on a JSON serialisation error and was duplicating work already done here)*
2. **Design** — three independent topologies from different angles, adversarially judged
   on watchdog connectivity and on fidelity/animation, then synthesized. *(designs done;
   the judging and synthesis stages hit the account session limit and did not run, so the
   three proposals were read and mined directly instead — see "Things tried and rejected")*
3. **Generate** — placement rules evaluated against the SMPL mesh, then handed to
   `optimize_spheres` for centre/radius refinement.
4. **Validate** — connectivity, walk-break rate, overlap, coverage, protrusion, and a pose
   sweep over the random poses and the mocap take.
5. **Iterate** against the report until the thresholds are met.

---

## Baseline — what `data/template.json` scores today

Measured with `src/topology/validate.py` on the neutral male, over the 3725 vertices the
template is actually meant to cover (hands, feet, skull and everything past the
wrist/ankle/neck truncation planes excluded). **This is the bar to beat.**

```
60 spheres · 40 doubles · 18 triples · 0 quads → 166 bumpers
                                                   (60 spheres, 88 cones, 18 prysmoids)

fidelity   RMS 6.12 mm     worst gap 35.5 mm    0.83% of vertices > 15 mm off
           worst protrusion 44.9 mm             8.63% of vertices inside the collider
watchdog   walk-break 3.68% of surface steps need > 3 hops, 0 unreachable
           mean walk 0.55 hops
posed      worst protrusion 59.8 mm over 54 poses, mean worst-per-pose 44.8 mm
structure  1 sphere component, 1 bumper component     ✓
           7 connections own no surface (cone[5,13], cone[11,13], cone[20,32],
           cone[32,49], cone[37,45], cone[38,43], …) — pure cost
symmetry   38 of 60 spheres have no exact x-mirror twin (drift 3–4 mm)
```

Per region — the complaint about the torso is confirmed, and quantified:

| region | verts | RMS | worst gap | worst protrusion | walk-break |
| --- | ---: | ---: | ---: | ---: | ---: |
| R_hip | 239 | 12.6 mm | 17.1 mm | **44.9 mm** | 12.1% |
| L_hip | 233 | 11.1 mm | 11.7 mm | **43.4 mm** | 13.7% |
| spine3 | 606 | 6.6 mm | 18.2 mm | 20.3 mm | **20.3%** |
| pelvis | 400 | 4.5 mm | 15.1 mm | 13.6 mm | 13.5% |
| L_elbow (forearm) | 230 | **2.6 mm** | 9.8 mm | 6.0 mm | 0.0% |
| L_shoulder (upper arm) | 271 | **2.9 mm** | 13.0 mm | 5.1 mm | ~0% |
| L_ankle | 21 | 29.3 mm | 74.9 mm | 7.1 mm | 0.0% |

The arms really are the good part — 2.6–2.9 mm RMS and a clean walk. The hips and lower
trunk carry 40+ mm of protrusion and a fifth of their surface steps lose the walk.

### Two things found in your current template, worth knowing regardless

1. **13 of 60 spheres carry stale LBS bindings.** Recomputing `rest_offset` with the
   tool's own current rule reproduces 47 of 60 exactly, so the rule is understood — but
   sphere 44 is bound to `R_elbow` while sitting *past* the `R_wrist` joint (269 mm of
   rest-offset error), and spheres 34/45/37 are off by exactly 3–4 mm, the signature of a
   centre nudged after its binding was computed. `nearest_vertex_index` and `rest_radius`
   are stale on ~48 of 60. Those spheres animate from the wrong rest frame today.
2. **`secondary_joint` / `lbs_weight` in the file come from the *old* rule** (the
   skinning-weight version now commented out at `src/sphere_manager.py:755`), not the
   current one. Harmless while the blend is disabled, but it means the fields are not what
   the current code would write.

---

## Progress

### Done and tested

- `src/topology/bumpers.py` — collision primitives, their **exact** signed distance, and
  the bumper adjacency graph. `scripts/test_topology_bumpers.py`: 30 checks, all passing,
  every closed form verified against brute-force minimisation over the medial simplex
  (agreement to 1e-12 for cones, 4e-6 for slabs — the latter limited by the reference
  grid, not by the formula).
- `src/topology/anatomy.py` — body measurement: cross-sections, medial loci, bone
  stations, and the tool's exact binding and posing rules.
  *Already earned its keep:* the true medial point of a limb section is off the bone axis
  by up to **36 mm** (shin) and **27 mm** (thigh), so spheres placed on the bone axis
  inherit that as protrusion on one side and a gap on the other.
- `src/topology/validate.py` — the full check suite (see the baseline above).
- `src/topology/export.py` — tool JSON with correct bindings, and engine text with quads.
  Its text output is **byte-identical** to `scripts/json2txt.py` on `template.json`.

- `src/topology/generate.py` + `scripts/build_topology.py` — the generator. Placement
  rules, not coordinates: `chain` walks a bone and drops spheres on each cross-section's
  *medial* point, `sheet` lays a grid along a horizontal cut's medial locus, `stub`
  terminates a truncated chain, `mirror` reflects the left half, `symmetrize` forces exact
  bilateral symmetry after every geometry pass.

### In flight

- Design fan-out: 3 independent topologies being judged (measurement phase done, 5 of 6 —
  the sixth died on a JSON serialization error, and it was the one duplicating analysis I
  had already done here).

---

## What the measurements changed about the design

These are the findings that overturned something I had already built. All measured.

**1. The lower trunk is not a sheet.** Width/depth aspect is only 1.21–1.45 from the
pelvis to y = +0.10, and the largest inscribed circle is 95–100% of the half-depth — it is
a fat, near-circular *tube*, and its medial axis is only 36–57% of the half-width. A wide
multi-column sheet there is spent on nothing. It flattens only above y = +0.13 (aspect
1.49 → 2.29 at the clavicle), where a real sheet earns its keep. So the trunk is now **two
structures**: a two-column ladder below, a three-column sheet above, seamed with triangles.

**2. Small triangles are pure cost.** A slab's gain over its own three edges is
`√(ρ² + r²) − r`, with ρ the medial inradius. With 120 mm torso spheres and 40×55 mm
cells, each of my 32 quads was buying **1.7 mm**. Same verdict on the hand-made template:
**15 of its 18 triangles buy under 5 mm**, and `template_male_2.0.json` has 68 triangles
of which 42 are under 5 mm. Triangles must be *big relative to their spheres* — at r = 100
mm you need ρ ≈ 46 mm, an equilateral medial triangle of side ~160 mm. The design now uses
**8 large quads** instead of 32 small ones.

**3. Never let a bumper cross the elbow.** Rigid single-joint binding is essentially free
*as long as a bumper stays inside one bone* — control RMS is 0.3–14 mm everywhere. Across
the elbow it is 260 mm RMS: p95 error is already 77 mm two centimetres past the joint and
483 mm at the wrist, and the elbow's median local rotation is 132°, so a *typical* pose
already folds the arm double. Your arms are the one region already split at the elbow,
which is exactly why they read as the good part.

**4. `spine1` and `neck` are dead joints.** No SMPL vertex assigns `spine1` more than
0.734 of its skinning, or `neck` more than 0.712 — against 0.98–1.00 for hips, knees,
elbows, shoulders, `spine3`. **No rigid sphere can ever track the lumbar region or the
neck**; on an interior grid, 2722 points bind to `spine1` and not one reaches 0.7. That is
a floor on accuracy there, not something to tune away.

**5. The tool's binding rule disagrees with the skin 21.4% of the time.** On a 15 mm
interior lattice, 21.4% of points are bound to a joint that is not their skinning-dominant
one, and it is not a boundary artefact (20.8% even with ≥20 mm clearance). Roughly a third
of the resulting drift is the rule picking the wrong joint — fixable by placement — and
two thirds is inherent to rigid single-joint binding. Every large disagreement cluster is
torso, pelvis or head; the limbs contribute almost nothing.

**6. The gluteal band belongs to nobody.** The thigh has no cross-section separable from
the pelvis below t = 0.167 of the femur, and the skin no thigh chain can own is **12.6% of
the whole body**. It has to be covered by a pelvis cluster explicitly connected to the top
thigh sphere, or the watchdog loses every node sliding off the buttock. A buttock sphere
also cannot sit on the medial locus — the largest circle inscribed in a pelvic section sits
120 mm *in front of* the buttock, so a dense 900-sphere medial cover still leaves a 48 mm
hole there. It is the wrong locus, not too few spheres.

**7. `spheremesh_distance` under-reports coverage — and the cone term, not the missing
slab, is the bigger error.** It evaluates the cone at the orthogonal projection rather than
the true minimiser, which reports up to **23.8 mm of phantom gap** on points lying exactly
on the cone surface (mean 0.67 mm, p95 2.9 mm over body vertices) — about 60× the missing
slab's contribution on the current template. It is left untouched, as `plan.md` requires;
`src/topology/bumpers.py` supplies the exact metric everything here is judged with.

---

## Where it is still short, honestly

8 of 16 checks pass. The thresholds are aspirational, not a claim about what is reachable.

- **3.54% of vertices sit more than 15 mm from the collider** (target 2%), worst 32.2 mm.
  It started at 20.9%. The residual is the armpit and the groin — deep concavities where a
  cone spanning them is either outside the skin or short of it. The `blobs` builder can
  place spheres inside a concavity and there are 20 spheres of budget left; that is the
  next thing to spend them on.
- **Protrusion 15.9 mm worst** (target 5 mm), down from 44.9 mm. It trades directly against
  the gap figure via `--min-radius-fraction`. Measured curve, floor -> (worst protrusion,
  fraction >15 mm off): 0.50 -> (13.3 mm, 4.01%), 0.55 -> (13.5, 3.82), 0.60 -> (15.9,
  3.54), 0.65 -> (19.2, 3.04), 0.80 -> (27.9, 2.07). 0.60 ships; move it if you would
  rather have coverage than clearance.
- **Posed protrusion 46.7 mm worst over 54 poses** (target 20 mm). Close to a floor rather
  than a bug: rigid single-joint binding carries 13.5 mm RMS of inherent error at the
  shoulder even for a bumper wholly inside its own bone, and the offenders here are the
  upper-arm cones whose spheres bind correctly (`L_shoulder`, skin weight 0.94-0.96).
  Enabling the secondary-joint blend commented out at `src/sphere_manager.py:826` is the
  real fix.
- **Walk-break 1.84%** against a 1% target, 0 unreachable. What remains is between the two
  inner thighs and around the hip/glute cluster.
- **Non-adjacent overlap still fails**, and all three design agents hit the same wall
  independently: 50-54% of bumpers on every design *including the hand-made template* have
  a third of their surface buried inside a bumper 3+ hops away. On a 1-D chain that is
  structural rather than a fault, so the 3-hop radius in this check is probably
  mis-calibrated rather than the topologies being wrong. Worth deciding deliberately.

## Things tried and rejected, with the numbers

- **Ring instead of ladder for the lower trunk.** Per cross-section a free ring of 4 disks
  beats a collinear row decisively (4.8 mm against 10.9 mm at y = -0.160). In the full 3D
  union it does not: 4.87% of vertices beyond 15 mm against the ladder's 4.62%, walk-break
  1.71% against 1.62%, for **ten more spheres and fourteen more quads**. The ladder's
  left-to-right slabs already reach the front and back walls the ring buys separately.
  Kept as `--lower-structure ring` so the comparison can be re-run.
- **A fine 3-column x 8-row medial sheet for the whole trunk** (40 spheres, 32 quads):
  8.4 mm RMS. The current two-structure trunk does 6.6 mm with 26.
- **Dropping the midline column from the lower ladder.** Two columns either side reach the
  full half-width, so it looked redundant. It is not: the slab between them is the plane
  tangent to both, which at x = 0 sits 17-19 mm behind the true belly. Restoring it took
  the worst gap from 49.5 mm to 33.1 mm and RMS from 7.4 mm to 6.6 mm for five spheres.
- **Deleting the pubic node** because its sphere is contained in its neighbour: cost 10 mm
  of worst gap. The coverage was never coming from the sphere — it came from its cones down
  to the hips and thighs. Kept, and moved low enough not to be swallowed.
- **A flat 50/50 shrink when relieving protrusion.** Charging the excess to both ends of a
  cone equally cost 5 mm of RMS and doubled the badly-covered fraction, against charging
  each sphere its actual share of the offending surface.

## Reproducing

```
.venv/Scripts/python.exe scripts/build_topology.py --max-offset 0.010 --protrusion-target 0.012
.venv/Scripts/python.exe scripts/build_topology.py --gender female --name auto_female
.venv/Scripts/python.exe scripts/test_topology_bumpers.py      # 30 checks on the SDF + graph
```

Nothing in the generator is a coordinate, so `--gender female` re-measures and re-places
rather than re-using the male numbers.

## Open questions / decisions taken

- **Quads**: engine-side only, `.json` gets the triangle split. *(answered)*
- **Pose validation**: both `data/random_pose.csv` and `00163_take2_fps30.pkl`, automated.
  *(answered)*
- Head/hands/feet truncation depth: my call — stubs that keep the graph connected and
  terminate cleanly, no articulated fingers/toes/skull.
