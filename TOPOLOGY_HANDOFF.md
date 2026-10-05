# Implementation handoff — automated sphere-mesh topology for SMPL

> **Superseded in part (2026-09-04).** A second session rebuilt the pipeline on top of this
> code: exact-SDF fitter (`src/topology/fit.py`), automatic gap filling and walk-break
> linking (`src/topology/repair.py`), mesh-plane sections (`src/topology/sections.py`), a
> design search, renders, and an extended validator. The current state, numbers and
> decisions are in `TOPOLOGY_NOTES.md` (session-2 sections at the top). Sections 4 and 13
> of this document (the measured facts) remain valid; sections 7, 8, 14, 15 and 17 describe
> work that has since been done.

Written for an agent restarting this task from scratch. It has the original request but
none of my session memory. Everything below is what I learned by actually doing it.

Read §13 and §14 first if you read nothing else.

---

## 1. Original user objectives

### Explicit requirements

Rebuild the **topology** of the sphere-mesh collision template — "the connections and the
number and position of spheres" — automatically, replacing a manual workflow where the user
placed spheres in empty space and connected them by hand.

1. Fit the **SMPL neutral male** body surface.
2. **Under 100 spheres.**
3. **Under ~20 three-sphere connections.**
4. Do not let bumpers (spheres, cones, planes) **overlap too much** — "being careful to not
   overlap too much the bumpers … since there is an optimization that leverages on the
   correct connectivity of the sphere-mesh".
5. The bumper graph must support the solver's **watchdog walk**. The user supplied a long
   explanation, and it is the core of the task: the solver caches per cloth node the bumper
   it last resolved against; next substep it tests that bumper, and if the node has left it,
   walks the **explicit adjacency graph** (sphere → its cones → prysmoids/quads →
   neighbouring prysmoids) until it finds the containing bumper. Only if the walk fails does
   it brute-force rescan. **Geometric overlap is not a substitute for a topological link.**
   Two cones that interpenetrate but belong to disjoint parts of the topology lose the node.
   "The watchdog assumes the bumper graph is a connected cover of the collider's surface."
6. **No animation artifacts.** An artifact is sphere-mesh geometry that "pops out" when the
   skeleton moves — the user's own example: "if a cone is placed from the shoulder to the
   wrist, when the elbow bends a part of the cone will be completely out of the human".
7. **Keep the character of the arms** from `data/template.json` ("I really like the arms
   modeling"); **rebuild the torso** ("I do not like it on the torso").
8. **Hands, feet and head are not modelled** — "truncated or super-simplified".
9. The user granted full authority on method and asked to be told if I got stuck.

Two clarifications the user gave when asked (2026-09-02):

- **Quads allowed engine-side.** 4-sphere connections may appear in the `.txt` the engine
  reads; the `.json` must carry a triangle split because `SphereManager.load_from_json`
  only accepts 2- and 3-index connections.
- **Pose validation: both** `data/random_pose.csv` and `00163_take2_fps30.pkl`, automated.

Mid-session the user twice asked to **see intermediate output** ("save your tentatives
somewhere I can see", "I still not see generated output, even partial is fine"). Treat
producing an openable artifact early as a hard requirement, not a nicety.

### Implicit requirements I inferred

- The result must **load in their existing tool** and round-trip through
  `scripts/json2txt.py`. That constrains the JSON schema and the text format exactly.
- It must be **reproducible and re-runnable**, not a one-off file — "the most automated way
  possible" means a generator, not coordinates.
- It should be **bilaterally symmetric**. The tool has mirror-edit and `check_x_symmetry`,
  and their own template is *not* symmetric (38 of 60 spheres unpaired by 3–4 mm), which
  reads as drift they would rather not have.
- The existing fitter (`optimize_spheres`) should be **used, not replaced** — the user
  volunteered its existence.
- `plan.md` (open in their editor) states `spheremesh_distance` **must not change**: "it is
  the external metric and the comparison to `fit_beta_offsets.py` depends on it staying
  exactly as it is".

### Constraints discovered during implementation

These were not in the request and cost real time to find. See §4 for detail.

- Sphere→joint binding is **rigid and single-joint**; the blend is commented out.
- The tool **rounds sphere centres to 3 decimals on load**.
- `spheremesh_distance` is **approximate in two ways**, one of them large.
- SMPL's `spine1` and `neck` are **"dead joints"** — no vertex weights them above ~0.73.
- A sphere placed exactly at a joint centre binds **non-deterministically**.

### Quality expectations inferred

The user writes long, precise, technical prose and their repo (`plan.md`, `sphere_fit.py`)
contains rigorous numerical briefs with acceptance criteria, silent-bug lists and
justified defaults. They expect **measured claims, not assertions**, and they expect
failures to be reported honestly. `plan.md` explicitly says of one metric: "The reported RMS
will get worse, and that is correct. … Do not treat the increase as a regression." Match
that standard.

---

## 2. Final state

### What exists

New code, none of it modifying existing files:

| File | Lines | Purpose |
| --- | ---: | --- |
| `src/topology/bumpers.py` | 591 | Collision primitives, **exact** signed distance, bumper adjacency graph |
| `src/topology/anatomy.py` | 605 | SMPL measurement: sections, medial loci, bone stations, the tool's binding/posing rules, pose sets |
| `src/topology/generate.py` | 799 | Placement builders (`chain`, `sheet`, `ring`, `blobs`, `stub`), `mirror`, `symmetrize`, `refine`, `relieve_protrusion` |
| `src/topology/validate.py` | 661 | 16-check validation suite incl. walk-break rate and pose sweep |
| `src/topology/export.py` | 208 | Tool JSON (with LBS bindings) and engine text (with quads) |
| `scripts/build_topology.py` | 586 | The design + CLI |
| `scripts/test_topology_bumpers.py` | 315 | 30 checks on the SDF and graph, all passing |
| `TOPOLOGY_NOTES.md` | 309 | Running log, baseline, findings, rejected approaches |

Outputs in `data/generated/`: `auto_male.{json,txt}` (the deliverable),
`auto_male_ring.*` / `auto_male_ladder.*` (the two lower-trunk families compared),
`auto_female.*` (proof the rules are body-agnostic), and `reports/*.txt`.

### What works

- **The whole pipeline runs end to end** in about 30 s including a 54-pose sweep, and
  `--gender female` re-measures and re-places from scratch with no hard-coded coordinates.
- The exact SDF is **verified against brute-force minimisation** over the medial simplex:
  1e-12 agreement for cones, ~4e-6 for slabs (limited by the reference grid, not the
  formula).
- The engine text writer is **byte-identical** to `scripts/json2txt.py` on `template.json`.
- The JSON's LBS bindings **reproduce the tool's own stored values** for 47 of 60 spheres
  on `template.json` (the other 13 are stale in *their* file — see §4).
- Symmetry survives the tool's load-time 3-decimal rounding.

### Result vs the hand-made template

Both measured identically, exact metric, 3725 modelled vertices, 54 poses.

| | hand-made | generated |
| --- | ---: | ---: |
| spheres / bumpers | 60 / 166 | 79 / 225 |
| connections | 40×2, 18×3 | 67×2, 14×3, 12×4 |
| worst protrusion (rest) | 44.9 mm | **15.9 mm** |
| vertices inside collider >5 mm | 8.63 % | **4.42 %** |
| walk steps needing >3 hops | 3.68 % | **1.84 %** |
| unreachable steps | 0 | 0 |
| worst gap | 35.5 mm | **32.2 mm** |
| RMS | **6.12 mm** | 6.64 mm |
| vertices >15 mm off | **0.83 %** | 3.54 % |
| posed worst / mean protrusion | 59.8 / 44.8 mm | **46.7 / 35.7 mm** |
| mirror-symmetric | no | **yes** |

### Partially working / incomplete

- **8 of 16 validator checks fail.** Thresholds are aspirational; nothing regressed.
- **Coverage fraction is the clear regression**: 3.54 % of vertices >15 mm off against
  0.83 %. Concentrated in the armpit and the groin.
- **Posed protrusion 46.7 mm** against a 20 mm target. Near the floor for rigid binding
  (see §4), not obviously fixable by topology.
- **`watchdog/non-adjacent-overlap` fails on every design including the user's own.** All
  three design agents hit this independently; I believe the check's 3-hop radius is
  mis-calibrated rather than the topologies being wrong. **Unresolved — decide deliberately.**
- **No search.** Every design parameter was chosen by hand-iteration. See §7.

### Known limitations / workarounds

- `relieve_protrusion` is a blunt shrink pass with a floor. It trades coverage for
  clearance along a measured curve; the operating point is a flag, not a principle.
- The armpit and groin are simply not modelled by any structure.
- `--lower-structure ring` exists only to keep a measured comparison reproducible.
- `data/generated/` is committed output; there is no cleanup story.

---

## 3. Architectural / implementation decisions

### D1 — Write an exact SDF instead of using `spheremesh_distance`

**Decision.** Implement `src/topology/bumpers.py` with the closed-form exact signed distance
to spheres, tapered cones and triangular slabs, and validate the topology with it.

**Alternatives.** (a) Use `spheremesh_distance` from `sphere_fit.py`. (b) Fix
`spheremesh_distance`. (c) Sample geometry with PyVista and use a mesh distance.

**Why.** `plan.md` forbids changing `spheremesh_distance`, and it is wrong in ways that
matter for exactly this task: it never evaluates a triangle's slab, so a sheet-based torso
would be scored as a hole. A design optimising against a metric that cannot see its own
primitives is worthless.

**Trade-offs.** Gained a metric I can trust and that catches real geometry errors. Cost:
~600 lines and a real debugging session. Also created a **two-metric situation** — the tool
and the fitter still use the approximate one, so numbers reported by the tool will not match
mine.

**Confidence: very high.** This was the single best decision. It is also where the *most
valuable measurement of the whole project* came from (the slab-gain law, §4).

### D2 — Ownership resolves to the lowest-dimensional bumper attaining the minimum

**Decision.** `BumperGraph.owners` takes the minimum, then among bumpers within 1 nm of it
prefers sphere over cone over slab.

**Alternatives.** Plain `argmin`. Or restrict each bumper's domain explicitly (a cone owns
only the band between its tangent circles).

**Why.** A cone's exact SDF includes its spherical end caps, so it **ties exactly** with its
end sphere wherever that cap is the nearest surface. A plain `argmin` hands every cap to
whichever adjacent cone rounds a hair lower, and ownership flips with floating-point noise —
which is precisely the instability the watchdog cannot absorb. The rendered geometry agrees:
`Spheremesh._build_combined` clips each end sphere at its base plane and keeps only the band.

**Trade-offs.** One tolerance constant. Equivalent to domain restriction but far simpler.

**Confidence: high.** A test in `test_topology_bumpers.py` pins it.

### D3 — The walk metric is a hop *count*, not an adjacency test

**Decision.** For every SMPL mesh edge, find the owning bumper at each end and report the
graph hop distance; fail on the fraction exceeding a budget (default 3) and on unreachable
pairs.

**Alternatives.** Binary adjacency (my first version). Or a simulation of the actual walk.

**Why.** Consecutive cones along a limb share a sphere and are **two hops apart through it**.
The binary version scored the user's template at 12.99 % "broken" and condemned every
well-formed chain. With hop counts the same file reads 3.68 % over budget, mean walk 0.55
hops — a number that means something.

**Trade-offs.** The budget of 3 is a judgement call. I have no ground truth for what the
real solver affords.

**Confidence: high on the metric, medium on the budget.** Ask the user what the real walk
depth limit is — it is a one-line answer that would calibrate this properly.

### D4 — Placement rules, never coordinates; spec keyed by name; mirror the left half

**Decision.** `TopologySpec` holds named spheres and connections between names. Only the
left half and midline are measured; `mirror` generates the right half and every connection
touching it; `symmetrize` then forces exactness after each geometry pass.

**Alternatives.** Emit coordinates. Or measure both halves. Or mirror by nearest-neighbour
matching (what the tool's mirror-edit does).

**Why.** Names survive insertion, make connection lists readable, and make mirroring
mechanical. Measuring both halves produces *near*-mirrors: SMPL's vertex sampling is not
exactly symmetric, so sections come out a few millimetres lopsided.

**Trade-offs.** Slightly more machinery. One real trap: a connection written to a bare
column index attached **both** thighs to the same side of the body, and the bug was invisible
until I read a protruding-bumper report. Naming would have prevented it; I introduced naming
only for the sheet columns after that.

**Confidence: very high.** Do this from the start.

### D5 — Radii come from the skin, not from a fit

**Decision.** Every sphere is placed with radius = distance to the nearest body vertex (the
maximal inscribed sphere at that centre), then `optimize_spheres` refines, then a shrink
pass relieves residual protrusion, then symmetrize.

**Alternatives.** Let the fitter choose radii from scratch.

**Why.** A sphere at the inscribed radius **cannot** protrude, by construction. That removes
the whole protrusion problem for spheres and leaves only the interesting one — cones and
slabs crossing concavities. It is also the same quantity
`SphereManager._compute_touch_radius_and_nearest_vertex` uses, so tool and generator agree.

**Trade-offs.** The refine pass then partly undoes it (see §5, §6).

**Confidence: high on the principle, low on the pipeline order.** See §7 — I would invert
refine and relieve, or replace both.

### D6 — Two structures for the trunk, not one

**Decision.** Lower trunk (y ≤ +0.075): a 3-column ladder of rungs joined by quads. Upper
trunk (y ≥ +0.14): a 3-column medial sheet. Seamed with triangles.

**Alternatives.** One sheet everywhere (my first attempt). A ring cage (tested, rejected).

**Why.** Measured: the lower trunk has width/depth aspect 1.21–1.45 and its largest
inscribed circle is 95–100 % of the half-depth. It is a **fat near-circular tube**, not a
slab; its medial axis is only 36–57 % of the half-width. The section genuinely flattens only
above y = +0.13 (aspect rising to 2.29 at the clavicle).

**Trade-offs.** Two code paths and a seam. Worth it: the one-sheet version was 8.4 mm RMS
with 40 trunk spheres; the two-structure version is 6.6 mm with 26.

**Confidence: high.**

### D7 — Quads engine-side, triangle split in the JSON, `quad_groups` recorded

**Decision.** `write_engine_text` emits real quads; `build_payload` splits each across its
0–2 diagonal for the GUI and records the original grouping under `quad_groups`.

**Alternatives.** Triangles everywhere. Or patch the tool to load quads.

**Why.** The user chose engine-side quads. `SphereManager._build_from_payload` accepts only
2/3. The split must match `Bumper.triangles()` so the two files describe identical geometry —
verified by a test.

**Trade-offs.** The two files disagree on triangle count on purpose (14 vs 26). **`json2txt.py`
does not read `quad_groups`**, so a round trip through the GUI silently loses the quads.
That is unfixed technical debt (§17).

**Confidence: high on the mechanism, medium on the ergonomics.**

### D8 — Use `optimize_spheres` rather than write a fitter

**Decision.** `refine()` is a thin wrapper.

**Why.** It exists, it is well-documented, `plan.md` describes its objective in detail, and
the user pointed at it.

**Trade-offs.** It fits each sphere to the vertices nearest **its own centre**, knowing
nothing about cones or slabs. So it can reduce sphere-only error while *increasing*
sphere-mesh protrusion — measured: 28.0 mm → 39.8 mm on one configuration. And it collapses
spheres whose partition is a small awkward patch.

**Confidence: medium.** Right call for reuse; wrong tool for the actual objective. See §7.

---

## 4. Important discoveries

**Ranked by how much time they would save you.**

### 4.1 Binding is rigid, single-joint, and chosen by closest *bone*

`SphereManager.assign_parent_joint` (`sphere_manager.py:697`) finds the kintree bone
`(p → c)` whose segment is closest to the sphere centre and binds the sphere to **p**, the
parent. `update_sphere_positions` (`:826`) then does

```
centre' = joints'[p] + R'[p] @ (centre − joints_rest[p])
```

and **the blend onto the secondary joint is commented out**. `secondary_joint`,
`secondary_rest_offset` and `lbs_weight` are written but unused. Everything about animation
artifacts must be reasoned against *this* rule, not against SMPL's LBS.

### 4.2 Rigid binding is nearly free inside a bone and catastrophic across one

Measured over 174 poses (24 random + 150 mocap), per joint, comparing full LBS against rigid
posing by the parent transform:

- Bumper **inside its own bone**: RMS 0.3–14 mm everywhere (elbow 9.5, shoulder 13.5/14.1,
  hip 6.1, knee 3.6, wrist 1.2, neck 2.7).
- Bumper **spanning the elbow**: RMS **260 mm**, p95 474 mm, max 497 mm.

Distance-banded span error past a joint (median/p95 mm) — read as a budget:

| joint | 0–2 cm | 2–4 cm | 4–6 cm | 6–10 cm | 10–15 cm |
| --- | --- | --- | --- | --- | --- |
| L_elbow | 35/77 | 63/101 | 97/133 | 149/187 | 223/267 |
| L_shoulder | 16/34 | 26/46 | 35/55 | 51/81 | 78/122 |
| L_knee | 9/21 | 12/26 | 16/30 | 21/37 | 29/47 |
| L_hip | 14/30 | 12/30 | 11/31 | 14/38 | 21/54 |

The elbow's **median** local rotation is 132° — a typical pose already folds the arm double.
The user's arms are the one region already split at the elbow; that is *why* they read as
good.

### 4.3 `spheremesh_distance` is wrong twice, and the bigger error is not the one you expect

It is a one-sided approximation: `d_code ≥ d_true` at every vertex, so it **under-estimates
coverage and under-reports protrusion**, never the reverse.

- **Cone term (dominant).** It evaluates `|P−c(t)|−r(t)` at the *orthogonal* projection, but
  the true minimiser slides toward the larger ball. On points lying exactly on the true cone
  surface it reports up to **23.77 mm of phantom gap**. Over body vertices: mean 0.67 mm,
  p95 2.88 mm, max 7.10 mm.
- **Missing slab.** Mean 0.011 mm, p95 0.00 mm, max 1.87 mm on `template.json`.

Roughly **60:1** in favour of the cone error. I had assumed the missing slab was the problem;
it is not, *on that template* — because its triangles are tiny (see 4.4).

### 4.4 The slab-gain law — the most useful single fact in this project

A triangular slab's gain over its own three edge cones is

```
gain = sqrt(rho² + r_in²) − r_in   ≈  rho² / (2r)   when rho ≪ r
```

with `rho` the medial triangle's inradius. Verified to 0.66 mm mean / 2.27 mm max over 29
triangles. Consequences:

- To buy 10 mm at r = 50 mm you need rho ≈ 32 mm; at r = 100 mm, rho ≈ 46 mm — an
  equilateral medial triangle of side ~160 mm.
- On `data/template.json`, **15 of 18 triangles buy under 5 mm**. On
  `data/template_male_2.0.json`, **42 of 68**.
- My own first torso used 32 quads between 120 mm spheres in 40×55 mm cells: **1.7 mm each**.

**Triangles must be big relative to their spheres or they are pure cost.** This inverts the
naive intuition that a sheet-based torso wants a fine grid.

### 4.5 The lower trunk is a tube; the upper trunk is a sheet

Cross-sections, mesh-plane cut (not a vertex band — see 4.10):

| y | width | depth | aspect | max inscribed r | medial half-extent |
| --- | --- | --- | --- | --- | --- |
| −0.23 (belly) | 357 | 271 | 1.32 | 135 | 87 |
| −0.05 (ribs) | 317 | 263 | 1.21 | 125 | 56 |
| +0.13 (axilla) | 375 | 252 | 1.49 | 123 | 185 |
| +0.25 (clavicle) | 345 | 170 | 2.04 | 82 | 173 |

The max inscribed circle is 95–100 % of the half-depth for every slice from −0.35 to +0.10.

### 4.6 A 2-column sheet is ~18 mm too thin at the midline

For a prysmoid the flat face is the plane through `c_i + n·r_i`, so with two symmetric
columns the slab depth at x = 0 is exactly `2r` **regardless of spacing**. Measured
`D_true(0) − D_slab(0)`: +17.6 mm at y = −0.29, +18.9 at −0.23, +18.3 at −0.17, +17.4 at
−0.08, +10.4 at +0.01. **The lower trunk needs a midline column.** I removed mine as
"redundant" and it cost 17 mm of worst gap; restoring it took worst gap 49.5 → 33.1 mm and
RMS 7.4 → 6.6 mm for five spheres. This was my single most expensive mistake.

### 4.7 The bone axis is not the limb's medial axis

Cross-section centroid offset from the bone line: **shin 21.8 mm posterior (max 33.4)**,
upper arm 10.7 mm inferior (max 36.2 at the deltoid), thigh 9.6 mm anterior + 5.1 lateral,
clavicle 18.4 inferior + 11.8 posterior. Only the forearm is centred (4.0 mm).

Cost of snapping centres to the bone axis, RMS / max protrusion: shin **17.5 / 43.8 mm**
against 2.6 / 8.7 free; thigh 12.6 / 35.4 against 3.5 / 11.0; clavicle 12.2 / 55.4 against
4.8 / 13.4.

### 4.8 Three to four spheres per limb segment; more is measurably wasted

Chain RMS vs N: upper-arm shaft 4.8 / 3.0 / 2.9 / 2.6 mm for N = 2/3/4/5; forearm 8.4 / 3.0 /
2.9 / 2.8; thigh 5.2 / 4.0 / 3.8 / 3.5; shin 8.6 / 3.3 / 2.8 / 2.6. Every knee is at N = 3–4.

**The limit is not the taper — it is circularity.** The minimum-width annulus of the true
cross-section is the irreducible max error of any chain: 4.0 mm mean on the forearm, 6.0 on
the upper arm, 6.3 on the thigh, 13.6 on the clavicle. A 5 mm max-error budget is
**unreachable** on 64 % of upper-arm stations and 100 % of clavicle stations at any sphere
count. The honest target is "95 % of the surface within 5 mm".

### 4.9 `spine1` and `neck` are dead joints

Maximum weight *any* SMPL vertex assigns: `spine1` 0.734, `neck` 0.712, against 0.98–1.00 for
hips, knees, elbows, shoulders, `spine3`, pelvis. On an interior grid, 2722 points bind to
`spine1` and **none** reaches 0.7. **No rigid sphere can ever track the lumbar region or the
neck.** That is a floor on accuracy there, not something to tune away.

Related: the tool's binding rule disagrees with the skinning-dominant joint for **21.4 %** of
interior points (20.8 % even with ≥20 mm clearance, so not a boundary artefact). Every large
disagreement cluster is torso, pelvis or head; the limbs contribute almost nothing. Roughly a
third of the resulting drift is the rule picking the wrong joint (fixable by placement) and
two thirds is inherent.

### 4.10 A vertex band is not a cross-section

Using dominant-joint-filtered vertices in a ±10 mm band gives only 25–42 points across a ~1 m
section perimeter. That under-reports depth by **45 mm** at y = −0.05 and misses the entire
lateral pelvis at y = −0.35 (0.287 vs 0.371 true width). **Cut the mesh by the plane and use
the resulting closed loops**, interpolating skinning weights along the cut edges. I used the
vertex band throughout and it is a real weakness of my measurement layer.

### 4.11 The gluteal band belongs to nobody, and its medial locus is in the wrong place

The thigh has no cross-section separable from the pelvis below t = 0.167 of the femur; the
skin no limb chain can own is **0.2469 m² = 12.6 % of the body**. Worse: the largest circle
inscribed in a pelvic cross-section sits **120 mm in front of** the buttock, so a dense
900-sphere medial cover *still* leaves a 48 mm hole across the seat. It is the wrong locus,
not too few spheres.

### 4.12 A sphere at a joint centre binds non-deterministically

Placed exactly on a joint, 20 of 24 joints bind to the kintree parent, but floating-point
ordering flips it to the child at `L_hip`, `R_hip` and `spine2`. **Offsetting 1 mm back along
the incoming bone already makes every case deterministic; 10–20 mm is the safe authoring
value.**

### 4.13 `Trispheremesh` never raises on a degenerate triangle

It silently drops both caps when `|cross(v1,v2)| ≤ 1e-6` and returns just the three cones —
`n_cells` falls 56 → 54 and nothing else signals it. There are **three different degeneracy
thresholds** in the codebase (1e-6, 1e-12, 1e-10) and none is meaningful. The real condition
for a slab to exist is that a plane tangent to all three spheres exists, i.e. the in-plane
part of the tangent normal is shorter than unit — which fails when **radii vary faster than
the centres are spaced**, not only when centres are collinear. Three of my triangles failed
this with perfectly good centre geometry (radii 85/126/135 mm, |w|² = 1.82).

Also: the rendered prysmoid uses the *centre triangle's* normal, not the tangent plane, so
its flat caps are chords when radii differ — sag 1.85 mm at 2:1 radii, 5.80 mm at 4:1.

### 4.14 The user's template has 13 stale bindings

Recomputing `rest_offset` with the tool's current rule reproduces **47 of 60 exactly**.
Sphere 44 is bound to `R_elbow` while sitting *past* the `R_wrist` joint — 269 mm of
rest-offset error. Spheres 34/45/37 are off by exactly 3–4 mm, the signature of a centre
nudged after binding. `nearest_vertex_index` and `rest_radius` are stale on ~48 of 60. Also,
`secondary_joint`/`lbs_weight` in the file come from the **old** rule (the skinning-weight
version now commented out at `sphere_manager.py:755`). Worth telling the user regardless.

### 4.15 Small mechanical facts

- The tool **rounds centres to 3 decimals on load** (`_trim_center_xyz_3decimals`). Symmetry
  survives because rounding is sign-symmetric — but verify it.
- `rest_radius` is a **ratio** (radius ÷ distance to nearest vertex), not a length.
- `+X is the LEFT side` in this SMPL model (joint 16, L_shoulder, at x = +0.199).
- SMPL male neutral: 6890 verts, 13776 faces, watertight (V−E+F = 2), `weights` (6890, 24).
- Body z-extent is only 309 mm total, and the belly front reaches z = +174 mm at y = −0.10.

---

## 5. Failed approaches and dead ends

### 5.1 A fine multi-column medial sheet for the whole trunk

**Tried:** 3 then 5 columns × 8–10 rows, joined by 18–32 quads. **Why:** the medial-axis view
says a triangle covers a sheet on both sides, so a torso should be cheap in triangles.
**Result:** 40 trunk spheres, 32 quads, RMS 8.4 mm. **Why it failed:** two independent
reasons — the lower trunk is a tube not a sheet (4.5), and small slabs buy nothing (4.4).
**Salvage:** the sheet is right for the *upper* trunk and is still used there.
**Instead:** two structures, big cells.

### 5.2 A ring cage for the lower trunk

**Tried:** 4 spheres per row arranged around the section, joined into a quad tube.
**Why:** solving the per-section minimax placement of K disks, a free ring beats a collinear
row decisively — 4.8 mm vs 10.9 mm at y = −0.160, 5.8 vs 10.1 at y = −0.230.
**Result:** in the full 3-D union it *loses*: 4.87 % of vertices beyond 15 mm against the
ladder's 4.62 %, walk-break 1.71 % vs 1.62 %, for **10 more spheres and 14 more quads**.
**Why it failed:** the per-section argument ignores the slabs. The ladder's left-to-right
slab already reaches the front and back walls that the ring buys with separate spheres.
**Salvage:** the builder is kept as `--lower-structure ring`; the lesson (2-D section
reasoning does not predict 3-D union quality) is worth more than the code.

### 5.3 Deleting the pubic node because its sphere was contained

**Tried:** removing it when the validator flagged it as swallowed by the bottom rung's 135 mm
sphere. **Result:** worst gap went 33.1 → 43.1 mm. **Why it failed:** the coverage was never
coming from the sphere — it came from its **cones** down to the hips and thighs.
**Lesson:** "sphere is contained in another" and "this connection is useless" are different
statements. My validator conflated them; I fixed it (a dead connection now counts against you
only if it is also not load-bearing for the walk).

### 5.4 Flat 50/50 shrink when relieving protrusion

**Tried:** charging a cone's excess protrusion equally to both endpoint spheres.
**Result:** 5 mm of extra RMS and roughly double the badly-covered fraction.
**Fix:** charge each sphere its actual share — the barycentric weight of the medial point
nearest the offending surface point. **Still imperfect** but much better.

### 5.5 Two rejected column-placement rules

- **Fraction of the medial span.** Collapses on a row whose span is short: at the top of the
  trunk the medial segment is 33 mm while the spheres are 87 mm in radius, so columns at 0
  and 0.55 land 18 mm apart with radii 18 mm different — one sphere entirely inside the other.
- **Target radius ratio.** Better, but still only bounds separation indirectly and fails on a
  steep profile (a column asked for 0.82 of the peak landed 18.2 mm out with its radius
  18.3 mm smaller — inside by 0.1 mm).
- **What works: reach fractions.** Write `reach(p) = p + r(p)`. Sphere k is contained in an
  inner sphere j exactly when `reach(p_k) ≤ reach(p_j)`. So placing columns by *how far each
  reaches* makes a swallowed sphere **impossible by construction**, whatever the section's
  shape. Use this from the start.

### 5.6 Containment test for medial-axis maximality

**Tried:** filtering the clearance ridge to circles not contained in another.
**Why it failed:** for an ellipse the containment at the medial endpoint is exactly tangent,
so on a polygonal hull the test decides by rounding — it returned the whole symmetry segment
(±182 mm instead of ±70 mm on the trunk). **Fix:** the actual medial criterion — the circle
touches the outline at two places whose normals differ by more than ~50°.

### 5.7 Placing the buttock from a masked cross-section

**Tried:** restricting the pelvis mask to posterior vertices and taking the section's medial
point. **Result:** spheres of **14.7 mm radius** hugging the skin — the thin slice's inscribed
circle is tiny. **Fix:** `blobs`, which grids a box, keeps interior points (sign of the
nearest vertex normal), scores by distance to the nearest surface vertex, and greedily takes
separated maxima. Cap the box's z so the search cannot wander forward into the pelvis where
the genuinely biggest spheres are.

### 5.8 The multi-agent workflow, partially

Sixteen agents: 5 of 6 measurement probes returned (one died on a `StructuredOutput` JSON
parse failure — 13.7 KB of output with unescaped characters), all 3 designs returned, and
**all 6 judges plus the synthesis failed on the account session limit**. The measurement
phase was extremely valuable (most of §4 comes from it). The design phase was worth reading
but I had to mine it by hand. **Lesson:** run measurement early and let it *finish before
designing* — I started designing on my own assumptions and the results overturned them.
Keep structured-output schemas small and put long prose in files, not in the return value.

---

## 6. Obstacles

### 6.1 The prysmoid inside-triangle test rejected every point under one face

**Symptom:** exactly one probe point in 150 disagreed with brute force, by 3.28 mm.
**Root cause:** the tangency triangle is wound one way against the top face's normal and the
other way against the bottom's, so demanding all-positive edge cross products silently
rejected every point under one of the two faces. **Diagnosis:** brute-force minimisation over
a dense barycentric grid, then printing the worst offenders and hand-checking the geometry.
**Fix:** require the three turn values to *agree in sign*, not to be positive. **Robust.**
**Will reappear** in any fresh implementation of a slab SDF — it is a one-character-class bug
that only a numerical reference catches.

### 6.2 The medial-axis extraction returned the symmetry segment

Covered in 5.6. Cost about 45 minutes and produced a trunk with dead spheres.

### 6.3 The ring's anchor fell to floating-point noise

The clearance field of a symmetrised section is symmetric in x and its maximum is on x = 0,
but where the maximum is nearly flat `argmax` lands tens of millimetres off-centre — three
rows out of five came out asymmetric. **Fix:** pin the anchor to the midline
(`best_transverse([0.0])`). Same class of bug as 6.2: *never take an argmax over a field
whose maximum you can characterise analytically.*

### 6.4 Radius floors compounding across pipeline stages

**Symptom:** the pubic sphere was 68 mm placed, 45 mm after `optimize_spheres`, **27 mm**
after the shrink pass — then swallowed by its neighbour and no longer a bumper at all.
**Root cause:** the shrink pass floored at 70 % of the radius *it was handed*, which was
already 70 % of the placed radius. **Fix:** pass the floor as an explicit array derived from
the **placed** radii. **Pragmatic, not robust** — the real fix is a fitter that understands
cones (§7).

### 6.5 Both thighs attached to the same side of the body

A connection written to a bare grid index `grid[(row, 0)]` was not renamed by `mirror`, so
the mirrored connection referenced the *same* trunk sphere. The result was a cone straight
across the crotch, 27.5 mm outside the skin. **Diagnosed** only by listing which bumpers owned
the worst-protruding vertices. **Fix:** name every sheet column with an `L_`/`M_` prefix.
**Will reappear** in any mirror-by-name scheme; make it structurally impossible.

### 6.6 Environment friction

- A ~25 KB Bash heredoc silently failed to parse. Use the `Write` tool for anything large.
- `pathlib.read_text()` defaults to cp1252 on this Windows box and choked on the notes file.
  Always pass `encoding="utf-8"`.
- Foreground Python runs of the full sweep exceeded the 2-minute tool timeout; long sweeps
  must be backgrounded.

### 6.7 What consumed the most effort, honestly

**Hand-iterating the design.** Roughly two thirds of the session was edit-a-constant,
re-run, read-the-report. Rows, columns, station parameters, masks, thresholds, floors — each
change a separate 30 s cycle, each result requiring interpretation. This is the thing to fix
(§7.1).

---

## 7. With hindsight — what I would do differently

### 7.1 I would have built a scoring loop and searched, instead of hand-tuning

This is the big one. I built an excellent validator and then used it as a **report** rather
than an **objective**. Every design parameter — 5 row heights, column reach fractions, 4×4
station parameters per limb, 3 mask thresholds, the radius floor, the protrusion target — was
chosen by hand.

With a `score(params) → float` wrapping the validator and even a crude search (coordinate
descent, or random restarts over a few dozen evaluations), I would have found better
configurations in less wall-clock time and *known* they were better. The pipeline runs in
~4 s without poses; hundreds of evaluations are affordable.

**Do this on day one.** Define a scalar objective early — something like
`w1·max(0, gap_frac − target) + w2·max(0, protrusion − target) + w3·walk_break + w4·spheres/100`
— and make every structural choice a searchable parameter.

### 7.2 I would have measured the body *before* designing anything

I launched the measurement fan-out and then started designing while it ran. When the results
came back they overturned two of my three trunk decisions (sheet-vs-tube, fine-vs-coarse
slabs) and I rebuilt the torso three times. **Measure first. Design second.** The measurement
phase is cheap and its findings are load-bearing.

### 7.3 I would use mesh-plane cuts, not vertex bands, for cross-sections

See 4.10. My `section_at` takes vertices within a slab, which under-samples badly (25–42
points across a 1 m perimeter) and gave a 45 mm depth error at one height. Cutting the
watertight mesh by the plane and interpolating skinning weights along cut edges is barely
more code and is correct. **This is the weakest part of my measurement layer.**

### 7.4 I would replace `optimize_spheres` for this use, or wrap it very differently

`optimize_spheres` fits each sphere to the vertices nearest its own centre and has no idea
cones or slabs exist. Measured, it *increased* sphere-mesh protrusion 28.0 → 39.8 mm on one
configuration while reducing its own objective, and it collapses spheres with small
partitions. I then bolted on a shrink pass and a floor to undo the damage — two workarounds
stacked on a tool solving a different problem.

A better refiner optimises **radii only** (centres come from the medial construction, which
is already right) against the **exact sphere-mesh** distance with a one-sided protrusion
penalty. That is a small, well-conditioned, convex-ish problem in ~80 variables and it would
replace both `refine` and `relieve_protrusion`. **I would build this.**

### 7.5 I would have plotted coverage by direction on day one

The missing-midline-column bug (4.6, 17 mm of worst gap for five spheres) was visible in a
single number: *the gap vertices were all at z > +0.108*. I found it late, by listing gap
clusters with their bounding boxes. A per-direction or per-region coverage summary printed on
every run would have caught it in the first iteration. **Make the report say *where*, not just
*how much*, from the start.**

### 7.6 I would name every sphere from the first line of code

I introduced column names only after the both-thighs-same-side bug. Names are free and
prevent a whole class of silent errors.

### 7.7 I would ask the user two calibration questions up front

- **What walk depth does the solver actually afford before falling back?** My budget of 3 is
  a guess and it sets the headline metric.
- **What is the acceptable protrusion, in millimetres?** The whole protrusion↔coverage
  trade-off is a single scalar the user could set in one sentence, and I spent a lot of
  effort exploring a curve they may have a firm opinion about.

### 7.8 I underestimated, and overestimated

**Underestimated:** how much the *shape* of each region matters (tube vs sheet vs bulge) and
how badly a uniform structure serves a non-uniform body. Also how much of the remaining error
is a floor rather than a fault (dead joints, annulus floor, rigid binding).

**Overestimated:** the value of triangles. I designed the whole first torso around "a slab
covers both sides", which is true and almost entirely irrelevant at the scales involved.

### 7.9 What is acceptable but not ideal

The `relieve_protrusion` shrink pass. It works, it terminates, its floor is principled, and
its blame attribution is now correct — but it is fundamentally a patch for using the wrong
fitter (7.4).

---

## 8. Things that work but could be better

### Important

**The refine → relieve pipeline.** *Current:* medial placement, `optimize_spheres`, floor,
weighted shrink, symmetrize. *Works* because each stage's damage is bounded by the next.
*Suboptimal* because two of the four stages exist to undo the third. *Better:* radius-only
optimisation against the exact sphere-mesh SDF with a one-sided penalty (7.4). *Worth it* —
it would likely take both the protrusion and gap numbers below the hand-made template.

**`section_at` uses a vertex band.** *Better:* mesh-plane cut (7.3). *Worth it* — it is the
input to every placement decision.

**The `non_adjacent_overlap` check.** *Current:* flags bumper pairs sharing no sphere and
more than 3 hops apart that interpenetrate. *Suboptimal:* it fails on every design ever
tested, including the user's, at 50–54 % of bumpers. Either the radius is wrong or the metric
should be "fraction of a bumper's surface that is *never* the nearest" — which is the thing
that actually costs the solver. *Worth deciding* before shipping a validator that always
says no.

### Nice to have

**`blobs` is a grid search.** 6 mm resolution over a box, with a `cKDTree` query per point.
Fine at current scale; a coarse-to-fine refinement would be faster and more accurate.

**Sphere ordering in the output** is generation order. Grouping by body part would make the
file far easier to inspect in the tool.

**`json2txt.py` does not read `quad_groups`.** A GUI round trip loses the quads. Twenty lines
to fix.

### Stylistic

The `rung()` index-remapping that lets ring and ladder share the same seam code is clever and
hard to read. Two explicit code paths would be clearer.

---

## 9. Performance / scalability

**Measured on this machine** (Windows, CPU torch):

- Full pipeline, no poses: **~4 s**. With 54 poses: **~30 s**.
- Validation suite alone on one template, 54 poses: **4.2 s**.
- `BumperGraph.distances` for 3725 points × 225 bumpers: well under a second. The cone term
  allocates `(M, K, 3)` intermediates — 3725 × 120 × 3 floats ≈ 10 MB, fine; at 10× the
  bumper count you would want chunking.
- `hop_distances` (BFS all-pairs over ~225 bumpers): negligible, and cached.
- **`non_adjacent_overlap` is the expensive check**: it calls `graph.distances` once per
  bumper, so O(B) full distance evaluations — O(B²) work. It dominates the no-pose run. It
  short-circuits when a bumper has no far candidates, which helps.
- **`inscribed_radius` is O(N_spheres × 6890)** in a Python loop. Called on every placement.
  Use a `cKDTree` — I did in `blobs` but not here.
- `optimize_spheres` with `max_outer=30` on ~3700 band vertices: a couple of seconds.
- `brute_force_triangle` in the tests uses ~400 k barycentric samples per case; the test
  file takes ~30 s and that is fine for a correctness reference.

**Surprisingly cheap:** the exact SDF, the pose sweep (SMPL forward is ~50 ms), BFS.
**Surprisingly expensive:** the overlap check, and the *human* loop of hand-tuning.

**Not implemented but considered:** caching the SMPL forward per pose across runs;
vectorising `inscribed_radius`; a coarse-to-fine `blobs`; chunking `distances` by point block.

---

## 10. Correctness / edge cases

### Handled

- Degenerate cones (one sphere swallowing another) fall back to the enclosing sphere.
- Triangles with no tangent plane are detected and reported (the real condition, not a
  cross-product threshold).
- Near-collinear centres, zero-area triangles.
- Interior points get clearance to the *nearest* surface, not the farthest (a real trap: the
  two tangent planes of a slab are not independent candidates — the correct one is chosen by
  which side of the centre triangle the point lies on).
- Quad rim vs diagonal: a quad's rim is its perimeter; the 0–2 diagonal is an interior split
  for distance only, never an adjacency edge.
- Ties between a sphere and its cone (D2).
- Sections too sparse to hull, medial loci that come back empty, columns that would collapse
  (all raise with an actionable message).

### Not handled / silent-failure risks

- **Non-convex cross-sections.** Every section is convex-hulled. Fine for trunk and limb
  sections; **wrong at the crotch and the armpit**, exactly where my worst residual errors
  are. A hull across the crotch bridges the two thighs and puts an inscribed circle in
  mid-air. I dodged this with `|x|` and side masks — a workaround, not a fix.
- **`blobs`' inside test** is the sign of the nearest vertex's normal. Exact enough in a box
  chosen to be interior; it would misclassify near a thin feature or a fold.
- **Quads are assumed planar-ish.** A non-planar quad is split into two triangles for
  distance but treated as one bumper for adjacency. If the engine treats it differently, the
  two disagree.
- **`walk_budget = 3` is unvalidated** against the real solver.
- **The pose sweep subsamples vertices** (`pose_vertex_stride=3`), so the reported worst-case
  protrusion is a lower bound.
- **`optimize_spheres` can move a sphere across a joint boundary** and change its binding.
  `max_offset` bounds this but nothing checks it afterwards. **Add a post-fit assertion that
  no sphere's `assign_parent_joint_like_tool` result changed.**
- No check that a cone spans at most one bone (the JOINT-FIRST design verified 0 of 108
  doubles spanned more than one bone; mine is unverified).

### Should be explicitly tested next time

- Every sphere's tool-assigned joint equals the intended one (a one-line assertion; the
  JOINT-FIRST agent reported 63/63, mine is unknown).
- Both spheres of every 2-connection bound to the same joint or a joint and its direct child.
- Round trip: generate → write JSON → load with the tool's own parser → re-export → compare.
- Symmetry after the tool's 3-decimal rounding (I do check this, ad hoc).
- The generator on a non-neutral `beta`, which nothing has exercised.

---

## 11. Testing and validation

**What exists.** `scripts/test_topology_bumpers.py` — 30 checks, all passing. Every closed
form is checked against brute-force minimisation of `min over the simplex of |p − c(λ)| −
r(λ)` on a dense λ grid: 1e-12 agreement for cones, ~4e-6 for slabs. Graph structure,
quad rim/diagonal semantics, ownership tie-breaks, and quad-vs-split-triangle equivalence are
all pinned. **This test caught a real bug** (6.1) that no amount of reading would have.

**Other validation performed:**

- Engine text output **byte-identical** to `scripts/json2txt.py` on `template.json`.
- Computed LBS bindings reproduce the tool's stored values for 47/60 spheres (the rest are
  stale in their file — verified individually).
- JSON keys match exactly what `_build_from_payload` reads.
- Symmetry survives the tool's load-time rounding (0 unpaired after `round(x, 3)`).
- The generator runs on the **female** body, confirming no coordinates are baked in.
- Baseline established on `data/template.json` before touching anything, and re-measured
  identically at the end.

**Not validated.**

- **Nothing has been opened in the actual tool.** `SphereManager` needs a PyVista plotter and
  I ran headless. The JSON is schema-correct but unproven in the GUI. **Do this first.**
- **The walk-break metric is not validated against the real solver.** The adjacency rules are
  inferred from the user's prose and from `Spheremesh`/`Trispheremesh`; the hop budget is a
  guess.
- **No visual inspection.** I never rendered the sphere-mesh. Every judgement is numerical.
  Several bugs (both thighs on one side; the ring's asymmetry) would have been obvious in one
  screenshot.
- The pose sweep uses subsampled vertices and stride-5 mocap frames.
- `blobs`, `ring` and the `--gender female` path have no unit tests at all.

**Things that "seem to work" but are not rigorously validated:** the whole `generate.py`
builder layer. It is exercised only end to end.

---

## 12. Codebase / file map

### Existing code you must read

| File | Why it matters |
| --- | --- |
| `src/sphere_manager.py:697` `assign_parent_joint` | **The** binding rule. Closest kintree bone → bind to its parent. Secondary joint = that bone's child; the old skinning-weight version is commented out at `:755`. |
| `src/sphere_manager.py:826` `update_sphere_positions` | Posing. The blend is commented out — it is rigid single-joint. |
| `src/sphere_manager.py:34` `_trim_center_xyz_3decimals` | Centres are rounded to mm **on load**. |
| `src/sphere_manager.py:1292` `_build_from_payload` | Exactly which JSON keys are read; 2- and 3-index connections only. |
| `src/sphere_manager.py:92,119` radius scaling | `rest_radius` is a *ratio* to the nearest-vertex distance. |
| `src/spheremesh.py` | `Spheremesh` builds the true tangent cone (`l = √(d²−Δr²)`, base planes) but renders only the frustum — the end caps are dead code in a triple-quoted string at `:36–43`. `Trispheremesh` builds two triangle faces at `c_i ± n·r_i` using the *centre triangle's* normal. |
| `src/sphere_fit.py:357` `_segment_pairs` | Decomposes a 3-connection into edges; the slab is never evaluated. |
| `src/sphere_fit.py:375` `spheremesh_distance` | The external metric. **Do not change it** (`plan.md`). Approximate — see 4.3. |
| `src/sphere_fit.py:665` `optimize_spheres` | The existing fitter. Read its docstring; it is excellent and states its objective precisely. |
| `scripts/json2txt.py` | The engine text format: title; `spheres triples doubles quads`; sphere lines truncated (not rounded) to 4 decimals; connections grouped triples, doubles, quads. |
| `plan.md` | The user's own brief for the fitter. Sets the expected standard of rigour and forbids changing `spheremesh_distance`. |
| `data/template.json` | The baseline. Its arms are the good part. |

### Code I added

| File | Key contents |
| --- | --- |
| `src/topology/bumpers.py` | `round_cone_distances` (exact closed form), `tangent_plane`, `prysmoid_distances`, `BumperGraph` (`distances`, `owners`, `hop_distances`, `components`). The docstring explains the union-of-balls formulation. |
| `src/topology/anatomy.py` | `Body`, `Section` (`half_planes`, `best_transverse`, `medial_axis`), `Station`, `bone_stations`, `section_at`, `assign_parent_joint_like_tool`, `pose_spheres_like_tool`, `random_poses`, `mocap_poses`. |
| `src/topology/generate.py` | `TopologySpec`, `chain`, `sheet`, `ring`, `blobs`, `stub`, `mirror`, `symmetrize`, `refine`, `relieve_protrusion`, `_medial_weights`. |
| `src/topology/validate.py` | `Thresholds`, `validate`, `modelled_vertex_mask` (with truncation planes), `surface_edges`, `walk_breaks`, `non_adjacent_overlap`, `degenerate_primitives`, `mirror_asymmetry`. |
| `src/topology/export.py` | `compute_bindings`, `build_payload`, `split_quads`, `write_tool_json`, `write_engine_text`. |
| `scripts/build_topology.py` | The design itself, in the constants at the top and `build_spec`. |

**Surprising:** `bumpers.py` and `validate.py` are the durable assets. `build_topology.py` is
the disposable part — it is one design among many and its constants are unsearched guesses.

---

## 13. Technical knowledge to start with

> If I could send one page:

1. **Binding is rigid and single-joint** (`sphere_manager.py:697`, blend commented out at
   `:826`). Closest kintree *bone*, bind to that bone's *parent*. All artifact reasoning uses
   this, not SMPL LBS.
2. **Never let a bumper cross a joint.** Inside one bone: 0.3–14 mm RMS. Across the elbow:
   260 mm. p95 is already 77 mm two centimetres past the elbow.
3. **Never place a sphere exactly on a joint centre** — the binding tie-break is
   floating-point noise at `L_hip`, `R_hip`, `spine2`. Back off 10–20 mm.
4. **Slab gain = √(ρ² + r²) − r.** Small triangles between fat spheres buy nothing. 15 of 18
   of the user's triangles buy under 5 mm. Make slabs *big relative to their spheres* or use
   cones.
5. **`spheremesh_distance` under-reports coverage**, dominated by its cone approximation
   (up to 23.8 mm phantom gap; 60× the missing-slab error). Do not optimise against it.
   Write the exact SDF: for a sphere-mesh, `d(p) = min over the medial simplex of
   |p − c(λ)| − r(λ)` is exact **inside and outside**, because the sphere-mesh *is* the union
   of interpolated balls.
6. **The lower trunk is a tube, the upper trunk is a sheet.** Aspect 1.21–1.45 below y = 0.10,
   rising to 2.29 at the clavicle. Two structures, not one.
7. **A 2-column sheet is ~18 mm too thin at the midline** — the slab depth at x = 0 is exactly
   2r regardless of spacing. Always include a midline column on the lower trunk.
8. **The bone axis is not the medial axis** (up to 36 mm on the shin). Place on the section's
   inscribed-circle centre; snapping to the skeleton costs 17.5 mm RMS on the leg.
9. **3–4 spheres per limb segment.** More is measurably wasted; the floor is the
   cross-section's departure from a circle (4–14 mm), not the taper.
10. **`spine1` and `neck` are dead joints** (max weight 0.73/0.71). No rigid sphere tracks the
    lumbar region or the neck. Accept it.
11. **The gluteal band is 12.6 % of the body and no chain can own it**, and its medial locus
    is 120 mm in front of the buttock. It needs a largest-inscribed-sphere-in-a-box search.
12. **Ownership must prefer the lowest-dimensional bumper** on a tie; the walk metric must be
    a hop **count** (consecutive cones on a chain are 2 hops apart, which is normal).
13. **Mirror the spec by name; never match halves by proximity.** Force symmetry after every
    geometry pass. The tool rounds centres to 3 decimals on load.
14. **`+X is LEFT.** SMPL male: 6890 verts, watertight, `weights` (6890, 24).
15. **Cut the mesh by a plane for cross-sections** — a vertex band gives 25–42 points across a
    1 m perimeter and a 45 mm depth error.

---

## 14. Recommended reimplementation strategy

### Phase 0 — Understand (do not write generator code yet)

1. Read `sphere_manager.py:697` and `:826`, `spheremesh.py`, `sphere_fit.py`'s
   `_segment_pairs`/`spheremesh_distance`/`optimize_spheres` docstring, `json2txt.py`, and
   `plan.md`.
2. **Ask the user:** the solver's real walk-depth budget; the acceptable protrusion in mm;
   whether they want the `secondary_joint` blend re-enabled (it is the real fix for posed
   artifacts and it is *their* commented-out code).
3. Load `data/template.json`, open it in the tool, and look at it. I never did and it cost me.

### Phase 1 — Build the metric and the validator first

4. Exact SDF for sphere / cone / prysmoid / quad, **with a brute-force test from the first
   commit**. This is non-negotiable; it caught a real bug and every later number depends on it.
5. `BumperGraph` with hop distances and lowest-dimension ownership.
6. The validator: budgets, connectivity, degeneracy, coverage, protrusion, walk-break
   histogram, **per-region and per-direction breakdown**, pose sweep, symmetry.
7. **Baseline `data/template.json` with it** and calibrate the thresholds against that. If a
   check fails on the user's own working template, the check is probably wrong.

### Phase 2 — Measure the body, and finish measuring before designing

8. Mesh-plane cross-sections with interpolated weights, everywhere.
9. Per-region: aspect ratio, inscribed radius, medial extent, contact-angle medial axis.
10. Per-joint: dominance, dead joints, transition bands, rigid-vs-LBS span error over the
    pose set.
11. Only now decide the structures.

### Phase 3 — Generator as a parameterised, searchable spec

12. `TopologySpec` with **named** spheres from line one. Builders: `chain` (medial stations),
    `sheet`, `blobs`. Column placement by **reach fraction** (5.5) so swallowing is
    impossible. `mirror` + `symmetrize`.
13. **Define a scalar objective and wire a search over the parameters immediately.** Row
    heights, station parameters, mask thresholds, floors. Even coordinate descent over ~50
    evaluations will beat hand-tuning.
14. Refinement: **radii only, against the exact SDF, one-sided protrusion penalty.** Do not
    use `optimize_spheres` as the primary refiner and do not stack a shrink pass on it.
15. Post-fit assertion: no sphere's tool-assigned joint changed.

### Phase 4 — Export and prove it

16. JSON with bindings + `quad_groups`; engine text with quads; **patch `json2txt.py` to read
    `quad_groups`** so a GUI round trip is lossless.
17. Open the result in the tool. Screenshot it. Then iterate on numbers.

### Explicitly do NOT

- Do not modify `spheremesh_distance` (`plan.md` forbids it).
- Do not build a fine multi-column sheet for the whole trunk (5.1).
- Do not use small triangles for coverage (4.4).
- Do not place spheres on bone axes (4.7) or at joint centres (4.12).
- Do not trust an `argmax` over a field whose maximum you can characterise (6.3).
- Do not spend a ring-cage's worth of effort on 2-D section reasoning without checking the
  3-D union (5.2).

### Where experimentation is worthwhile

The **armpit and groin**, which are the entire remaining coverage residual and where the
convex-hull assumption breaks. And the **protrusion↔coverage operating point**, which is one
scalar the user can probably just tell you.

---

## 15. Priority of improvements

### Critical

1. **A search loop over design parameters instead of hand-tuning** (7.1). Biggest single win.
2. **Radius-only refinement against the exact SDF**, replacing `optimize_spheres` + shrink +
   floors (7.4). Removes two workarounds and probably beats the hand-made template outright.
3. **Mesh-plane cross-sections** instead of vertex bands (7.3). Everything downstream depends
   on this input being right.
4. **Open the output in the actual tool** before iterating on numbers (11).

### Important

5. Per-region *and per-direction* coverage in the report from the first run (7.5).
6. Post-fit binding-invariance assertion, and a "one bone per cone" check (10).
7. Resolve or recalibrate `non_adjacent_overlap` — it currently fails on everything (8).
8. Handle the armpit and groin explicitly; the convex hull is wrong there (10).
9. Patch `json2txt.py` for `quad_groups` (17).

### Optional

10. `cKDTree` in `inscribed_radius`; coarse-to-fine `blobs`.
11. Order spheres by body group in the output.
12. Unit tests for `sheet`/`ring`/`blobs`.

### Do not change

13. **The exact SDF and its brute-force test.** Correct, fast, and it earned its keep.
14. **Lowest-dimension ownership tie-break** and the **hop-count walk metric**.
15. **Placement rules keyed by name, left half mirrored, symmetrized after each pass.**
16. **Radii from the skin** (maximal inscribed) as the starting point.
17. **Two structures for the trunk**; 3–4 stations per limb segment; deltoid and buttock as
    their own spheres; chains stopping short of joints on both sides.
18. **`TOPOLOGY_NOTES.md` as a running, honest log.** The user asked for visible progress and
    it doubles as the argument for every constant.

---

## 16. Original goals → current results

| Original goal | Current result | Status | Notes |
| --- | --- | --- | --- |
| Automate topology (count, position, connections) | `scripts/build_topology.py` + `src/topology/`; no coordinates, runs on male and female | **COMPLETE** | But the design *parameters* are hand-chosen, not searched |
| Fit the SMPL male surface | RMS 6.64 mm vs hand-made 6.12; worst gap 32.2 vs 35.5 | **MOSTLY COMPLETE** | 3.54 % of vertices >15 mm off vs 0.83 % — the one clear regression |
| Under 100 spheres | 79 | **COMPLETE** | 20 spheres of headroom left unspent |
| Under ~20 three-sphere connections | 14 triples + 12 quads in the engine file | **COMPLETE** | JSON shows 26 triples because quads are split for the GUI |
| Don't overlap bumpers too much | Worst protrusion 15.9 mm vs 44.9; check still fails at the 3-hop radius | **PARTIAL** | The check fails on every design incl. the user's — likely mis-calibrated |
| Bumper graph = connected cover for the watchdog | 1 component; **0 unreachable** steps; 1.84 % over 3 hops vs 3.68 % | **MOSTLY COMPLETE** | 2× better than baseline; budget of 3 hops is unvalidated |
| No animation artifacts | Posed worst 46.7 mm vs 59.8; mean 35.7 vs 44.8 | **PARTIAL** | Near the floor for rigid binding; real fix is the commented-out blend |
| Keep the arms' character | 4 medial stations per segment, split at the elbow, deltoid its own sphere | **COMPLETE** | Elbow-crossing prohibition is now measured and enforced |
| Rebuild the torso | Two structures (ladder + sheet), 26 spheres | **COMPLETE** | Torso protrusion 44.9 → 15.9 mm |
| Truncate hands/feet/head | Wrist and ankle stubs, 2 neck spheres, no head | **COMPLETE** | Truncation planes defined in `validate.py` |
| Quads engine-side, triangles in JSON | Both, with `quad_groups` recorded | **COMPLETE** | `json2txt.py` doesn't read `quad_groups` yet — **WORKAROUND** |
| Automated pose validation, both sources | 54 poses (24 random + 30 mocap), in the suite | **COMPLETE** | Vertex-subsampled, so worst-case is a lower bound |
| Show intermediate output | `data/generated/` + `TOPOLOGY_NOTES.md` from early on | **COMPLETE** | The user had to ask twice before I did this |
| Mirror symmetry (inferred) | Exact, and survives the tool's rounding | **COMPLETE** | Their template has 38/60 unpaired |
| Use the existing fitter (inferred) | Used via `refine()` | **WORKAROUND** | It optimises the wrong objective; two passes exist to undo it |

---

## 17. Known technical debt

| Debt | Why introduced | Risk | Remove? | Better solution |
| --- | --- | --- | --- | --- |
| `relieve_protrusion` shrink pass | To undo protrusion that `optimize_spheres` introduces | Medium — trades coverage for clearance along a curve nobody chose | **Yes** | Radius-only fit against the exact SDF with a one-sided penalty |
| Radius floors (`--min-radius-fraction`, explicit `floor` array) | To stop the shrink pass and the fitter collapsing spheres | Medium — a magic number that silently caps the fit | **Yes**, with the above | Same |
| `json2txt.py` ignores `quad_groups` | Out of scope at the time; the generator writes both files directly | Low but sharp — a GUI round trip silently downgrades quads to triangles | **Yes** | 20 lines in `json2txt.py` |
| Convex-hull sections dodged with `|x|` and side masks at crotch/pelvis | The hull bridges the legs; masking was the quick fix | Medium — masks are tuned to this body and may not transfer | Yes | Mesh-plane cut with connected-component selection |
| `walk_budget = 3` | No ground truth for the solver's real limit | Medium — it sets the headline metric | Ask the user | A number from the engine |
| `--lower-structure ring` kept though rejected | To keep a measured comparison reproducible | Low | Optional | Move the comparison into the notes and delete the code |
| `rung()` index remapping shared by ring and ladder | To avoid duplicating seam code | Low, but genuinely confusing | Optional | Two explicit paths |
| `data/generated/` committed | The user asked to see output | Low | No | Fine |
| `inscribed_radius` is an O(N × 6890) Python loop | Expedient | Low at this scale | Optional | `cKDTree` |
| No test coverage for `generate.py` | Ran out of session before writing it | **High** — the builder layer is exercised only end to end | **Yes** | Unit tests per builder with synthetic bodies |

---

## 18. Final handoff summary

### What I would keep

The **exact SDF with its brute-force test** — it is correct, fast, caught a real bug, and
every trustworthy number in this project came through it. The **hop-count walk metric** and
**lowest-dimension ownership**, both of which took a wrong first version to get right. **Rules
not coordinates**, with **names** and **spec-level mirroring**. **Radii from the skin.** **Two
structures for the trunk.** And the **running notes file**, which the user asked for and which
turned out to be the best defence of every constant in the design.

### What I would change

**Search instead of hand-tune.** I built a validator and then used it as a report. Two thirds
of the session went into edit-a-constant / re-run / interpret. **Replace `optimize_spheres`**
for this use with a radius-only fit against the exact SDF, and delete the two passes that
exist to undo it. **Cut the mesh by planes** instead of banding vertices. **Report where the
error is, not just how much** — the 17 mm midline bug was one printed bounding box away from
being obvious on the first run.

### What I would investigate first

Open `data/template.json` **in the actual tool** and look at it, then open a generated file
and confirm it loads. Ask the user the solver's real walk-depth budget and their acceptable
protrusion in millimetres — two sentences that would have saved hours of exploring a
trade-off curve. Check whether they want the `secondary_joint` blend re-enabled; it is their
own commented-out code and it is the only real fix for the remaining posed protrusion.

### Biggest lesson

**Measure the shape before choosing the structure, and measure the whole 3-D result rather
than a 2-D proxy.** I designed a medial sheet for a torso that turned out to be a tube, and
then nearly replaced it with a ring because rings win per-cross-section — they lose in the
union. Both errors came from reasoning about a plane when the object is a solid. Every
structural decision that survived was one I made *after* the numbers came back.

### Biggest opportunity

**Close the loop.** The validator already computes everything a search needs and the pipeline
runs in four seconds. A stronger implementation that defines a scalar objective and searches
the design parameters — row heights, station positions, column reaches, the
protrusion/coverage operating point — should beat the hand-made template on *both* coverage
and protrusion simultaneously, which my hand-tuned version does not. The 20 unspent spheres
and the unmodelled armpit and groin are sitting right there.
