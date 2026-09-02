# Implementation Brief — Fixing radius inflation in the sphere fitter

Incremental changes to the module built from `brief_sphere_fit.md`. The numerics there are
correct; the objective is not. Do not rewrite the module — the Gauss-Newton core, the
projection, the freeze rules and the reporting all stay.

## Diagnosis

The fitter inflates radii across the whole body. Three causes, in descending order of
contribution.

**1. The squared loss is symmetric, so the optimum is the mean distance, not the minimum.**
For a fixed centre, minimising `Σ (|v_j − c| − r)²` gives `r* = mean_j |v_j − c|`. No point
set drawn from a real surface is equidistant from a sphere centre, so `mean > min` always,
and the sphere inflates until it protrudes through the nearest vertices. On a cylinder of
radius `R = 30 mm` with vertices spanning `±h` along the axis, distances are `√(R² + a²)`
and the fitted radius is:

| axial extent `h` | fitted `r` | protrusion |
|---|---|---|
| 10 mm | 30.55 mm | 0.55 mm |
| 15 mm | 31.21 mm | 1.21 mm |
| 20 mm | 32.09 mm | 2.09 mm |
| 30 mm | 34.44 mm | 4.44 mm (+15 %) |
| 45 mm | 38.99 mm | 8.99 mm (+30 %) |

This is systematic, not an artefact of a bad neighbourhood. The intended object — a sphere
tangent to the surface from inside — is the maximal inscribed sphere, which no amount of
restricting the vertex set will produce from a symmetric loss.

**2. The assignment rule feeds back on the radius.** With `a(j) = argmin_i (|v_j − c_i| −
r_i)`, a larger sphere wins more vertices, which widens its assigned set, which raises the
mean distance, which enlarges it further. `E` stays monotonically non-increasing throughout —
the existing correctness invariant does not detect this — but the minimiser it converges to
is genuinely a set of oversized spheres.

**3. The assigned sets are too wide.** Real, and it compounds cause 1, but it is the weakest
of the three: halving the axial extent from 30 to 15 mm cuts protrusion from 4.44 to
1.21 mm, whereas fixing the loss at unchanged extent cuts it to 0.38 mm.

## Phase 0 — Instrumentation only, no behaviour change

Add to `info` and to the per-iteration report, and change nothing else:

- Per sphere: assigned vertex count, median residual `f`, **fraction of residuals with
  `f < 0`**, min and max `f`, and the axial spread of the assigned set (largest pairwise
  distance between assigned vertices).
- Aggregate: fraction of all assigned vertices with `f < 0`, and the mean over spheres of
  `median(f)`.

The negative-residual fraction is the primary instrument for everything below. Run it on the
current implementation and record the numbers before touching the objective.

**Acceptance:** a table of baseline numbers on the male template at the current default `δ`,
with the per-sphere breakdown for the ten worst spheres. Bit-identical fitted output to the
current implementation — this phase adds reporting only.

## Phase 1 — Asymmetric loss

Replace the squared loss with a one-sided penalty. Residual sign convention as established
in Phase 0 of the previous brief: `f_j = |v_j − c_i| − r_i`, positive when the vertex lies
outside the sphere (a gap), negative when the sphere protrudes through the surface.

```
ρ(f) = f²          f ≥ 0
     = (κ·f)²      f < 0
```

`ρ` is convex and C1 at the origin (both one-sided derivatives vanish there), so Gauss-Newton
with a line search is as well-behaved as before. Only the curvature jumps.

Implementation is IRLS and reuses the weight slot already in `H` and `g`:

```
w_j = 1      if f_j ≥ 0
    = κ²     if f_j < 0
```

recomputed from the current residuals at the start of **every inner Gauss-Newton iteration**.

New parameter `protrusion_weight: float = 10.0` (this is `κ`, not `κ²`). Values above roughly
50 scale the normal matrix by `κ²` on the interior points and make `H` badly conditioned;
clamp with a warning rather than letting it through silently. As `κ → ∞` the problem tends to
`max r subject to r ≤ min_j |v_j − c|`, the maximal inscribed sphere.

Expected effect at unchanged assignment, cylinder `R = 30 mm`, `h = 30 mm`:

| `κ` | fitted `r` | protrusion |
|---|---|---|
| 1 | 34.44 mm | 4.44 mm |
| 2 | 32.44 mm | 2.44 mm |
| 5 | 30.90 mm | 0.90 mm |
| 10 | 30.38 mm | 0.38 mm |
| 20 | 30.16 mm | 0.16 mm |

**The reported RMS will get worse, and that is correct.** RMS is the symmetric squared error;
deliberately declining to minimise it is the whole point of this phase. Do not treat the
increase as a regression, do not tune `κ` to keep RMS down. Judge this phase on the
negative-residual fraction and on median `f` per sphere.

**Acceptance:**
- `protrusion_weight = 1.0` reproduces the Phase 0 output **bit-identically**. This is the
  regression test that matters most.
- Synthetic cylinder, 4001 samples, `R = 30`, `h = 30`: reproduces the table above to better
  than 0.02 mm at every `κ`.
- Synthetic cylinder, `κ = 1000`: `r → 30 mm` to within the sampling resolution.
- Fixed-centre cross-check: with `c` held, the returned `r` equals the fixed point of the
  iterated weighted mean `r ← Σ w(r)·d / Σ w(r)`, to solver tolerance, whenever the radius
  bound is inactive.
- `E` remains monotonically non-increasing on the real template.
- On the real template, the aggregate negative-residual fraction falls substantially from the
  Phase 0 baseline.

## Phase 2 — Euclidean assignment, distance cap, frozen partition

Three coupled changes to how vertices reach spheres.

**2a. Assignment by Euclidean distance to centre**, `a(j) = argmin_i |v_j − c_i|`. Dropping
the `− r_i` term removes the feedback loop in cause 2 and makes the assignment a genuine
partition. It also means a KD-tree over the centres is now the *correct* structure, not a
wrong one: silent bug 1 of the previous brief is retired by this change, and the note about
Apollonius diagrams should be updated rather than left to confuse a later reader.

This is also what makes the intended geometric behaviour fall out for free. The Voronoi
bisector between two consecutive spheres along a limb is a plane perpendicular to the line
joining their centres, which along a limb is approximately perpendicular to the bone. The
partition therefore produces per-sphere axial slabs whose thickness derives from the actual
sphere spacing, with no thickness parameter to choose and no need to define a bone axis for
the pelvis, spine, clavicle or skull, where no such axis exists.

**2b. Absolute distance cap.** Discard the assignment when `|v_j − c_i| > β·r_i`. New
parameter `assign_radius_scale: float = 1.75`. This covers the two cases Voronoi alone does
not: terminal spheres at the end of a chain, whose cell is unbounded on the outward side, and
regions the template does not model. Unassigned vertices contribute nothing — report their
count, and report which spheres lost the most.

**2c. Freeze the partition.** Compute the assignment once from the **input** centres and
radii, then hold it fixed for the whole run. Justification: centres move by at most `δ`,
which is small by construction, so the partition is nearly static anyway; and a fixed
partition restores a well-defined objective over a fixed point set.

This matters for correctness testing. With 2b active and the partition recomputed, a sphere
that shrinks drops vertices out of its own set, and `E` falls because the summand count fell
rather than because the fit improved. The monotonicity invariant becomes meaningless. Freezing
the partition keeps it sharp.

Keep the outer loop structure but default `reassign_passes = 0`. As a diagnostic only,
recompute the assignment once at the end and report the fraction that would change, without
acting on it. A large fraction means `δ` is too loose for a frozen partition to be defensible.

**Acceptance:**
- The assignment computed by `cKDTree` matches a naive Python loop over a 200-vertex subset,
  exactly.
- Assigned-set axial spreads drop substantially versus Phase 0, most visibly on limb spheres.
- `E` monotone, now over a fixed point set.
- Negative-residual fraction improves further, or holds while the axial spread shrinks.
- The end-of-run reassignment diagnostic reports a small fraction; state the number.
- A run with `assign_radius_scale = ∞` and `reassign_passes` large reproduces Phase 1
  behaviour on spheres whose cells are bounded.

## Phase 3 — Shared-bone filter (conditional)

**Only implement this if Phase 2 leaves visible leakage between adjacent chains** — a thigh
sphere taking vertices from the other thigh when the legs are close, a chest sphere taking arm
vertices. Voronoi separates along a chain but not across chains that pass near each other.
Check the Phase 2 per-sphere reports before starting; if there is no leakage, skip and say so.

No new geometry is needed. SMPL ships per-vertex skinning weights in the `.npz`, and each
template sphere carries `parent_joint`. Admit the pair `(i, j)` only when

```
W[j, parent_joint(i)] ≥ bone_weight_min          default 0.1
```

and apply the Euclidean argmin within the admitted candidates. Confirm the weights key name
and its shape in the `.npz` during recon rather than assuming; report both.

**Acceptance:** leakage cases identified in Phase 2 are resolved; no sphere loses so many
vertices that it trips the freeze rules; the unassigned-vertex count does not rise sharply.

## Phase 4 — Re-sweep the offset cap

With the symmetric loss, `δ` was partly acting as a brake on inflation. Under an asymmetric
loss that pressure is gone and `δ` returns to its intended job of protecting the skinning, so
the value chosen before is probably no longer the right one — most likely too tight.

Re-run the sweep over `δ ∈ {0, 1, 2, 5, 10, 20} mm` and report, per `δ`: negative-residual
fraction, median `f`, spheres pinned at the `δ` boundary, spheres at `r_min`, mean and max
centre displacement, and both RMS columns.

**Acceptance:** `δ = 0` still returns centres bit-identical to the input. A recommendation for
the default, argued from the table rather than asserted.

## Silent bugs specific to these changes

1. **`E` evaluated with stale IRLS weights.** The line search must evaluate `Σ ρ(f)` directly
   from the current residuals, not `Σ w_j f_j²` with `w` frozen from the previous iterate.
   Using frozen weights makes `E` a different function on each iteration, and the monotonicity
   test will pass while the true objective rises.
2. **Weights recomputed per outer iteration instead of per inner iteration.** The sign of `f`
   changes as the sphere shrinks; that is the mechanism doing the work here.
3. **Residual sign convention inverted**, so `κ` penalises gaps instead of protrusions. The
   run will look plausible and produce spheres larger still. Guard with an assertion on the
   synthetic cylinder test, where the correct direction is known.
4. **`κ` applied instead of `κ²`** in the weight, giving a much weaker effect than the table
   predicts and inviting an unnecessarily large `κ`.
5. **The old signed-distance assignment left in place** somewhere — in the metric, in a
   diagnostic, in a test helper — while the fit uses the new Euclidean one, so the two
   disagree about which sphere owns a vertex.
6. **Unassigned vertices silently counted as zero residual** rather than excluded, which
   rewards shrinking spheres until they own nothing.
7. **`spheremesh_distance` altered** to match the new assignment rule. It must not change: it
   is the external metric and the comparison to `fit_beta_offsets.py` depends on it staying
   exactly as it is.
8. **Freeze rules evaluated against the pre-cap assignment**, so a sphere that lost most of
   its vertices to the `β·r` cap is solved anyway on 2 or 3 points.

## Scope

Unchanged from the previous brief: spheres only, no segments or slabs as optimisation targets,
no topology changes, no modification to `fit_beta_offsets.py`, `extract_beta_offsets.py` or
the CSV format. The two-column RMS report stays, with the caveat above about how to read it.

Update the module docstring to state the objective in its new form, and to record that
minimising signed distance symmetrically and producing an inscribed medial sphere are
different problems — the second is what this module now targets, and `protrusion_weight`
interpolates between them.