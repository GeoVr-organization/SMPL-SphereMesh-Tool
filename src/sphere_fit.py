"""Geometric fitting of sphere centres and radii against a target surface.

What this module does
---------------------
Given a sphere-mesh's spheres and a point cloud sampled from a target surface (SMPL
vertices), it returns new centres and radii that fit those points, subject to a hard cap on
how far each centre may move from its input position.

The objective is

    E(c, r) = sum_j w_j * rho( f_j )     f_j = |v_j - c_a(j)| - r_a(j)

    rho(f)  = f^2            f >= 0      the point lies outside the sphere, a gap
            = (kappa*f)^2    f <  0      the sphere protrudes through the surface

    a(j)    = argmin_i |v_j - c_i|,  discarded when |v_j - c_a(j)| > beta * r_a(j)

subject to |c_i - c_i^input| <= delta and r_min <= r_i <= r_max. The sign convention is the
one used throughout the repository (SphereMesh.signedDistance, ManagedSphere.get_distance):
positive means the point lies outside the sphere. kappa is protrusion_weight and beta is
assign_radius_scale.

Fitting a surface and inscribing a medial sphere are different problems
----------------------------------------------------------------------
This is the reason for the asymmetry in rho, and it is worth stating plainly because the
symmetric version of this module looked correct and was not.

With the plain squared loss (kappa = 1), the minimiser in r at a fixed centre is the *mean*
assigned distance, r* = sum w |v_j - c| / sum w. No point set drawn from a real surface is
equidistant from a sphere centre, so mean > min always, and the fitted sphere protrudes
through its own nearest vertices by a margin that grows with how much of the surface the
sphere is asked to cover. On a cylinder of radius 30 mm sampled over an axial extent of
+-h, the fitted radius is 30.55 mm at h = 10 mm, 34.44 mm at h = 30 mm and 38.99 mm at
h = 45 mm. That is systematic, not a bad neighbourhood, and no restriction of the vertex set
removes it: it is what a symmetric loss means.

The object actually wanted is the maximal inscribed sphere, tangent to the surface from the
inside. rho interpolates between the two. kappa = 1 is the pure surface fit; as kappa grows
the problem tends to "maximise r subject to r <= min_j |v_j - c|", the inscribed sphere. On
the same cylinder at h = 30 mm the fitted radius falls to 32.44 mm at kappa = 2, 30.90 mm at
kappa = 5, 30.38 mm at kappa = 10 and 30.16 mm at kappa = 20.

rho is convex and C1 at the origin, both one-sided derivatives vanishing there, so
Gauss-Newton with a line search is as well-behaved as with the squared loss; only the
curvature jumps. It is realised by IRLS through the weight slot already present in H and g.

**The reported RMS gets worse as kappa rises, and that is correct.** RMS is the symmetric
squared error, and declining to minimise it is the entire point. Judge a run by the
negative-residual fraction (info["negative_fraction_after"], the share of assigned points
the spheres protrude through) and by the per-sphere median residual, not by RMS. On the male
template at delta = 5 mm the negative-residual fraction fell from 0.516 under the symmetric
loss to 0.117 under the defaults here, while RMS rose -- both are the intended direction.

Choosing max_offset
-------------------
Under the symmetric loss delta was doing two jobs: protecting the skinning, and acting as
the only brake on radius inflation. protrusion_weight now does the second job directly, so
delta was re-swept. Male template, neutral body, 4522 band vertices, kappa = 10, beta = 1.75,
frozen partition, max_outer = 60:

    delta   neg-frac  median f  pinned  disp mean  dr mean  RMS sph   RMS full  drift
    0 mm    0.092     7.82 mm   0       0.0 mm     -0.91 mm 0.018009  0.012922  0.00 %
    1 mm    0.109     7.14 mm   59      1.0 mm     -0.08 mm 0.017484  0.012600  1.24 %
    2 mm    0.114     7.04 mm   58      2.0 mm      0.25 mm 0.017072  0.012243  2.14 %
    5 mm    0.117     6.63 mm   48      4.6 mm      1.06 mm 0.016425  0.011872  3.40 %
    10 mm   0.105     6.63 mm   37      8.1 mm      2.11 mm 0.015718  0.011405  5.57 %
    20 mm   0.103     6.17 mm   26     13.1 mm      3.77 mm 0.014561  0.010571  10.35 %

    template baseline: neg-frac 0.117, RMS sph 0.019547, RMS full 0.015006
    the same fit under the symmetric loss: neg-frac 0.516, dr mean +8.95 mm

The recommended default is **10 mm**, and the argument is the last column. Every other
column improves monotonically with delta, so nothing in the fit itself picks a value; what
picks it is drift, the share of points that would change owner if the partition were
recomputed at the end. Freezing the partition is only defensible while that stays small.
It is 5.6 % at 10 mm and 10.4 % at 20 mm, and 10 % is where a frozen partition stops being a
fair description of the problem being solved. Against that, 10 mm already captures 78 % of
the total full-metric gain available between 0 and 20 mm (0.012922 -> 0.011405 of a possible
0.012922 -> 0.010571), while displacing centres by 8.1 mm on average and growing radii by
only 2.1 mm. Going beyond 10 mm buys the remaining 22 % at the cost of the assumption that
makes the objective well defined; if that trade is wanted, raise reassign_passes as well so
the partition is not stale.

Note that delta is still binding at 10 mm: 37 of 60 spheres sit pinned at the boundary, as
the degenerate direction predicts for spheres on the chest, back and thighs.

Why it separates
----------------
With a hard assignment each residual row of the Jacobian touches exactly one sphere, so
J^T W J is block-diagonal with 4x4 blocks and the problem splits into N independent
4-parameter fits. No global sparse system is built. Per sphere, with d_j = v_j - c_i and
n_j = d_j / |d_j|:

    df_j/dc = -n_j^T      df_j/dr = -1

    H = [ sum w n n^T   sum w n ]        g = [ sum w f n ]
        [ (sum w n)^T    sum w  ]            [ sum w f   ]

and the damped step solves (H + mu*diag(H)) * delta = g. The sign follows from
grad_c E = -2 sum w f n and Gauss-Newton solving H*delta = -grad E / 2. The w in those sums
is the product of the caller's per-point weight and the IRLS weight of rho, which is 1 on a
gap and kappa^2 on a protrusion, recomputed from the current residuals at the start of every
inner iteration. The line search evaluates sum rho(f) directly from each candidate's own
residuals: reusing weights frozen from the previous iterate would make E a different
function on every iteration, and the monotonicity test would then pass while the true
objective rose.

The degenerate direction
------------------------
sum w n n^T loses rank when the assigned normals do not span three dimensions. On a
locally flat patch all n_j are nearly parallel and the null direction of H is "translate c
along the common normal and grow r by the same amount" -- the plane limit, where the best
fit sphere has infinite radius. This is the expected behaviour on the chest, back and
thighs, not an edge case. Consequently max_offset acts as the regulariser there, and
spheres on flat regions are expected to sit pinned at the delta boundary; the count is
reported. Levenberg-Marquardt damping is mandatory, and no additional ridge is applied on
top of it, so that the effect of delta stays readable from the results.

Deviation from the brief
------------------------
The specification asks for the full-metric column to import spheremesh_distance from
scripts/fit_beta_offsets.py. That script does not exist on this branch.
spheremesh_distance was first written here as a vectorised mirror of the GUI's old
approximate cone distance (clamped point-segment distance minus the interpolated radius,
triangles reduced to their three edges). Since 2026-10-12 it is the exact SphereMesh
distance of src/sphere_mesh.py instead, slabs included, on the user's decision to make the
tool calculate like that class. The "RMS full" figures quoted above were measured with the
old approximation and are not comparable with runs after that change.

SphereManager.get_heatmap uses the same union signed distance (plain signed minimum).

Consequences of this approach
-----------------------------
Linear composition in beta is lost. The beta-offset CSV pipeline produces centre modes
that are the exact Jacobian dc_i/dbeta_k and compose to float64 epsilon, the radius rule
being the only non-linear part. Running this optimiser once per beta destroys that
property for the centres too, because the assignment step is a discrete function of the
configuration. If these results are ever written as betaN.csv, the sliders stop being
additive.

Minimising signed distance is not preserving mediality. A medial sphere that inflates
until it touches the surface everywhere scores sd ~ 0 and has stopped being medial. Under
the symmetric loss max_offset was the only thing resisting this, and it constrains the
centre rather than the radius, which is why delta was doing double duty as an inflation
brake. protrusion_weight now resists inflation directly, so delta is free to return to its
intended job of protecting the skinning. The reported RMS remains a partial view of quality.

Spheres-only optimisation against a full-mesh target is a stopgap. The tapered segment and
triangular slab terms are the intended next phase; the two-column report exists precisely
to quantify how much that phase matters. Because the segments already cover much of the
body, spheres pulled outward to reduce the spheres-only residual can protrude through a
region the cones covered fine, increasing the full metric while the objective decreases.
That is not a failure of the run, it is the signal that the segment terms are needed.

Out of scope by design: segments and slabs as optimisation targets (measured, never
differentiated), any topology change (sphere count, order, connections, parent_joint,
rest_offset, lbs_weight are invariant), and beta-space composition.

Monotonicity of E, and why the obvious argument for it is wrong
---------------------------------------------------------------
The fit step provably decreases E: under a fixed assignment each residual touches exactly
one sphere, so the per-sphere objectives sum to E and each is decreased by its own line
search. The reassignment step, however, does *not* provably decrease E, contrary to the
natural argument that a(j) is an argmin. It is the argmin of the *signed* distance, while E
sums the *square* of that distance. The two disagree the moment a sphere runs away along
the degenerate direction: an over-large sphere becomes the signed-argmin for distant points
at a hugely negative f, and those squared residuals explode. Measured on the male template
with max_offset = inf, E fell 9.158 -> 0.396 -> 0.361 -> 0.350 -> 0.344 and then jumped to
645.2 on the fifth reassignment.

Monotonicity is therefore enforced rather than assumed. Each outer iteration is accepted
only if the reassigned E did not exceed the previous iterate; otherwise the configuration
is rolled back to the start of that iteration and the run stops with
info["rolled_back"] = True. E returned by this function is monotonically non-increasing by
construction, and a rollback is a visible signal that the cap is too loose for the data
rather than a silent failure. The optional max_radius cap addresses the same runaway at
source.

Under the current defaults the partition is frozen (reassign_passes = 0), so there is no
reassignment step at all and the line search alone guarantees the decrease. The rollback
guard is retained for reassign_passes > 0 and for the legacy assignment mode.

The partition, and a retired warning
------------------------------------
Points reach spheres by a plain Euclidean argmin over the centres, capped at beta*r. An
earlier version of this module used the additively weighted argmin_i (|v - c_i| - r_i) and
warned at length that a KD-tree over the centres would be wrong for it, that query being an
Apollonius diagram rather than a Euclidean one. That warning no longer applies to the fit:
under the Euclidean rule a KD-tree is exactly the right structure, and
scripts/test_sphere_fit_inflation.py asserts that the dense partition, cKDTree and a naive
loop all agree. The dense matrix is kept only because at this size it costs nothing and it
serves both assignment modes from one code path.

The warning does still apply to sphere_signed_distances and assign_points, which implement
the union signed distance used for the reported metric and remain an Apollonius query. Those
are deliberately untouched by the partition parameters, so the objective and the external
metric disagree about which sphere owns a vertex. That is intended: one is what is being
minimised, the other is what has to stay comparable across versions of this module.

Dropping the -r_i term also breaks a feedback loop that the symmetric loss depended on. With
the signed rule a larger sphere wins more points, which widens its assigned set, which
raises the mean distance, which enlarges it further. E stayed monotone throughout, so the
existing correctness invariant never detected it; the minimiser it converged to was simply a
set of oversized spheres.
"""

from __future__ import annotations

import math
import warnings

import numpy as np

from src.sphere_mesh import SphereMesh

__all__ = [
    "sphere_signed_distances",
    "assign_points",
    "assign_partition",
    "spheres_only_distance",
    "spheremesh_distance",
    "select_band",
    "optimize_spheres",
]

_DEGENERATE_EPS = 1e-12

MAX_PROTRUSION_WEIGHT = 50.0
_MAX_EXACT_SPREAD_POINTS = 2000


def sphere_signed_distances(
    points: np.ndarray, centers: np.ndarray, radii: np.ndarray
) -> np.ndarray:
    """Full (M, N) matrix of signed distances from every point to every sphere.

    Computed by broadcasting rather than with a KD-tree on purpose. argmin_i (|v - c_i| -
    r_i) is an additively weighted nearest-neighbour query -- an Apollonius diagram, not a
    Euclidean one -- so a KD-tree over the centres silently returns the wrong sphere
    whenever a larger nearby sphere wins on the -r term, which is exactly the situation in
    the torso. At this problem size the dense matrix is trivial.
    """
    points = np.ascontiguousarray(points, dtype=np.float64)
    centers = np.ascontiguousarray(centers, dtype=np.float64)
    radii = np.ascontiguousarray(radii, dtype=np.float64)
    diff = points[:, None, :] - centers[None, :, :]
    return np.sqrt(np.einsum("mnk,mnk->mn", diff, diff)) - radii[None, :]


def assign_points(
    points: np.ndarray, centers: np.ndarray, radii: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Assign each point to the sphere minimising its signed distance.

    Returns (assignment, signed_distance), both of length M.
    """
    signed = sphere_signed_distances(points, centers, radii)
    assignment = np.argmin(signed, axis=1)
    return assignment, signed[np.arange(signed.shape[0]), assignment]


def assign_partition(
    points: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    *,
    assignment: str = "euclidean",
    assign_radius_scale: float = 1.75,
) -> tuple[np.ndarray, np.ndarray]:
    """Partition the points among the spheres for fitting purposes.

    Returns ``(owner, uncapped_owner)``, both (M,) int64. ``owner`` is -1 where the point
    was discarded by the distance cap; ``uncapped_owner`` is the same assignment before the
    cap, so callers can report which spheres lost the most.

    ``assignment="euclidean"`` uses argmin_i |v - c_i|. Dropping the -r_i term breaks the
    feedback loop by which a larger sphere wins more points, which widens its assigned set,
    which raises the mean distance, which enlarges it further. It also makes the partition a
    genuine Voronoi diagram: the bisector between two consecutive spheres along a limb is a
    plane perpendicular to the line joining their centres, which along a limb is roughly
    perpendicular to the bone, so the cells are axial slabs whose thickness follows the
    actual sphere spacing. No slab thickness has to be chosen and no bone axis has to be
    defined for the pelvis, spine, clavicle or skull, where none exists.

    Because the Euclidean rule is a plain nearest-neighbour query, a KD-tree over the
    centres would now be the correct structure rather than a wrong one. The dense matrix is
    kept because at this size it costs nothing and it keeps one code path for both modes.

    ``assignment="signed"`` restores the additively weighted argmin_i (|v - c_i| - r_i) that
    the first version of this module used, for regression testing only. That rule is an
    Apollonius diagram, not a Euclidean one, and a KD-tree over the centres is wrong for it.

    ``assign_radius_scale`` (beta) discards the assignment when |v - c_i| > beta*r_i. This
    covers what a Voronoi partition alone does not: terminal spheres at the end of a chain,
    whose cell is unbounded on the outward side, and regions the template does not model.
    Discarded points contribute nothing at all; they are never treated as zero residual,
    which would reward a sphere for shrinking until it owned nothing.
    """
    points = np.ascontiguousarray(points, dtype=np.float64)
    centers = np.ascontiguousarray(centers, dtype=np.float64)
    radii = np.ascontiguousarray(radii, dtype=np.float64)
    diff = points[:, None, :] - centers[None, :, :]
    euclidean = np.sqrt(np.einsum("mnk,mnk->mn", diff, diff))
    if assignment == "euclidean":
        owner = np.argmin(euclidean, axis=1)
    elif assignment == "signed":
        owner = np.argmin(euclidean - radii[None, :], axis=1)
    else:
        raise ValueError(f"Unknown assignment rule {assignment!r}; expected 'euclidean' or 'signed'")
    owner = owner.astype(np.int64)
    uncapped_owner = owner.copy()
    if np.isfinite(assign_radius_scale):
        rows = np.arange(points.shape[0])
        beyond_cap = euclidean[rows, owner] > assign_radius_scale * radii[owner]
        owner = np.where(beyond_cap, np.int64(-1), owner)
    return owner, uncapped_owner


def _irls_weights(residual: np.ndarray, protrusion_weight: float) -> np.ndarray:
    """IRLS weights realising the one-sided loss rho(f) = f^2, or (kappa*f)^2 when f < 0.

    The weight is kappa **squared**, not kappa: rho(f) = w*f^2, so w = kappa^2 reproduces
    (kappa*f)^2. Using kappa itself gives a far weaker effect than intended and invites an
    unnecessarily large setting.

    Must be recomputed from the current residuals at every inner iteration. The sign of f
    flips as a sphere shrinks, and that flip is the mechanism doing the work.
    """
    return np.where(residual < 0.0, protrusion_weight * protrusion_weight, 1.0)


def _pairwise_spread(points: np.ndarray) -> float:
    """Largest pairwise distance within a point set, the assigned set's axial spread.

    Evaluated exactly for sets up to _MAX_EXACT_SPREAD_POINTS; larger sets are strided down
    to that size first, which can only under-report. This is a diagnostic, never an input to
    the fit.
    """
    if points.shape[0] < 2:
        return 0.0
    if points.shape[0] > _MAX_EXACT_SPREAD_POINTS:
        stride = int(np.ceil(points.shape[0] / _MAX_EXACT_SPREAD_POINTS))
        points = points[::stride]
    diff = points[:, None, :] - points[None, :, :]
    return float(np.sqrt(np.einsum("ijk,ijk->ij", diff, diff)).max())


def spheres_only_distance(
    points: np.ndarray, centers: np.ndarray, radii: np.ndarray
) -> np.ndarray:
    """Signed distance from each point to the union of spheres."""
    return np.min(sphere_signed_distances(points, centers, radii), axis=1)


def spheremesh_distance(
    points: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    connections=None,
) -> np.ndarray:
    """Signed distance to the union of the spheres and their capsuloids and prysmoids.

    The capsuloid and prysmoid terms are SphereMesh.signedDistances from
    src/sphere_mesh.py, the implementation the GUI renders and measures with: the exact
    distance to each cone (tangent cone between its two spheres) and each triangular slab
    (tangent planes plus edge cones). The spheres term keeps spheres that belong to no
    connection in the union. Passing connections=None degrades to the spheres-only metric.
    """
    points = np.ascontiguousarray(points, dtype=np.float64)
    centers = np.ascontiguousarray(centers, dtype=np.float64)
    radii = np.ascontiguousarray(radii, dtype=np.float64)
    best = spheres_only_distance(points, centers, radii)
    if connections is None or len(connections) == 0:
        return best
    sphere_mesh = SphereMesh.fromSpheresAndConnections(centers, radii, connections)
    return np.minimum(best, sphere_mesh.signedDistances(points))


def select_band(
    points: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    connections=None,
    *,
    band_max: float = 0.05,
) -> np.ndarray:
    """Indices of the target points that the sphere-mesh actually models.

    Hands, feet and the skull cap are covered by no sphere; leaving them in drags the
    nearest spheres toward them and wrecks the torso. The band is every point whose full
    metric distance to the template configuration is within band_max.

    This must be evaluated once, on the neutral body, and the resulting index array reused
    for every subsequent run. Recomputing it per shape changes the population between runs
    and makes the reported RMS columns incomparable.
    """
    distance = spheremesh_distance(points, centers, radii, connections)
    return np.flatnonzero(np.abs(distance) <= float(band_max))


def _fit_one_sphere(
    center: np.ndarray,
    radius: float,
    assigned_points: np.ndarray,
    weights: np.ndarray,
    center_input: np.ndarray,
    max_offset: float,
    min_radius: float,
    max_radius: float,
    max_inner: int,
    tol: float,
    protrusion_weight: float,
) -> tuple[np.ndarray, float, float, int, bool]:
    """Projected Levenberg-Marquardt Gauss-Newton for a single sphere.

    Returns (center, radius, condition_number, degenerate_point_count, moved), where moved
    says whether any step was accepted. A sphere that accepted nothing has settled: given the
    same assigned points it would repeat this call and reach the same answer, so the caller
    may skip it on later outer iterations for as long as its point set is unchanged.

    The loss is the one-sided rho of _irls_weights, realised by IRLS: the weight slot
    already present in H and g carries kappa^2 on the protruding points. The weights are
    recomputed from the current residuals at the start of every inner iteration, and the
    line search evaluates sum(rho(f)) directly from the candidate's own residuals rather
    than sum(w f^2) with w frozen from the previous iterate. Freezing the weights would make
    E a different function on each iteration, and the monotonicity test would then pass
    while the true objective rose.

    The feasible set for the centre is a ball, which is convex, so projection is exact and
    projected Gauss-Newton is sound. The line search evaluates the objective at the
    projected and clamped candidate, never at the raw step: evaluating before projection
    can accept a step that increases the objective and breaks the monotonicity invariant
    with no visible cause. Projection is always measured against center_input, the original
    template centre, never the previous iterate, which would let a sphere walk arbitrarily
    far in small legal steps.
    """
    center = np.array(center, dtype=np.float64, copy=True)
    radius = float(radius)
    damping = 1e-6
    degenerate_count = 0
    condition_number = np.inf

    def objective(candidate_center: np.ndarray, candidate_radius: float) -> float:
        residual = np.linalg.norm(assigned_points - candidate_center, axis=1) - candidate_radius
        loss_weights = _irls_weights(residual, protrusion_weight)
        return float(np.sum(weights * loss_weights * residual * residual))

    def project(candidate_center: np.ndarray) -> np.ndarray:
        if not np.isfinite(max_offset):
            return candidate_center
        displacement = candidate_center - center_input
        length = math.sqrt(float(displacement @ displacement))
        if length <= max_offset:
            return candidate_center
        if length < _DEGENERATE_EPS:
            return np.array(center_input, dtype=np.float64, copy=True)
        return center_input + (max_offset / length) * displacement

    current_energy = objective(center, radius)
    moved = False

    for _ in range(max_inner):
        delta_vectors = assigned_points - center
        distances = np.linalg.norm(delta_vectors, axis=1)
        valid = distances > _DEGENERATE_EPS
        degenerate_count = int(np.count_nonzero(~valid))
        if np.count_nonzero(valid) < 4:
            break
        normals = delta_vectors[valid] / distances[valid, None]
        residuals = distances[valid] - radius
        point_weights = weights[valid] * _irls_weights(residuals, protrusion_weight)

        weighted_normals = point_weights[:, None] * normals
        hessian = np.empty((4, 4), dtype=np.float64)
        hessian[:3, :3] = normals.T @ weighted_normals
        normal_sum = weighted_normals.sum(axis=0)
        hessian[:3, 3] = normal_sum
        hessian[3, :3] = normal_sum
        hessian[3, 3] = float(point_weights.sum())

        gradient = np.empty(4, dtype=np.float64)
        gradient[:3] = (point_weights * residuals) @ normals
        gradient[3] = float(np.sum(point_weights * residuals))

        with np.errstate(over="ignore", invalid="ignore"):
            condition_number = float(np.linalg.cond(hessian))

        accepted = False
        for _ in range(12):
            damped = hessian + damping * np.diag(
                np.maximum(np.diag(hessian), _DEGENERATE_EPS)
            )
            try:
                step = np.linalg.solve(damped, gradient)
            except np.linalg.LinAlgError:
                damping = min(damping * 10.0, 1e2)
                continue
            if not np.all(np.isfinite(step)):
                damping = min(damping * 10.0, 1e2)
                continue

            scale = 1.0
            improved = False
            for _ in range(20):
                candidate_center = project(center + scale * step[:3])
                candidate_radius = min(max(radius + scale * step[3], min_radius), max_radius)
                candidate_energy = objective(candidate_center, candidate_radius)
                if candidate_energy < current_energy:
                    improvement = current_energy - candidate_energy
                    accepted = improvement > tol * current_energy
                    center = candidate_center
                    radius = candidate_radius
                    current_energy = candidate_energy
                    damping = max(damping / 10.0, 1e-12)
                    improved = True
                    moved = True
                    break
                scale *= 0.5
            if improved:
                break
            damping = min(damping * 10.0, 1e2)

        if not accepted:
            break

    return center, radius, condition_number, degenerate_count, moved




def _aggregate_negative_fraction(
    points: np.ndarray, owner: np.ndarray, centers: np.ndarray, radii: np.ndarray
) -> float:
    """Fraction of assigned points the spheres protrude through.

    The primary instrument for radius inflation. Under a symmetric squared loss the fitted
    radius is the mean assigned distance rather than the minimum, so a converged sphere
    protrudes through roughly half its own assigned points and this figure sits near 0.5.
    Driving it down is what the one-sided loss is for.
    """
    assigned = owner >= 0
    if not np.any(assigned):
        return float("nan")
    owned = owner[assigned]
    residual = np.linalg.norm(points[assigned] - centers[owned], axis=1) - radii[owned]
    return float(np.count_nonzero(residual < 0.0) / residual.size)


def _owner_drift_fraction(owner: np.ndarray, final_owner: np.ndarray) -> float:
    """Fraction of points that would change sphere, among those assigned in both partitions.

    Separated from the raw disagreement because two different effects are mixed there. A
    point can change owner because the centres moved, which is the partition drift that
    decides whether freezing is defensible; or it can cross the beta*r cap in either
    direction because the radii changed, which says nothing about drift. Reporting only the
    combined figure overstates the drift whenever the fit shrinks the radii, which is
    precisely what the one-sided loss is for.
    """
    both_assigned = (owner >= 0) & (final_owner >= 0)
    if not np.any(both_assigned):
        return 0.0
    changed = np.count_nonzero(owner[both_assigned] != final_owner[both_assigned])
    return float(changed / np.count_nonzero(both_assigned))


def _per_sphere_diagnostics(
    points: np.ndarray,
    owner: np.ndarray,
    uncapped_owner: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    sphere_count: int,
    include_spread: bool = True,
) -> dict:
    """Per-sphere residual and geometry statistics for the fit partition.

    Axial spread is the largest pairwise distance within a sphere's assigned set. It measures
    how wide a slab the partition handed the sphere, which is what compounds inflation: on a
    cylinder of radius R the mean assigned distance grows with the axial extent, so a wide
    slab inflates the fitted radius even when the loss is doing its job.

    Spread is quadratic in the assigned set size and dominates the cost of this function, so
    include_spread is left off for the per-iteration record and switched on only for the
    final one. Every other statistic here is linear and cheap enough to track per iteration.
    """
    counts = np.zeros(sphere_count, dtype=np.int64)
    median_residual = np.full(sphere_count, np.nan, dtype=np.float64)
    negative_fraction = np.full(sphere_count, np.nan, dtype=np.float64)
    min_residual = np.full(sphere_count, np.nan, dtype=np.float64)
    max_residual = np.full(sphere_count, np.nan, dtype=np.float64)
    axial_spread = np.zeros(sphere_count, dtype=np.float64)

    uncapped_counts = np.bincount(uncapped_owner, minlength=sphere_count).astype(np.int64)
    assigned = owner >= 0
    if np.any(assigned):
        counts = np.bincount(owner[assigned], minlength=sphere_count).astype(np.int64)
    lost_to_cap = uncapped_counts - counts

    order = np.argsort(np.where(assigned, owner, sphere_count), kind="stable")
    sorted_owner = np.where(assigned, owner, sphere_count)[order]
    boundaries = np.searchsorted(sorted_owner, np.arange(sphere_count + 1))
    for sphere_index in range(sphere_count):
        members = order[boundaries[sphere_index]:boundaries[sphere_index + 1]]
        if members.size == 0:
            continue
        member_points = points[members]
        residual = (
            np.linalg.norm(member_points - centers[sphere_index], axis=1) - radii[sphere_index]
        )
        median_residual[sphere_index] = float(np.median(residual))
        negative_fraction[sphere_index] = float(np.count_nonzero(residual < 0.0) / residual.size)
        min_residual[sphere_index] = float(residual.min())
        max_residual[sphere_index] = float(residual.max())
        if include_spread:
            axial_spread[sphere_index] = _pairwise_spread(member_points)

    return {
        "count": counts,
        "lost_to_cap": lost_to_cap,
        "median_residual": median_residual,
        "negative_fraction": negative_fraction,
        "min_residual": min_residual,
        "max_residual": max_residual,
        "axial_spread": axial_spread,
    }


def optimize_spheres(
    centers: np.ndarray,
    radii: np.ndarray,
    points: np.ndarray,
    *,
    max_offset: float,
    min_radius: float = 1e-3,
    max_radius: float = np.inf,
    max_outer: int = 20,
    max_inner: int = 10,
    tol: float = 1e-9,
    loss: str = "l2",
    huber_delta: float = 0.01,
    protrusion_weight: float = 10.0,
    assignment: str = "euclidean",
    assign_radius_scale: float = 1.75,
    reassign_passes: int = 0,
    weights: np.ndarray | None = None,
    connections=None,
    frozen: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit sphere centres and radii to a target point cloud.

    Point selection is the caller's responsibility: band filtering, subsampling and area
    weighting all belong upstream, so that the optimiser stays testable on synthetic data.
    It fits whatever it is handed.

    Both returned arrays are new allocations; the inputs are never mutated, so a second run
    starting from the same inputs reproduces the first exactly.

    protrusion_weight is kappa of the one-sided loss described in the module docstring.
    kappa = 1 recovers the plain squared loss exactly. Values above MAX_PROTRUSION_WEIGHT
    scale the normal matrix by kappa^2 on the protruding points and leave H badly
    conditioned, so they are clamped with a warning rather than passed through silently.

    The fit partition is separate from the reporting metric. assignment, assign_radius_scale
    and reassign_passes govern which points a sphere is fitted to; spheres_only_distance and
    spheremesh_distance, which produce the two reported RMS columns, are untouched by them
    and continue to use the signed union distance. The two therefore disagree about which
    sphere owns a vertex, deliberately: one is the objective, the other is the external
    metric that has to stay comparable across versions.

    reassign_passes is the number of outer iterations after the first in which the partition
    is recomputed. The default of 0 freezes the partition at the input configuration for the
    whole run. Centres move by at most max_offset, which is small by construction, so the
    partition is nearly static anyway, and a fixed partition restores a well-defined
    objective over a fixed point set. That matters for correctness testing: with the distance
    cap active and the partition recomputed, a sphere that shrinks drops points out of its
    own set and E falls because the summand count fell rather than because the fit improved,
    which would make the monotonicity invariant meaningless. Whatever the setting, the
    partition is recomputed once at the end as a diagnostic only, and the fraction that would
    change is reported as info["would_reassign_fraction"]; a large fraction means max_offset
    is too loose for a frozen partition to be defensible.

    Frozen spheres are passed through unchanged rather than omitted, preserving index
    correspondence with the skeleton. A sphere is frozen for an outer iteration when fewer
    than four points are assigned to it *after* the distance cap has been applied, four
    non-coplanar points being the minimum that determine a sphere; below that the 4x4 block
    is structurally singular. Spheres with no assigned points at all are reported separately,
    because a large count there means the template has spheres the partition cannot see,
    which is information about the template rather than a fault in the optimiser. No sphere
    borrows points from its neighbours.

    connections is used only to report the full metric alongside the spheres-only metric; it
    contributes no residuals and no Jacobian entries.

    frozen is an optional (N,) boolean mask of spheres to hold at their input values, for
    callers with an external reason to pin one -- the interactive tool passes its locked
    spheres. They still take part in the partition, so they keep covering their region of the
    surface, they are simply never solved for.

    E is monotonically non-increasing across the returned iterations by construction: an
    outer iteration whose E exceeds the previous one is rolled back and the run stops with
    info["rolled_back"] set. With the partition frozen this cannot trigger, since the line
    search alone guarantees the decrease; it remains as a guard for reassign_passes > 0. See
    the module docstring for why a reassignment step cannot simply be assumed to decrease E.

    Convergence is tested on E, never on partition stability: points equidistant from two
    spheres can flip forever without changing E, and a partition-based stopping rule spins
    until max_outer.
    """
    if loss != "l2":
        raise NotImplementedError(
            "Only the squared and one-sided losses are implemented; Huber via IRLS is a "
            "later phase."
        )
    protrusion_weight = float(protrusion_weight)
    if protrusion_weight < 1.0:
        raise ValueError("protrusion_weight must be at least 1.0")
    if protrusion_weight > MAX_PROTRUSION_WEIGHT:
        warnings.warn(
            f"protrusion_weight {protrusion_weight} exceeds {MAX_PROTRUSION_WEIGHT} and "
            f"leaves the normal matrix badly conditioned; clamping to "
            f"{MAX_PROTRUSION_WEIGHT}.",
            RuntimeWarning,
            stacklevel=2,
        )
        protrusion_weight = MAX_PROTRUSION_WEIGHT

    centers_input = np.array(centers, dtype=np.float64, copy=True)
    radii_input = np.array(radii, dtype=np.float64, copy=True)
    points = np.ascontiguousarray(points, dtype=np.float64)
    if centers_input.ndim != 2 or centers_input.shape[1] != 3:
        raise ValueError("centers must have shape (N, 3)")
    if radii_input.shape != (centers_input.shape[0],):
        raise ValueError("radii must have shape (N,)")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (M, 3)")
    if points.shape[0] == 0:
        raise ValueError("points is empty; nothing to fit against")

    sphere_count = centers_input.shape[0]
    point_count = points.shape[0]
    if weights is None:
        weights = np.ones(point_count, dtype=np.float64)
    else:
        weights = np.ascontiguousarray(weights, dtype=np.float64)
        if weights.shape != (point_count,):
            raise ValueError("weights must have shape (M,)")

    if frozen is None:
        frozen_mask = np.zeros(sphere_count, dtype=bool)
    else:
        frozen_mask = np.asarray(frozen, dtype=bool)
        if frozen_mask.shape != (sphere_count,):
            raise ValueError("frozen must have shape (N,)")

    working_centers = centers_input.copy()
    working_radii = np.clip(radii_input.copy(), min_radius, max_radius)
    working_radii[frozen_mask] = radii_input[frozen_mask]

    condition_numbers = np.full(sphere_count, np.inf, dtype=np.float64)
    settled = np.zeros(sphere_count, dtype=bool)
    frozen_empty: list[int] = []
    frozen_sparse: list[int] = []
    degenerate_points = 0

    def partition_energy(
        owner: np.ndarray, centers_now: np.ndarray, radii_now: np.ndarray
    ) -> float:
        assigned = owner >= 0
        if not np.any(assigned):
            return 0.0
        owned = owner[assigned]
        offset = points[assigned] - centers_now[owned]
        residual = np.sqrt(np.einsum("mk,mk->m", offset, offset)) - radii_now[owned]
        loss_weights = _irls_weights(residual, protrusion_weight)
        return float(np.sum(weights[assigned] * loss_weights * residual * residual))

    owner, uncapped_owner = assign_partition(
        points,
        working_centers,
        working_radii,
        assignment=assignment,
        assign_radius_scale=assign_radius_scale,
    )
    iterations: list[dict] = []
    initial_energy = partition_energy(owner, working_centers, working_radii)
    previous_energy = initial_energy
    rolled_back = False
    converged = False

    for outer_index in range(max_outer):
        restore_centers = working_centers.copy()
        restore_radii = working_radii.copy()
        reassigned = 0
        if 0 < outer_index <= reassign_passes:
            previous_owner = owner
            owner, uncapped_owner = assign_partition(
                points,
                working_centers,
                working_radii,
                assignment=assignment,
                assign_radius_scale=assign_radius_scale,
            )
            reassigned = int(np.count_nonzero(owner != previous_owner))
            settled[:] = False

        assigned = owner >= 0
        counts = np.bincount(owner[assigned], minlength=sphere_count).astype(np.int64)
        frozen_empty = [int(i) for i in np.flatnonzero((counts == 0) & ~frozen_mask)]
        frozen_sparse = [
            int(i) for i in np.flatnonzero((counts > 0) & (counts < 4) & ~frozen_mask)
        ]
        degenerate_points = 0

        order = np.argsort(np.where(assigned, owner, sphere_count), kind="stable")
        sorted_owner = np.where(assigned, owner, sphere_count)[order]
        boundaries = np.searchsorted(sorted_owner, np.arange(sphere_count + 1))
        for sphere_index in range(sphere_count):
            if frozen_mask[sphere_index] or settled[sphere_index]:
                continue
            member_indices = order[boundaries[sphere_index]:boundaries[sphere_index + 1]]
            if member_indices.size < 4:
                continue
            new_center, new_radius, condition, degenerate, moved = _fit_one_sphere(
                working_centers[sphere_index],
                working_radii[sphere_index],
                points[member_indices],
                weights[member_indices],
                centers_input[sphere_index],
                max_offset,
                min_radius,
                max_radius,
                max_inner,
                tol,
                protrusion_weight,
            )
            working_centers[sphere_index] = new_center
            working_radii[sphere_index] = new_radius
            condition_numbers[sphere_index] = condition
            degenerate_points += degenerate
            settled[sphere_index] = not moved

        if reassign_passes > 0:
            measured_owner, _ = assign_partition(
                points,
                working_centers,
                working_radii,
                assignment=assignment,
                assign_radius_scale=assign_radius_scale,
            )
            energy = partition_energy(measured_owner, working_centers, working_radii)
        else:
            energy = partition_energy(owner, working_centers, working_radii)
        if energy > previous_energy:
            working_centers = restore_centers
            working_radii = restore_radii
            rolled_back = True
            break

        diagnostics = _per_sphere_diagnostics(
            points,
            owner,
            uncapped_owner,
            working_centers,
            working_radii,
            sphere_count,
            include_spread=False,
        )
        observed = np.isfinite(diagnostics["median_residual"])
        displacement = np.linalg.norm(working_centers - centers_input, axis=1)
        metric_spheres = spheres_only_distance(points, working_centers, working_radii)
        metric_full = spheremesh_distance(points, working_centers, working_radii, connections)
        iterations.append(
            {
                "E": energy,
                "rms_spheres": float(np.sqrt(np.mean(metric_spheres ** 2))),
                "rms_full": float(np.sqrt(np.mean(metric_full ** 2))),
                "reassigned": reassigned,
                "unassigned": int(np.count_nonzero(~assigned)),
                "negative_fraction": _aggregate_negative_fraction(
                    points, owner, working_centers, working_radii
                ),
                "mean_median_residual": (
                    float(np.mean(diagnostics["median_residual"][observed]))
                    if observed.any()
                    else float("nan")
                ),
                "at_max_offset": (
                    int(np.count_nonzero(displacement >= max_offset * (1.0 - 1e-9)))
                    if np.isfinite(max_offset) and max_offset > 0.0
                    else 0
                ),
                "at_min_radius": int(np.count_nonzero(working_radii <= min_radius + 1e-15)),
            }
        )

        decrease = previous_energy - energy
        converged = decrease <= tol * previous_energy
        previous_energy = energy
        if converged:
            break

    final_owner, _ = assign_partition(
        points,
        working_centers,
        working_radii,
        assignment=assignment,
        assign_radius_scale=assign_radius_scale,
    )
    diagnostics = _per_sphere_diagnostics(
        points, owner, uncapped_owner, working_centers, working_radii, sphere_count
    )
    baseline_owner, baseline_uncapped = assign_partition(
        points,
        centers_input,
        radii_input,
        assignment=assignment,
        assign_radius_scale=assign_radius_scale,
    )
    baseline_diagnostics = _per_sphere_diagnostics(
        points, baseline_owner, baseline_uncapped, centers_input, radii_input, sphere_count
    )

    displacement = np.linalg.norm(working_centers - centers_input, axis=1)
    radius_change = working_radii - radii_input
    baseline_spheres = spheres_only_distance(points, centers_input, radii_input)
    baseline_full = spheremesh_distance(points, centers_input, radii_input, connections)
    final_spheres = spheres_only_distance(points, working_centers, working_radii)
    final_full = spheremesh_distance(points, working_centers, working_radii, connections)

    info = {
        "iterations": iterations,
        "initial_E": initial_energy,
        "final_E": previous_energy,
        "rolled_back": rolled_back,
        "converged": converged,
        "protrusion_weight": protrusion_weight,
        "assignment": assignment,
        "assign_radius_scale": float(assign_radius_scale),
        "reassign_passes": int(reassign_passes),
        "frozen_requested": [int(i) for i in np.flatnonzero(frozen_mask)],
        "frozen_empty": frozen_empty,
        "frozen_sparse": frozen_sparse,
        "point_counts": diagnostics["count"],
        "per_sphere": diagnostics,
        "per_sphere_before": baseline_diagnostics,
        "condition_numbers": condition_numbers,
        "degenerate_points": degenerate_points,
        "unassigned": int(np.count_nonzero(owner < 0)),
        "would_reassign_fraction": float(np.count_nonzero(final_owner != owner) / point_count),
        "would_reassign_owner_fraction": _owner_drift_fraction(owner, final_owner),
        "would_recap_fraction": float(
            np.count_nonzero((owner >= 0) != (final_owner >= 0)) / point_count
        ),
        "negative_fraction_before": _aggregate_negative_fraction(
            points, baseline_owner, centers_input, radii_input
        ),
        "negative_fraction_after": _aggregate_negative_fraction(
            points, owner, working_centers, working_radii
        ),
        "max_displacement": float(displacement.max()) if sphere_count else 0.0,
        "mean_displacement": float(displacement.mean()) if sphere_count else 0.0,
        "max_radius_change": float(np.abs(radius_change).max()) if sphere_count else 0.0,
        "mean_radius_change": float(radius_change.mean()) if sphere_count else 0.0,
        "rms_spheres_before": float(np.sqrt(np.mean(baseline_spheres ** 2))),
        "rms_spheres_after": float(np.sqrt(np.mean(final_spheres ** 2))),
        "rms_full_before": float(np.sqrt(np.mean(baseline_full ** 2))),
        "rms_full_after": float(np.sqrt(np.mean(final_full ** 2))),
        "num_points": point_count,
        "num_spheres": sphere_count,
    }
    return working_centers, working_radii, info
