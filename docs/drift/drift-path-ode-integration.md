# Drift-path ODE integration: the method and its accuracy limits

This note answers two questions about how `test/test-full-3d-pixel.sh` produces
drift paths:

1. **What numerical method solves the trajectory ODE `dr/dt = v(r)`?**
2. **Given that the velocity `v` is available at any point in the volume, what is
   the accuracy limit of that method?**

It complements [`potential-based-drift-velocity.md`](potential-based-drift-velocity.md),
which explains *how `v` is computed*; here the focus is *how the trajectory is
integrated* and *what bounds its accuracy*.

## 1. Call path — from the shell script to the integrator

`test/test-full-3d-pixel.sh` runs (abbreviated):

```
pochoir drift --starts starts/drift3d --velocity velocity/drift3d \
              --paths paths/drift3d_tight '0*us,320*us,0.05*us'
```

The `drift` command (`pochoir/__main__.py:561-666`) then:

- passes **no `--engine`**, so `engine = "numpy"`;
- passes **no `-L/-T` diffusion keys**, so `use_sde = False`;
- finds that the `velo` command stamped `potential` and `temperature` into the
  velocity array's metadata (`__main__.py:471-472`), so it sets
  `use_potential = True` (`__main__.py:623-624`) and dispatches to
  `drift_numpy.solve_potential` (`__main__.py:642-644`):

```python
# pochoir/__main__.py
use_potential = (engine == "numpy" and not use_sde
                 and 'potential' in md and 'temperature' in md)
...
elif use_potential:
    path = drift_numpy.solve_potential(dom, point, pot, temp, ticks,
                                       method=interp_order, verbose=verbose)
```

So for this script the trajectory is integrated by **`solve_potential`**, which
re-derives velocity from the scalar drift potential on the fly rather than reading
the stored velocity vector components.

## 2. The ODE being solved

The drift path is the solution of the first-order, **time-independent** initial
value problem

```
dr/dt = v(r),      r(t0) = r_start
```

where `v(r)` is the local liquid-argon drift velocity. Because `v` does not depend
on `t` explicitly, this is an autonomous ODE; the right-hand side handed to the
solver, `PotentialField.__call__(time, pos)` (`drift_numpy.py:195-205`), ignores
`time` and returns `v(pos)`.

## 3. The numerical method

`solve_potential` (`pochoir/drift_numpy.py:208-226`) integrates with SciPy's
`solve_ivp` using the **`Radau`** method:

```python
# pochoir/drift_numpy.py
def solve_potential(domain, start, potential, temperature, times,
                    method='linear', verbose=False):
    ...
    func = PotentialField(domain, potential, temperature,
                          method=method, verbose=verbose)
    res = solve_ivp(func, [times[0], times[-1]], start, t_eval=times,
                    rtol=0.0000000001, atol=0.0000000001,
                    method='Radau',
                    )
    return res['y'].T
```

Key properties of this choice:

- **`Radau` = implicit Runge-Kutta (Radau IIA), formal order 5.** It is an
  *A-stable / L-stable, adaptive-step* scheme intended for stiff problems. The
  step size is chosen internally by the error controller; `first_step` and
  `max_step` are left commented out, so the stepper is free.
- **Tolerances `rtol = atol = 1e-10`** — an extremely tight error target.
- **`t_eval = times` is only the *output* sampling grid.** In the shell command
  the time argument `'0*us,320*us,0.05*us'` is parsed
  (`__main__.py:600-605`) into
  `ticks = linspace(0, 320 µs, 6400, endpoint=False)`, i.e. a 0.05 µs cadence.
  Radau computes a dense continuous solution and *interpolates it onto these
  ticks*. **The 0.05 µs does not set the integration step or its accuracy** — it
  only controls how finely the resulting path is reported.
- The loop that "advances position over time" is therefore entirely inside
  `scipy.integrate.solve_ivp`; pochoir supplies only the RHS and the tolerances.

### How the step error and tolerance are computed

The adaptive error control is **not** the difference between two consecutive
solutions in time (`y(t_{n+1}) − y(t_n)` is just the step increment, never an
error), and it is **not** step-doubling / Richardson extrapolation (comparing a
step of `h` against two steps of `h/2`). `Radau` uses an **embedded per-step
local error estimate**:

- At **each single step** `t_n → t_n + h`, two estimates of the solution *at the
  same time* `t_n + h` are formed from the *same* stage evaluations: the main
  Radau IIA solution `y_{n+1}` (order 5) and an **embedded lower-order (order 3)**
  estimate. Their difference is the local error estimate `err`. For `Radau`
  (Hairer & Wanner's RADAU5) that raw difference is additionally *stabilized* by
  one solve with `(I − h·γ·J)⁻¹` — reusing the already-factored implicit iteration
  matrix (`J` = Jacobian) — which makes the estimate reliable for stiff systems.
- So the estimate is **local** (committed inside one step), **at a single time
  point**, from an **embedded pair of different-order formulas** — *not* a
  comparison across two consecutive output times.

The estimate is then measured against a per-component tolerance scale and reduced
to a scalar by a weighted RMS norm:

```
scale_i    = atol + rtol · max(|y_n,i|, |y_{n+1},i|)
error_norm = sqrt( mean_i ( err_i / scale_i )² )
```

- Accept the step if `error_norm ≤ 1`; otherwise **reject**, shrink `h`, and retry.
- The next step size is set from the same estimate:
  `h_new ≈ h · safety · (1/error_norm)^{1/(p+1)}` (safety ≈ 0.9, `p` the estimator
  order), clipped to bounded growth/shrink factors.

With `rtol = atol = 1e-10`, every accepted step keeps this **local** (per-step)
error at the `~1e-10` level — this bounds the *local* error, not the accumulated
*global* error, though both stay far below the field-interpolation error discussed
in §5.

For context, the file contains other integrators that are **not** used by this
script:

| Path | Method | Notes |
|---|---|---|
| `solve_potential` (active) | `solve_ivp`, Radau, order 5, adaptive, `1e-10` | potential-based `v` |
| `solve` / `Simple` (legacy) | `solve_ivp`, Radau, `1e-10` | component-wise velocity interpolation |
| `solve_sde` | **forward Euler** + Gaussian diffusion increment | only if `-L`/`-T` given (diffusion); the only hand-written stepper |
| engine `numpyold` | `odeint` (LSODA), `rtol=atol=0.01` | inactive |
| engine `torch` | `torchdiffeq.odeint` (`dopri5`), `rtol=atol=0.01` | inactive |

## 4. How `v` enters the solver

At each position the integrator queries, the RHS recomputes `v` from the scalar
potential (`drift_numpy.py:195-205`, detailed in
[`potential-based-drift-velocity.md`](potential-based-drift-velocity.md)):

1. interpolate the scalar potential φ (trilinear `linear` by default; `cubic`
   optional) with a `RegularGridInterpolator`;
2. differentiate the interpolant by a **central finite difference with step
   ½·grid-spacing** to get `E = ∇φ` (`efield`, `drift_numpy.py:168-193`);
3. `v = μ(|E|, T) · E` using the LAr mobility model (`lar.py:10-50`).

This matters for the accuracy discussion below: the solver never sees a closed-form
`v`; it sees the output of *interpolate-then-differentiate* on a grid.

### Is `v` re-derived directly, or is there still an interpolation?

A natural question about the new (potential-based) path is whether the velocity is
computed *directly* — since the drift **E-field is deliberately never
interpolated** — or whether an interpolation step still remains. The precise
answer:

- **The scalar potential φ is still interpolated** — and this is required, not a
  leftover from the old sequence. A drift-path query point is off-grid, while φ is
  only stored on grid nodes, so φ (or its gradient) simply cannot be evaluated at
  an arbitrary point without interpolating φ first. The interpolator is built once
  in `PotentialField.__init__` (`drift_numpy.py:156-157`) and evaluated by
  `potential_at` (`drift_numpy.py:165-166`).
- **The drift E-field is never interpolated.** Interpolating the *vector* field
  component-by-component is exactly the step that broke `∇ × E = 0` in the old
  sequence, and it has been removed. So "no interpolation of the E-field" is
  correct; "no interpolation at all" is not achievable off-grid (short of snapping
  query points to grid nodes).
- **The differentiation is numerical, not analytic.** `E` is obtained by
  *re-sampling* the φ-interpolant at `pos ± ½·spacing` and taking a central
  finite difference (`efield`, `drift_numpy.py:179-192`), not by differentiating a
  closed-form expression. For the default `linear` (trilinear) interpolant one
  *could* differentiate the per-cell interpolation formula analytically to obtain
  the exact in-cell gradient (a truly "direct" calculation); the code instead
  re-evaluates and differences. Because the `±½·spacing` sample points can straddle
  a neighbouring cell, the result is a hybrid slope rather than the exact analytic
  in-cell gradient.

In short: **φ is interpolated (necessarily and correctly); E is not; and E is a
numerical finite-difference of the φ-interpolant, not an analytic derivative.**

## 5. Accuracy limit of the method

It is essential to separate two different error sources. The headline finding is
that they are wildly mismatched: **the ODE method is far more accurate than the
velocity field it is integrating.**

### (a) Temporal-integration error — effectively negligible

If `v` were a smooth function of position, the Radau integrator, being order 5
with adaptive stepping driven to `rtol = atol = 1e-10`, would drive the
time-integration (truncation) error down to the tolerance floor, i.e. near
floating-point round-off. In that idealised sense the ODE solver is *not* the
accuracy bottleneck — it is deliberately over-provisioned.

### (b) The real limit: smoothness of the supplied `v`

**Smoothness is not required to *run* — it is required to *achieve the advertised
order*.** The finite-difference gradient and the RK stepper will produce numbers
for *any* `v`; nothing crashes. But "Radau is order 5" is a *theorem* proved by
Taylor-expanding the solution within a step, and an order-`p` method attains local
error `O(Δt^{p+1})` **only if the solution is `p+1` times differentiable**. The
same is true of the central difference: `[φ(x+½h) − φ(x−½h)]/h ≈ φ′(x) + O(h²)`
holds only where `φ ∈ C³`; straddling a kink it returns roughly the average of two
one-sided slopes — an `O(1)` error, not `O(h²)`. So the non-smoothness does not
break executability; it invalidates the accuracy guarantees.

Here `v` is derived from a **trilinear (`linear`) interpolant of φ**, which is only
**C⁰**: its gradient — and therefore `E`, and therefore `v` — **jumps across every
grid-cell face**. Along a path:

- On a step that stays inside one cell, `v` is smooth → Radau is genuinely order 5,
  takes a large step, cheap.
- On a step that **crosses a face**, `v` jumps mid-step. The Taylor expansion is
  invalid, so the embedded order-5 and order-3 estimates disagree strongly and the
  **error estimate spikes**. The adaptive controller then **rejects the step and
  shrinks `Δt`** to force the estimate back under tolerance (see §3). So it does
  *not* silently return a low-order-wrong answer — it **crawls** across each face
  with tiny steps (efficiency collapse), with the error estimator working at the
  edge of its assumptions.

It helps to separate **two distinct "orders"**, which are easy to conflate:

- *Order in the time step `Δt`* — what "Radau order 5" means. It degrades on
  face-crossing steps, but adaptive step control masks the loss by shrinking `Δt`.
- *Order in the grid spacing `h`* — how fast the trajectory converges to the *true
  physical path* as the field is refined. With a `C⁰` (linear) field the derivative
  error is `O(h)`, so the path converges only at **first order in `h`**, no matter
  how tight the ODE tolerance. **This is the limit that matters physically**, and
  no `rtol`/`atol` can beat it.

Switching to `--interp-order cubic` produces a `C²` field and a smooth RHS, which
both removes the step-size collapse and improves the path (see §6).

### (c) Further practical limits

- **Fixed gradient step.** `E` is a central difference at ½·grid-spacing, so it
  carries its own O(h²) FD truncation error *on top of* the interpolation error.
- **Domain edges → one-sided differences.** Near the bounding box, `efield`
  clamps the sample points and degrades to a one-sided difference, dropping to
  O(h) and biasing `E` at the edge.
- **Velocity cliff at the boundary.** The RHS returns exactly `0` outside the
  domain (`inside` check, `drift_numpy.py:200-201`), a hard discontinuity. There
  is **no `solve_ivp` event** to terminate the path cleanly when it reaches the
  anode, so behaviour at the end of the drift is integrator-dependent (the path
  can stall against the boundary rather than stop on it).
- **Very tight `atol` in physical units.** `atol = 1e-10` is measured in the
  system-of-units (mm-scale positions), which is aggressive; combined with the
  non-smooth RHS it mostly costs runtime rather than buying real accuracy.

### Summary table

| Error source | Approx. magnitude | Mitigation |
|---|---|---|
| Radau time integration | ~ `rtol`/`atol` = 1e-10 (round-off floor) | already negligible |
| φ interpolation → `E` (trilinear, C⁰) | **O(h)**, dominant | refine grid `h`; use `--interp-order cubic` |
| Central-difference gradient (½·spacing) | O(h²) | refine grid `h` |
| One-sided difference at domain edge | O(h) at edges | keep paths away from bounds; refine |
| Velocity cliff / no anode event | endpoint-dependent | add a `solve_ivp` event to stop at the anode |

## 6. Interpolation order versus grid resolution — two different knobs

Cubic interpolation and a finer grid are often confused as "the same accuracy
improvement," but they fix genuinely different things.

First, a correction to a common premise: **cubic interpolation improves accuracy,
not only smoothness** — *provided the true field is smooth*, which the drift
potential is (it solves Laplace and is analytic in the bulk). For smooth data on
spacing `h`:

| Scheme | value error | derivative (`E`) error | continuity |
|---|---|---|---|
| linear (trilinear) | `O(h²)` | `O(h)` | `C⁰` (kinks at faces) |
| cubic | `O(h⁴)` | `O(h³)` | `C²` (smooth) |

Cubic adds no new sampled data, but it does not need to: it exploits the
*smoothness prior* of the true field to reconstruct between nodes more accurately —
the same reason Simpson's rule beats the trapezoid rule on the same samples. So at
an off-node position cubic gives both a **smoother** and a **more accurate** `E`.

**The floor cubic cannot cross:** the node values themselves come from the FDM
solve, whose discretization error is `~O(h²)`, fixed by the grid. Interpolation
only controls the error introduced *between* nodes; it can never be more faithful
than the nodes it interpolates. That is exactly the difference between the two
knobs:

- **Interpolation order (linear → cubic):** improves the *between-node
  reconstruction* — smoothness (`C⁰`→`C²`) and inter-nodal accuracy
  (`O(h)`→`O(h³)` in `E`) — for a **fixed set of node values**. Bounded above by
  node fidelity.
- **Grid resolution (`h`):** improves the *node values themselves* (denser
  sampling + more accurate FDM solve, `O(h²)`), lifting the floor that cubic is
  bounded by. Cost scales `~1/h³` in memory (3D), worse in solve time.

**Crucially — and this is the key point — a finer grid does *not* make `v` smooth.**
With linear interpolation `v` is `C⁰`-with-kinks at *every* resolution; refining
`h` only makes the jumps *smaller* (`O(h)`) and *more frequent* — it never removes
them. So the adaptive integrator keeps fighting kinks (just smaller ones) at every
scale. **Only changing the interpolation scheme (cubic) removes the artificial
non-smoothness; refining the grid cannot.**

So cubic interpolation affects the paths through two independent gains:

1. **Smoothness gain (integrator-side).** A `C²` `v` lets Radau regain its high
   order and stops the step controller from choking at cell faces — no step-size
   collapse, trustworthy error estimate, faster runs. This gain exists even if
   accuracy were unchanged.
2. **Accuracy gain (physics-side).** The reconstructed `E` between nodes is closer
   to the true smooth field, so the path is closer to the true path for the *same*
   grid.

### ⚠️ The cubic failure mode near the pixel plane (strong geometry)

**Cubic's accuracy advantage holds only where the true field is smooth. It breaks
down exactly where we care most — at the pixel plane.** The smoothness argument
above assumes the underlying potential is analytic. Near the pixel pads, grid
lines, and chamfers the *true* field is **nearly singular**: φ has a near-kink at
every electrode edge (a corner in the potential, an near-step in `E`). A `C²`
cubic spline **cannot** represent a kink without **overshooting and oscillating**
around it (the Gibbs / Runge phenomenon). So precisely in the strong-geometry
layer at the plane — the region that sets where every drift path lands — cubic is
at its *least* reliable, not its most.

Why this is not a minor edge effect but the dominant concern here:

- **It is a guaranteed artifact, not a modelling choice.** The drift potential is
  harmonic, so it obeys a **maximum principle**: between two grid nodes the true φ
  can never exceed the range of the surrounding nodal values. Linear interpolation
  honors this bound; **cubic provably violates it** near the pad edge (measured:
  overshoot on 23.6 % of sub-cell points, peak 0.58 V — see §8). Any such
  excursion is pure numerical ringing with no physical counterpart.
- **The error lands where it hurts.** Differentiating the φ overshoot produces a
  **spurious `E` spike right at the pad edge**, which gives the drift paths an
  artificial transverse kick and **over-focuses them onto the pads**. Because all
  the transverse field lives in this thin layer, this thin-layer artifact sets the
  final landing position — the quantity the whole calculation exists to produce.
- **It is large.** On the active 0.1 mm store this moved path landings by a
  **median of 345 µm (up to ~950 µm, i.e. 3–9 grid cells)** relative to linear
  (§8). This is not a rounding-level perturbation; it is a first-order change to
  the result driven substantially by ringing.
- **Refining the grid does not save cubic here.** A finer grid makes the kink
  *steeper*, not smoother — the pad edge is a genuine geometric feature, so the
  near-discontinuity in the true field survives at every resolution. Cubic will
  keep ringing at the edge until the geometry itself is smoothed (which it is
  not).

**Practical guidance.** Use cubic freely in the smooth bulk, where it is both
smoother and more accurate. **Do not trust cubic in the strong-geometry layer at
the pixel plane** without one of: (i) a locally **refined grid** *plus* a
**monotone / shape-preserving** (overshoot-free, e.g. PCHIP-style) interpolant, or
(ii) treating the near-plane field with a scheme that respects the electrode
boundary. Plain global cubic is precisely the wrong tool at the electrode edge.

## 7. Practical recommendations

- **⚠️ Do not trust `--interp-order cubic` near the pixel plane.** In the
  strong-geometry layer at the electrode edges cubic **overshoots and rings**
  (§6, §8) — a guaranteed, non-physical artifact (harmonic φ cannot leave its
  nodal envelope) that produces a spurious `E` spike and **over-focuses the drift
  paths onto the pads** (measured median landing shift 345 µm, up to ~950 µm on
  the active store). This is the single most important caveat in this document.
  Cubic is safe and beneficial in the **smooth bulk**; it is the *wrong* tool at
  the pad edge unless paired with a locally refined grid **and** a monotone /
  shape-preserving (overshoot-free) interpolant.
- **Refining the grid or using `--interp-order cubic` improves drift-path accuracy
  far more than tightening the ODE tolerances** *in the smooth bulk*, because the
  trajectory error there is dominated by the piecewise-linear interpolation of φ
  (O(h) in `E`), not by the Radau integration. (Near the plane, see the warning
  above — refining the grid alone does not remove cubic ringing.)
- Consider adding a `solve_ivp` **event** so integration terminates cleanly when a
  path reaches the anode plane, instead of relying on the velocity dropping to
  zero at the domain boundary.
- The current `rtol = atol = 1e-10` is well below the field-interpolation error
  floor; loosening it (e.g. to `1e-8`) would likely speed up integration with no
  measurable loss of path accuracy — worth benchmarking if runtime matters.

## 8. Empirical linear-vs-cubic study on the active store (measured)

The theoretical caveat in §6 was tested directly on the active
`test-full-3d-pixel.sh` store (44×44×300 grid, 0.1 mm spacing, pixel plane at
`z = 10.1 mm`). The drift was re-run on the **same** starts and the **same**
stored scalar potential, changing **only** `--interp-order` (linear vs cubic).
Three scripts under `test/` reproduce the analysis:

- `interp_linear_vs_cubic_summary.py` → `interp_linear_vs_cubic_summary.pdf`
  (field-level view: φ, `E`, `v` along lines and in an (x,z) slice).
- `paths_linear_vs_cubic_summary.py` → `paths_linear_vs_cubic_summary.pdf`
  (path-level view: landing pattern, separation-vs-z, overlays).
- `interp_overshoot_test.py` → `interp_overshoot_test.pdf`
  (reference-free overshoot/ringing test).

**Findings.**

1. **Bulk agreement.** Above the plane the two orders are indistinguishable —
   linear and cubic paths overlie for the entire ~20 mm drift and separate only
   in the last few tenths of a mm at the plane. The bulk value difference in φ is
   `~10⁻⁴ V`.

2. **Large near-plane path change.** Transverse landing shift `|cubic − linear|`
   over 100 paths: **median 345 µm, mean 390 µm, max 954 µm** — i.e. **3–9 grid
   cells** (cell = 100 µm). The cubic paths focus **coherently inward onto the
   pixel pads** more strongly than linear.

3. **Integrator stiffness of linear.** The linear run took **~3× longer** (~16
   min vs ~6 min CPU) — the C⁰ staircase step-collapses the adaptive Radau
   integrator exactly as predicted in §4b.

4. **Cubic overshoot is real and quantified (reference-free).** Because the drift
   potential is harmonic it obeys a **maximum principle**: between grid nodes the
   true φ can never leave the interval spanned by the surrounding nodal values.
   Linear interpolation respects this; **cubic violates it near the pad edge**,
   overshooting the nodal envelope on **23.6 %** of sub-cell sample points with a
   **peak overshoot of 0.58 V (4.2 % of the 13.8 V transverse span at the
   plane)**, with the classic adjacent under-shoot lobes (Gibbs-like ringing).
   Differentiated, this φ overshoot becomes a **spurious `E` spike** at the pad
   edge that gives cubic paths extra transverse kick — a substantial part of the
   345 µm focusing shift is therefore **artifact, not physics**.

**Interpretation.** On this grid *neither* order is trustworthy right at the pad:
linear is monotone-safe but staircased (step-collapse, `O(h)` `E`), cubic is
smooth but **rings** at the near-singular edge. The root cause is the 0.1 mm grid
**under-resolving** the pad edge (the coarse solve even flattens the pad bottom
to `−7.61, −7.61 V` where the geometry has real structure). The robust fix is a
**finer grid near the plane** so the node data resolves the edge; then
interpolation order becomes a second-order choice. A **monotone/limited cubic**
(overshoot-free, e.g. PCHIP-style) is the alternative if global refinement is too
costly. The `--interp-order cubic` flag in `test-full-3d-pixel.sh` should be used
with this caveat in mind — it removes the integrator stiffness and improves the
smooth bulk, but its near-plane focusing is partly ringing until the grid is
refined there.

## Source references

- `pochoir/drift_numpy.py:208-226` — `solve_potential` (Radau, `1e-10`, `t_eval`).
- `pochoir/drift_numpy.py:195-205` — `PotentialField.__call__` (the RHS `v(r)`).
- `pochoir/drift_numpy.py:168-193` — `efield` (`E = ∇φ`, ½-spacing central diff,
  edge clamping).
- `pochoir/drift_numpy.py:156-157` — the scalar-potential `RegularGridInterpolator`.
- `pochoir/drift_numpy.py:262-327` — `solve_sde` (inactive; forward Euler +
  diffusion).
- `pochoir/__main__.py:600-605` — time-grid / `ticks` construction.
- `pochoir/__main__.py:622-644` — engine/solver selection (`use_potential`).
- `pochoir/lar.py:10-50` — LAr mobility model `μ(|E|, T)`.
