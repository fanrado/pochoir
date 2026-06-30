# From potential to induced current in `pochoir`

This document traces how `pochoir` turns a solved **drift potential** and a
solved **weighting potential** into an **induced current**, with particular
attention to *where interpolation happens* in the chain and what numerical
methods are used at each step. Every claim is anchored to a `file:line`
reference so it can be re-verified against the source.

It answers three concrete questions:

1. **Drift field** — is the field obtained by interpolating the *potential* and
   then differentiating, or by differentiating on the grid and then
   interpolating the *field*?
2. **Induced current** — is it computed via the Ramo/Shockley theorem
   (E&#8407;<sub>w</sub>·v dot product), or as d(weighting potential)/dt?
3. **Drift integration** — Runge–Kutta, Euler, or something else?

---

## 1. The pipeline at a glance

The signal chain is a sequence of CLI subcommands (see `pochoir/__main__.py`),
each consuming the output of the previous one out of the persistent store:

```
                 (FDM Poisson/Laplace solve)
  geometry  ──►  fdm / solve  ──►  potential/drift3d   (drift potential on grid)
                              └──►  potential/weight3d  (weighting potential on grid)

  potential/drift3d ──►  velo  ──►  velocity/drift3d   (velocity VECTOR field on grid)
                                     efield/drift3d     (E field on grid, byproduct)

  starts  ──────────────►  drift  ──►  paths/...        (drift trajectories, r(t))
  velocity/drift3d ───────►

  paths/... ───────────►  induce-pixel  ──►  current/...   (Method A: dQ/dt)
  potential/weight3d ──►
        — or —
  paths/... ───────────►  srdot         ──►  current/...   (Method B: Ramo E_w·v)
  potential/weight3d ──►
  velocity/drift3d ────►
```

Two facts dominate everything below:

- **Differentiation (gradient) is always done on the regular grid**, using
  finite differences, *before* any interpolation.
- **Interpolation is always trilinear**
  (`scipy.interpolate.RegularGridInterpolator(method="linear")`), evaluated at
  arbitrary points along the drift path.

The full-3D-pixel production driver
(`test/test-full-3d-pixel.sh:124-160`) uses the chain
`velo` → `starts` → `drift` (default engine) → `induce-pixel`.

---

## 2. Q1 — Drift field: gradient on the grid first, then interpolate the field

**Answer: the field is computed on grid points first (finite-difference gradient
of the potential), and only later interpolated to points along the path.** This
is the user's option (b), not option (a).

The work is done by the `velo` command, `pochoir/__main__.py:380-484`:

```python
# __main__.py:392 — gradient of the drift potential, evaluated on the grid
efield = pochoir.arrays.gradient(pot, *dom.spacing)
```

`arrays.gradient` (`pochoir/arrays.py:96-111`) is a thin wrapper over
`numpy.gradient`:

```python
def gradient(array, *spacing):
    if isinstance(array, numpy.ndarray):
        return numpy.array(numpy.gradient(array, *spacing))
    ...
```

`numpy.gradient` uses **second-order accurate central differences** in the
interior of the grid and one-sided differences at the edges. The result is a
vector field (`efield[0]`, `efield[1]`, `efield[2]`) sampled **on the same
grid** as the potential.

Conductor cells are then forced to zero field using the boundary mask
(`__main__.py:393-402`):

```python
flag = barr == 1            # boundary/conductor mask
efield[0][flag] = 0
efield[1][flag] = 0
efield[2][flag] = 0
# plus zeroing of the normal component on the domain walls, lines 399-402
```

After scaling to physical units (`efield = efield*units.V`,
`__main__.py:425`), the field is converted to a **velocity field on the grid**
(see §3) and stored. The `efield` itself is also persisted as a byproduct
(`__main__.py:474-475`).

Interpolation to off-grid path points happens **much later**, inside the
drifter (`pochoir/drift_numpy.py:39-60`), once per velocity component:

```python
# drift_numpy.py:39-41
self.interp = [
    RGI(points, component, fill_value=0.0)
    for component in vfield]
```

So the order is unambiguous: **differentiate Φ → (mobility) → velocity field on
grid → interpolate** at each path position. No potential is interpolated for the
drift field.

---

## 3. Q1b — Velocity model: field-dependent liquid-argon mobility

The velocity field is **not** `E` directly; it is `v = μ(|E|, T) · E`, where the
mobility μ depends on the local field magnitude and the temperature
(`__main__.py:432-438`):

```python
emag = pochoir.arrays.vmag(efield)          # |E| on the grid
mu   = pochoir.lar.mobility(emag, temp)     # field-dependent LAr mobility
...
varr = [e*mu/units.mm**2 for e in efield]   # v = mu * E, per component
```

`pochoir.lar.mobility` is a vectorized `mobility_function`
(`pochoir/lar.py:10-50`) implementing the BNL liquid-argon transport
parametrization (<https://lar.bnl.gov/properties/trans.html>): a rational
function of |E| expressed in kV/cm,

```
       a0 + a1·E + a2·E^(3/2) + a3·E^(5/2)
μ = ───────────────────────────────────────── ,   Trel = T / 89 K
     (1 + (a1/a0)·E + a4·E² + a5·E³) · Trel^(3/2)
```

with coefficients `a0..a5` from `lar.py:25-31`. The velocity field thus carries
the correct LAr drift physics, evaluated grid-cell by grid-cell.

Optionally, the same command produces **longitudinal and transverse diffusion
coefficient fields** (`lar.diff_longit`, `lar.diff_tran`,
`__main__.py:434-437`) that feed the stochastic drift path (see §5).

### What is interpolated: the velocity, not E (and the nonlinearity caveat)

It is worth being explicit about *which* quantity is interpolated to off-grid
path points, because there are two conceivable orderings:

- **Option 1 — compute velocity on the grid, then interpolate the velocity.**
  ← *this is what `pochoir` does.*
- **Option 2 — interpolate E to the path point, then evaluate `v = μ(|E|,T)·E`
  there.** ← *not done.*

The `velo` command evaluates the entire chain `E → |E| → μ(|E|,T) → v = μ·E`
**once, on the grid** (`__main__.py:432-438`), and stores the finished
**velocity vector field** `varr`. The drifter then builds one
`RegularGridInterpolator` per **velocity** component (`drift_numpy.py:39-41`)
and evaluates them at each path position. The mobility μ is never re-evaluated
at off-grid points; E is interpolated directly only in the `srdot`/Ramo path
(§4, Method B), where the *weighting* field components are interpolated.

This distinction matters because μ(|E|) is **nonlinear** in E. Linearly
interpolating the nonlinear velocity field (option 1) is therefore *not*
identical to interpolating E and recomputing v (option 2):

```
interp( μ(|E|)·E )   ≠   μ(|interp(E)|)·interp(E)
```

The difference is a discretization error that vanishes as the grid spacing
shrinks (and where μ varies slowly with |E|). For the drift field, then: the
grid-sampled quantity that is interpolated to arbitrary points is the
**velocity**, with the field physics already folded in.

---

## 4. Q2 — Induced current: two methods (the pixel driver uses dQ/dt)

`pochoir` implements **two** ways to compute the induced current. They are
mathematically equivalent in the continuum (Ramo's theorem says
*I = q·E&#8407;<sub>w</sub>·v = q·dΦ<sub>w</sub>/dt*), but they discretize that
identity differently.

### Method A — d(weighting potential)/dt — `induce` / `induce-pixel` / `induce-30deg`

This is the method the production pixel driver uses
(`test/test-full-3d-pixel.sh:157-160` calls `induce-pixel`).

The **weighting potential** (a scalar) is interpolated at each path point, then
differenced in time (`__main__.py:839-852`, `induce`; the `induce_pixel` body at
`885+` is identical in form):

```python
Q  = charge * rgi(shifted_paths)     # interpolate weighting potential along path
dQ = Q[:, 1:] - Q[:, :-1]            # finite difference in time
dT = ticks[1:] - ticks[:-1]
I_tot = dQ / dT                      # induced current
```

Here `rgi` is the linear `RegularGridInterpolator` (§6). The order is
**interpolate the potential, then take the time difference**. No spatial
gradient of the weighting potential is computed.

The `shifted_paths` machinery (`__main__.py:814-837`) replicates/translates the
drift paths to evaluate the current on neighboring strips/pixels exploiting the
detector symmetry; `induce` also supports averaging groups of paths
(`__main__.py:854-865`).

### Method B — Ramo/Shockley E&#8407;<sub>w</sub>·v — `srdot`

The `srdot` command (`__main__.py:1255-1274`) computes the **weighting field**
on the grid via the same finite-difference gradient used for the drift field:

```python
# __main__.py:1264
sol_Ew = pochoir.arrays.gradient(pot, dom_Ew.spacing)   # weighting field on grid
res = pochoir.srdot.dotprod(dom_Ew, dom_Drift, sol_Ew, sol_Drift, velo)
```

Then `pochoir/srdot.py:6-42` interpolates **both** the weighting-field
components and the velocity components at each path point and takes the dot
product:

```python
ew_interp   = [RGI(points_ew, ew_i) for ew_i in pcb_3Dstrips_sol]   # E_w components
velo_interp = [RGI(points_v,  v_i)  for v_i  in velo]               # v components
q = -1
for path in pcb_drift:
    for point in path:
        V = numpy.array([velo_interp[k](point)[0] for k in range(3)])
        E = numpy.array([ew_interp[k](point)[0]   for k in range(3)])
        i = q * numpy.dot(E, V)        # I = q · (E_w · v)
```

So this is the explicit Ramo/Shockley dot product: differentiate Φ<sub>w</sub>
on the grid → interpolate the field → dot with the interpolated velocity.

### Comparison

| Aspect | Method A — `induce` / `induce-pixel` | Method B — `srdot` |
|---|---|---|
| Formula | *I = dQ/dt*, with *Q = q·Φ<sub>w</sub>(r(t))* | *I = q·E&#8407;<sub>w</sub>·v = q·∇Φ<sub>w</sub>·v* |
| What is interpolated | weighting **potential** (scalar) | weighting **field** + **velocity** (vectors) |
| What is differentiated | the path samples, **in time** (finite diff) | the potential, **in space** (grid gradient) |
| Gradient on grid? | no | yes (`numpy.gradient`, 2nd order) |
| Interpolation | linear RGI | linear RGI, per component |
| CLI command | `induce`, `induce-pixel`, `induce-30deg` | `srdot` |
| Source | `__main__.py:791+`, `885+`, `1062+` | `__main__.py:1255+`, `srdot.py:6-42` |
| Used by pixel driver? | **yes** (`test-full-3d-pixel.sh:157`) | no |

Practical note: Method A needs only the weighting potential and the path; it
inherits its time resolution from the drift tick spacing. Method B needs the
velocity field as well and depends on the grid-gradient quality of the weighting
potential. Both rely on the same trilinear interpolation.

---

## 5. Q3 — Drift integration: adaptive implicit Runge–Kutta (Radau) by default

**Answer: the default drift uses an adaptive implicit Runge–Kutta scheme
(Radau) via `scipy.integrate.solve_ivp`. There is no hand-rolled forward-Euler
for the deterministic path; Euler–Maruyama is used only for the optional
stochastic (diffusion) path.**

The `drift` command (`__main__.py:561-618`) selects an engine
(`--engine`, default `numpy`, `__main__.py:574-576`) and dispatches to
`pochoir.drift.solve_<engine>` (`__main__.py:597`). The RHS of the ODE is, in
every engine, the **interpolated velocity field** — time-independent: given a
position, return *v(r)* (`drift_numpy.py:65-83`, `Simple.__call__`).

| Engine | Solver | Method | Source |
|---|---|---|---|
| `numpy` (default) | `scipy.integrate.solve_ivp` | **Radau** — implicit RK, stiff-capable, adaptive internal step | `drift_numpy.py:87-115` |
| `torch` | `torchdiffeq.odeint` | adaptive ODE (GPU-capable), `rtol=atol=0.01` | `drift_torch.py:63-75` |
| `numpyold` | `scipy.integrate.odeint` | LSODA (Adams/BDF auto-switch) | `drift_numpyold.py` |

The default engine (`drift_numpy.py:106-111`):

```python
res = solve_ivp(func, [times[0], times[-1]], start, t_eval=times,
                rtol=1e-10, atol=1e-10,
                method='Radau')
```

Key points about stepping:

- **Radau** is a fully implicit Runge–Kutta method (good for stiff fields, e.g.
  near electrodes where velocity changes sharply). Its **internal step size is
  adaptive**, chosen to meet the tight `rtol`/`atol` tolerances.
- The user-supplied `ticks` (a uniform `linspace` from `start,stop,step`,
  `__main__.py:590-595`) are passed as `t_eval`. They control only the **output
  sampling** of the trajectory, **not** the integrator's internal steps.
- There is **no early termination** on hitting a boundary. The integration runs
  to `stop`. Out-of-domain velocity queries return `fill_value=0.0`
  (`drift_numpy.py:41`), so a particle that leaves the domain simply stops
  moving.

### Stochastic (diffusion) path — Euler–Maruyama

When both diffusion keys are supplied (`--diff-longitudinal` and
`--diff-transverse`), the `drift` command instead calls
`drift_numpy.solve_sde` (`__main__.py:602-617`, `drift_numpy.py:152-217`). This
is an **explicit Euler–Maruyama** integrator on the fixed tick grid:

```python
for i, dt_time in enumerate(dt_array):           # fixed time steps
    v_drift   = vel_interp(t, pos)               # interpolated velocity
    delta_pos = v_drift * dt_time                # deterministic drift step
    # anisotropic Gaussian increment split into longitudinal (∥v) and transverse (⟂v)
    noise = sqrt(2·dt_time·d_long)·z_par·u + sqrt(2·dt_time·d_tran)·z_perp
    pos = pos + delta_pos + noise
```

The diffusion coefficients `d_long`, `d_tran` are themselves interpolated from
grid fields (`ScalarField`, `drift_numpy.py:119-149`) produced by the `velo`
command from the LAr diffusion model.

---

## 6. Interpolation, in one place

- Engine: `scipy.interpolate.RegularGridInterpolator` with **`method="linear"`**
  (trilinear in 3D). Defined in `arrays.rgi` (`arrays.py:162-177`); used for
  velocity in `drift_numpy.Simple` (`drift_numpy.py:39-41`), for weighting
  potential in `induce`/`induce-pixel` (`__main__.py:813,839`), and for both
  weighting field and velocity in `srdot` (`srdot.py:22-23`).
- The `torch` engine uses `torch_interpolations.RegularGridInterpolator`
  (`drift_torch.py:11,39-41`), also linear.
- Grid axes for the interpolators come from `domain.linspaces`
  (`pochoir/domain.py`), one coordinate array per dimension built from
  `origin`, `spacing`, `shape`.
- Out-of-domain queries: `fill_value=0.0` in the numpy drifter and scalar fields
  (`drift_numpy.py:41,138`).

There is **no higher-order (cubic/spline) spatial interpolation** anywhere in
the drift/induce chain — accuracy in space is first-order (linear), while
accuracy in *time* along the trajectory comes from the high-order adaptive ODE
solver.

---

## 7. Consolidated file/line reference

| Topic | File:line | Symbol |
|---|---|---|
| Drift field = grad(Φ) on grid | `pochoir/__main__.py:392` | `velo` |
| Finite-difference gradient | `pochoir/arrays.py:96-111` | `gradient` |
| Conductor masking of field | `pochoir/__main__.py:393-402` | `velo` |
| Velocity = μ(|E|,T)·E | `pochoir/__main__.py:432-438` | `velo` |
| LAr mobility model | `pochoir/lar.py:10-50` | `mobility_function` |
| LAr diffusion models | `pochoir/lar.py:52+` | `*_diffusion` |
| Velocity interpolation (RHS) | `pochoir/drift_numpy.py:39-83` | `Simple` |
| Drift dispatcher + time grid | `pochoir/__main__.py:561-618` | `drift` |
| Default solver (Radau) | `pochoir/drift_numpy.py:87-115` | `solve` |
| Torch solver | `pochoir/drift_torch.py:63-75` | `solve` |
| Legacy LSODA solver | `pochoir/drift_numpyold.py` | `solve` |
| SDE / Euler–Maruyama | `pochoir/drift_numpy.py:152-217` | `solve_sde` |
| Induced current, dQ/dt | `pochoir/__main__.py:791-868` | `induce` |
| Induced current, dQ/dt (pixel) | `pochoir/__main__.py:885+` | `induce_pixel` |
| Induced current, Ramo E·v | `pochoir/__main__.py:1255-1274` | `srdot` |
| Ramo dot product | `pochoir/srdot.py:6-42` | `dotprod` |
| Linear interpolator factory | `pochoir/arrays.py:162-177` | `rgi` |
| Grid coordinate axes | `pochoir/domain.py` | `linspaces` |
| Production driver | `test/test-full-3d-pixel.sh:124-160` | — |

---

## 8. Why does the induced-current integral come out ≈ 0.96 instead of 1?

A natural sanity check is that the time-integral of the induced current for a
fully collected unit charge equals 1 (one electron). In practice, integrating
the `induce-pixel` waveforms for the central pixel
(`test/Paths_at_centralpixBorder.ipynb`,
`simpson(fr, dx=0.05µs, axis=2)`) gives **≈ 0.96**, not 1. This section explains
why — and it is **not** a bug in the induce/dQ-dt code, a stalled path, or an
interpolation artifact.

### The integral telescopes to a difference in weighting potential

Method A computes `I = dQ/dt` with `Q = q·Φ_w(r(t))`
(`__main__.py:839-852`). Integrating over the full waveform therefore
telescopes:

```
∫ I dt  =  ∫ (dQ/dt) dt  =  q·[ Φ_w(r_end) − Φ_w(r_start) ]
```

So the integral is exactly the **change in weighting potential between the
charge's birthplace and its collection point** (times q). It equals 1 only if
Φ_w sweeps the full 0 → 1.

### What the data shows (dataset `store_0.1mmSpacing_0.05usTimeStep_0.1mmPixPlaneWidth`)

Reading the stored intermediate `tmp/interpolated_phiW.npy` (the weighting
potential sampled along each of the 625 paths):

| Quantity | Value |
|---|---|
| Φ_w at collection (collected paths) | **1.0000** (paths do reach the pixel) |
| Φ_w at the **start** of every path | **≈ 0.039** (range 0.0370–0.0391) |
| ⇒ integral per collected path | `1 − 0.039 ≈ 0.961` |
| max per-path integral over all 625 paths | **≈ 0.96** (none ≥ 0.99) |
| hard ceiling `Φ_w(max) − Φ_w(start)` | **0.9611** |

So **no single-pixel path integrates to 1** — the best any path can do is
≈ 0.96, and the ≈0.039 offset is uniform across all paths (they all start on the
same plane). Paths collected by a *neighbor* integrate to ≈ `0 − 0.039 = −0.039`
(a small negative baseline). If a cell ever looked like "1" in a plot, that is
the colorbar scale/rounding, not a true unit integral.

### Root cause: the weighting-potential cathode boundary is not grounded

The deficit is entirely `Φ_w(start) ≈ 0.039 ≠ 0`. The drift charges start at
**z = 29.8 mm** (the top of the domain; the pixel is at z = 10 mm). The
weighting potential there is ≈ 0.039 because the solve never forces Φ_w → 0 on
the drift-entrance (cathode) side. The Φ_w profile along z at the pixel center:

```
 z = 0 mm  (backplane)  : Φ_w = 0.000   ← grounded
 z = 8 mm               : Φ_w = 0.358
 z = 10 mm (pixel)      : Φ_w = 1.000   ← readout pixel
 z = 12 mm              : Φ_w = 0.385
 z = 20 mm              : Φ_w = 0.051
 z = 28 mm              : Φ_w = 0.039
 z = 30 mm (top)        : Φ_w = 0.039   ← plateau, NOT 0
```

Below the pixel Φ_w decays to 0 (grounded backplane), but above it Φ_w decays
only to a **flat ≈ 0.039 plateau** and never reaches 0. The boundary mask
confirms why: in `boundary/weight3d.npz`, the bottom plane `z=0` has all 48400
cells fixed and the pixel plane `z=100` has 35300 fixed, but the **top plane
`z=299` has 0 fixed cells** — it is a free (Neumann) boundary. The generator
grounds only the bottom (`pochoir/gen_pcb_pixel_with_grid.py:223`,
`barr[:,:,0]=1`); there is no `barr[:,:,-1]=1`. With an insulating top, the
Laplace solution flattens (∂Φ_w/∂z → 0) into a nonzero constant rather than
decaying to 0.

By Ramo's theorem, the missing ≈ 0.039 was *already* induced earlier — during
the (un-simulated) drift from the true Φ_w = 0 cathode down to the z = 29.8 mm
start plane. The simulation only captures the portion of the signal from the
start plane onward, hence `1 − 0.039`.

This is the same class of issue as the earlier fix "weighting potential floating
in bulk: ground cathode plane" (commit `de786c1`), which was not applied to this
generator path / dataset.

### Note: finer grid does not fix it

This deficit is a **boundary-condition / domain-height effect, not a
discretization error**. Refining the grid (e.g. 0.1 mm → 0.05 mm) leaves the
≈ 0.039 plateau essentially unchanged. (Contrast with the §3 nonlinear-velocity
interpolation caveat, which *does* shrink with finer spacing.)

### Recommended fix (follow-up, not done here)

To make the response integrate to ≈ 1:

1. **Ground the cathode/top plane in the weighting-potential geometry** — add
   `barr[:,:,-1]=1` (with value 0) alongside the existing `barr[:,:,0]=1` in
   `pochoir/gen_pcb_pixel_with_grid.py:223`, then **regenerate** the `weight3d`
   weighting potential. Φ_w then decays to ≈ 0 at the start plane and every
   collected path's integral rises from ≈ 0.96 to ≈ 1.0 — a uniform ~4 %
   rescaling, not a per-path correction. (Equivalently/additionally, start the
   drift exactly at the grounded cathode where Φ_w ≈ 0.)
2. **Band-aid only:** if just the normalized waveform *shape* is needed,
   baseline-subtract `Φ_w(start)` (or renormalize the integral to 1). This does
   not correct the small shape error near the start of the drift.

---

### TL;DR

1. **Drift field:** gradient on the grid first (2nd-order finite differences),
   converted to a velocity field, *then* trilinearly interpolated along the
   path. (Option b.)
2. **Induced current:** both exist. The pixel production path uses
   *I = dQ/dt* from the interpolated weighting **potential** (`induce-pixel`);
   `srdot` is the explicit Ramo *q·E&#8407;<sub>w</sub>·v* variant using the
   interpolated weighting **field** and velocity.
3. **Integration:** adaptive **implicit Runge–Kutta (Radau)** via `solve_ivp` by
   default (torchdiffeq / LSODA alternatives); explicit Euler–Maruyama only for
   the optional diffusion (SDE) path.
4. **Integral ≈ 0.96, not 1:** `∫I dt = q·[Φ_w(end) − Φ_w(start)]`, and
   Φ_w(start) ≈ 0.039 (not 0) because the weighting solve leaves the top/cathode
   plane ungrounded (`gen_pcb_pixel_with_grid.py:223` grounds only `barr[:,:,0]`).
   No path reaches 1; the ceiling is `1 − Φ_w(start) ≈ 0.96`. Fix: ground the
   cathode plane and regenerate `weight3d`. See §8.
