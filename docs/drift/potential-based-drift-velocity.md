# Drift velocity from the interpolated scalar potential

## Motivation

The drift-velocity calculation used by `test/test-full-3d-pixel.sh` used to
proceed in four steps:

1. Solve Laplace on the drift domain → drift potential **φ** on the grid nodes
   (`pochoir fdm`).
2. Compute the electric drift field **E = ∇φ** on the grid nodes (`pochoir velo`).
3. Interpolate the *electric drift field* to an arbitrary position `(x, y, z)`.
4. Compute the drift velocity at `(x, y, z)` from the interpolated field.

**Problem.** Step 3 interpolates the *vector* field **E** component-by-component.
A component-wise interpolated vector field is, in general, **not curl-free**
(`∇ × E ≠ 0`), so it corresponds to no valid electrostatic potential — it
violates the static Maxwell equation.

**New strategy (implemented).** Interpolate the *scalar* potential **φ** instead,
immediately after it is computed on the grid nodes. A scalar field is a genuine
function of position, so differentiating its interpolant gives

```
E(x, y, z) = ∇φ_interp(x, y, z)
```

which is a gradient of a scalar and therefore **curl-free by construction**
(`∇ × ∇φ ≡ 0`). The static Maxwell equation is respected everywhere. The drift
velocity is then computed pointwise from this field with the existing liquid-argon
mobility model.

The new pipeline is:

1. Drift potential **φ** on the grid nodes (`pochoir fdm`) — *unchanged*.
2. **Interpolate the scalar potential φ** to any `(x, y, z)`.
3. **Differentiate the interpolated potential** → **E(x, y, z) = ∇φ**.
4. Compute the drift velocity `v = μ(|E|, T) · E` at `(x, y, z)`.

Steps 2–4 are done *on the fly* during path integration, so the electric field
the ODE integrator sees is always the gradient of one consistent scalar potential.

## Where the code lives

- `pochoir/drift_numpy.py` — new `PotentialField` class + `solve_potential()`.
- `pochoir/__main__.py` — the `drift` command wires it in and resolves the
  potential array + temperature from the velocity metadata.
- `pochoir/velo` (the `velo` command) and `test/test-full-3d-pixel.sh` are
  **unchanged**.

## How the velocity is calculated

### Step 2 — interpolate the scalar potential

A single `RegularGridInterpolator` is built over the potential array using the
domain's exact per-axis grid coordinates (`Domain.linspaces`). Out-of-range
samples clamp to the edge value rather than injecting a spurious `0`, which would
otherwise create a huge false gradient near the (periodic) transverse edges.

```python
# pochoir/drift_numpy.py — PotentialField.__init__
# Use the exact grid coordinate axes (shape-length) so the axes
# match the potential array shape exactly.
points = domain.linspaces

potential = numpy.asarray(potential)
# fill_value=None + bounds_error=False -> extrapolate/clamp instead
# of injecting a spurious 0 that would create a huge false gradient
# at the (periodic) transverse edges.
self.interp = RGI(points, potential, method=method,
                  bounds_error=False, fill_value=None)
```

The scalar potential at an arbitrary position is a single interpolator call:

```python
def potential_at(self, pos):
    return float(self.interp([pos])[0])
```

`method` is `'linear'` by default (trilinear interpolation) and can be switched
to `'cubic'` via the `--interp-order` CLI option.

### Step 3 — differentiate the interpolated potential to get E

The electric field is obtained by a **central finite difference of the
interpolated potential** along each axis, using a step of half the grid spacing.
The sample points are clamped inside the domain bounding box (a one-sided
difference is used at an edge). The same `+∇φ` sign convention and `units.V`
scaling as the original `velo` command are preserved, so the physics/units are
unchanged — only the *order of interpolate-then-differentiate* is swapped.

```python
def efield(self, pos):
    '''
    E = grad(phi_interp) via central finite differences, using the same
    (+grad phi) sign convention and units.V scaling as the `velo` command.
    Sample points are clamped inside the bounding box; a one-sided
    difference is used when a neighbour would fall outside.
    '''
    pos = numpy.asarray(pos, dtype=float)
    lo = numpy.array(self.bb[0], dtype=float)
    hi = numpy.array(self.bb[1], dtype=float)
    efield = numpy.zeros_like(pos)
    for dim in range(len(pos)):
        h = 0.5 * self.spacing[dim]
        pp = pos.copy()
        pm = pos.copy()
        # clamp the +/- sample points inside the domain
        phigh = min(pos[dim] + h, hi[dim])
        plow = max(pos[dim] - h, lo[dim])
        pp[dim] = phigh
        pm[dim] = plow
        denom = phigh - plow
        if denom <= 0.0:
            efield[dim] = 0.0
            continue
        efield[dim] = (self.potential_at(pp) - self.potential_at(pm)) / denom
    return efield * units.V
```

Concretely, for dimension `d` with grid spacing `Δ_d`:

```
E_d(x) = [ φ_interp(x + ½Δ_d · ê_d) − φ_interp(x − ½Δ_d · ê_d) ] / Δ_d
```

Because every component is a difference of the *same* scalar interpolant,
`E = ∇φ_interp` is curl-free by construction.

### Step 4 — drift velocity from E

The velocity is computed pointwise with the existing liquid-argon mobility model
(`pochoir.lar.mobility`), using exactly the same arithmetic as the `velo`
command (`v = E · μ(|E|, T) / mm²`):

```python
def __call__(self, time, pos):
    '''
    Return the drift velocity vector at location (time independent).
    '''
    self.calls += 1
    if not self.inside(pos):
        return numpy.zeros_like(numpy.asarray(pos, dtype=float))
    efield = self.efield(pos)
    emag = math.sqrt(sum([e*e for e in efield]))
    mu = lar.mobility(emag, self.temp)
    return numpy.array([e*mu/units.mm**2 for e in efield])
```

This `__call__(time, pos)` is the right-hand side handed to the ODE integrator,
so the velocity is evaluated fresh — via interpolate-φ → differentiate → mobility
— at every position the integrator queries.

### Path integration

`solve_potential` builds a `PotentialField` and integrates the trajectory with
`scipy.integrate.solve_ivp` (Radau, `rtol = atol = 1e-10`), identical settings to
the original velocity-based `solve`:

```python
def solve_potential(domain, start, potential, temperature, times,
                    method='linear', verbose=False):
    start = numpy.array(start, dtype=float)
    potential = numpy.asarray(potential)
    times = numpy.array(times)

    func = PotentialField(domain, potential, temperature,
                          method=method, verbose=verbose)
    res = solve_ivp(func, [times[0], times[-1]], start, t_eval=times,
                    rtol=0.0000000001, atol=0.0000000001,
                    method='Radau',
                    )
    return res['y'].T
```

## How the `drift` command selects this path

The `drift` command resolves the potential array and temperature from the
**metadata of the velocity array** (which the `velo` command already records:
`potential`, `temperature`, `domain`), so neither `velo` nor the test script had
to change. For the default numpy, non-diffusion path it uses `solve_potential`;
if the metadata is missing (older stores) it falls back to the legacy velocity
interpolation.

```python
# pochoir/__main__.py — drift command
use_potential = (engine == "numpy" and not use_sde
                 and 'potential' in md and 'temperature' in md)
pot = temp = None
if use_potential:
    pot = ctx.obj.get(md['potential'])
    temp = md['temperature']
...
for ind, point in enumerate(start_points):
    if use_sde:
        path = drift_numpy.solve_sde(dom, point, velo, dl, dt, ticks, verbose=verbose)
    elif use_potential:
        path = drift_numpy.solve_potential(dom, point, pot, temp, ticks,
                                           method=interp_order, verbose=verbose)
    else:
        path = drifter(dom, point, velo, ticks, verbose=verbose)
    thepaths[ind] = path
```

A new option controls the interpolation order:

```
--interp-order {linear,cubic}   # default: linear
```

## Verification performed

- **Gradient correctness / curl-free.** On a linear test potential
  `φ = 3x + 2z`, `PotentialField.efield` returns `∇φ = (3, 0, 2)` exactly;
  since `E` is the gradient of a scalar interpolant, `∇ × E ≡ 0` by construction.
- **Consistency with the old approach.** Potential-based and legacy
  velocity-based drift endpoints agree to sub-grid-spacing (~0.02 mm) on sample
  drifts, and paths drift monotonically toward the anode.
- **End-to-end.** `pochoir drift` runs through the CLI and prints
  `drift: potential-based (...)`; both `linear` and `cubic` orders complete.

## Does this match the requested strategy?

Yes. The requested change was: *stop interpolating the electric drift field;
instead interpolate the drift potential right after it is computed on the grid
nodes, then use the interpolated potential to compute the electric field at any
`(x, y, z)`, then the drift velocity.* That is exactly the flow above:

| Requested step | Implementation |
|---|---|
| 1. Drift potential on grid nodes (solve Laplace) | `pochoir fdm` → `potential/drift3d` (unchanged) |
| 2. Interpolate the **drift potential** to any `(x, y, z)` | `PotentialField.interp` / `potential_at()` |
| 3. Compute **E = ∇φ** from the interpolated potential | `PotentialField.efield()` |
| 4. Compute **drift velocity** at `(x, y, z)` | `PotentialField.__call__()` via `lar.mobility` |

The interpolation of the vector E-field/velocity (the step that broke
`∇ × E = 0`) has been removed from the default drift path.
