# Pochoir Algorithm Explanation

*Read-only examination. No code was modified. All claims are supported by file:line citations.*

---

## 1. The Laplace Boundary-Value Problem

The goal of the FDM step is to solve **Laplace's equation** on a uniform orthogonal grid:

```
∇²u = 0
```

with Dirichlet boundary conditions: certain grid cells (electrodes, cathode, walls) are held at fixed voltages, and the interior cells converge to a harmonic function. The grid is N-dimensional (2D or 3D), uniform spacing, axis-aligned.

### Discretisation

On a uniform grid with spacing `h`, the standard second-order central-difference approximation gives, for the 2D case:

```
u[i,j] ≈ (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1]) / 4
```

and for 3D:

```
u[i,j,k] ≈ (u[i±1,j,k] + u[i,j±1,k] + u[i,j,k±1]) / 6
```

The general form for N dimensions is: each interior cell is the average of its 2N nearest neighbours. Stated as code in `pochoir/fdm_generic.py:48`:

```python
norm = 1/(2*nd)   # nd = number of dimensions
```

The stencil (`fdm_generic.py:35–67`) accumulates `+1` and `−1` offset views of the padded array for each dimension, then multiplies by `norm`.

---

## 2. Jacobi Iteration

The algorithm iterates, replacing every interior cell simultaneously by the average of its neighbours. This is the **Jacobi method** — the classical double-buffering Gauss-Seidel is *not* used. Key properties:

- **Global convergence guaranteed** for the Laplace equation with Dirichlet BCs.
- **Convergence rate**: the spectral radius ρ ≈ 1 − π²/N² for an N×N grid. For the production `weight3d` domain (350 × 66 × 2000), ρ ≈ 1 − O(10⁻⁷) — extremely slow.
- **No red-black decomposition**, no Successive Over-Relaxation (SOR), no Chebyshev acceleration, no multigrid. All five backends are pure Jacobi brute force.

The test script `test/test-full-3d-50L.sh` runs `--epoch 5000000 --nepochs 4` with `--precision 2e-10` — up to 20 million iterations for convergence. An optimal SOR scheme (ω = 2/(1+sin(π/N))) would reduce this by roughly 10–100×.

---

## 3. Dirichlet Boundary Condition Enforcement

Rather than branching on every cell, all backends use a **masked update** trick:

```
bi    = iarr * barr          # fixed boundary values (0 everywhere else)
mu    = ~barr                # 1 at free cells, 0 at boundary cells
u ← bi + mu * stencil(u_padded)
```

Concrete implementations:
- `fdm_numpy.py:41`: `bi_core = amod.array(iarr)` (before the boundary is set, but effectively equivalent after `set_core2`)
- `fdm_cupy.py:41`: `bi_core = cupy.array(iarr*barr)`
- `fdm_cupy.py:42`: `mutable_core = cupy.invert(barr)`
- `fdm_cupy.py:65`: `iarr_pad[core] = bi_core + mutable_core*tmp_core`

The numpy backend additionally calls `set_core2(iarr, fixed, ifixed)` (`fdm_numpy.py:62`) after every step to re-assert Dirichlet values at all boundary-flagged cells. The GPU backends (`fdm_cupy.py`, `fdm_cumba.py`, `fdm_torch.py`) **omit this second enforcement step** — see `02-bugs.md §5` for the implication.

---

## 4. Padding and Edge Conditions

The array is padded by one cell on all sides before iteration (`iarr_pad = pad(iarr, 1)`). The padding border is managed by `edge_condition` (`fdm_generic.py:3–32`), which is called at the end of each step.

Two edge modes per dimension, controlled by `--edges` CLI option (`__main__.py:321`):

- **Fixed** (default): `arr[0] ← arr[1]` and `arr[-1] ← arr[-2]` (mirror from one cell inside the boundary). This effectively forces a zero-gradient (Neumann-like) condition at the outer boundary of the padded region. Actual Dirichlet values come from the `barr` mask, not the padding.
  - `fdm_generic.py:30–32`:
    ```python
    arr[tuple(dst1)] = arr[tuple(src2)]   # border ← 1-cell-inside
    arr[tuple(dst2)] = arr[tuple(src1)]   # other border ← other 1-cell-inside
    ```

- **Periodic**: `arr[0] ← arr[-2]` and `arr[-1] ← arr[1]` (wrap from the true interior).
  - `fdm_generic.py:27–29`:
    ```python
    arr[tuple(dst1)] = arr[tuple(src1)]   # border ← opposite interior
    arr[tuple(dst2)] = arr[tuple(src2)]
    ```

Note: `edge_condition` acts on the **padded** array, so indices `n-2:n-1` and `1:2` select the outermost interior cells.

---

## 5. Convergence Check and Epoch Structure

To avoid the overhead of a convergence check every single step, the outer loop is over **epochs** (`nepochs`) and the inner loop over **steps per epoch** (`epoch`):

```
for iepoch in range(nepochs):       # outer
    for istep in range(epoch):      # inner
        if last step of epoch:
            prev = copy(iarr[core])
        stencil → update → edge_condition
        if last step of epoch:
            err = iarr[core] - prev
            maxerr = max(|err|)
            if prec and maxerr < prec: return early
```

(`fdm_numpy.py:52–74`, same structure in all backends.)

The convergence criterion is **absolute max-norm**: the maximum pointwise change over one epoch must be smaller than `prec`. Only the final step of each epoch is stored as `prev`, so the error represents the full epoch's drift, not a single-step change.

On the GPU backends, `if prec and maxerr < prec` requires comparing a 0-D GPU scalar against a Python float, which **forces a device-to-host synchronisation** every epoch. See `03-gpu-efficiency.md §2` for the impact.

---

## 6. The Drift Pipeline (Post-FDM)

### 6.1 Velocity Field Computation (`velo` command, `__main__.py:337–360`)

After the potential is solved, the electron drift velocity field is computed as:

```
E = −∇u      (the sign is implicitly handled by the application of mobility)
v = μ(|E|) · E
```

where μ(|E|) is the field-dependent LAr electron mobility from `pochoir/lar.py:10–50` (the Walkowiak parameterisation). The gradient is computed by `pochoir.arrays.gradient(pot, *dom.spacing)` (`arrays.py:96–111`), which wraps `numpy.gradient`. For torch arrays this involves a GPU→CPU→GPU round trip (see `03-gpu-efficiency.md §5`).

The mobility function (`lar.py:10–50`) uses scalar Python `math.sqrt` and is vectorised via `numpy.vectorize` (`lar.py:51`), which is a Python-level per-element loop — a significant CPU bottleneck for large grids.

### 6.2 Electron Drift Paths (`drift` command, `__main__.py:403–447`)

Given starting positions and a velocity field, electron trajectories are integrated with an ODE solver. The `--engine` option selects:

- **numpy** (`drift_numpy.py`): `scipy.integrate.solve_ivp` with the Radau (implicit RK45-like) method. The velocity field is interpolated with `scipy.interpolate.RegularGridInterpolator` on CPU. Has bounding-box check (`Simple.inside`, `drift_numpy.py:39–43`) and returns zero velocity outside the domain (`extrapolate`, `drift_numpy.py:58–59`).

- **torch** (`drift_torch.py`): `torchdiffeq.odeint` with Dormand-Prince (RK45) by default. The velocity field is interpolated with `torch_interpolations.RegularGridInterpolator`. **Hardcoded `device='cpu'`** (`drift_torch.py:67`) — nothing runs on GPU. No bounding-box check.

The trajectory `start`, `velocity`, and `times` arrays are stored per-start-point; the outer loop in `__main__.py:441–443` calls the solver once per starting position, re-transferring velocity data to the solver each call.

### 6.3 Induced Current (Shockley–Ramo)

The induced current at electrode k from a drifting electron at position **r** with velocity **v** is:

```
i_k(t) = q · v(t) · E_w,k(r(t))
```

where **E_w,k** is the weighting field for electrode k. The `srdot` and `induce` commands implement this dot product along each trajectory and for each electrode.

### 6.4 Output Format

`convertfr` exports the computed response functions to Wire-Cell Toolkit's `fr.json` JSON format. The most recent git commit `ab9dadd` targeted this conversion.

---

## 7. Algorithmic Limitations Summary

| Aspect | Current state |
|--------|---------------|
| PDE solver | Jacobi iteration (no SOR, no multigrid) |
| Convergence rate | O(N²) iterations for grid size N |
| Convergence check | Absolute max-norm per epoch |
| GPU loop structure | Host Python loop, one (or N) kernel(s) per step |
| ODE integrator (numpy) | Radau (implicit, L-stable, high accuracy) |
| ODE integrator (torch) | Dormand–Prince RK45 (explicit, lower accuracy) |
| Interpolation (numpy) | scipy RGI, CPU only |
| Interpolation (torch) | torch_interpolations RGI, CPU only (device hardcoded) |
| Mobility computation | Scalar Python loop via numpy.vectorize |
