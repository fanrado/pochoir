# `wogrid` Branch — Algorithm Documentation

*All line numbers refer to `origin/wogrid` as of 2026-04-17.*

---

## 1. Overview

The `wogrid` branch retains the master-branch Jacobi FDM solver and the full
Shockley–Ramo drift pipeline (see `../01-algorithm.md` for those details).
The new algorithmic work is:

1. A **Poisson stencil** (`stencil_poisson` in `fdm_generic.py`) that adds a
   right-hand-side source term to the Jacobi update.
2. A **two-step mixed-precision solver** in `fdm_torch.py` that uses the
   Poisson stencil to correct the floating-point error accumulated during a
   float32 Laplace solve.

---

## 2. Jacobi Relaxation Refresher

The standard Jacobi update for the Laplace equation ∇²φ = 0 on a uniform
grid with spacing h is:

```
φᵢ(k+1) = (1 / 2N) * Σⱼ∈neighbours φⱼ(k)
```

where N is the number of spatial dimensions (N=3 for 3D) and the sum is over
the 2N direct neighbours. This is what `stencil()` in `fdm_generic.py`
computes (`fdm_generic.py:42–82`).

---

## 3. Poisson Stencil (`stencil_poisson`)

**File:** `pochoir/fdm_generic.py`, lines 84–140.

For the Poisson equation ∇²φ = −f, the Jacobi update is:

```
φᵢ(k+1) = (1 / 2N) * Σⱼ∈neighbours φⱼ(k)  +  (h² / 2N) * fᵢ
```

The additional term `(h²/2N) * f` comes from rearranging the finite-difference
Laplacian: `(Σneighbors − 2N·φ) / h² = −f` → `φ = (Σneighbors + h²·f) / 2N`.

`stencil_poisson` implements this as:

```python
res  = stencil_neighbour_sum(array) * norm   # (1/2N) * Σneighbors
res -= spacing**2 * source * norm            # − (h²/2N) * source
```

Note the **subtraction** (line 136): this means the function solves
`∇²φ = +source` (note positive sign), whereas the docstring at line 87 says
the function solves `∇²φ = -f`. There is an inconsistency between docstring
and code; see `02-bugs.md § B-1` for details.

When `source=None`, `stencil_poisson` degenerates to the plain `stencil`
(Laplace update). The `_compiled_step` wrapper always calls `stencil_poisson`
regardless of whether Poisson mode is active (`fdm_torch.py:36–44`).

---

## 4. Two-Step Mixed-Precision Solver

**Files:** `pochoir/__main__.py` (lines ~333–375), `pochoir/fdm_torch.py`.

### 4.1 Motivation

A float32 Laplace solve is roughly 2× faster and 2× cheaper in GPU memory
than float64, but it saturates at a residual of ~10⁻⁷. The two-step scheme
recovers float64 accuracy without running the full solve in float64.

### 4.2 Step 1 — Float32 Laplace

```
solve(iarr, barr, edges, prec, epoch, nepochs,
      _dtype=torch.float32, phi0=None)
  → (φ₀, err₀)
```

This is a standard Jacobi solve of ∇²φ₀ ≈ 0 at float32 precision.
The solution `φ₀` has a residual `∇²φ₀ ≠ 0` limited by fp32 rounding.

### 4.3 Step 2 — Float64 Poisson Correction

The residual `∇²φ₀` is computed in float64 directly from the stencil:

```python
phi0_ = pad(phi0.to(float64))
edge_condition(phi0_, *periodic)
s = stencil(phi0_)                       # (1/2N) * Σneighbors(φ₀)
source = -(6/h²) * (s − phi0) * mutable_core   # ≈ −∇²_FD φ₀
```

The source construction (`fdm_torch.py:117–126`) uses the identity:

```
∇²_FD φ₀ = (1/h²) * (Σneighbors − 2N·φ₀) = (2N/h²) * (stencil(φ₀) − φ₀)
```

so `source = −(2N/h²) * (stencil(φ₀) − φ₀) ≈ −∇²_FD φ₀`.

Step 2 then solves the Poisson equation:

```
∇²δ = ∇²φ₀    (i.e., f_RHS = source ≈ −∇²φ₀)
```

with zero initial values (`iarr*0`) and float64, yielding the fp32-error
correction `δ`.

### 4.4 Final Combination

In `__main__.py` (lines ~366–374):

```python
arr = phi_0 + delta_phi          # φ_final = φ₀ + δ
err = sqrt(err_phi0**2 + err_delta_phi0**2)
```

The combined uncertainty is the quadrature sum of the two step errors.

### 4.5 Physical Interpretation

This approach is a **defect-correction method**: given an approximate
solution φ₀ to ∇²φ = 0, the defect is `d = ∇²φ₀` (should be zero), and
Step 2 solves for the correction δ that satisfies `∇²δ = −d`. The final
solution satisfies `∇²(φ₀+δ) = ∇²φ₀ + ∇²δ = d − d = 0` (to the precision
of the float64 solve).

---

## 5. Convergence Check Cadence

**File:** `pochoir/fdm_torch.py`, lines 152–178.

The inner loop structure is:

```python
for istep in range(epoch):
    if istep % 1000 == 0:               # A
        prev = iarr_pad.clone()
    _compiled_step(...)                  # B: apply one Jacobi update
    if (istep % 1000 == 0) and (istep != 0):  # C
        err = iarr_pad[core] − prev[core]
        maxerr = max(|err|)
        if prec and maxerr < prec:
            return early
```

Important: line A saves `prev` at the **same** iteration that line C computes
`err`. This means `err` is always the **one-step change**, not the 1000-step
change. At `istep=1000`, `prev` is saved (state before step 1000), step 1000
is applied, then `err = state_after_step_1000 − state_before_step_1000`.

Effectively, the convergence check fires once per 1000 steps and measures
how much a single step changed the solution. This is a valid convergence
criterion (converged when Δφ per step < prec), but the 1000-step cadence
means if `epoch < 1000` the check **never fires** and `err` remains `None`.
See `02-bugs.md § B-5` for the resulting crash.

---

## 6. New Pixel/PCB Geometry

### 6.1 Drawing Algorithm

The pixel geometry generators all use the **Bresenham mid-point circle
algorithm** (`draw_quarter_circle` in `gen_pcb_drift_pixel_with_grid.py`,
lines 9–30) to rasterize a quarter-circle arc, then mirror it to the other
three quadrants to produce a rounded-corner square hole in the PCB.

The boundary array (`barr`) marks cells that are fixed electrodes. The
geometry is repeated on a pixel pitch using Python nested loops
(`draw_3Dstrips_sq`, `draw_pixel_plane`). These loops iterate over each pixel
individually in Python, which is slow for large pixel counts but is a
one-time setup cost before the GPU solve.

### 6.2 Domain Sizes (from test scripts)

From `test/test-full-3d-pixel.sh`:

| Solve | Grid shape | N cells | fp64 per array |
|---|---|---|---|
| Drift (`drift3d`) | 44 × 44 × 1500 | 2.9 M | 23.4 MB |
| Weighting (`weight3d`) | 220 × 220 × 1500 | 72.6 M | **581 MB** |

The weighting-field domain at 72.6 M cells is significantly larger than the
master-branch tutorial grid (46.2 M cells). Memory implications are detailed
in `04-memory.md`.

---

## 7. Diffusion Physics (`lar.py`)

Two new vectorised functions were added:

- `longitudanal_diffusion(Emag, T)` — longitudinal diffusion coefficient DL
  using the BNL parameterisation (3 parameters b0, b1, b2, b3).
- `transverse_diffusion(Emag, T)` — transverse diffusion DT derived from DL
  and the Einstein relation: `DT = DL / (1 + μ'·E/μ)`.

These are not yet connected to the drift path integrator; their presence in
`lar.py` suggests they are planned for a future diffusion-aware drift step.

---

## 8. Unit System Change

The `units.py` file was replaced with the full Wire-Cell unit system. The
base unit `mm = 1.0` is unchanged, so all existing dimensioned quantities are
numerically compatible. The time base changed from "millisecond = 1.0" (old)
to "nanosecond = 1.0" (new), matching Wire-Cell convention. Any code that
relied on `units.ms == 1.0` will silently compute wrong time values.
The critical constant `Kelvin = 1.0` (line 285) is still defined as before.
