# Pochoir `wogrid` Branch — Examination Overview

*Read-only examination. No code was modified. All claims are supported by
file:line citations against `origin/wogrid` (fetched 2026-04-17).*

*This document is a companion to `../00-overview.md` (master-branch
examination). Content that is identical to master is not repeated here;
this document focuses on what changed.*

---

## What Is the `wogrid` Branch?

The `wogrid` branch (`github.com/fanrado/pochoir`, branch `wogrid`) is an
active development fork targeting **pixel-readout field-response
calculations** for the DUNE Near Detector. The name "wogrid" refers to a
pixel geometry without an explicit wire-grid layer.

`git diff origin/master origin/wogrid --stat` shows **76 files changed,
+9 061 / −312 lines**. The changes fall into three categories:

| Category | Files | Lines |
|---|---|---|
| New pixel/PCB geometry generators | 5 new `.py` | +832 |
| FDM solver upgrades (fdm_torch, fdm_generic) | 2 modified | +420 |
| Supporting infrastructure (units, lar, plots, persist) | 4 modified | +734 |
| Test scripts, configs, notebooks, benchmark | 65 new/modified | +7 075 |

---

## What Changed vs Master

### 1. Mixed-Precision Two-Step Solver (`fdm_torch.py`, `__main__.py`)

The most significant algorithmic addition is a **two-step mixed-precision
approach** activated by `--multisteps yes` on the `pochoir fdm` CLI:

- **Step 1** — Solve ∇²φ₀ = 0 (Laplace) with `float32` on GPU. This is
  fast but limited in precision by fp32 rounding (~1×10⁻⁷ relative).
- **Step 2** — Compute the residual `f ≈ ∇²φ₀` from Step 1, then solve
  the Poisson equation `∇²δ = −∇²φ₀` with `float64`. The correction `δ`
  recovers the lost fp32 precision.
- **Final** — `φ_final = φ₀ + δ`, combined error as `√(err₀² + errδ²)`.

This is implemented in `pochoir/__main__.py` (the `fdm` command, lines
~333–375) and depends on the new `stencil_poisson` function in
`fdm_generic.py`.

### 2. Poisson Stencil (`fdm_generic.py`)

`stencil_poisson(array, source, spacing, res)` extends the existing
`stencil()` by subtracting a source-term contribution `(h²/(2N)) * source`
from the Jacobi neighbor average. See `01-algorithm.md` for the derivation.

### 3. New Geometry Generators

Five new generators handle the pixel/PCB geometry:

| File | Role |
|---|---|
| `pochoir/gen_pcb_pixel_with_grid.py` | Weighting-field domain for pixel anode (square-hole PCB) |
| `pochoir/gen_pcb_pixel_with_grid_dense.py` | Dense variant of the above |
| `pochoir/gen_pcb_drift_pixel_with_grid_dense.py` | Drift-field domain with pixel PCB (dense) |
| `pochoir/gen_pcb_quarter_30deg.py` | 30° rotated geometry |
| `pochoir/gen_pcb_quarter_90deg.py` | 90° geometry |
| `pochoir/gen_pcb_drift_pixel_with_grid.py` | (pre-existing, base drift generator) |

### 4. Expanded Unit System (`units.py`)

The minimal 8-constant `units.py` from master was replaced with a full
Wire-Cell units file (+286 lines). Base units are unchanged
(`mm = 1.0`, `cm = 10.0*mm`), but hundreds of derived constants are added.
The `Kelvin` alias (line 285) and time units (`nanosecond = 1.0`, etc.) used
by `lar.py` are now defined here instead of implicitly.

### 5. LAr Physics Additions (`lar.py`)

Added `longitudanal_diffusion()` and `transverse_diffusion()` functions (+117
lines). These implement the BNL parameterisation for longitudinal and
transverse electron diffusion coefficients as a function of E-field and
temperature. Note: the function name `longitudanal_diffusion` contains a
typo ("longitudanal" vs "longitudinal").

### 6. Benchmark / Tooling

- `test/bench_fdm.py` — standalone FDM benchmark script with trace-analysis
  and plotting capability (PyTorch profiler JSON parsing).
- `test/parse_maxerr.py` — parses log files to extract convergence curves
  and produce summary PDFs.
- `test/for_pixel/validate_FR.ipynb`, `velocity_vs_err.ipynb` — validation
  notebooks comparing FR output to references.

---

## Key Source Files (GPU/FDM Scope)

| File | Status vs master | Role |
|---|---|---|
| `pochoir/fdm_torch.py` | **Major rewrite** | Primary GPU solver |
| `pochoir/fdm_generic.py` | Modified | Shared stencil; new `stencil_poisson` |
| `pochoir/fdm.py` | Unchanged | Dispatcher |
| `pochoir/fdm_cupy.py` | Unchanged | CuPy backend |
| `pochoir/fdm_cumba.py` | Unchanged | CuPy + Numba CUDA JIT |
| `pochoir/fdm_numba.py` | Unchanged | Numba CPU JIT |
| `pochoir/fdm_numpy.py` | Unchanged | Reference CPU |
| `pochoir/__main__.py` | Modified | CLI; two-step solver logic |
| `pochoir/units.py` | **Replaced** | Wire-Cell unit system |
| `pochoir/lar.py` | Extended | LAr diffusion added |

---

## Navigation

- `01-algorithm.md` — Mathematical description of the two-step Poisson solver
- `02-bugs.md` — Potential bug catalog with file:line citations and severity
- `03-gpu-efficiency.md` — GPU running-efficiency analysis
- `04-memory.md` — Memory footprint analysis and reduction opportunities
- `05-summary.md` — Prioritised punch list

*For content identical to master (drift path ODE, Shockley–Ramo, backend
comparison), refer to `../01-algorithm.md` and `../05-cross-backend-comparison.md`.*
