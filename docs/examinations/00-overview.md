# Pochoir GPU Field Response Code — Overview

*Read-only examination. No code was modified. All claims are supported by file:line citations.*

---

## Scientific Purpose

`pochoir` computes **detector field-response functions** for Liquid Argon Time Projection Chambers (LArTPCs), specifically targeting the DUNE Near Detector pixel/strip geometry. The name is French for "stencil" — a nod to the stencil-iteration method at the heart of the solver.

A field-response function describes the electrical signal induced on a readout electrode by a single drifting electron traversing the detector. The calculation follows the **Shockley–Ramo theorem** and requires three distinct physics computations:

1. **Drift potential** — Solve the Laplace equation for the real electrostatic field (cathode and anode bias potentials as Dirichlet boundary conditions). The gradient of this potential, multiplied by the LAr electron mobility, gives the drift-velocity field.

2. **Ramo (weighting) potentials** — For each sensitive electrode, solve Laplace again with that electrode at 1 V and all others at 0 V. These *weighting fields* are not physical fields; they encode the electrode's geometric coupling to charge at any position.

3. **Drift paths + Shockley–Ramo dot product** — Integrate electron trajectories through the drift-velocity field (ODE initial-value problem), then evaluate the induced current on each electrode as the Shockley–Ramo dot product of the trajectory velocity with the electrode's weighting field. The result is a discrete current waveform per electrode, which is the field-response function used by Wire-Cell Toolkit.

---

## Full Computational Pipeline

The pipeline is a sequence of CLI invocations (each a separate process), stored in NPZ/JSON files:

```
pochoir domain     → defines grid shape, spacing, origin
pochoir gen        → paints electrode geometry → initial + boundary arrays (iarr, barr)
pochoir fdm        → solves Laplace BVP → potential array + increment (error) array
pochoir velo       → gradient(potential) × mobility → velocity vector field
pochoir starts     → stores initial particle positions
pochoir drift      → integrates trajectories → path array
pochoir srdot      → (intermediate weighting field dot products)
pochoir induce     → Shockley–Ramo induced current waveforms
pochoir convertfr  → export to Wire-Cell Toolkit JSON format (fr.json)
```

The `bc-interp` command is an optional step that projects a 2D far-field Ramo potential onto the boundaries of a 3D near-field domain, exploiting translation symmetry along the strip direction.

---

## FDM Engine Zoo

Five FDM solver backends share the same interface `solve(iarr, barr, periodic, prec, epoch, nepochs)`:

| Engine | File | Stencil implementation | Target hardware |
|--------|------|------------------------|-----------------|
| `numpy` | `fdm_numpy.py` | generic slice-ops via `fdm_generic.stencil` | CPU |
| `numba` | `fdm_numba.py` | `@numba.stencil` + `@numba.njit`, CPU JIT | CPU (JIT) |
| `torch` | `fdm_torch.py` | generic slice-ops via `fdm_generic.stencil` | CPU or CUDA (auto) |
| `cupy` | `fdm_cupy.py` | generic slice-ops via `fdm_generic.stencil` | CUDA GPU |
| `cumba` | `fdm_cumba.py` | fused `@cuda.jit` kernel | CUDA GPU |

All five implement the same **Jacobi iteration** algorithm. The dispatcher (`fdm.py`) imports all five at startup and exposes them as `solve_numpy`, `solve_torch`, etc.; the CLI selects via `--engine`.

---

## Typical Grid Sizes

From `test/test-full-3d-50L.sh` (small/medium production test) and `tutorial/tutorial.jsonnet` (full production tutorial):

| Domain | Shape | Cells | Float64 size/array |
|--------|-------|-------|--------------------|
| `drift3d` (test) | 25 × 17 × 700 | ~298 K | 2.4 MB |
| `weight3d` (test) | 350 × 32 × 700 | ~7.8 M | 62.7 MB |
| `weight2d` (test) | 1092 × 700 | ~765 K | 6.1 MB |
| `elects3d` (tutorial) | 84 × 56 × 1000 | ~4.7 M | 37.6 MB |
| `weight3d` (tutorial) | 350 × 66 × 2000 | ~46.2 M | **369.6 MB** |
| `weight2d` (tutorial) | 1092 × 2500 | ~2.73 M | 21.8 MB |

The tutorial `weight3d` domain at 46.2 M cells × float64 is the demanding case. The FDM solver simultaneously holds 7–9 device arrays of this size (see `04-memory.md`), putting total GPU residency at **2.5–3.5 GB** for float64 — a tight fit on a 16 GB GPU; see the memory examination for details.

---

## Key Source Files

| File | Role |
|------|------|
| `pochoir/fdm_generic.py` | Shared stencil + edge-condition used by numpy/cupy/torch |
| `pochoir/fdm_numpy.py` | Reference CPU implementation (also used by numba) |
| `pochoir/fdm_torch.py` | Torch GPU/CPU solver |
| `pochoir/fdm_cupy.py` | CuPy GPU solver |
| `pochoir/fdm_cumba.py` | CuPy + Numba CUDA JIT solver |
| `pochoir/fdm_numba.py` | Numba CPU JIT solver |
| `pochoir/fdm.py` | Dispatcher module |
| `pochoir/drift_torch.py` | Drift path solver (torch + torchdiffeq) |
| `pochoir/drift_numpy.py` | Drift path solver (scipy Radau) |
| `pochoir/arrays.py` | Backend-agnostic array utilities |
| `pochoir/bc_interp.py` | 2D→3D boundary-condition interpolation |
| `pochoir/lar.py` | LAr electron mobility function |
| `pochoir/__main__.py` | CLI commands (fdm:274, velo:337, drift:403, bc-interp:450) |
| `pochoir/geom.py` | Initial/boundary array initialisation |
| `pochoir/npz.py` | NPZ-based persistent store |

---

## Navigation

- `01-algorithm.md` — Mathematical description of the Jacobi FDM and drift pipeline
- `02-bugs.md` — Potential bug catalog with file:line citations and severity
- `03-gpu-efficiency.md` — GPU running-efficiency analysis
- `04-memory.md` — Memory footprint analysis and reduction opportunities
- `05-cross-backend-comparison.md` — Side-by-side table of all five FDM engines
- `06-summary-and-recommendations.md` — Prioritized punch list
