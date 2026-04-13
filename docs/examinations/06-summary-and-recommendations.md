# Summary and Recommendations

*Read-only examination. No code was modified. This document is a synthesis of findings from 01–05.*

---

## Executive Summary

`pochoir` is a functional Laplace-equation solver for LArTPC field-response calculations with five FDM engine backends and two drift-path solvers. The GPU infrastructure is architecturally sound (host-driven Jacobi loop with device kernels) but has several concrete bugs, a systematic memory-management inefficiency, and an algorithmic bottleneck that limits scalability regardless of hardware.

**The three highest-impact issues to address, in order:**

1. **BUG-01** — The 3D `cumba` kernel is broken (wrong `cuda.grid` dimensionality).
2. **BUG-02** — The `torch` FDM backend crashes on modern NumPy (removed `numpy.bool`).
3. **The Jacobi algorithm** — No acceleration scheme means O(N²) iteration counts; GPU parallelism does not compensate.

---

## Prioritized Punch List

### Priority 1: Correctness Bugs (broken at runtime)

| ID | Severity | Fix location | Description |
|----|----------|-------------|-------------|
| BUG-01 | HIGH | `fdm_cumba.py:24` | Change `cuda.grid(2)` → `cuda.grid(3)` for 3D kernel. Also update `blockspergrid` to a 3-tuple (L43–48). |
| BUG-02 | HIGH | `fdm_torch.py:46` | Replace `numpy.bool` → `numpy.bool_` or `bool`. Better: use `arrays.invert(barr)` which is already correct. |
| BUG-04 | MED | `fdm_cupy.py:73–75`; `fdm_cumba.py:89–91` | Add `.get()` on the early-return path to match the late-return path. One-line fix each. |

### Priority 2: Silent Behavioural Bugs (wrong results or crash under specific conditions)

| ID | Severity | Fix location | Description |
|----|----------|-------------|-------------|
| BUG-07 | MED | `drift_torch.py:35`; `drift_numpy.py:32` | Replace `torch.arange(start, stop, step)` with `torch.linspace(start, start+(shape-1)*step, shape)` to avoid float accumulation in axis length. |
| BUG-08 | MED | `drift_torch.py` (class `Simple`) | Add `inside()` and `extrapolate()` methods matching `drift_numpy.py:39–59`. |

### Priority 3: Performance and Memory Bugs

| ID | Severity | Fix location | Description |
|----|----------|-------------|-------------|
| BUG-05 | MED | `fdm_cumba.py:82` | Change `iarr_pad = bi_pad + mutable_pad*tmp_pad` to in-place: `cupy.add(bi_pad, cupy.multiply(mutable_pad, tmp_pad), out=iarr_pad)`. Eliminates per-step allocation. |
| BUG-06 | MED | `fdm_cupy.py:45,47,51–52` | Remove dead allocations of `err` (L45), `barr_pad` (L47), `ifixed` (L51), `fixed` (L52). Saves ~3–4 full arrays of GPU memory. |
| BUG-06b | MED | `fdm_cumba.py:60` | Remove `barr_pad` (L60 — never used). |
| BUG-12 | LOW | `lar.py:10–51` | Rewrite `mobility_function` using array-level NumPy operations (no `math.sqrt`, no `numpy.vectorize`). 10–100× CPU speedup for the `velo` step. |
| BUG-13 | INFO | `__main__.py:333` | Store only `float(maxerr)` as a metadata key instead of the full `err` array; halves disk footprint of FDM outputs. |

### Priority 4: Robustness and Compatibility

| ID | Severity | Fix location | Description |
|----|----------|-------------|-------------|
| BUG-03 | HIGH (misleading) | `drift_torch.py:67` | Either implement GPU-resident drift integration, or rename the module `drift_torchdiffeq.py` and document that it is CPU-only. |
| BUG-09 | LOW | `fdm_torch.py:47` | Add `dtype=bi_core.dtype` to `torch.zeros(...)` to match precision of other arrays. |
| BUG-10 | LOW | `drift_torch.py:50` | Wrap the print in `if self.verbose:` (as done in `drift_numpy.py:75–76`). |
| BUG-11 | LOW | `bc_interp.py:5–6` | Replace direct `from scipy.interpolate import RGI` with `from pochoir.arrays import rgi` to allow future GPU acceleration. |

---

## Algorithmic Recommendations (Structural, Not Bug Fixes)

These require more significant changes and should be considered after the bugs are resolved:

### A. Accelerate Convergence: SOR or Red-Black Gauss-Seidel

The Jacobi iteration converges as ρ^k where ρ ≈ 1 − π²/N² for the smallest dimension N. For the `weight3d` production domain (N_min = 66), converging to 2×10⁻¹⁰ absolute precision requires roughly 12–20 million Jacobi iterations.

**Successive Over-Relaxation (SOR)** with ω_opt = 2/(1 + sin(π/N_min)) reduces ρ to ω_opt − 1, cutting required iterations by ~50×. On the same GPU hardware this would reduce a 20 M-iteration run to ~400 K iterations. SOR requires only a small change to the update rule and is compatible with the existing stencil infrastructure.

**Red-Black Gauss-Seidel** (updating even and odd cells alternately) does not itself accelerate convergence but allows SOR to be applied without race conditions and can be implemented efficiently with two GPU kernels per step.

### B. Fuse the Iteration Loop

The current Python `for` loop over `nepochs × epoch` steps dispatches many small kernels per step. For the `cumba` backend (fused CUDA stencil), the bottleneck is Python loop overhead: at ~1 μs per Python iteration, 20 M steps costs ~20 seconds of Python overhead alone, before GPU compute.

Wrapping the inner loop in a CUDA kernel (or using a CuPy `RawKernel` with a device-side for-loop) would eliminate this overhead. The convergence check (once per epoch) could be done with a device-side reduction that writes a scalar to pinned memory, checked asynchronously.

### C. Use `numpy.gradient` Replacement on GPU

The `arrays.gradient()` function (`arrays.py:96–111`) does a full D2H transfer of the potential, computes the gradient on CPU via `numpy.gradient`, then transfers back. For the tutorial `weight3d` potential (370 MB float64), this round-trip costs ~740 MB of transfers.

PyTorch provides `torch.gradient` (as of PyTorch 1.11) which operates natively on tensors. CuPy provides `cupy.gradient` with the same signature as `numpy.gradient`. Either would eliminate the round-trip.

### D. Single-Process Pipeline Driver

The current CLI design runs each command in a separate process, destroying the GPU context and flushing all device memory between steps. The `fdm` → `velo` → `drift` sequence involves unnecessary serialise-to-disk and reload cycles.

A single-process driver (a Python script or Snakemake rule that calls the solver functions directly rather than via CLI subprocesses) would allow the potential array to remain on GPU between `fdm` and `velo`, saving two full array transfers per domain per electrode.

---

## Memory Reduction Summary

| Action | Savings (weight3d, float64) |
|--------|-----------------------------|
| Remove dead allocations in `fdm_cupy` (BUG-06) | ~1.5 GB GPU |
| Switch all FDM arrays to float32 | ~50% of all above |
| Fix in-place `iarr_pad` in `fdm_cumba` (BUG-05) | eliminates transient alloc per step |
| Store only `max|err|` scalar (BUG-13) | 370 MB disk per FDM run |
| Use `savez_compressed` in `npz.py` | 3–5× disk reduction |
| GPU gradient (recommendation C) | ~740 MB transfers per `velo` |

---

## Quick Reference: Bug Locations

| Bug | File | Line(s) |
|-----|------|---------|
| BUG-01: 3D cumba `cuda.grid(2)` | `pochoir/fdm_cumba.py` | 24 |
| BUG-02: `numpy.bool` removed | `pochoir/fdm_torch.py` | 46 |
| BUG-03: drift_torch CPU-only | `pochoir/drift_torch.py` | 67 |
| BUG-04: inconsistent return type | `pochoir/fdm_cupy.py` | 73–75 |
| BUG-04: inconsistent return type | `pochoir/fdm_cumba.py` | 89–91 |
| BUG-05: iarr_pad realloc per step | `pochoir/fdm_cumba.py` | 82 |
| BUG-06: dead allocations | `pochoir/fdm_cupy.py` | 45, 47, 51–52 |
| BUG-06b: dead barr_pad | `pochoir/fdm_cumba.py` | 60 |
| BUG-07: torch.arange rounding | `pochoir/drift_torch.py` | 33–35 |
| BUG-08: no bounding-box check | `pochoir/drift_torch.py` | 43–60 |
| BUG-09: mixed float32/float64 | `pochoir/fdm_torch.py` | 47 |
| BUG-10: chatty print in RHS | `pochoir/drift_torch.py` | 50 |
| BUG-11: scipy RGI hardcoded | `pochoir/bc_interp.py` | 5–6 |
| BUG-12: numpy.vectorize mobility | `pochoir/lar.py` | 51 |
| BUG-13: full err array persisted | `pochoir/__main__.py` | 333, 439 |
