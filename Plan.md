# Pochoir GPU Acceleration Plan

## Context
Pochoir solves FDM (Laplace/Poisson) for LAr TPC detectors. `fdm_torch.py` has solid two-pass (float32→float64) architecture and `@torch.compile` stencil but leaks performance via CPU↔GPU transfers, excessive synchronization, and poor memory reuse. Goal: maximize GPU throughput while preserving correctness and multi-backend design.

---

## Phase 1 — Eliminate CPU↔GPU Bottlenecks in `fdm_torch.py`

**Target file:** `pochoir/fdm_torch.py`

**Actions:**
- Remove all `torch.cuda.synchronize()` calls from hot path; keep only around timed benchmarks behind `--profile` flag
- Keep all tensors on device throughout iteration loop; defer `.cpu().numpy()` to single call after convergence
- Replace per-epoch numpy error checks with `torch.max(torch.abs(...))` on-device; no CPU transfer until final report
- Preallocate `tmp_core` once before loop; reuse in-place with `out=` parameter in stencil ops
- Remove leftover debug `print()` calls; unify logging to standard `logging` module

**Keep:** `@torch.compile`, two-pass float32→float64 strategy, auto device detection

---

## Phase 2 — Optimize Stencil in `fdm_generic.py`

**Target file:** `pochoir/fdm_generic.py`

**Actions:**
- Annotate `stencil_poisson` / `stencil` with `torch.compile(fullgraph=True)` if not already
- Replace slicing-based stencil with `torch.nn.functional.conv3d` (or `conv2d` for 2D) with fixed Laplacian kernel → single fused CUDA kernel instead of 6 slice operations
- Precompute Laplacian kernel as registered buffer on first call; cache by dtype+device
- Verify `edge_condition` stays GPU-side (no hidden numpy ops)

---

## Phase 3 — IDW Initialization (`InverseDistanceWeight_torch.py`)

**Target file:** `pochoir/InverseDistanceWeight_torch.py`

**Actions:**
- Replace hardcoded `batch=1024` with dynamic batch sizing based on `torch.cuda.get_device_properties().total_memory`
- Use `torch.cdist` for pairwise distance computation (fused CUDA) instead of manual loops
- Enable `torch.compile` on the main IDW kernel function
- Pin CPU tensors with `pin_memory=True` before `.to(device)` to overlap H2D transfer with compute

---

## Phase 4 — Drift Solver (`drift_torch.py`)

**Target file:** `pochoir/drift_torch.py`

**Actions:**
- Ensure `torchdiffeq` ODE solver runs fully on GPU without intermediate CPU callbacks
- Batch multiple drift trajectories in single ODE call (vectorize over particles axis)
- Use `torch.compile` on the drift RHS function
- Profile and replace `torch_interpolations` if it forces CPU fallback

---

## Phase 5 — Memory & Precision Management

**Cross-cutting:**
- Add `torch.cuda.empty_cache()` + `torch.cuda.reset_peak_memory_stats()` at start of each `solve()` call
- Add optional `--amp` flag (Automatic Mixed Precision via `torch.autocast`) for float16 compute on supported GPUs
- Add `--profile` CLI flag that wraps solve in `torch.profiler.profile` and exports Chrome trace
- Document GPU memory requirements in README for typical grid sizes

---

## Phase 6 — Testing & Validation

**Target:** `test/`

**Actions:**
- Add regression test: compare torch result vs numpy result within tolerance (1e-5) for 2D and 3D cases
- Add benchmark script `bench_fdm.py`: measures solve time and GPU utilization (via `nvidia-smi`) per phase
- Validate pixelated readout workflow end-to-end after changes

---

## Critical Files

| File | Phase |
|------|-------|
| `pochoir/fdm_torch.py` | 1 |
| `pochoir/fdm_generic.py` | 2 |
| `pochoir/InverseDistanceWeight_torch.py` | 3 |
| `pochoir/drift_torch.py` | 4 |
| `pochoir/__main__.py` | 5 (CLI flags) |
| `test/` | 6 |

## What NOT to change
- Multi-backend router (`fdm.py`, `arrays.py`) — keep numpy/cupy/numba paths intact
- Two-pass float32→float64 solve strategy
- `domain.py`, `persist.py`, `schema.py`, `lar.py` — unrelated to GPU path
- CLI command structure and metadata tracking
