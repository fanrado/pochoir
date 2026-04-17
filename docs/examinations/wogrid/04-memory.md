# `wogrid` Branch — GPU/CPU Memory Analysis

*All line numbers refer to `origin/wogrid` as of 2026-04-17.*

*Compare with `../04-memory.md` (master-branch analysis) for the master
baseline.*

---

## 1. Reference Domains

Two representative grids from `test/test-full-3d-pixel.sh`:

| Name | Shape | N cells | fp32/cell | fp64/cell |
|---|---|---|---|---|
| `drift3d` | 44 × 44 × 1500 | 2.90 M | 11.6 MB | 23.2 MB |
| `weight3d` | 220 × 220 × 1500 | 72.6 M | 290 MB | **581 MB** |

The padded versions (used in the solver) add 2 cells per dimension:

| Name | Padded shape | N cells padded |
|---|---|---|
| `drift3d_pad` | 46 × 46 × 1502 | 3.18 M |
| `weight3d_pad` | 222 × 222 × 1502 | 74.0 M |

For memory table computations the weight3d domain is used as the
representative worst case (as in the master examination).

---

## 2. Standard Mode (`--multisteps no`, Single `solve()` Call)

### 2.1 Tensor Inventory

All tensors created inside `fdm_torch.py` `solve()` for a single run,
excluding the dead allocations:

| Tensor | Shape | dtype | Note |
|---|---|---|---|
| `bi_core` | 72.6 M | `_dtype` | Fixed boundary values |
| `mutable_core` | 72.6 M | `_dtype` | Float mask (0/1); could be bool |
| `tmp_core` | 72.6 M | `_dtype` | Scratch for stencil output |
| `iarr_pad` | 74.0 M | `_dtype` | Padded working array |
| `prev` | 74.0 M | `_dtype` | Clone for convergence check |

**Dead (never read) tensors also allocated:**

| Tensor | Shape | dtype | Bug ref |
|---|---|---|---|
| `barr_pad` | 74.0 M | `_dtype` | B-4 |

### 2.2 Total GPU Memory (Standard Mode)

Using `weight3d` domain:

| dtype | Live tensors | Dead (barr_pad) | Total |
|---|---|---|---|
| fp64 | 5 × ~73 M cells × 8 B ≈ **2.86 GB** | +0.59 GB | **3.45 GB** |
| fp32 | 5 × ~73 M cells × 4 B ≈ **1.43 GB** | +0.30 GB | **1.73 GB** |

The fp32 option roughly halves GPU memory vs fp64. On a 16 GB GPU, the fp64
standard mode is comfortable. On an 8 GB GPU, fp64 is tight (3.45 GB leaves
~4.5 GB for CUDA context, framework overhead, and batched operations).

*Master comparison:* The master-branch `torch` backend on the same domain
size (tutorial `weight3d` 350×66×2000 ≈ 46.2 M cells) uses ~2.5–3.5 GB at
fp64 (see `../04-memory.md §3`). The `wogrid` weight3d domain is larger
(72.6 M vs 46.2 M cells), so total memory is higher.

---

## 3. Poisson Mode (Two-Step, `--multisteps yes`)

Step 2 runs `solve()` with `phi0 ≠ None`. The additional tensors created
inside the `if phi0 is not None` block:

| Tensor | Shape | dtype | Note |
|---|---|---|---|
| `iarr_pad_source` | 74.0 M | `_dtype` | Dead; never used (B-5) |
| `non_padded_phi0` | 72.6 M | `_dtype` | Copy of input phi0 on device |
| `phi0_` | 74.0 M | **fp64** | Padded phi0 for stencil |
| `s` | 72.6 M | **fp64** | `stencil(phi0_)` output |
| `source` | 72.6 M | `_dtype` | Poisson source (cast from fp64) |
| `source_FP32` (checkpoint) | 72.6 M | `_dtype` | Saved to store; CPU copy made |

These tensors are all created **before the iteration loop** and persist for
its duration. The `ctx.obj.put` calls copy `s`, `source`, and `phi0_` to CPU
for checkpointing (three `~581 MB` CPU copies).

### 3.1 Peak GPU Memory (Poisson Mode, Step 2)

All standard-mode live tensors (5) + Poisson-init tensors (5 live + 1 dead):

| dtype | Live | Dead | Total |
|---|---|---|---|
| fp64 solve + fp64 phi0_ | 10 × ~73 M × 8 B ≈ **5.71 GB** | +1.18 GB | **6.89 GB** |

Note: `phi0_` and `s` are always fp64 regardless of `_dtype` (hard-coded on
lines 112 and 113). In a fp32 Step 2 run, `source` would be fp32 but the
intermediates are fp64.

For the `drift3d` domain (smaller, 2.90 M cells) all modes fit easily in a
few hundred MB.

---

## 4. Memory Held Outside the Solver

### 4.1 Checkpoint CPU Copies

In Poisson mode the solver calls `ctx.obj.put(..., tensor.cpu(), ...)` to
save intermediate arrays for debugging:

```
padded_phi0  → ~74.0 M × 8 B = 592 MB on CPU
stencil      → ~72.6 M × 8 B = 581 MB on CPU
source       → ~72.6 M × 8 B = 581 MB on CPU
phi0_withBC  → ~74.0 M × 8 B = 592 MB on CPU
source_FP32  → ~72.6 M × 4–8 B = 290–581 MB on CPU
```

These five checkpoints together require **~2.6 GB of CPU RAM** for the
weight3d domain. They are written to the NPZ store (`ctx.obj.put`) which may
buffer them in memory before flushing to disk. On a machine with limited RAM,
this can cause paging.

### 4.2 Two-Step Orchestration (`__main__.py`)

After Step 1 returns `(phi_0, err_phi0)`, both tensors remain in Python scope
(as CPU tensors) while Step 2 runs. The total CPU RAM during Step 2:

| Array | Size (weight3d, fp64) |
|---|---|
| `phi_0` (Step 1 result) | 581 MB |
| `err_phi0` (Step 1 error) | 581 MB |
| `delta_phi` (Step 2 result, returned) | 581 MB |
| `err_delta_phi0` (Step 2 error) | 581 MB |
| Solver checkpoints (see §4.1) | ~2.6 GB |
| **Total CPU RAM during Step 2** | **~4.9 GB** |

This does not count the NumPy input arrays (`iarr`, `barr`).

---

## 5. Memory Reduction Opportunities

| Opportunity | Saving (weight3d, fp64) | Complexity |
|---|---|---|
| Delete `barr_pad` (dead, B-4) | 592 MB GPU | Trivial (`del barr_pad`) |
| Delete `iarr_pad_source` (dead, B-5) | 592 MB GPU | Trivial |
| Delete `phi0_`, `s` after source computation | 2×592 MB GPU (Poisson mode) | Simple (`del phi0_, s`) |
| Use `bool` for `mutable_core` instead of float | 72.6 MB GPU (1/8 of fp64 size) | Simple |
| Disable checkpoint `ctx.obj.put` in non-debug runs | ~2.6 GB CPU RAM | Add a `--debug-checkpoints` flag |
| Return `err` from Step 1 lazily (avoid holding two error tensors) | ~581 MB CPU | Refactor orchestration |

The two dead-tensor deletions alone save **~1.2 GB of GPU memory** at fp64
on the weight3d domain — enough to substantially reduce peak GPU memory in
Poisson mode.

---

## 6. Comparison with Master (`torch` Backend)

| Metric | Master (`weight3d` 46.2 M cells, fp64) | `wogrid` (`weight3d` 72.6 M cells, fp64) |
|---|---|---|
| Live tensors in solve | 5 | 5 (standard) / 10 (Poisson) |
| Dead tensors | 1 (`barr_pad`) | 2 (`barr_pad`, `iarr_pad_source`) |
| Peak GPU (standard mode) | ~2.5 GB | ~3.5 GB |
| Peak GPU (Poisson mode) | N/A | ~6.9 GB |
| Checkpoint CPU copies | None | ~2.6 GB |

The `wogrid` branch introduces higher memory pressure primarily due to: (a)
the larger domain size used in pixel geometry tests, (b) the Poisson-mode
intermediate tensors, and (c) the debug checkpoint copies.
