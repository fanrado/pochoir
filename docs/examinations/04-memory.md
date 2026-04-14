# Memory Footprint Analysis

*Read-only examination. No code was modified. All claims are supported by file:line citations.*

---

## 1. Data Type (dtype) Policy

There is **no unified dtype policy** across the codebase. The default numpy dtype is `float64` (8 bytes/element), which propagates through almost all paths.

| File | Location | dtype |
|------|----------|-------|
| `geom.py:41` | `barr = numpy.zeros(dom.shape, dtype=bool)` | bool (1 byte) |
| `geom.py:45` | `iarr = numpy.zeros(dom.shape, dtype='f4') + ambient` | **float32** (4 bytes) |
| `fdm_torch.py:47` | `tmp_core = torch.zeros(iarr.shape, ...)` | **float32** (torch default) |
| `fdm_torch.py:45` | `bi_core = torch.tensor(iarr*barr, ...)` | follows `iarr` dtype (float32 from geom, but numpy default float64 if iarr re-cast) |
| `fdm_cupy.py:43` | `tmp_core = cupy.zeros(iarr.shape)` | **float64** (cupy default) |
| `fdm_cumba.py:64` | `tmp_pad = cupy.zeros_like(iarr_pad)` | follows `iarr_pad` dtype |
| `arrays.py:15` | `zeros = numpy.zeros` | float64 by default |
| `__main__.py:439` | `pochoir.arrays.zeros((n_starts, n_ticks, ndim))` | float64 (no dtype arg) |

The initial array `iarr` is created as float32 in `geom.py:45`, but is then passed to `fdm_cupy.py:38` where `cupy.array(iarr)` preserves float32. However:
- `fdm_cupy.py:43`: `tmp_core = cupy.zeros(iarr.shape)` — creates a **new float64** array regardless of `iarr` dtype.
- `fdm_cumba.py:64`: `tmp_pad = cupy.zeros_like(iarr_pad)` — correctly inherits dtype.

The mixed float32/float64 situation means memory footprint is unpredictable and depends on the path taken.

---

## 2. Per-Backend Simultaneous Device Array Inventory

### 2a. `fdm_cupy.py` — device arrays at peak residency

During the main iteration loop, the following arrays exist simultaneously on the GPU:

| Variable | Shape | dtype | Notes |
|----------|-------|-------|-------|
| `iarr` | core `(N)` | as input | uploaded from host |
| `barr` | core `(N)` | bool | uploaded from host |
| `bi_core` | core `(N)` | as `iarr` | `iarr*barr` computed on device |
| `mutable_core` | core `(N)` | bool | `~barr` on device |
| `tmp_core` | core `(N)` | float64 | result of stencil |
| `err` | core `(N)` | as `iarr` | allocated upfront (L45), **never used**; overwritten at L70 |
| `barr_pad` | padded `(N+2)` | bool | allocated (L47), **never used** |
| `iarr_pad` | padded `(N+2)` | as `iarr` | main working array |
| `ifixed` | padded `(N+2)` | bool | boolean index mask, **never used** (L51) |
| `fixed` | 1D (n_boundary,) | as `iarr` | boundary values, **never used** (L52) |
| `prev` | core `(N)` | as `iarr` | per-epoch snapshot; allocated once per epoch |

**Peak simultaneous**: 11 arrays. Of these, 4 are dead allocations (`err`, `barr_pad`, `ifixed`, `fixed`). Removing them reduces to 7.

### 2b. `fdm_torch.py` — device arrays at peak residency

| Variable | Shape | dtype | Notes |
|----------|-------|-------|-------|
| `bi_core` | core `(N)` | follows `iarr` | `iarr*barr` on host, then tensor to device |
| `mutable_core` | core `(N)` | bool | `~barr` on host, then tensor to device |
| `tmp_core` | core `(N)` | float32 | **float32** regardless of `iarr` dtype |
| `barr_pad` | padded `(N+2)` | bool | `numpy.pad(barr,1)` on host, tensor to device |
| `iarr_pad` | padded `(N+2)` | follows `iarr` | `numpy.pad(iarr,1)` on host, tensor to device |
| `prev` | padded `(N+2)` | follows `iarr` | **full padded clone** once per epoch (L61) |

**Peak simultaneous**: 6 arrays. On epoch boundaries, `iarr_pad` and `prev` coexist (2 × padded-size).

Note: `barr_pad` is computed and transferred to device but never used in the iteration loop (similar to `fdm_cupy.py`). It is used only implicitly to derive `core` slices, which are computed from `iarr_pad.shape` anyway.

### 2c. `fdm_cumba.py` — device arrays at peak residency

| Variable | Shape | dtype | Notes |
|----------|-------|-------|-------|
| `iarr_pad` | padded `(N+2)` | as `iarr` | main working array; rebound each step (BUG-05) |
| `barr_pad` | padded `(N+2)` | bool | allocated (L60), **never used** |
| `bi_pad` | padded `(N+2)` | as `iarr` | `iarr*barr` on host, then padded on device |
| `mutable_pad` | padded `(N+2)` | bool | `~barr` unpadded on device, then padded |
| `tmp_pad` | padded `(N+2)` | as `iarr_pad` | stencil output |
| `err` | core `(N)` | as `iarr` | allocated upfront (L66), overwritten at L86 |
| `prev` | padded `(N+2)` | as `iarr_pad` | **full padded clone** once per epoch (L78) |

**Peak simultaneous**: 7 arrays, all at padded shape. `barr_pad` is a dead allocation.

Plus: on every step, an **additional temporary** `bi_pad + mutable_pad*tmp_pad` exists briefly (the new `iarr_pad`) before the old one is dereferenced.

---

## 3. Memory Footprint Estimates

Using the production domain sizes from `tutorial/tutorial.jsonnet`:

### 3a. `weight3d` tutorial domain: 350 × 66 × 2000

| Quantity | Value |
|----------|-------|
| Core cells (N) | 350 × 66 × 2000 = 46,200,000 |
| Padded cells | 352 × 68 × 2002 = 47,922,688 |
| Float64 per array (core) | 46.2 M × 8 B = **370 MB** |
| Float64 per array (padded) | 47.9 M × 8 B = **383 MB** |
| Float32 per array (core) | **185 MB** |
| Float32 per array (padded) | **192 MB** |

Backend peak GPU residency (float64):

| Backend | Live arrays (cleaned) | Peak residency |
|---------|----------------------|----------------|
| `fdm_cupy` (as coded, 11 arrays) | 9 (core) + 1 (padded) + 1 (prev core) | ~9×370 + 1×383 + 370 ≈ **4.1 GB** |
| `fdm_cupy` (dead arrays removed, 7 arrays) | 4 (core) + 1 (padded) + 1 (prev core) | ~4×370 + 383 + 370 ≈ **2.2 GB** |
| `fdm_torch` (6 arrays) | 3 (core) + 2 (padded) + 1 (prev padded) | ~3×370 + 3×383 ≈ **2.3 GB** |
| `fdm_cumba` (7 arrays, all padded) | 6×padded + 1 extra/step | ~7×383 ≈ **2.7 GB** |

At float32 (half the footprint): `fdm_cupy` cleaned → ~1.1 GB; `fdm_torch` → ~1.15 GB; `fdm_cumba` → ~1.35 GB.

### 3b. `weight3d` test domain: 350 × 32 × 700

| Quantity | Value |
|----------|-------|
| Core cells | 350 × 32 × 700 = 7,840,000 |
| Padded cells | 352 × 34 × 702 = 8,398,368 |
| Float64 per array (core) | **62.7 MB** |
| Float64 per array (padded) | **67.2 MB** |

Backend peak GPU residency (float64):

| Backend | Peak residency (as coded) | Peak residency (cleaned, float32) |
|---------|--------------------------|-----------------------------------|
| `fdm_cupy` (11 arrays) | ~9×62.7 + 2×67.2 ≈ **699 MB** | ~5×31.4 + 2×33.6 ≈ **224 MB** |
| `fdm_torch` (6 arrays) | ~3×62.7 + 3×67.2 ≈ **389 MB** | ~**195 MB** |
| `fdm_cumba` (7 padded) | ~7×67.2 ≈ **471 MB** | ~**235 MB** |

These are comfortably within 16 GB GPU limits for the test domain, but the tutorial `weight3d` at 2–4 GB is tight on a 16 GB GPU and will OOM on a 4 GB or 8 GB card.

---

## 4. Boundary Encoding Overhead

The boundary condition uses two dense arrays:

1. `barr` (`geom.py:41`): bool array over the full domain — 1 byte per voxel regardless of electrode coverage.
2. `bi_core = iarr * barr`: float array over the full domain — nonzero only on electrode surfaces.

For the `weight3d` domain (350 × 66 × 2000 = 46.2 M cells), the electrode surfaces occupy a small fraction of the total volume (PCB strips with holes, occupying perhaps 5–10% of cells). The dense encoding is therefore 10–20× larger than a sparse representation would require.

This is not a correctness issue, but for very large 3D domains the dense boundary encoding contributes ~400 MB per pair (`barr` + `bi_core`), which could be reduced to a few MB with a sparse (COO or CSR) representation.

---

## 5. Persistence Overhead

### 5a. Uncompressed NPZ

`pochoir/npz.py:73`: `numpy.savez(dp.resolve(), **arrs)` — uncompressed ZIP.

Numpy's `savez_compressed` applies zlib compression. For FDM potential arrays (slowly varying, high spatial correlation), compression ratios of 3–5× are typical. A 370 MB float64 potential file would compress to ~75–120 MB. Over many solve runs (many electrodes), the disk savings are significant.

### 5b. Full increment array stored alongside potential

`__main__.py:333`:
```python
ctx.obj.put(increment, err, taxon="increment", **params)
```

The full-shape error array (`err`) is persisted. For `weight3d` (370 MB float64), this doubles the disk footprint of every FDM output. The only apparent use of the stored increment is diagnosis (inspecting how well the solver converged). A scalar summary `max|err|` stored in the metadata JSON would serve the same purpose at ~100 bytes.

---

## 6. Copy Overhead in `arrays.py`

Several utility functions always copy:

- `arrays.to_numpy(array)` (`arrays.py:67–77`): calls `numpy.array(array)` even when the input is already a numpy array — creates an unnecessary copy.
- `arrays.to_torch(array, device)` (`arrays.py:79–84`): calls `torch.tensor(array, device=device)` — always copies. The non-copying equivalent is `torch.as_tensor(array)`.
- `arrays.dup(array)` (`arrays.py:126–133`): explicit full copy — correct by design, but should only be used when a copy is actually needed.
- `arrays.vmag(vfield)` (`arrays.py:113–123`): allocates `numpy.zeros_like(c2s[0])` and accumulates — unavoidable but note it operates on host only.

---

## 7. Ranked Memory Savings (Future Work, No Code Changed)

1. **Force float32 throughout** (modify dtype in `fdm_cupy.py`, `fdm_cumba.py`, `geom.py` already uses f4 for `iarr`): halves all FDM array footprints. ~2× reduction in GPU residency.

2. **Remove dead allocations in `fdm_cupy.py`** (`err` at L45, `barr_pad` at L47, `ifixed` at L51, `fixed` at L52): eliminates ~4 full core/padded arrays. For `weight3d` (float64), saves ~4 × 370 MB = **1.5 GB**.

3. **Remove `barr_pad` in `fdm_cumba.py`** (L60): saves 1 padded-size array ≈ **383 MB** (float64) or **192 MB** (float32).

4. **In-place update in `fdm_cumba.py`** (fix BUG-05): eliminates the transient new allocation of `iarr_pad` each step (~383 MB briefly each step, though pool recycles immediately).

5. **Clone only core in `fdm_torch.py`** (change `iarr_pad.clone()` at L61 to `iarr_pad[core].clone()`): saves (padded − core) cells per epoch. For `weight3d`: 383 − 370 = 13 MB per epoch — modest but correct.

6. **Store only `max|err|` scalar in increment** (`__main__.py:333`): saves 370 MB (float64) or 185 MB (float32) per FDM output on disk.

7. **Switch to `numpy.savez_compressed`** in `npz.py:73`: 3–5× disk space reduction for potential/velocity/path arrays.

8. **Keep arrays resident across commands** (requires a single-process orchestrator instead of separate CLI invocations): eliminates all H2D/D2H copies between `fdm` → `velo` → `drift`. Structural change.
