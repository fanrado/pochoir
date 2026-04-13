# Potential Bugs — Catalog

*Read-only examination. No code was modified. Severity: HIGH = likely causes incorrect results or crash; MED = behavioural defect or reliability risk; LOW = correctness risk under specific conditions; INFO = quality/maintenance concern.*

---

## BUG-01 [HIGH] `fdm_cumba.py:24` — `cuda.grid(2)` for 3D kernel

**File:** `pochoir/fdm_cumba.py:22–29`

```python
@cuda.jit
def stencil_numba3d_jit(arr, out):
    i, j, k = cuda.grid(2)     # ← BUG: cuda.grid(2) returns exactly 2 ints
    l, n, m = arr.shape
    if 1 <= i < l - 1 and 1 <= j < n - 1 and 1 <= k < m - 1:
        out[i, j, k] = (1/6.0)*(...)
```

`numba.cuda.grid(ndim)` returns a tuple of `ndim` integers representing the absolute thread index in a grid. Calling `cuda.grid(2)` returns `(i, j)` — exactly two values. Attempting to unpack this into three names `i, j, k` will raise a Python-level `ValueError: not enough values to unpack` at JIT compilation or at first call, crashing any 3D `cumba` solve.

The 2D kernel at line 14–20 correctly uses `cuda.grid(2)`:
```python
@cuda.jit
def stencil_numba2d_jit(arr, out):
    i, j = cuda.grid(2)     # ← correct for 2D
```

**Fix direction (not applied):** Change line 24 to `i, j, k = cuda.grid(3)` and make sure the launch configuration at `fdm_cumba.py:43–48` passes a 3-tuple `blockspergrid`.

**Impact:** The 3D cumba path (`--engine cumba` on a 3D domain) is entirely broken and has never worked.

---

## BUG-02 [HIGH] `fdm_torch.py:46` — `numpy.bool` removed in NumPy ≥ 1.24

**File:** `pochoir/fdm_torch.py:46`

```python
mutable_core = torch.tensor(
    numpy.invert(barr.astype(numpy.bool)),   # ← numpy.bool is removed
    requires_grad=False
).to(device)
```

`numpy.bool` was deprecated in NumPy 1.20 and **removed in NumPy 1.24**. On any system running NumPy ≥ 1.24, this line raises:

```
AttributeError: module 'numpy' has no attribute 'bool'
```

This is an import-time breakage: any call to `fdm_torch.solve(...)` will fail immediately regardless of array contents.

The correct replacement is either `numpy.bool_` (always valid) or plain Python `bool` (since `barr` is already a numpy bool array).

Note: the `arrays.py:182` module already provides `arrays.invert(arr)` which handles torch/numpy uniformly and uses `arr.logical_not()` for torch tensors — the correct approach that avoids this issue entirely.

**Impact:** `--engine torch` is broken on any modern NumPy stack. Commit `4da650d "Fix torch version of FDM"` apparently addressed other issues but left this one.

---

## BUG-03 [HIGH] `drift_torch.py:67` — drift integration hardcoded to CPU

**File:** `pochoir/drift_torch.py:63–75`

```python
def solve(domain, start, velocity, times, **kwds):
    device = 'cpu'                                        # ← hardcoded
    start = torch.tensor(start, dtype=torch.float32, device=device)
    velocity = [torch.tensor(v, dtype=torch.float32, device=device) for v in velocity]
    times = torch.tensor(times, dtype=torch.float32, device=device)
    func = Simple(domain, velocity)
    res = odeint(func, start, times, rtol=0.01, atol=0.01)
    return res.cpu().numpy()
```

Despite the file being named `drift_torch.py` and being selected via `--engine torch` in the `drift` CLI command, all tensors are explicitly placed on CPU. The `odeint` integrator and the `RegularGridInterpolator` both run on CPU. The final `.cpu()` call is a no-op.

Users expecting GPU-accelerated drift integration when `--engine torch` is selected will not get it.

**Impact:** Misleading naming. The torch drift path provides `torchdiffeq`'s autodiff-capable ODE solver (potentially useful for adjoint methods), but provides no GPU acceleration for the drift step.

---

## BUG-04 [MED] `fdm_cupy.py:75` and `fdm_cumba.py:91` — Inconsistent return types

**File:** `pochoir/fdm_cupy.py:73–79`

```python
if prec and maxerr < prec:
    print(f'fdm reach max precision: {prec} > {maxerr}')
    return (iarr_pad[core], err)          # ← returns cupy arrays

print(f'fdm reach max epoch ...')
res = (iarr_pad[core], err)
return tuple([r.get() for r in res])      # ← returns numpy arrays
```

**File:** `pochoir/fdm_cumba.py:89–95` (same pattern):

```python
return (iarr_pad[core], err[core])        # ← early exit: cupy arrays
...
return tuple([r.get() for r in res])      # ← late exit: numpy arrays
```

The early-return path (precision reached) returns cupy arrays. The late-return path (epoch limit reached) returns numpy arrays. The caller in `__main__.py:325–326` does:

```python
arr, err = solve(iarr, barr, bool_edges, precision, epoch, nepochs)
ctx.obj.put(potential, arr, ...)
```

`npz.Store.put` calls `numpy.savez(dp.resolve(), **arrs)`. If `arr` is a cupy array, `numpy.savez` will fail or silently produce wrong output depending on NumPy/CuPy version.

The torch backend (`fdm_torch.py:73,75`) is consistent: both paths call `.cpu()`.

**Impact:** When precision is reached before the epoch limit (the desired and common case in production runs), `fdm_cupy` and `fdm_cumba` return cupy arrays, likely causing a crash or silent data corruption in the store.

---

## BUG-05 [MED] `fdm_cumba.py:82` — `iarr_pad` rebound to new array every step

**File:** `pochoir/fdm_cumba.py:80–83`

```python
stencil(iarr_pad, tmp_pad)             # stencil reads iarr_pad, writes tmp_pad
iarr_pad = bi_pad + mutable_pad*tmp_pad   # ← NEW allocation every step; old iarr_pad discarded
edge_condition(iarr_pad, *periodic)
```

The assignment `iarr_pad = ...` **rebinds the Python variable** to a freshly allocated cupy array produced by `bi_pad + mutable_pad*tmp_pad`. The previously allocated `iarr_pad` array becomes garbage and will be collected at some future point. This means:

1. A new device memory allocation is made every single step.
2. CuPy's memory pool will reuse blocks, but the GC pressure is significant.
3. The `prev = cupy.array(iarr_pad)` clone at line 78 captures the *current* `iarr_pad` reference correctly (before the rebind), so the epoch-error computation is technically correct, but only by accident.

The intent is clearly to update `iarr_pad` in-place using `bi_pad` and `tmp_pad`. The correct approach would be `cupy.add(bi_pad, mutable_pad*tmp_pad, out=iarr_pad)` or a combination of in-place ops.

**Impact:** GPU memory allocation churn every iteration. For 5,000,000 iterations per epoch (as in `test-full-3d-50L.sh`), this allocates and immediately discards a full padded-size array 5 million times. Even with CuPy's allocator pool, this is significant overhead.

---

## BUG-06 [MED] `fdm_cupy.py:45,47,51–52` — Dead allocations (copy-paste residue)

**File:** `pochoir/fdm_cupy.py:38–52`

```python
iarr = cupy.array(iarr)
barr = cupy.array(barr)
bi_core = cupy.array(iarr*barr)
mutable_core = cupy.invert(barr)
tmp_core = cupy.zeros(iarr.shape)
err = cupy.zeros_like(iarr)          # L45: allocated, then overwritten at L70; never used as allocated
barr_pad = cupy.pad(barr, 1)         # L47: allocated, never used after this line
iarr_pad = cupy.pad(iarr, 1)
ifixed = barr_pad == True            # L51: allocated, never used
fixed = iarr_pad[ifixed]             # L52: allocated, never used
```

These four lines are copy-paste from `fdm_numpy.py` where:
- `ifixed` and `fixed` are passed to `set_core2(iarr, fixed, ifixed)` to re-assert boundary values each step (line 62 of `fdm_numpy.py`).
- `barr_pad` is used implicitly to derive `ifixed`.

In `fdm_cupy.py`, the `set_core2` call was removed when the masked-update approach was adopted, but the dead variable assignments were left behind. `ifixed` is a boolean index array over a padded-size domain — potentially large.

**Secondary concern:** Because the cupy/cumba backends drop the `set_core2` step, Dirichlet values at boundary cells that fall on the **padded border** (i.e., electrode cells at the edge of the domain) are not explicitly re-asserted each step. The masked-update `bi_core + mutable_core*tmp_core` works for interior boundary cells, but `edge_condition` may overwrite the padded border cells after every step regardless of whether they are electrodes. This is a fragile semantic divergence from `fdm_numpy`.

**Impact:** Wasted GPU memory (3 extra full-size arrays) and silent inconsistency between numpy and GPU backends for boundary cells at domain edges.

---

## BUG-07 [MED] `drift_torch.py:33–35` — `torch.arange` for float grid axes

**File:** `pochoir/drift_torch.py:33–37`

```python
for dim in range(len(domain.shape)):
    start = origin[dim]
    stop  = origin[dim] + shape[dim] * spacing[dim]
    rang = torch.arange(start, stop, spacing[dim])   # ← float accumulation risk
    points.append(rang)
```

`torch.arange(start, stop, step)` with floating-point arguments is documented to be susceptible to rounding errors: the number of elements generated equals `ceil((stop-start)/step)`, which may differ from `shape[dim]` by 1 due to floating-point representation. If `rang.shape[0] != vfield[dim].shape[dim]`, `torch_interpolations.RegularGridInterpolator` will either raise a shape-mismatch exception or silently misalign the interpolation grid.

For comparison, `drift_numpy.py:31–33` has the same pattern with `numpy.arange` — equally susceptible.

The `arrays.py` module provides a comment at line 10 suggesting that the interpolator axis generation "can come from `arrays.rgi()`", which wraps a proper `linspace`-based construction. Neither drift backend uses it.

**Impact:** Sporadic `IndexError` or shape-mismatch during drift integration, likely triggered only for specific combinations of origin, spacing, and shape.

---

## BUG-08 [MED] `drift_torch.py` — No bounding-box check; divergent behaviour from `drift_numpy`

**File:** `pochoir/drift_torch.py:16–60` (class `Simple`)

The `drift_torch.py` `Simple` class has no `inside()` method and no `extrapolate()` fallback. When the ODE integrator steps outside the interpolation domain, `torch_interpolations.RegularGridInterpolator` raises an exception.

`drift_numpy.py:39–59` handles this explicitly:
```python
def inside(self, point): ...          # bounding-box check
def extrapolate(self, pos):
    return numpy.zeros_like(pos)      # zero velocity outside domain
def __call__(self, time, pos):
    if self.inside(pos):
        velo = self.interpolate(pos)
    else:
        velo = self.extrapolate(pos)
```

This means:
- With `--engine numpy`, a trajectory that exits the domain is smoothly stopped (zero velocity).
- With `--engine torch`, the same trajectory causes a crash.

This is a silent behavioural divergence between officially-supported backends.

**Impact:** `--engine torch` drift runs will crash rather than gracefully stop when particles exit the domain boundary.

---

## BUG-09 [LOW] `fdm_torch.py:47` — Mixed precision: float32 `tmp_core` vs float64 `bi_core`

**File:** `pochoir/fdm_torch.py:45–47`

```python
bi_core = torch.tensor(iarr*barr, requires_grad=False).to(device)    # dtype follows iarr (numpy default: float64)
mutable_core = torch.tensor(numpy.invert(barr.astype(numpy.bool)), ...) # bool
tmp_core = torch.zeros(iarr.shape, requires_grad=False).to(device)   # torch default: float32
```

`torch.zeros(...)` without an explicit `dtype` creates a **float32** tensor. `bi_core` will be float64 if `iarr` arrived as numpy float64 (the default everywhere). The update:

```python
iarr_pad[core] = bi_core + mutable_core*tmp_core
```

mixes float64 (`bi_core`) and float32 (`tmp_core`). PyTorch will auto-upcast to float64 during the arithmetic, so the result is numerically correct, but `tmp_core` unnecessarily occupies float32 memory while the update implicitly creates float64 temporaries.

**Impact:** Precision is not actually degraded (upcast preserves values), but memory usage is inconsistent and the asymmetry may mask intentional future changes to force float32 throughout.

---

## BUG-10 [LOW] `drift_torch.py:50` — Chatty debug print inside ODE RHS

**File:** `pochoir/drift_torch.py:50`

```python
def __call__(self, tick, tpoint):
    print(f'drift: point={tpoint} tick={tick}')   # ← runs every RHS evaluation
    ...
```

The ODE RHS is evaluated dozens of times per time step by the adaptive integrator (`torchdiffeq.odeint`). For a drift of 1500 μs at 0.1 μs step (15,000 steps) with an integrator calling the RHS ~6 times per step, this produces ~90,000 print lines per start point. This will dominate wall-clock time and I/O.

The numpy backend (`drift_numpy.py:75–76`) guards its print behind `if self.verbose:`, which is the correct pattern.

**Impact:** Severe performance degradation and unusable console output for any practical drift integration.

---

## BUG-11 [LOW] `bc_interp.py:5–6` — scipy RGI hardcoded, ignores `arrays.rgi()` dispatcher

**File:** `pochoir/bc_interp.py:5–6`

```python
# fixme: this can come from arrays.rgi()
from scipy.interpolate import RegularGridInterpolator as RGI
```

The `# fixme` comment itself acknowledges this. The `arrays.rgi()` function (`arrays.py:162–177`) is the intended dispatch point that would select `torch_interpolations.RGI` for torch arrays and `scipy.interpolate.RGI` for numpy arrays. Using the scipy import directly means `bc_interp.interp` is always CPU-bound, even if a GPU-backed pipeline were implemented in future.

Additionally, `bc_interp.py:34–41` builds interpolation point lists using a Python `for` loop:
```python
for j in range(dom3D.shape[1]):
    points3D_i = list()
    points3D_f = list()
    for z in points3D_z:
        points3D_i.append(numpy.array([...]))
        points3D_f.append(numpy.array([...]))
    arr3D[0,j,:] = func_interp(points3D_i)
    arr3D[-1,j,:] = func_interp(points3D_f)
```

This is O(N_y × N_z) Python operations. For `weight3d` (shape 350 × 32 × 700), that is 32 outer iterations × 700 per inner list-build = 22,400 Python operations. Slow but not catastrophic.

**Impact:** `bc-interp` is always CPU-only regardless of backend; marginally slow due to Python loops.

---

## BUG-12 [LOW] `lar.py:51` — `numpy.vectorize` for per-voxel mobility

**File:** `pochoir/lar.py:51`

```python
mobility = numpy.vectorize(mobility_function)
```

`numpy.vectorize` is explicitly documented as a convenience wrapper for Python-level loops — it provides no SIMD or NumPy-style vectorisation. For the production `weight3d` domain (≈7.84 M cells), this calls the Python `mobility_function` approximately 7.84 million times sequentially. `mobility_function` itself uses `math.sqrt` (scalar), regular Python arithmetic, and float division.

Approximate timing: at ~1 μs per Python call, that is ~8 seconds just for the mobility computation, on CPU, for a single `velo` invocation.

A properly vectorised form using array-level NumPy operations would reduce this to milliseconds.

**Impact:** Significant CPU bottleneck in the `velo` step, especially for large 3D domains.

---

## BUG-13 [INFO] `__main__.py:439–440` — Path buffer defaults to float64; increment array persisted at full size

**File:** `pochoir/__main__.py:439–440`

```python
thepaths = pochoir.arrays.zeros((len(start_points), len(ticks), len(dom.shape)))
```

`pochoir.arrays.zeros` re-exports `numpy.zeros` (`arrays.py:15`), which defaults to float64. For 100 start points × 42,500 ticks × 3 dims = 12.75 M elements × 8 bytes ≈ 102 MB. Float32 would halve this.

**File:** `pochoir/__main__.py:333`

```python
ctx.obj.put(increment, err, taxon="increment", **params)
```

The full-shape error array from the FDM solver is persisted to disk alongside the potential. For `weight3d`, that is another 62.7 MB (float64) written to disk, even though only the scalar maximum error (`maxerr`) is needed to assess convergence. No downstream command is known to read the increment array; it appears to be stored only for diagnostic purposes.

**Impact:** Approximately doubles the disk footprint of every FDM output and wastes one full-array transfer back from the device.

---

## Summary Table

| ID | Severity | File | Line(s) | Short description |
|----|----------|------|---------|-------------------|
| BUG-01 | HIGH | `fdm_cumba.py` | 24 | `cuda.grid(2)` for 3D kernel — 3D cumba broken |
| BUG-02 | HIGH | `fdm_torch.py` | 46 | `numpy.bool` removed in NumPy ≥ 1.24 — torch broken |
| BUG-03 | HIGH | `drift_torch.py` | 67 | `device='cpu'` hardcoded — torch drift is CPU-only |
| BUG-04 | MED | `fdm_cupy.py`, `fdm_cumba.py` | 75 / 91 | Early-exit returns cupy, late-exit returns numpy |
| BUG-05 | MED | `fdm_cumba.py` | 82 | `iarr_pad` rebound every step — VRAM churn |
| BUG-06 | MED | `fdm_cupy.py` | 45,47,51–52 | Dead allocations; dropped `set_core2` divergence |
| BUG-07 | MED | `drift_torch.py` | 33–35 | `torch.arange` float rounding → axis length mismatch |
| BUG-08 | MED | `drift_torch.py` | 43–60 | No bounding-box check; crash vs zero-velocity divergence |
| BUG-09 | LOW | `fdm_torch.py` | 47 | `tmp_core` float32 vs `bi_core` float64 mixed precision |
| BUG-10 | LOW | `drift_torch.py` | 50 | Debug print in ODE RHS — ~90K prints per drift |
| BUG-11 | LOW | `bc_interp.py` | 5–6 | scipy RGI hardcoded; `arrays.rgi()` ignored |
| BUG-12 | LOW | `lar.py` | 51 | `numpy.vectorize` per-voxel mobility — CPU bottleneck |
| BUG-13 | INFO | `__main__.py` | 333, 439 | Path buffer float64; full increment array persisted |
