# GPU Running Efficiency Analysis

*Read-only examination. No code was modified. All claims are supported by file:line citations.*

---

## 1. Loop Structure: Host-Driven Iteration

All GPU backends (`fdm_cupy.py`, `fdm_torch.py`, `fdm_cumba.py`) share the same outer structure:

```python
for iepoch in range(nepochs):           # Python loop on host
    for istep in range(epoch):          # Python loop on host
        # one or more GPU kernel dispatches
        stencil(iarr_pad, tmp)
        iarr_pad[core] = bi + mu * tmp
        edge_condition(iarr_pad, ...)
        # once per epoch: compute maxerr and maybe return
```

**Every iteration of both loops executes on the host.** There is no on-device while-loop; each iteration dispatches one or more GPU kernels and returns. For `--epoch 5000000 --nepochs 4` (the test script `test/test-full-3d-50L.sh`), this executes the Python inner loop body 20 million times. Even at 1 μs of Python overhead per iteration, that is 20 seconds of pure Python loop overhead before any GPU work is counted.

A production GPU code would fuse the iteration into a single kernel launch (or a device-side loop) and return only when done. The pochoir architecture does not support this without significant restructuring.

---

## 2. Per-Epoch Host Synchronisation

**Every epoch**, all backends force a device→host synchronisation:

- `fdm_cupy.py:71`: `if prec and maxerr < prec:` — `maxerr` is a 0-D `cupy.ndarray`; comparing it to a Python float implicitly calls `.item()`, blocking until the GPU finishes and transferring a scalar back to CPU.
- `fdm_torch.py:71`: same — `maxerr` is a torch scalar; `maxerr < prec` performs an implicit `.item()`.
- `fdm_cumba.py:87`: same pattern.

This synchronisation happens even when `prec=0.0` (precision check disabled), because the condition is `if prec and maxerr < prec` — when `prec=0.0`, `if prec` is `False` so the sync is **avoided**. However, the `maxerr` computation at the preceding line still runs:

- `fdm_cupy.py:71`: `maxerr = cupy.max(cupy.abs(err))`
- This dispatches two GPU kernels (abs, then max) and produces a 0-D cupy array.

The sync only fires on the comparison, so with `prec=0.0`, the two extra kernels fire but the blocking `.item()` is skipped. The epoch print statement (`fdm_numpy.py:54`) is always executed on the host, which is another Python I/O stall.

**Practical impact:** For `--epoch 5000000`, the per-epoch overhead occurs once every 5 million steps — negligible if the kernel throughput fills that gap. For `--epoch 1000` (the default), the sync overhead recurs every 1000 steps and becomes non-trivial.

---

## 3. Kernel Launch Overhead: `fdm_generic.stencil` (cupy and torch)

For the `cupy` and `torch` backends, the stencil is computed by `fdm_generic.stencil` (`fdm_generic.py:35–67`), which uses element-wise array operations:

```python
for dim, n in enumerate(array.shape):
    pos = list(slices)
    pos[dim] = slice(2,n)
    res += array[tuple(pos)]    # one kernel launch per dimension

    neg = list(slices)
    neg[dim] = slice(0,n-2)
    res += array[tuple(neg)]    # one kernel launch per dimension

res *= norm                     # one more kernel launch
```

For N dimensions this issues `2N + 1` GPU kernel launches per stencil call. In 3D that is 7 kernel launches (6 adds + 1 multiply). With the subsequent update and `edge_condition`:

```
update:   iarr_pad[core] = bi_core + mutable_core*tmp_core   → 2 launches (multiply + add)
edge_cond: per dimension, 2 slice-assign ops → 2 × ndim launches
total per step (3D): 7 + 2 + 6 = 15 kernel launches
```

At a GPU kernel launch overhead of ~5–10 μs (CUDA launch latency on modern hardware), 15 launches per step × 20 M steps ≈ 150 M–300 M μs of launch overhead alone, or **150–300 seconds**, before any compute is counted. For small grids where each kernel completes in <10 μs, launch overhead dominates.

The `cumba` backend (`fdm_cumba.py:14–48`) replaces the stencil with a **single fused `@cuda.jit` kernel** per step (one kernel for 2D, one for 3D — when the 3D bug is fixed). This eliminates the 7 separate launches for the stencil, reducing launch overhead by ~50% per step.

---

## 4. Per-Step and Per-Epoch Allocation Churn

### `fdm_cumba.py:82` — per-step reallocation

```python
iarr_pad = bi_pad + mutable_pad*tmp_pad   # creates a new cupy array every step
```

This allocates a new device buffer of `iarr_pad.nbytes` bytes on every iteration. CuPy's memory pool reuses previously-freed blocks, so actual `cudaMalloc` calls are rare after warmup. However:

1. The Python variable rebind still causes reference counting overhead.
2. The old `iarr_pad` must be garbage-collected or explicitly freed; this can cause pool fragmentation.
3. In-place updates (`cupy.add(bi_pad, mutable_pad*tmp_pad, out=iarr_pad)`) would eliminate this entirely.

### `fdm_torch.py:61` — per-epoch full padded clone

```python
prev = iarr_pad.clone().detach().requires_grad_(False)
```

This clones the **full padded array** (not just the core). For `weight3d` (350×32×700 padded to 352×34×702 ≈ 8.4 M float64 cells ≈ 67.4 MB), one extra allocation of 67.4 MB occurs per epoch. With `--nepochs 4` this is minor, but with `--nepochs 200` (tutorial `elects3d`) this occurs 200 times.

The cupy backend (`fdm_cupy.py:61`) does this more efficiently: `prev = cupy.array(iarr_pad[core])` copies only the core (unpadded) portion — a smaller allocation.

---

## 5. Host↔Device Data Transfer

### FDM setup (one-time per `fdm` command invocation)

All backends upload the initial and boundary arrays at the start of `solve()`:
- `fdm_cupy.py:38–39`: `cupy.array(iarr)`, `cupy.array(barr)` — full arrays transferred H2D.
- `fdm_torch.py:45–50`: `torch.tensor(iarr*barr, ...)`, `torch.tensor(numpy.pad(iarr, 1), ...)` etc. — multiple H2D transfers. Note that `numpy.pad` is called on the host first, so the padded array is created in host RAM and then uploaded to GPU.
- `fdm_cumba.py:59–62`: `cupy.array(iarr)`, `cupy.array(barr)` called inside `cupy.pad(...)` — upload then pad on device.

**No per-iteration H2D/D2H copy occurs** in any backend during the iteration loop. This is correct.

### `arrays.gradient()` — GPU→CPU→GPU round trip

**File:** `pochoir/arrays.py:96–111`

```python
def gradient(array, *spacing):
    if isinstance(array, numpy.ndarray):
        return numpy.array(numpy.gradient(array, *spacing))
    # ... for torch:
    a = array.to('cpu').numpy()         # ← D2H transfer
    gvec = numpy.gradient(a, spacing)   # CPU computation
    g = numpy.array(gvec)
    return to_torch(g, device=array.device)  # ← H2D transfer
```

The comment in the source reads: *"At the cost of possible GPU→CPU→GPU transit, for now we do the dirty"*. This is called from `__main__.py:354` (`velo` command). For a 7.84 M-cell float64 potential array that is ≈62.7 MB transferred D2H, computed on CPU, and ≈188 MB (3-component gradient) transferred H2D. Total: ~250 MB of unnecessary transfers per `velo` invocation.

For the tutorial `weight3d` (46.2 M cells, float64): ≈370 MB D2H + ≈1.1 GB H2D per `velo` call.

This is the worst single-point data movement bottleneck in the post-FDM pipeline.

### No persistent GPU context across commands

Each CLI command (`pochoir fdm`, `pochoir velo`, `pochoir drift`) is a **separate OS process**. The CUDA context is created and destroyed each time. Arrays are reloaded from NPZ files on disk for every command. The GPU never holds data between commands.

For the pipeline in `test-full-3d-50L.sh`:
```
pochoir fdm → exits → GPU context destroyed, arrays freed
pochoir velo → new process → reload potential from disk → upload to GPU (if needed) → gradient (round-trip) → exit
pochoir drift → new process → reload velocity from disk → CPU integration
```

The potential solved in `fdm` cannot be kept on GPU for use in `velo` because the process boundary forces a D2H copy (to write NPZ) and then an H2D copy (to reload). This is an architectural choice in the CLI design, not a bug per se, but it prevents any multi-step GPU pipeline optimisation.

---

## 6. `edge_condition` — Python Slice Loop

**File:** `pochoir/fdm_generic.py:3–32`

```python
for dim, per in enumerate(periodic):
    # ... compute src1, src2, dst1, dst2 as slices
    arr[tuple(dst1)] = arr[tuple(src2)]   # one element-wise GPU op
    arr[tuple(dst2)] = arr[tuple(src1)]   # one element-wise GPU op
```

For 3D arrays, this loop runs 3 times per call, each iteration dispatching 2 slice-assign operations → 6 kernel launches per `edge_condition` call. `edge_condition` is called once per iteration step, so for 20 M total steps that is 120 M kernel launches just for edge handling.

These 6 launches could be fused into a single custom kernel that handles all borders simultaneously, eliminating most of the overhead.

---

## 7. `fdm_numba.py` (CPU JIT) — Stencil Semantics Issue

**File:** `pochoir/fdm_numba.py:7–34`

The `@numba.stencil` decorator generates a full-array stencil kernel that allocates a new result array. The `solve` function delegates to `fdm_numpy.solve` with the numba stencil substituted:

```python
def solve(iarr, barr, periodic, prec, epoch, nepochs,
          stencil = stencil):
    return solve_numpy(iarr, barr, ..., stencil=stencil)
```

The numba `stencil()` function (`fdm_numba.py:27–34`):
```python
def stencil(a):
    if a.ndim == 2:
        a = stencil_numba2d_jit(a)    # numba returns a new array
    else:
        a = stencil_numba3d_jit(a)
    slices = tuple([slice(1,s-1) for s in a.shape])
    return a[slices]                  # return core slice
```

The numba `@numba.stencil` kernels operate on the **padded** array and return a full-sized array (same shape, with border cells zeroed out by numba's boundary handling). The `slice(1,s-1)` extracts the core — this is consistent with what `fdm_generic.stencil` returns (core shape).

However, `fdm_numpy.solve` passes `iarr` (padded) to `stencil(iarr)` at line 60, expecting a core-shaped result. The numba stencil of a padded array uses the border cells (which contain edge-conditioned values), so it correctly includes one-cell border effects. This is the intended double-buffering.

No bug here, but the interface for the numba stencil differs from `fdm_generic.stencil` (the former takes a pre-padded array and returns a core-shaped result; the latter also takes padded but requires `res` to be pre-allocated of core shape). The `fdm_numpy.solve` loop uses `tmp = stencil(iarr)` which allocates a new core-shaped array every step when using the generic stencil (since `res=None`). With the numba stencil, the allocation is inside the JIT kernel.

---

## 8. Convergence Algorithm Efficiency

As noted in `01-algorithm.md`, all backends implement **Jacobi iteration** with no acceleration. The spectral radius of the Jacobi iteration for the Laplace equation on a grid of size N in each dimension is:

```
ρ = 1 - π²/(2N²)   (approximately, for large N)
```

For the production `weight3d` domain (350 × 66 × 2000), the smallest dimension is 66, giving:

```
ρ ≈ 1 - π²/(2 × 66²) ≈ 1 - 0.00113 ≈ 0.99887
```

To reduce the error by a factor of 10⁶ (from ~1 V initial error to ~1 μV precision), one needs approximately:

```
k ≈ log(10⁻⁶) / log(ρ) ≈ 6 × log(10) / 0.00113 ≈ 12,200,000 iterations
```

The test script `test-full-3d-50L.sh` runs up to 20 M iterations with precision 2×10⁻¹⁰ — consistent with this estimate. A **red-black Gauss-Seidel** iteration has the same spectral radius as Jacobi (no improvement). **SOR with optimal relaxation parameter** ω_opt = 2/(1 + sin(π/N_min)) would give:

```
ρ_SOR = ω_opt - 1 ≈ (1 - sin(π/66)) ≈ 0.9524
```

reducing the required iterations by a factor of ~log(ρ_SOR)/log(ρ) ≈ 50. A **multigrid V-cycle** would reduce this to O(N log N) total work, potentially 1000× faster.

GPU throughput does not compensate for 50–1000× more iterations than necessary.

---

## 9. Expected Backend Performance Ranking

For 2D (once BUG-01 is fixed for 3D cumba):

| Backend | Stencil kernels/step | Per-step alloc | Fused? | Expected rank |
|---------|----------------------|----------------|--------|---------------|
| `cumba` | 1 | yes (BUG-05) | yes | 1st (despite alloc bug) |
| `cupy` | 7 (3D) | no | no | 2nd |
| `torch` | 7 (3D) | no | no | 3rd (parity with cupy) |
| `numba` | 1 (CPU JIT) | yes (in JIT) | yes (CPU) | best CPU |
| `numpy` | 7 (3D) | yes | no | slowest |

Note: the toy benchmark in `toy/stencil2d.py` (referenced in `toy/README.org`) reported ~1561 Hz for torch on a 2000×2000 grid on a 3080 Ti. At 1561 iterations/second, a 20 M iteration run would take ~3.6 hours — demonstrating that GPU throughput is not the binding constraint; iteration count is.
