# Cross-Backend Comparison

*Read-only examination. No code was modified. All claims are supported by file:line citations.*

---

## FDM Backend Comparison Table

| Attribute | `fdm_numpy` | `fdm_numba` | `fdm_torch` | `fdm_cupy` | `fdm_cumba` |
|-----------|-------------|-------------|-------------|------------|-------------|
| **File** | `fdm_numpy.py` | `fdm_numba.py` | `fdm_torch.py` | `fdm_cupy.py` | `fdm_cumba.py` |
| **Target** | CPU numpy | CPU JIT | CPU or CUDA | CUDA GPU | CUDA GPU |
| **Stencil** | `fdm_generic.stencil` (slice ops) | `@numba.stencil` JIT | `fdm_generic.stencil` (slice ops) | `fdm_generic.stencil` (slice ops) | fused `@cuda.jit` kernel |
| **Kernel launches / stencil (3D)** | 7 (element-wise) | 1 (JIT) | 7 (element-wise) | 7 (element-wise) | 1 (fused CUDA) |
| **Algorithm** | Jacobi | Jacobi | Jacobi | Jacobi | Jacobi |
| **Working array scope** | core + padded | core + padded (via numpy backend) | core + padded | core + padded | all padded |
| **`iarr_pad` dtype** | inherits `iarr` | inherits `iarr` | inherits `iarr` | inherits `iarr` | inherits `iarr` |
| **`tmp` array dtype** | inherits (from `zeros_like`) | new alloc in JIT | **float32** (bug: L47) | **float64** (cupy default) | inherits `iarr_pad` |
| **Dirichlet re-assertion** | `set_core2` each step (L62) | via numpy backend | **omitted** | **omitted** | **omitted** |
| **`barr_pad` allocated but unused** | no (used for `ifixed`) | n/a | yes (L49) | yes (L47) | yes (L60) |
| **Dead `err` allocation** | no (`err` is used) | n/a | no | yes (L45) | effectively yes (L66, overwritten at L86) |
| **Dead `ifixed`/`fixed`** | no (used) | n/a | no | yes (L51–52) | no |
| **Per-step `iarr_pad` realloc** | no | no | no | no | **yes** (L82, BUG-05) |
| **`prev` scope** | core only (L58) | core only | **full padded** (L61) | core only (L61) | **full padded** (L78) |
| **Host sync per epoch** | yes (maxerr comparison) | yes | yes | yes | yes |
| **Return type — early exit** | numpy | numpy | torch CPU tensor | **cupy** (BUG-04) | **cupy** (BUG-04) |
| **Return type — late exit** | numpy | numpy | torch CPU tensor → `.cpu()` | numpy (`.get()`) | numpy (`.get()`) |
| **Return types consistent** | yes | yes | yes | **no** | **no** |
| **Modern NumPy compat** | yes | yes | **broken (`numpy.bool`, BUG-02)** | yes | yes |
| **3D support** | yes | yes | yes | yes | **broken** (BUG-01, `cuda.grid(2)`) |
| **Expected performance (2D)** | slowest | better (JIT) | moderate | moderate | fastest (fused stencil) |
| **Expected performance (3D)** | slowest | better (JIT) | moderate | moderate | **broken** |
| **GPU acceleration** | no | no | yes (device auto-select, L44) | yes | yes |

---

## Drift Backend Comparison Table

| Attribute | `drift_numpy` | `drift_torch` |
|-----------|---------------|---------------|
| **File** | `drift_numpy.py` | `drift_torch.py` |
| **ODE solver** | `scipy.solve_ivp` Radau (implicit, L-stable) | `torchdiffeq.odeint` Dormand-Prince (explicit RK45) |
| **Velocity interpolation** | `scipy.interpolate.RegularGridInterpolator` | `torch_interpolations.RegularGridInterpolator` |
| **Device** | CPU | **CPU only** (`device='cpu'` hardcoded, BUG-03) |
| **Bounding-box check** | yes (`inside()`, L39–43) | **no** (BUG-08) |
| **Out-of-domain behaviour** | returns zero velocity (`extrapolate()`, L58–59) | **raises exception** |
| **Grid axis generation** | `numpy.arange(start, stop, spacing)` (float rounding risk) | `torch.arange(start, stop, spacing)` (float rounding risk, BUG-07) |
| **Debug print in RHS** | guarded by `verbose` flag (L75) | **always prints** (BUG-10, L50) |
| **RHS argument order** | `(time, pos)` — scipy convention | `(tick, tpoint)` — torchdiffeq convention |
| **rtol / atol** | 1e-4 / 1e-4 (Radau, L93) | 0.01 / 0.01 (odeint, L73) — lower accuracy |

---

## Structural Observations

### Near-Duplication with Silent Divergences

All five FDM backends implement the same algorithm but were written by copying and adapting `fdm_numpy.py`. Several differences were introduced unintentionally or left incomplete:

1. **Dirichlet re-assertion** (`set_core2`): present in `fdm_numpy.py` (L62), absent in all GPU backends. Whether this matters depends on whether any boundary cell ever falls on the padded border — if so, the GPU backends will produce different results from the numpy backend at those cells.

2. **`prev` scope**: `fdm_numpy.py` and `fdm_cupy.py` snapshot only the core; `fdm_torch.py` and `fdm_cumba.py` snapshot the full padded array. The epoch-error computation uses only `iarr[core] - prev[core]` in all backends, so the result is the same — but `torch` and `cumba` allocate more memory for `prev` than needed.

3. **Dead `barr_pad`**: allocated in `fdm_torch.py` (L49), `fdm_cupy.py` (L47), and `fdm_cumba.py` (L60); in none of these is it used after construction. In `fdm_numpy.py` it is never created (padding happens in-place on the local copy).

4. **`tmp` array scope**: `fdm_cupy.py` uses a core-sized `tmp_core` (shape `N`), then calls `stencil(iarr_pad, tmp_core)` — the generic stencil writes core-shaped output from a padded input. `fdm_cumba.py` uses a padded-sized `tmp_pad` and calls the CUDA kernel with both `iarr_pad` and `tmp_pad` at padded shape (the kernel reads interior, writes interior, leaving borders at their initial zero). This is a deliberate difference but not documented.

5. **Return types**: `fdm_torch.py` is the only GPU backend that consistently returns host tensors (`.cpu()` on both exit paths). The cupy and cumba backends are inconsistent.

### Interface Contract

All five backends share the same function signature:
```python
def solve(iarr, barr, periodic, prec, epoch, nepochs) -> (arr, err)
```
where:
- `iarr`: numpy array, initial values
- `barr`: numpy bool array, True at boundary cells
- `periodic`: list of bool, per-dimension edge condition
- `prec`: float, precision threshold (0 = disabled)
- `epoch`: int, steps per precision check
- `nepochs`: int, max epochs

All backends accept numpy inputs and should return numpy arrays. The inconsistent return types in `fdm_cupy` and `fdm_cumba` violate this contract on the early-exit path.
