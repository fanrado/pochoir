# `wogrid` Branch — Potential Bug Catalog

*Read-only examination. No code was modified.*
*All line numbers refer to `origin/wogrid` as of 2026-04-17.*

Severity scale: **Critical** (crash or silent wrong result under normal use),
**High** (wrong result in a specific mode), **Medium** (wrong result in an
edge case), **Low** (code quality / latent risk).

---

## B-1 · Docstring/Code Sign Mismatch in `stencil_poisson`

**File:** `pochoir/fdm_generic.py:87–110` (docstring), line 136 (code)
**Severity:** Medium (reader confusion, latent correctness risk)

The docstring states:
```
φ_new = (1/(2N)) * Σ(neighbours) + (spacing² / (2N)) * f
solves  ∇²φ = -f
```

The code does:
```python
res -= (spacing ** 2) * source * norm   # line 136
```

which is `res = (1/2N)*Σ − (h²/2N)*source`, i.e., it solves `∇²φ = +source`
(the sign of the PDE is opposite to what the docstring describes). A comment
on line 135 says "corrected the sign here, since we are solving ∇²φ = -f =
source", but the code implements the convention `∇²φ = +source`.

The overall two-step algorithm gives the correct answer because the source
passed in from `fdm_torch.py` already has the appropriate sign (it is
constructed as `−(2N/h²)*(stencil(φ₀) − φ₀)`, which is `−∇²_FD φ₀`).
However, the inconsistency makes the sign convention of the API ambiguous.
Any future use of `stencil_poisson` must carefully track whether `source`
means `f` (docstring convention) or `−f` (code convention).

---

## B-2 · Hard-Coded `6` Assumes 3-D (`fdm_torch.py`)

**File:** `pochoir/fdm_torch.py:117` (source construction),
`pochoir/fdm_torch.py:221` (post-loop source reconstruction)
**Severity:** High (silent wrong result in 2-D Poisson mode)

The source term is computed as:
```python
source = -(6/spacing**2) * (s - non_padded_phi0) * mutable_core  # line 117
```

The value `6 = 2N` is correct only for 3-D (`N=3`). For a 2-D domain
(`N=2`), the correct factor is `4 = 2*2`. The same hard-coding appears in
the post-loop block at line 221.

The standard `stencil()` function correctly uses `norm = 1/(2*nd)` where
`nd = len(array.shape)`, so it is dimensionally general. The source
construction should read:

```python
nd = len(iarr.shape)
source = -(2*nd / spacing**2) * (s - non_padded_phi0) * mutable_core
```

Currently, running `--multisteps yes` on a 2-D problem will produce a
wrong source term and, therefore, a wrong correction δ.

---

## B-3 · `err = None` Crash When `epoch < 1000`

**File:** `pochoir/fdm_torch.py:152–185`
**Severity:** Critical (guaranteed crash on the benchmark's default settings)

`err` is initialised to `None` (line 60) and is only assigned inside the
convergence-check block:
```python
if (istep % 1000 == 0) and (istep != 0):   # fires first at istep=1000
    err = iarr_pad[core] − prev[core]
```

If `epoch < 1000`, the condition `istep % 1000 == 0 and istep != 0` is
**never True**, so `err` remains `None`. The function then reaches:
```python
return (iarr_pad[core].cpu(), err.cpu())   # line 195 or 207
```
and crashes with `AttributeError: 'NoneType' object has no attribute 'cpu'`.

**Triggered by:**
- `test/bench_fdm.py` default `--steps 500` (epoch=500 < 1000).
- Any test script specifying `--epoch N` with N < 1000.
- Single-epoch runs used for debugging.

**Fix direction:** Initialise `err` to a zero tensor of the appropriate shape
before the loop, or relax the convergence-check condition to fire at least
once regardless of `epoch`.

---

## B-4 · `barr_pad` Allocated on GPU but Never Read

**File:** `pochoir/fdm_torch.py:85`
**Severity:** Medium (wasted GPU memory; scales with domain size)

```python
barr_pad = torch.tensor(numpy.pad(barr, 1), requires_grad=False, dtype=_dtype).to(device)
```

`barr_pad` is created and sent to the GPU but is **never referenced again**
in the function. It is a full-size padded copy of the boundary mask occupying
the same GPU memory as `iarr_pad`. For the `weight3d` domain (220×220×1500
padded to 222×222×1502 ≈ 74.2 M cells), this wastes **595 MB** at fp64 or
**297 MB** at fp32.

The equivalent allocation existed in the master branch but was also unused there.

---

## B-5 · `iarr_pad_source` Allocated but Never Used

**File:** `pochoir/fdm_torch.py:92`
**Severity:** Low (wasted GPU memory; dead code)

```python
iarr_pad_source = iarr_pad.clone().detach().requires_grad_(False).to(device)
```

This is a full `iarr_pad` clone. Only commented-out code references it (line
125, commented). It should be removed or gated behind a `if phi0 is not None`
block at minimum.

---

## B-6 · `mutable_core` Applied Inconsistently to Source

**File:** `pochoir/fdm_torch.py:117` vs line 221
**Severity:** Medium (inconsistent treatment of boundary cells in two-step mode)

During initialisation (phi0 path, line 117):
```python
source = -(6/spacing**2) * (s - non_padded_phi0) * mutable_core  # multiplied
```

During post-loop source reconstruction (line 221):
```python
source = -(6/spacing**2) * (s - non_padded_phi0)  # NOT multiplied by mutable_core
```

The `mutable_core` mask zeros the source at fixed-boundary cells (where
`barr == True`). Its absence in the post-loop block means boundary cells get
a non-zero source, which is physically meaningless. Although boundary cells
are overwritten by `bi_core` in every iteration, an incorrect source at
boundary cells wastes a small amount of computation and makes the code
logically inconsistent.

---

## B-7 · `numpy.bool` Deprecated / Removed

**File:** `pochoir/fdm_torch.py:84`
**Severity:** High (runtime crash on NumPy ≥ 1.24)

```python
mutable_core = torch.tensor(numpy.invert(barr.astype(numpy.bool)), ...)
```

`numpy.bool` was deprecated in NumPy 1.20 and **removed in NumPy 1.24**.
On any modern environment this line raises:
```
AttributeError: module 'numpy' has no attribute 'bool'
```

The fix is `barr.astype(bool)` (Python builtin) or `barr.astype(numpy.bool_)`.
This same issue exists in the master branch (`fdm_torch.py`).

---

## B-8 · `ctx`, `potential`, `params` Silently Required When `phi0 is not None`

**File:** `pochoir/fdm_torch.py:109–137`
**Severity:** Medium (crash with unhelpful traceback)

The `solve()` signature declares `ctx=None, potential=None, params=None`
as optional, but when `phi0 is not None` the body immediately calls:
```python
path_to_padded_phi0 = potential.split('/')[0] + '/padded_phi0'  # AttributeError if potential=None
ctx.obj.put(path_to_padded_phi0, ...)                           # AttributeError if ctx=None
```

There is no guard or helpful error message. `test/bench_fdm.py` correctly
passes `phi0=None`, so it avoids this, but any caller attempting to use
Poisson mode without passing `ctx` and `potential` will get a confusing
`AttributeError: 'NoneType' object has no attribute 'split'`.

---

## B-9 · Unused Module-Level Imports in `fdm_torch.py`

**File:** `pochoir/fdm_torch.py:7–13`
**Severity:** Low (import overhead; potential display issues on headless servers)

```python
import torch.nn.functional as F        # line 7 — never used
import matplotlib.pyplot as plt        # line 8 — never used at module level
import numpy as np                     # line 9 — never used (numpy imported as 'numpy' below)
from torch.profiler import profile, record_function, ProfilerActivity  # line 12 — only in comments
```

`matplotlib.pyplot` is the most consequential: on a headless GPU server
without a display, this import can fail or produce a `UserWarning` unless the
backend is set to `Agg` before import. It also adds ~0.3–0.5 s to module
import time.

---

## B-10 · Typo in Function Name `longitudanal_diffusion`

**File:** `pochoir/lar.py:61`
**Severity:** Low (API naming consistency)

The function is named `longitudanal_diffusion` (extra 'a'). The correct
spelling is `longitudinal_diffusion`. The vectorised alias on line 89 is
`diff_longit`, which is fine, but any code calling the function by its full
name will need updating if it is ever corrected.

---

## B-11 · Unit System Time-Base Change: Potential Incompatibility

**File:** `pochoir/units.py` (entire file)
**Severity:** Medium (silent numerical change for any code that uses time units)

The master branch defines `second = 1.0` (millimeter-second system).
The `wogrid` branch replaces this with the Wire-Cell unit system where
`nanosecond = 1.0` (millimeter-nanosecond system), so `second = 1e9`.

| Quantity | Master | `wogrid` |
|---|---|---|
| `units.second` | 1.0 | 1 × 10⁹ |
| `units.ms` | 1 × 10⁻³ | 1 × 10⁶ |
| `units.us` | 1 × 10⁻⁶ | 1 × 10³ |
| `units.ns` | 1 × 10⁻⁹ | 1.0 |

All _internal_ unit arithmetic remains self-consistent as long as quantities
are always expressed as `value * units.X`. The risk is in:
1. Hard-coded numerical thresholds (e.g., precision `--precision 2e-8` is
   unit-free and unaffected).
2. Any absolute time comparison or printout that assumed `second = 1`.
3. Mixed codepaths that load data from a file produced with master and
   process it with wogrid (or vice versa) — the physical time values will
   differ by 10⁹.

The test scripts specify times symbolically (e.g., `'0*us,100*us,0.05*us'`)
which evaluate consistently under both unit systems, so the canonical test
pipelines are safe.

---

## B-12 · `err` Convergence Check Measures 1-Step Change, Not 1000-Step Change

**File:** `pochoir/fdm_torch.py:152–165`
**Severity:** Low (semantic mismatch, not necessarily wrong)

See `01-algorithm.md §5` for the detailed analysis. The check at
`istep % 1000 == 0` saves `prev`, applies one step, then measures `err`.
The measured `err` is therefore the **one-step change**, not the change over
1000 steps. For well-converged Jacobi iteration these are proportional (one-
step change → 0 iff 1000-step change → 0), so the convergence criterion is
still valid. However, the `prec` threshold is interpreted as a per-step
tolerance rather than a 1000-step tolerance, which may make it harder to
choose an appropriate `prec` value.

---

## Summary Table

| ID | File:Line | Severity | Description |
|---|---|---|---|
| B-1 | `fdm_generic.py:87,136` | Medium | Docstring/code sign mismatch in `stencil_poisson` |
| B-2 | `fdm_torch.py:117,221` | High | Hard-coded `6` fails in 2-D Poisson mode |
| B-3 | `fdm_torch.py:60,195` | **Critical** | `err=None` crash when `epoch < 1000` |
| B-4 | `fdm_torch.py:85` | Medium | `barr_pad` allocated on GPU but never read |
| B-5 | `fdm_torch.py:92` | Low | `iarr_pad_source` dead allocation |
| B-6 | `fdm_torch.py:117,221` | Medium | `mutable_core` inconsistently applied to source |
| B-7 | `fdm_torch.py:84` | High | `numpy.bool` removed in NumPy ≥ 1.24 |
| B-8 | `fdm_torch.py:109` | Medium | `ctx`/`potential` silently required in Poisson mode |
| B-9 | `fdm_torch.py:7–13` | Low | Unused imports incl. `matplotlib` at module level |
| B-10 | `lar.py:61` | Low | Typo `longitudanal_diffusion` |
| B-11 | `units.py` | Medium | Time-base change (ns=1 vs s=1) — latent incompatibility |
| B-12 | `fdm_torch.py:152` | Low | `err` measures 1-step change, misleadingly named |
