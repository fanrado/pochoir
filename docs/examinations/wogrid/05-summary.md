# `wogrid` Branch — Summary and Prioritised Recommendations

*Read-only examination. No code was modified.*

---

## What Was Examined

The `wogrid` branch (`github.com/fanrado/pochoir`, branch `wogrid`, commit
`d31ca1b`) was examined against the `master` branch. The examination focused
on the GPU field-response calculation: the FDM solver, the new Poisson two-
step correction, geometry generators, and supporting infrastructure. No code
was modified.

Detailed findings are in:
- `00-overview.md` — what changed vs master
- `01-algorithm.md` — algorithm derivation and explanation
- `02-bugs.md` — 12 potential bugs with severity ratings
- `03-gpu-efficiency.md` — GPU running efficiency
- `04-memory.md` — memory footprint tables

---

## Key Algorithmic Contributions

1. **Two-step mixed-precision solver**: A defect-correction scheme that solves
   the Laplace equation in fp32 (fast), computes the floating-point residual,
   then corrects it with a fp64 Poisson solve. This is a sound numerical
   approach that recovers fp64 accuracy at lower total cost.

2. **Poisson stencil** (`stencil_poisson`): A general extension of the Jacobi
   stencil for equations with source terms. The implementation is correct (the
   docstring has a sign inconsistency with the code, but the calling convention
   in `fdm_torch.py` compensates for it).

3. **Pixel geometry**: New PCB pixel generators with rounded-corner square holes,
   supporting 30°, 90°, and default orientations. The geometry construction
   is CPU-side Python and is a one-time setup cost.

---

## Priority Bug List

### Fix Immediately (blocks correct use)

| ID | Issue | Impact | Fix |
|---|---|---|---|
| B-3 | `err=None` crash when `epoch < 1000` | Crash on benchmark defaults (`--steps 500`) and any small test | Initialise `err` to zeros before the loop |
| B-7 | `numpy.bool` removed in NumPy ≥1.24 | Import crash on modern NumPy | Change to `bool` or `numpy.bool_` |

### Fix Before Production (silent wrong results)

| ID | Issue | Impact | Fix |
|---|---|---|---|
| B-2 | Hard-coded `6` assumes 3-D | Wrong source term in any 2-D Poisson solve | Replace `6` with `2*len(iarr.shape)` |
| B-6 | `mutable_core` inconsistently applied to source | Incorrect source at boundary cells in post-loop block | Apply `mutable_core` consistently |
| B-11 | Time unit base change (ns vs s) | Latent incompatibility with master-branch data files | Document clearly; add unit test |

### Improve for Robustness

| ID | Issue | Impact | Fix |
|---|---|---|---|
| B-1 | Docstring/code sign mismatch in `stencil_poisson` | Future misuse of the API | Fix docstring to match code |
| B-4 | `barr_pad` dead GPU allocation (592 MB wasted) | GPU memory waste | Delete the line |
| B-5 | `iarr_pad_source` dead GPU allocation (592 MB wasted) | GPU memory waste | Delete the line |
| B-8 | `ctx`/`potential` silently required in Poisson mode | Confusing crash | Add guard with helpful error message |

### Low Priority / Cosmetic

| ID | Issue | Fix |
|---|---|---|
| B-9 | Unused imports incl. `matplotlib` at module level | Remove `F`, `plt`, `np`, dead profiler imports |
| B-10 | Typo `longitudanal_diffusion` | Rename to `longitudinal_diffusion` |
| B-12 | `err` measures 1-step change (semantic mismatch) | Add comment clarifying the convention |

---

## GPU Efficiency Recommendations

1. **Enable `@torch.compile`** on `_compiled_step` (re-add the decorator at
   `fdm_torch.py:26`). This is the largest potential speedup: ~15 kernel
   launches/step → 1–3, reducing launch overhead by ~5–10× on large grids.
   Start with `mode="reduce-overhead"` and test for correctness.

2. **Use `fdm_cumba` for Laplace solves** (no source term): The fused CUDA JIT
   kernel in `fdm_cumba.py` is already 5–15× faster per step than the torch
   backend. The two-step solver could run Step 1 (Laplace) with `cumba` and
   Step 2 (Poisson, requires source term) with `torch`. This requires
   exposing the engine choice separately for each step.

3. **Remove the `torch.cuda.synchronize()` timing call** from inside the epoch
   loop (line 168). Timing is useful but can be replaced with `time.time()`
   before/after the full epoch loop, measured without a GPU stall.

---

## Memory Reduction Quick Wins

| Action | Savings (weight3d, fp64) |
|---|---|
| Delete `barr_pad` (B-4) | 592 MB GPU |
| Delete `iarr_pad_source` (B-5) | 592 MB GPU |
| Delete `phi0_`, `s` after source init in Poisson mode | 1.17 GB GPU |
| Use `bool` dtype for `mutable_core` | 72 MB GPU |
| Gate checkpoint `ctx.obj.put` behind `--debug` flag | 2.6 GB CPU RAM |
| **Total quick-win savings** | **~2.4 GB GPU + 2.6 GB CPU** |

These are all deletions or dtype changes — no algorithmic changes required.

---

## Next Steps Suggested

1. Fix B-3 and B-7 (crash bugs) so the code runs on modern environments.
2. Delete dead tensor allocations B-4 and B-5 (trivial, saves >1 GB GPU).
3. Add a correctness test for the two-step solver on a small analytic problem
   (e.g., linear potential in a box) to catch sign errors before running on
   real geometry.
4. Re-enable `@torch.compile` experimentally and measure speedup on the pixel
   geometry benchmark (`test/bench_fdm.py`).
5. Benchmark `--multisteps yes` vs `--multisteps no` on the same geometry to
   quantify whether the two-step approach is faster end-to-end.
