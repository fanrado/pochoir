# `wogrid` Branch — GPU Running-Efficiency Analysis

*All line numbers refer to `origin/wogrid` as of 2026-04-17.*

*For the master-branch baseline, see `../03-gpu-efficiency.md`.*

---

## 1. Summary Assessment

The GPU execution path in `wogrid` is architecturally identical to master for
the standard (Laplace, `--multisteps no`) mode: `fdm_torch.py` dispatches to
`stencil_poisson` (which, when `source=None`, is identical to `stencil`)
followed by an in-place update and `edge_condition`. No fused kernel exists;
efficiency is the same as master's `torch` backend.

The `wogrid`-specific additions that affect performance are:

| Change | Impact | Direction |
|---|---|---|
| `@torch.compile` commented out | Large potential speedup foregone | Negative |
| `torch.cuda.synchronize()` at every epoch | CPU stall once per epoch | Mild negative |
| `prev = iarr_pad.clone()` every 1000 steps | Extra GPU→GPU copy | Mild negative |
| Poisson source construction (one-time) | ~5 extra tensor ops at init | Negligible |
| Dead tensor allocations (barr_pad, iarr_pad_source) | Extra memory bandwidth at init | Negligible |
| Convergence check every 1000 steps (not per epoch) | More frequent early-exit possibility | Mild positive |

---

## 2. `@torch.compile` Is Commented Out

**File:** `pochoir/fdm_torch.py:26`

```python
# @torch.compile
def _compiled_step(iarr_pad, tmp_core, bi_core, mutable_core,
                   core, periodic, spacing=1.0, source=None):
```

The decorator is present but disabled. `torch.compile` (TorchDynamo) would
fuse the sequence of elementwise and slice operations inside `_compiled_step`
into a small number of CUDA kernels, reducing launch overhead and improving
memory bandwidth utilisation.

**Quantitative opportunity:** The inner step calls:
- `stencil_poisson`: 2·N slice-add ops + 1 multiply + 1 subtract → ~8 CUDA
  kernel launches for 3-D.
- `iarr_pad[core] = bi_core + mutable_core * tmp_core` → 2 elementwise ops.
- `edge_condition`: 2·N index copies → ~6 CUDA operations.

Total per iteration: ~16 kernel launches. Each launch has ~5–10 μs overhead
on modern hardware. For a typical run of 1 000 000 steps, launch overhead
alone is **~10–16 seconds**, not counting the actual compute.

`@torch.compile` with `mode="reduce-overhead"` or `mode="max-autotune"` would
fuse these into 1–3 kernels per step, reducing launch overhead by ~5–10×.
The main obstacle is that in-place slice assignment (`iarr_pad[core] = ...`)
must be written carefully for the compiler to trace it. The existing `_compiled_step`
function structure (explicit arguments, no Python-side control flow per step)
is already well-suited for compilation.

**Why it was disabled:** The commented-out profiler blocks at lines 29–45 and
lines 143–153 suggest the author was investigating compilation but encountered
issues (likely with in-place operations or the `edge_condition` loop) and
reverted to uncompiled mode.

---

## 3. `torch.cuda.synchronize()` Calls

**File:** `pochoir/fdm_torch.py:67, 168, 176, 182, 192`

`torch.cuda.synchronize()` blocks the CPU until all pending CUDA work
completes. The calls exist at:
- Line 67: before the timing `start_time`. Correct use.
- Line 168: at the start of each epoch (for timing). Forces a CPU stall.
- Line 176: inside the early-exit precision-hit branch.
- Line 182: inside the `maxerr == 0` branch.
- Line 192: after the epoch loop (for timing).

The call at line 168 runs **once per epoch**. With typical `nepochs=20`, that
is 20 synchronisation stalls, which is acceptable. The issue is that
synchronisation forces the CPU to wait for GPU completion — if the GPU is
still processing a large batch of work when the CPU calls `synchronize()`,
the CPU stalls. With `epoch=1 000 000` steps, the GPU processes 1 M Jacobi
steps without any CPU involvement, then stalls once. This is fine.

However, the `info_msg` call at line 150 (`info_msg(f'====== epoch: ...')`)
is executed on the CPU every epoch. If `info_msg` triggers any GPU operation
(it doesn't in the current code, as it just calls `logging.info`), it could
force synchronisation. As written, this is not an issue.

---

## 4. Cloning `prev` Every 1000 Steps

**File:** `pochoir/fdm_torch.py:155`

```python
if istep % 1000 == 0:
    prev = iarr_pad.clone().detach().requires_grad_(False)
```

A `clone()` copies the full padded array from GPU memory to GPU memory.
For the `weight3d` domain (222×222×1502 ≈ 74.2 M cells at fp64):
- Copy size: 74.2 M × 8 = 594 MB
- GPU memory bandwidth (A100): ~2 TB/s
- Clone time: ~0.3 ms

This clone runs every 1000 steps. For 1 000 000 total steps, it runs 1000
times, costing **~0.3 s total** — negligible compared to the solve time.

The `prev` clone retains one extra full tensor on the GPU throughout the run
(see `04-memory.md`).

---

## 5. Per-Step Kernel Launch Count

The `_compiled_step` function (uncompiled) launches approximately:

| Operation | CUDA kernels |
|---|---|
| `stencil_poisson`: 6 neighbor-sum adds (3-D) | 6 |
| `stencil_poisson`: 1 multiply by norm | 1 |
| `stencil_poisson`: 1 subtract source | 1 (if Poisson) |
| `iarr_pad[core] = bi_core + mutable_core * tmp_core` | 2 |
| `edge_condition`: 3 dims × 2 index copies | 6 |
| **Total per step (Laplace)** | **15** |
| **Total per step (Poisson)** | **16** |

For comparison, the `fdm_cumba.py` fused kernel runs **1 CUDA kernel per step**
(see `../03-gpu-efficiency.md §4`). On a 72.6 M-cell domain, the cumba
backend would be expected to be 5–15× faster per step than the torch backend.

The test scripts use `--engine torch` exclusively; the `cumba` backend is
available but not tested for the pixel geometry.

---

## 6. Mixed-Precision Two-Step Performance Profile

When `--multisteps yes`:

1. **Step 1** (fp32): All tensors at 4 bytes instead of 8 bytes → 2× memory
   bandwidth reduction → ~1.5–1.8× faster iteration (bandwidth-bound).
2. **Step 2** (fp64): Full fp64 computation on the error correction domain.

The total wall time is `T_step1(fp32) + T_step2(fp64)`. Step 2 solves for δ
which is a small correction; in principle it converges faster than the
original Laplace solve (the correction δ is numerically small, so it should
relax quickly). Whether this is faster than a direct fp64 Laplace solve
depends on the number of iterations each step needs.

The `test/bench_fdm.py` benchmarks single-step solves but does not benchmark
the two-step solver. A benchmark comparing `--multisteps no` (fp64) vs
`--multisteps yes` (fp32+fp64) would be valuable.

---

## 7. Profiler Infrastructure (Dead Code)

There is extensive commented-out profiler code in `fdm_torch.py`:

- Lines 29–45: `torch.profiler.profile` inside `_compiled_step`.
- Lines 143–153: same profiler inside `solve()`.
- Lines 83–84 in `fdm_generic.py`: `time.time()` timestamps.

This code was used during development to identify bottlenecks. The evidence
suggests the author saw that `stencil_poisson` dominated the CUDA time
budget and considered using `@torch.compile` (the commented decorator at
`fdm_torch.py:26`).

The profiler infrastructure can be re-enabled by uncommenting those blocks
to produce a Chrome trace (`stencil_poisson_runtime_profile.json`,
`fdm_step_trace_epoch.json`). The `test/bench_fdm.py --plot-trace` sub-
command can then generate visualisations from these traces.

---

## 8. Geometry Generation Performance

The pixel boundary array (`barr`) is constructed in Python using nested loops
in `gen_pcb_pixel_with_grid.py`:

```python
# draw_pixel_plane: draw_3Dstrips_sq inner loop
for i in range(0, n_pix):
    for j in range(0, n_pix):
        ...
```

For `n_pix = 5` (from the config) this runs 25 iterations, each filling a
2-D sub-array. This is a one-time CPU cost run before the GPU solve and is
negligible in absolute terms (~milliseconds).

If `n_pix` is increased to handle a larger tile (e.g., 50×50 pixels), the
loop would run 2500 iterations and may become noticeable, but still
sub-second.

---

## 9. Comparison vs Master (`torch` Backend)

| Aspect | Master | `wogrid` |
|---|---|---|
| Compiler | No | No (commented out) |
| Kernel launches / step | ~14 | ~15–16 |
| Convergence check | Every epoch | Every 1000 steps |
| Synchronize / epoch | None | 1 × per epoch (timing) |
| Extra `clone()` | None | 1 × per 1000 steps (`prev`) |
| Dead tensor alloc | `barr_pad` | `barr_pad` + `iarr_pad_source` |
| fp32 option | No | Yes (`_dtype` parameter) |

The `wogrid` torch backend is marginally slower per step than master's torch
backend due to the extra dead allocations and the `clone` overhead, but the
difference is small. The **fp32 option** in Step 1 of the two-step solver is
the main efficiency gain, offering roughly 2× faster iteration at the cost of
a second fp64 correction pass.
