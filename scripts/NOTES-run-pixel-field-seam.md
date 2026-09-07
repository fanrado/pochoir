# Near/far seam: cost, measurement and sweep-count study

Beads: `pochoir-xuit` (Phase 1/Step 1, cost), `pochoir-u91z` (Phase 1/Step 2,
seam measurement), `pochoir-k83d` (Phase 1/Step 3, sweep-count study).

All three runs are the **drift** hybrid solve exactly as `run-pixel-field.sh`
now invokes it (post `pochoir-5b50`), differing only in `--max-sweeps` and the
store name. Nothing in the solver, the runner or the configs was changed.

```
pochoir field-solve --hybrid yes --field drift --domain no \
  --coarse-config example_gen_pcb_drift_pixel_task13_coarse.json \
  --fine-config   example_gen_pcb_drift_pixel_task13_fine.json \
  --coarse-shape 20,20,361 --near-shape 44,44,199 --fine-shape 44,44,793 \
  --interface '19.8*mm' --coarse-spacing 0.22 --fine-spacing 0.1 \
  --band-cells 3 --max-sweeps {4,1,0} --schwarz-tol 2e-8 \
  --precision 0.00000002
```

| store | `--max-sweeps` |
|---|---|
| `store_seam_drift_band3` | 4 (the runner's value) |
| `store_seam_drift_sweeps1` | 1 |
| `store_seam_drift_sweeps0` | 0 (one-shot-pin fallback) |

Only the field solve was run — no PART B (velo/starts/drift). Stores are under
`scripts/`, not `/tmp` (`pochoir-rm4c`: host `/tmp` is at 89 %, 194 MB free).

> **Timing caveat.** The three solves were run **concurrently on the same host**
> (one GPU, FDM engine `torch`, `cuda:0`). The wall clocks below are therefore
> directly comparable *to each other* but are inflated relative to a solo run.
> The per-stage breakdown is read from the driver's own timestamps, so the
> *shape* of the cost is unaffected.
>
> Also note `/usr/bin/time` does not exist on this host; timing is shell
> `date`-based plus the driver's log timestamps.

---

## Step 1 — cost of the band-3 / 4-sweep drift solve

**Total wall clock: 250 s** (about 4 minutes). This is minutes, not hours, as
expected: `hybrid_iterate` does **not** re-solve the full-depth fine grid —
`_stitch` upsamples the coarse solution onto the 44×44×793 output lattice and
overwrites the first 199 planes with the near solution.

Per-stage, from the driver's log timestamps
(`store_seam_drift_band3/pochoir_driftfield.log`):

| stage | wall clock |
|---|---|
| domains (`domain/coarse`, `domain/near`, `domain/drift3d`) | 0.1 s |
| geometry generation, coarse (`initial/coarse`, `boundary/coarse`) | 21.8 s |
| geometry generation, near | 21.2 s |
| geometry generation, full-depth fine | 21.1 s |
| **coarse solve** (20×20×361) | **36.1 s** |
| `refine` coarse → near | 0.04 s |
| `near_bc` (pin the near top plane) | 0.01 s |
| **near solve** (44×44×199), sweep 0 | **25.4 s** |
| **Schwarz sweep** (4 iterations: far + near per iteration) | **120.2 s** |
| `stitch-near` (upsample coarse + overwrite 199 near planes) | 0.15 s |

Two things dominate: the **Schwarz sweep at 120 s (48 % of the run)** and the
three geometry generations at 64 s combined (26 %) — the latter is pure
generator cost, unrelated to the seam. The stitch itself is free.

Within a sweep the far re-solve costs ~21 s and the near re-solve ~7 s (the
first sweep's near re-solve took 11 s, the rest 7.2–7.4 s), so widening the
band would mostly buy against the ~21 s far side.

### Convergence of the two base solves

Both hit the requested `--precision 2e-8` rather than an epoch limit:

| solve | iterations | achieved maxerr | stop reason |
|---|---|---|---|
| coarse 20×20×361 | **876 000** | **1.9834e-08** | `fdm reach max precision: 2e-08 > 1.98e-08` |
| near 44×44×199 | **207 000** | **1.9457e-08** | `fdm reach max precision: 2e-08 > 1.95e-08` |

`pad-plane no-flux` is applied on both: `z_pad=46, gap_nodes=144` coarse and
`z_pad=100, gap_nodes=735` near.

The Schwarz geometry the driver reports:

```
schwarz: interface z=19.8 coarse idx 90 (inner 87), near top idx 198 (seed 197)
```

which matches the documented band exactly — coarse nodes 90…87 = z 19.80 /
19.58 / 19.36 / 19.14 mm.

### THE PER-SWEEP NEAR DELTA — the load-bearing number

```
schwarz iter 0: near delta = 0.5043925981392476
schwarz iter 1: near delta = 0.4658988697312907
schwarz iter 2: near delta = 0.43034471450971523
schwarz iter 3: near delta = 0.3975040926957263
schwarz hit max_iters=4 (last delta 0.3975040926957263)
```

| sweep | near delta | ratio to previous |
|---|---|---|
| 0 | 0.50439 | — |
| 1 | 0.46590 | 0.924 |
| 2 | 0.43034 | 0.924 |
| 3 | 0.39750 | 0.924 |

**Conclusion (stated as the issue asks): 4 sweeps under-converges the seam.**

* The tolerance **never gates**. It stopped on `max_iters=4`, and the final
  delta 0.3975 is **~7.3 orders of magnitude above** `--schwarz-tol 2e-8`. This
  confirms the premise recorded in `run-pixel-field.md` and `run-pixel-field.sh`:
  `--max-sweeps` is the binding limit and the sweep count alone decides seam
  quality.
* The delta is **still of order 1 at sweep 4** (0.3975 V on a 3878 V problem).
* The contraction ratio is a flat **0.924 per sweep** — essentially the
  ~0.95/sweep rate `nearfar.schwarz_solve`'s docstring attributes to the
  *single-cell* coupling. **Band 3 is barely improving on the 1-cell rate here.**
  At 0.924/sweep, reaching 2e-8 from 0.50 would need ~220 sweeps, i.e. ~1.8
  hours of sweeping.

The sweep count was **not** raised to make this look converged.

### Store keys written, and size

22 npz keys (each with a sibling `.json`):

```
domain/coarse      domain/near         domain/drift3d
initial/coarse     initial/near        initial/drift3d
initial/coarse_insulator  initial/near_insulator  initial/drift3d_insulator
initial/near_refined      initial/near_bc
boundary/coarse    boundary/near       boundary/near_bc   boundary/drift3d
increment/coarse   increment/near
potential/coarse   potential/near
potential/coarse_schwarz  potential/near_schwarz
potential/drift3d                      <- the stitched output
```

Total on disk: **66 MB** (`store_seam_drift_sweeps1` 66 MB,
`store_seam_drift_sweeps0` 62 MB — it writes no `*_schwarz` pair).

---

## Step 2 — what the seam actually does (band 3, 4 sweeps)

```
./env/bin/python test/seam_profile.py --store scripts/store_seam_drift_band3 \
    --potential potential/drift3d --interface 19.8 --design-field 56.0
```

The diagnostic picked the pad-centre axis at transverse index **(0, 0)**. That
is correct and worth recording, because it looks wrong: the drift domain is one
*periodic* pixel tile and the pad footprint **straddles the wrap** — metal at
x = 27…43 and 0…17, gap at 18…26 — so the pad centre is index 0, not 22.

### Interface placement

**19.8 mm is on a node**, index **198** of 793, `(z_iface - z0)/dz = 198.000000000`
exactly. No mis-specification.

### phi VALUE jump

| quantity | value |
|---|---|
| `phi(seam)` | −553.622488 V |
| linear extrapolation from BELOW | −553.622496 V |
| linear extrapolation from ABOVE | −553.622490 V |
| departure from the below-side trend | +7.80e-06 V |
| departure from the above-side trend | +2.13e-06 V |
| the two one-sided trends disagree by | 5.67e-06 V |

The value is continuous to ~1e-8 relative. That is expected and is **not**
evidence of a good seam: the Schwarz sweep ends on the near solve, so `phi` at
the interface is pinned to the far solution **by construction** (exact C0). The
`phi` table only shows the kink indirectly, through the disagreement of the two
one-sided trends.

### E_z GRADIENT KINK — the number that decides it

| quantity | value |
|---|---|
| E_z just BELOW the plane | 55.4034309 V/mm |
| E_z just ABOVE the plane | 55.9659261 V/mm |
| **kink (above − below)** | **+0.562495 V/mm** |
| as % of the local &#124;E_z&#124; (55.685) | **1.010 %** |
| **as % of the 56.0 V/mm design field** | **1.004 %** |

Note *which side* is wrong, which is exactly what the residual spike cannot
tell you: **the above-side (far/coarse) E_z, 55.966 V/mm, is close to the
coarse grid's own measured far value 55.996; the below-side (near/fine) E_z,
55.403, is well below the fine grid's own 55.885.** The near solution is being
dragged down at the top of its domain — consistent with a near field still
carrying the sweep-0 pin it has not yet iterated away.

### Transverse CORRUGATION

| z [mm] | index | max − min of phi [V] |
|---|---|---|
| 10.5 | 105 | 21.0994608 |
| 15.0 | 150 | 0.0335460796 |
| **19.8 (seam)** | **198** | **3.600e-05** |
| 30.0 | 300 | 1.910e-11 |
| 40.0 | 400 | 1.137e-12 |

The pad-plane corrugation has decayed by **~6 orders of magnitude** between
10.5 mm and the seam, and is at the numerical floor by 30 mm. The direction
agrees with the ~4.2 mm decay length in `NOTES-weighting-farfield.md`; the
measured decay over 10.5 → 15 → 19.8 mm here is steeper than a single 4.2 mm
exponential would give, which is not chased further in this step.

**So the interface at 19.8 mm does NOT sit inside the still-corrugated zone**
for the drift field — 3.6e-05 V of transverse structure on a 553 V potential is
7e-08 relative. Transverse corrugation is **not** the cause of this seam kink.
(This is the drift field only; the weighting probe is a separate question and is
not measured here.)

### Which of the two: floor, or real discontinuity?

The two known floors from `pochoir-u91z`:

* the grids model different pads (`pixelSize` 3.52 vs 3.5, +1.1 % area;
  `chamfer_r` 0.66 vs 0.7; `padThicknessCells` 3 is a *cell* count, so the pad
  block is 0.66 mm coarse against 0.30 mm fine)
* the configs' own measured far E_z: **55.996 coarse vs 55.885 fine = 0.199 %**

**The measured kink is 1.004 % of the design field — about 5× the 0.199 %
geometry floor.** This is the discontinuity being chased, not the floor. No fix
attempted here.

---

## Step 3 — sweep-count study: 0 and 1 against 4

Same command, `--max-sweeps` 0 / 1 / 4. `--band-cells` was **not** changed;
band-width and interface-placement experiments are Phase 2 and are deliberately
kept out of this evidence.

`--max-sweeps 0` is the one-shot-pin fallback: `_schwarz` is skipped entirely
(`hybrid-iterate: max_sweeps=0 -- skipping the Schwarz sweep, stitching the
one-shot near/coarse fields`) and the stitch uses the sweep-0 near solution,
where `near_bc` pinned the near top plane once and never revisited it.

| sweeps | phi jump (departure from below-side trend) | E_z kink (V/mm) | E_z kink (% of 56.0) | wall clock |
|---|---|---|---|---|
| 0 | 7.97e-06 V | 0.772708 | **1.3798 %** | 130 s |
| 1 | 7.81e-06 V | 0.713747 | **1.2745 %** | 165 s |
| 4 | 7.80e-06 V | 0.562495 | **1.0045 %** | 250 s |
| 20 | 7.85e-06 V | 0.157948 | **0.2821 %** | 647 s (Phase 2/Step 1, solo) |

E_z either side, for completeness:

| sweeps | E_z below | E_z above |
|---|---|---|
| 0 | 55.2234815 | 55.9961895 |
| 1 | 55.2739597 | 55.9877066 |
| 4 | 55.4034309 | 55.9659261 |

### CONCLUSION

Measured against the three outcomes `pochoir-k83d` sets out, this is the
**middle** one — with a strong lean toward the third:

**The sweep is working, but is badly under-converged at band 3.**

* It is **not** the first outcome. The kink does not fall steeply and is
  nowhere near the 0.199 % geometry floor at 4 sweeps: it is 1.0045 %, still
  ~5× the floor. **Phase 2 is therefore necessary.**
* It is **not quite** the third outcome either — 4 sweeps *is* measurably
  better than 1 (1.0045 % vs 1.2745 %), so the band **is** coupling the two
  domains. But only just: going from 1 sweep to 4 (three extra sweeps, +85 s,
  +52 % wall clock) removes only **21 %** of the kink, and the whole sweep
  machinery from 0 to 4 removes only **27 %** of the one-shot kink. The
  per-sweep near delta contracts at a flat **0.924**, essentially the
  single-cell rate.
* Direction of travel: every sweep moves the **below** side up (55.223 → 55.274
  → 55.403) toward the fine grid's own 55.885 while the **above** side barely
  moves (55.996 → 55.988 → 55.966). The iteration is converging to the right
  answer; it is simply far too slow to get there in 4 sweeps.

The prime suspect named in the issue is consistent with this: the band's
innermost plane at **19.14 mm is not a fine node**. At the 2.2 coarse:fine ratio
the grids share nodes only every 1.1 mm, so that far Dirichlet pin is
*interpolated* onto the near grid rather than exact, and only `band_cells` that
are multiples of 5 give an exact pin. A band whose innermost pin is smeared
would plausibly couple no better than a 1-cell band — which is what the flat
0.924 contraction looks like.

That remains a **hypothesis**, not a result: testing it means varying
`--band-cells` (3 vs 5 vs 10) and the interface placement, which is Phase 2.
Reporting and stopping here as instructed.


---

# Phase 2/Step 1 — drift seam at band 3 with 20 sweeps

Beads: `pochoir-7nyz`.

Identical to Phase 1/Step 1 in every respect — same geometry, same
`--band-cells 3` — except `--max-sweeps 20` and the store name
`store_seam_drift_sweeps20`. **`--band-cells` was not varied**, and the wider
bands (10, 20 coarse cells) were deliberately **not** run: pinning the coarse
far solve to fine data over a large fraction of its depth converges the seam by
turning the hybrid into the single-spacing solve, which defeats the method.

This run was **solo on the GPU**, unlike Phase 1's three concurrent runs, so its
wall clock is the clean one.

## Wall clock

**Total 647 s (10.8 min)**, close to the ~12 min estimate.

| stage | wall clock |
|---|---|
| domains | 0.1 s |
| geometry generation ×3 (coarse, near, full-depth fine) | 46.8 s |
| coarse solve | 31.0 s |
| refine + `near_bc` | 0.4 s |
| near solve (sweep 0) | 10.4 s |
| **Schwarz sweep, 20 iterations** | **556.7 s** |
| stitch | 0.19 s |

The sweep is now **86 %** of the run. Per sweep: **27.81 s** (measured across
the 19 gaps between logged deltas), matching the ~28 s estimate (far ~21 s +
near ~7 s). Store size 68 MB.

Solo, the base solves are noticeably faster than Phase 1's contended numbers
(coarse 31.0 s vs 36.1 s, near 10.4 s vs 25.4 s, geometry 46.8 s vs 64 s) —
which confirms Phase 1's caveat that those wall clocks were inflated.

## Stop reason

```
schwarz hit max_iters=20 (last delta 0.11162065008306854)
near-far-solve: 20 sweeps, final near delta=0.11162065008306854
```

**It stopped on `max_iters=20`, as expected — `--schwarz-tol 2e-8` still never
gates.** The final delta is 0.1116, ~6.7 orders of magnitude above the tol.

## THE FULL 20-ENTRY PER-SWEEP NEAR DELTA SERIES

| sweep | near delta | ratio to previous |
|---|---|---|
| 0 | 0.5043925981392476 | — |
| 1 | 0.4658988697312907 | 0.9237 |
| 2 | 0.4303447145097152 | 0.9237 |
| 3 | 0.3975040926957263 | 0.9237 |
| 4 | 0.3671696616370355 | 0.9237 |
| 5 | 0.3391501191784982 | 0.9237 |
| 6 | 0.3132688104536783 | 0.9237 |
| 7 | 0.2893625625489449 | 0.9237 |
| 8 | 0.2672806542069566 | 0.9237 |
| 9 | 0.2468838659827952 | 0.9237 |
| 10 | 0.2280436025694144 | 0.9237 |
| 11 | 0.2106410820555311 | 0.9237 |
| 12 | 0.1945665870460971 | 0.9237 |
| 13 | 0.1797187729263214 | 0.9237 |
| 14 | 0.166004028915836 | 0.9237 |
| 15 | 0.1533358878842819 | 0.9237 |
| 16 | 0.1416344812038233 | 0.9237 |
| 17 | 0.1308260352016077 | 0.9237 |
| 18 | 0.120842406037923 | 0.9237 |
| 19 | 0.1116206500830685 | 0.9237 |

**The ratio is 0.9237 to four decimal places for all nineteen gaps.** It does
not drift, does not improve, and shows no transient: the first gap and the
nineteenth are the same number.

**This answers the question the step was set to answer: the 0.924 coupling rate
is a property of the band, not a transient of the near solution settling.** The
iteration is a clean geometric contraction with a fixed rate.

The kink contracts at exactly the same rate: 0.157948 / 0.562495 = **0.2808**
over 16 sweeps, and 0.9237^16 = **0.2809**.

## The seam at 20 sweeps

Interface still on node 198 of 793. `phi` continuous to ~1e-8 (C0 by
construction, as before).

| quantity | value |
|---|---|
| E_z just BELOW the plane | **55.7497232 V/mm** |
| E_z just ABOVE the plane | **55.9076713 V/mm** |
| **kink (above − below)** | **+0.157948 V/mm** |
| as % of the local &#124;E_z&#124; (55.829) | 0.2829 % |
| **as % of the 56.0 V/mm design field** | **0.2821 %** |

Transverse corrugation on the seam plane is 3.62e-05 V — unchanged from Phase 1
and still six orders below the pad plane, so corrugation remains a non-factor.

## The falsifiable prediction, checked

The prediction was: below-side gap to the fine grid's 55.885 is 0.482 at sweep
4, so 16 more sweeps at 0.924 gives ~0.138 → E_z below ~55.75, kink ~0.22 V/mm
= ~0.39 % of design.

* **The below-side half of the prediction was exact.**
  55.885 − 0.482 × 0.9237^16 = **55.7496**, measured **55.7497**. The
  contraction did **not** improve, exactly as predicted.
* **The kink half was pessimistic**, because the prediction held the far side
  fixed at 55.966. It did not stay fixed: the above side also converged
  downward, 55.966 → **55.9077**, toward the fine grid's 55.885. Both sides move
  toward each other, so the kink came in at **0.2821 %** rather than 0.386 %.

So the measurement **partly contradicts** the prediction — not in the
contraction rate, which was dead-on, but in the assumption that only the near
side moves.

## WHICH OF THE TWO OUTCOMES

Neither cleanly, and the distinction matters:

* The issue's second branch — *"if it lands near 0.39 %, then band 3 cannot
  reach the floor at any practical sweep count"* — **is not what happened.** It
  landed at 0.2821 %, i.e. **1.42× the 0.199 % geometry floor** (0.111 V/mm),
  down from 5× at 4 sweeps.
* The issue's first branch — *"much better than that, the contraction
  improved"* — is **half right**: the kink is better than predicted, but **the
  contraction did not improve at all** (a flat 0.9237 over 19 gaps). The
  improvement came from the far side converging too, not from a better rate.

**Conclusion: more sweeps at band 3 IS a viable fix, and band 3 reaches the
geometry floor at a practical sweep count.** Extrapolating the same 0.9237
contraction from the measured kink 0.157948 down to the 0.111 V/mm floor needs
**4.4 more sweeps — about 25 sweeps in total, ~11.3 min of sweeping.**

That is a real cost (25 sweeps is ~700 s of sweeping against the 4 sweeps the
runner currently ships, ~111 s) but it is not the "impractical at any sweep
count" outcome, and it does **not** require touching `--band-cells`, the
interface placement, or the non-fine-node inner pin at 19.14 mm.

**The leading hypothesis from Phase 1 — that the interpolated, non-fine-node
inner pin at 19.14 mm is what caps the coupling — is not needed to explain
these numbers, and is not supported by them either way.** What the flat 0.9237
does establish is that whatever sets the rate is *fixed* for this band: it is
not a transient the iteration works through. Whether the rate would improve
with an exactly-pinned band (`band_cells` a multiple of 5) is untested and
remains a Phase 2 question — but it is now a question about *speed*, not about
*reachability*.

No fix attempted, no default in `pochoir/` changed.
