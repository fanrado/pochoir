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


---

# Phase 3/Step 1 — band 5 at 20 sweeps: does an exact fine-node pin improve the rate?

Beads: `pochoir-douz`. A **speed** question only — Phase 2 already settled
reachability.

Identical to Phase 2/Step 1 except `--band-cells 5` and the store name
`store_seam_drift_band5_sweeps20`. Solo on the GPU, so directly comparable to
Phase 2's solo numbers. **Bands 10 and 20 were not run**, for the reason already
recorded: pinning the coarse far solve to fine data over a large fraction of its
depth converges the seam by turning the hybrid into the single-spacing solve.

## The band moved — confirmed before trusting anything else

```
schwarz: interface z=19.8 coarse idx 90 (inner 85), near top idx 198 (seed 197)
```

**`inner 85`, not 87.** Coarse node 85 = z **18.70 mm** = fine node 187 exactly,
so this band's far Dirichlet pin lands on a **real fine node** and is exact,
against band 3's inner plane at 19.14 mm which is interpolated. The width
difference is only 0.44 mm (1.10 mm vs 0.66 mm).

## THE CONTRACTION RATIO — the whole point of the step

| sweep | near delta | ratio to previous |
|---|---|---|
| 0 | 0.8345370995033363 | — |
| 1 | 0.7291638228919055 | 0.8737 |
| 2 | 0.6370979895199298 | 0.8737 |
| 3 | 0.5566566820564276 | 0.8737 |
| 4 | 0.4863720582111455 | 0.8737 |
| 5 | 0.4249617162129198 | 0.8737 |
| 6 | 0.3713051710010404 | 0.8737 |
| 7 | 0.3244234121623322 | 0.8737 |
| 8 | 0.2834610411573522 | 0.8737 |
| 9 | 0.2476706638342421 | 0.8737 |
| 10 | 0.2163992535745365 | 0.8737 |
| 11 | 0.1890762362520491 | 0.8737 |
| 12 | 0.1652030796062718 | 0.8737 |
| 13 | 0.1443441970955064 | 0.8737 |
| 14 | 0.1261190002317107 | 0.8737 |
| 15 | 0.110194954418148 | 0.8737 |
| 16 | 0.09628151156380227 | 0.8737 |
| 17 | 0.08412480878041606 | 0.8737 |
| 18 | 0.073503036433749 | 0.8737 |
| 19 | 0.06422239103153515 | 0.8737 |

**0.8737, flat to four decimal places across all nineteen gaps** — the same
"fixed property of the band, no transient" character as band 3, at a different
value.

| band | inner plane | pin | contraction ratio |
|---|---|---|---|
| 3 | 19.14 mm | interpolated | **0.9237** |
| 5 | 18.70 mm | **exact fine node** | **0.8737** |

**The rate improves materially — 0.9237 → 0.8737, below the ~0.88 threshold
this step set for "the exact pin matters".**

Note the sweep-0 delta is *larger* for band 5 (0.8345 vs 0.5044): the two bands
start from different far pins, so only the rate is comparable, not the
magnitudes.

## The seam at 20 sweeps

| quantity | band 3 | band 5 |
|---|---|---|
| E_z below | 55.7497232 | **55.8404534** |
| E_z above | 55.9076713 | **55.8924068** |
| kink (V/mm) | 0.157948 | **0.0519533** |
| **kink as % of 56.0** | **0.2821 %** | **0.0928 %** |

Both sides are closer to their own grids' measured far values (fine 55.885,
coarse 55.996) than band 3 managed. Transverse corrugation on the seam plane is
3.638e-05 V — unchanged, still a non-factor.

**At 20 sweeps band 5 is already past the 0.199 % geometry floor**, sitting at
0.47× it.

> A kink *below* the floor is not "more accurate" in any deep sense: the floor is
> the systematic disagreement between what the two grids model (different pad
> area, chamfer and pad-block thickness). Once the seam discontinuity is smaller
> than that, the seam has stopped being the limiting error. It is a stopping
> criterion, not a target to beat.

## Cost

| | band 3 | band 5 |
|---|---|---|
| total wall clock | 647 s | **666 s** |
| per sweep | 27.81 s | **28.76 s** |
| geometry generation ×3 | 46.8 s | 45.5 s |
| coarse solve | 31.0 s | 31.7 s |
| near solve (sweep 0) | 10.4 s | 10.5 s |
| store size | 68 MB | 68 MB |

The wider band costs **3.4 % more per sweep** (28.76 vs 27.81 s), as expected —
the far re-solve carries a slightly deeper pin. Everything else is unchanged.

## TIME TO FLOOR — the comparison that decides it

Extrapolating each band's own measured rate down to the 0.111 V/mm floor:

| band | rate | sweeps to floor | sweeping wall clock |
|---|---|---|---|
| 3 | 0.9237 | 24.4 → **25** | 695 s (**11.6 min**) |
| 5 | 0.8737 | 14.4 → **15** | 431 s (**7.2 min**) |

**Band 5 wins on rate and on time-to-floor, despite costing more per sweep** —
which is exactly the comparison this step asked for. 15 sweeps at 28.76 s beats
25 sweeps at 27.81 s by **264 s, a 38 % saving**, and the saving compounds on
the weighting field where each sweep is far more expensive.

## CONCLUSION — which of the two outcomes

**The first: the rate improves materially, the exact pin matters, and band 5 is
the better default.**

The Phase 1 hypothesis is **supported, not dead**: the interpolated inner pin at
19.14 mm *was* capping the coupling, and landing the pin on a real fine node
recovers a meaningfully better rate. Phase 2 could only say the rate was fixed;
this step shows *what* fixes it.

**Recommendation for Phase 3/Step 3: `--band-cells 5` with `--max-sweeps 15`**
for the drift field, subject to the weighting field's cost measured in Phase
3/Step 2 (`pochoir-h92v`), which may force a different sweep count there.

Two caveats on that recommendation, both honest limits of this measurement:

* the 15 is an **extrapolation** from a clean geometric fit, not a measured
  15-sweep run. The fit is exact to four decimals over nineteen gaps, so it is
  a strong extrapolation — but it has not been confirmed at the endpoint.
* band 5 is the *first* exactly-pinned band. Whether 10 would be better still is
  untested **and deliberately not tested**, for the single-spacing-solve reason
  above.

No fix applied, `run-pixel-field.sh` not touched, no default in `pochoir/`
changed — setting the shipped values is Phase 3/Step 3.


---

# Phase 3/Step 2 — the weighting field: seam and per-sweep cost

Beads: `pochoir-h92v`. The cost question is decided here, not on the drift field.

**Band used: 5.** Phase 3/Step 1 showed the exact fine-node pin improves the
contraction rate materially (0.9237 → 0.8737), so band 5 is what this step
carries forward, as `pochoir-h92v` instructs.

Store `store_seam_weight_sweeps20`, solo on the GPU, weighting configs
(`example_gen_pixel_with_grid_*`), transverse shapes 5× wider than drift.
Band confirmed again: `schwarz: interface z=19.8 coarse idx 90 (inner 85)`.
Disk checked before starting: 1.1 TB free on `/nfs/data/1`.

## COST — and the result is the opposite of what was feared

The step's premise was that a naive 25× node scaling puts the weighting sweep at
hours. **It does not. The weighting field is CHEAPER per sweep than the drift
field.**

| | drift (band 5) | weighting (band 5) | ratio |
|---|---|---|---|
| coarse grid | 20×20×361 = 144 k | 100×100×361 = 3.6 M | 25× |
| near grid | 44×44×199 = 385 k | 220×220×199 = 9.6 M | 25× |
| output lattice | 44×44×793 = 1.5 M | 220×220×793 = 38 M | 25× |
| **per sweep** | **28.76 s** | **10.10 s** | **0.351×** |
| **total wall clock** | **666 s** | **324 s** | **0.487×** |
| store size | 68 MB | 1.6 GB | 24× |

**A 25× larger problem sweeps 2.85× FASTER.** GPU throughput is not merely
sublinear here — it inverts. The drift grids are small enough that each FDM
iteration is dominated by kernel-launch latency rather than arithmetic, so the
tiny drift solve wastes most of its time; the weighting grids actually fill the
device. This is exactly why the step said to measure rather than extrapolate.

Stage breakdown (weighting):

| stage | wall clock |
|---|---|
| domains | 0.1 s |
| geometry generation ×3 | 33.2 s |
| coarse solve (3.6 M) | 57.0 s |
| refine + `near_bc` | 1.4 s |
| near solve, sweep 0 (9.6 M) | 20.3 s |
| **Schwarz sweep, 20 iterations** | **205.2 s** |
| stitch (38 M output) | 4.6 s |

**Answering the step's question directly: 25 sweeps on the weighting field costs
252 s of sweeping (4.2 min). It is entirely affordable — cheaper than the same
25 sweeps on the drift field (695–719 s).** There is no need for the two fields
to take different sweep counts on cost grounds.

## Per-sweep near delta series

| sweep | near delta | ratio to previous |
|---|---|---|
| 0 | 0.000300782 | — |
| 1 | 0.000139035 | 0.4622 |
| 2 | 7.02999e-05 | 0.5056 |
| 3 | 3.97029e-05 | 0.5648 |
| 4 | 2.17265e-05 | 0.5472 |
| 5 | 1.45088e-05 | 0.6678 |
| 6 | 1.0069e-05 | 0.6940 |
| 7 | 7.40132e-06 | 0.7351 |
| 8 | 5.74085e-06 | 0.7757 |
| 9 | 4.65568e-06 | 0.8110 |
| 10 | 3.90535e-06 | 0.8388 |
| 11 | 3.35814e-06 | 0.8599 |
| 12 | 2.93954e-06 | 0.8753 |
| 13 | 2.59647e-06 | 0.8833 |
| 14 | 2.29654e-06 | 0.8845 |
| 15 | 2.03609e-06 | 0.8866 |
| 16 | 1.81054e-06 | 0.8892 |
| 17 | 1.61506e-06 | 0.8920 |
| 18 | 1.44516e-06 | 0.8948 |
| 19 | 1.29694e-06 | 0.8974 |

**The contraction is NOT flat here, and that is the important structural
difference from the drift field.** It starts at 0.4622 — far faster than any
drift ratio — and then *degrades* monotonically, rising through 0.55, 0.69,
0.81, 0.87 and reaching **0.8974** by the last gap, still rising.

So the weighting seam converges very quickly at first and then settles toward a
rate close to (and slightly worse than) the drift field's band-5 0.8737. The
early sweeps are doing most of the work.

Absolute magnitudes are not comparable to drift — this probe is dimensionless in
[0, 1] while the drift potential is in volts. The final delta is 1.297e-06,
which is ~1.8 orders above the 2e-8 tol; it stopped on **`max_iters=20`**, so
the tol still never gates, though the margin is far narrower than drift's 6.7
orders.

## The seam

Interface on node 198 of 793, as before.

| quantity | value |
|---|---|
| E_z just BELOW the plane | 0.00470598 1/mm |
| E_z just ABOVE the plane | 0.00451240 1/mm |
| kink (above − below) | −0.000193582 1/mm |
| **kink as % of the local &#124;E_z&#124; (0.0046092)** | **4.20 %** |
| phi departure from the below-side trend | 1.984e-04 |
| phi departure from the above-side trend | 1.309e-04 |

No `--design-field` was passed: there is no 56.0 V/mm reference for a
dimensionless probe.

**The weighting seam is far worse than the drift seam: 4.20 % of the local field
against the drift field's 0.093 % at the same band and sweep count.** Note also
that the kink is *negative* here (the far side is below the near side), the
opposite sign to drift.

## TRANSVERSE CORRUGATION — this is the finding

This had never been measured for the weighting probe, and it does **not** behave
like the drift field.

| z [mm] | index | max − min | W at pad centre | (max−min)/W |
|---|---|---|---|---|
| 10.5 | 105 | 0.786249 | — | — |
| 15.0 | 150 | 0.0896033 | — | — |
| **19.8 (seam)** | **198** | **0.0183202** | **0.045814** | **0.400** |
| 25.0 | 250 | 0.004007 | 0.032836 | 0.122 |
| 30.0 | 300 | 0.000964657 | 0.028000 | 0.034 |
| 40.0 | 400 | 6.06952e-05 | 0.021606 | 0.003 |

**At the seam plane the transverse corrugation is 40 % of the local W value.**
For the drift field the same quantity was 3.6e-05 V on 553 V — a factor of 7e-08.

**The interface at 19.8 mm sits deep inside the still-corrugated zone for the
weighting field, and does not for the drift field.** The corrugation only falls
below ~3 % of W by z ≈ 30 mm, which is where `NOTES-weighting-farfield.md` put
the end of the corrugated zone. The seam is 10 mm too shallow for this field.

This reframes the 4.20 % kink: it is being measured at a plane where the
transverse structure is the dominant feature, and the coarse grid must represent
that structure at 0.22 mm while the near grid has it at 0.1 mm. The two grids
disagreeing there is not surprising, and **more sweeps cannot fix it** — the
sweep converges the two domains to each other, not to the truth.

## KNOWN CONFOUND — the far-field BC (noted, not fixed)

The weighting solve's transverse edges are `fix,fix,fix`, which
`fdm_generic.py:8-34` implements as a **Neumann mirror, not Dirichlet zero**.
Per `scripts/NOTES-weighting-farfield.md` that makes the 5×5 patch behave as an
infinite periodic pad array.

**Confirmed here — the far side is a plateau/ramp, not a decay:**

| z [mm] | 19.8 | 25 | 30 | 40 | 50 | 60 | 70 | 79.2 |
|---|---|---|---|---|---|---|---|---|
| W at pad centre | 0.0458 | 0.0328 | 0.0280 | 0.0216 | 0.0159 | 0.0104 | 0.0049 | 0.0000 |

From 30 mm out, W falls essentially **linearly** to zero at the cathode — the
parallel-plate signature of an infinite pad array. A physically isolated pad's
weighting potential would decay far faster. So the entire far side of this seam
is unphysical in its tail, which plausibly contributes to the coarse side of the
kink. **The BC was not changed**, as instructed.

## Sweeps to floor, and what it would cost

Honest answer: **this cannot be computed the way it was for drift, and it should
not be faked.**

* There is **no measured coarse/fine far-field spread for the weighting field**.
  The 0.199 % floor is a *drift* number (55.996 vs 55.885 V/mm) and does not
  transfer to a dimensionless probe with a different generator and different
  edges. There is no target value to extrapolate to.
* The contraction is **not a fixed rate** here — it is still degrading at sweep
  19 — so the clean geometric extrapolation that worked for drift is not valid.

What can be said, using the asymptotic ~0.897:

| goal | extra sweeps | extra wall clock |
|---|---|---|
| halve the kink | 6.4 | 65 s |
| quarter the kink | 12.8 | 129 s |

So even aggressive extra sweeping is cheap in absolute terms (~1–2 min). **Cost
is not the obstacle for the weighting field; the corrugated interface placement
and the unphysical far-field BC are.**

## CONCLUSION

1. **25 sweeps is affordable on the weighting field** — 252 s of sweeping,
   *cheaper* than the same count on drift. The feared 25× cost blow-up does not
   exist; the GPU inverts it. **Both fields can ship the same sweep count.**
2. **But the weighting seam is not sweep-limited the way the drift seam is.** At
   4.20 % of local field, with 40 % transverse corrugation on the seam plane and
   a linear-ramp far tail from the Neumann-mirror BC, its error is dominated by
   *where the interface is* and *what the far BC does*, not by how many sweeps
   are run.
3. Recommendation for Phase 3/Step 3: **band 5 and the same sweep count for both
   fields** — cost permits it and it keeps the runner simple. But the shipped
   comment should not claim the weighting seam is converged: it is not, and more
   sweeps will not converge it.
4. Two follow-ups worth their own issues, **not attempted here**: measuring a
   coarse/fine far-field spread for the weighting probe so it has a floor to be
   judged against, and moving the weighting interface deeper (~30 mm, where
   corrugation is ~3 % of W) or fixing the transverse BC.

`run-pixel-field.sh` not touched; no default in `pochoir/` changed.


---

# Phase 4/Step 6 — the drift seam at 15 cm / 0.55 mm / band 2

Beads: `pochoir-jteg`. First solve on the retargeted geometry. Store
`store_15cm_drift_band2`, solo on the GPU. **Note the design field is now
50.0 V/mm**, so `--design-field 50.0`, not 56.0.

This step could not run until `pochoir-oq8a` fixed `_check_interface` (it
compared reconstructed float millimetres, so `--interface '29.7*mm'` was
rejected outright). Band confirmed from the log **before** trusting any number:

```
schwarz: interface z=29.7 coarse idx 54 (inner 52), near top idx 297 (seed 296)
```

**`inner 52`** — coarse node 52 = z 28.6 mm = fine node 286 exactly, so the far
Dirichlet pin *is* on a real fine node, as designed.

## 1. DID THE EXACT PIN SURVIVE THE RESPACING? **No — and this is the finding.**

| sweep | near delta | ratio to previous |
|---|---|---|
| 0 | 0.8968357064619568 | — |
| 1 | 0.8395385925741721 | 0.9361 |
| 2 | 0.7859046959684974 | 0.9361 |
| 3 | 0.7356971992733179 | 0.9361 |
| 4 | 0.6886972069187323 | 0.9361 |
| 5 | 0.6446964559281696 | 0.9361 |
| 6 | 0.6035100502684827 | 0.9361 |
| 7 | 0.5649548365148576 | 0.9361 |
| 8 | 0.5288627209429251 | 0.9361 |
| 9 | 0.4950763486317555 | 0.9361 |
| 10 | 0.463448417270115 | 0.9361 |
| 11 | 0.4338410349514561 | 0.9361 |
| 12 | 0.4061251189863242 | 0.9361 |
| 13 | 0.3801798331271584 | 0.9361 |
| 14 | 0.3558920607457594 | 0.9361 |

**The contraction ratio is 0.9361 — flat to four decimals across all fourteen
gaps, and WORSE than the 0.9237 that an *interpolated* pin gave at 0.22 mm.**

This is the outcome the step flagged as "a real finding": the pin is landing
exactly where it should (`inner 52`, verified), the band is the same 1.1 mm
physical width that produced 0.8737 at 0.22 mm coarse — and the rate did not
follow.

| configuration | band | physical width | pin | rate |
|---|---|---|---|---|
| 8 cm, coarse 0.22 | 3 | 0.66 mm | interpolated | 0.9237 |
| 8 cm, coarse 0.22 | 5 | 1.10 mm | **exact** | **0.8737** |
| 15 cm, coarse 0.55 | 2 | 1.10 mm | **exact** | **0.9361** |

So **the exact pin is not sufficient on its own**, and the Phase 3 conclusion —
that landing the pin on a fine node is what bought the rate — was
over-attributed. Holding the pin exact and the physical band width fixed, the
rate still got worse when the coarse grid was relaxed. Something else in the
respacing dominates. Two candidates, neither tested here:

* **the coarse grid itself is 2.5× coarser**, so the far solve the near domain
  is being pinned *to* is a much lower-fidelity field. The pin being on-node
  says nothing about the accuracy of the value being pinned.
* **the domain is 1.9× deeper** (149.6 vs 79.2 mm), and the Schwarz rate for
  this kind of alternating iteration generally degrades as the far domain grows
  relative to the overlap.

Distinguishing those needs one run at 15 cm with 0.22 mm coarse (isolating
depth) or one at 8 cm with 0.55 mm (isolating spacing). **Not attempted here** —
it is a new experiment, not this step's question.

## 2. IS THE SEAM CONVERGED AT 15 SWEEPS? **Yes — but only because the floor got
much worse.**

It stopped on `max_iters=15`, final delta 0.3559, still ~7.3 orders above the
tol; the tolerance does not gate.

Interface on node **297** of 1497, exactly.

| quantity | value |
|---|---|
| E_z just BELOW the plane | 49.6868501 V/mm |
| E_z just ABOVE the plane | 49.9924973 V/mm |
| kink (above − below) | **+0.305647 V/mm** |
| as % of the local &#124;E_z&#124; (49.840) | 0.6133 % |
| **as % of the 50.0 V/mm design field** | **0.6113 %** |

### THE NEW FLOOR, MEASURED — not carried over

The 0.199 % floor is an 8 cm / 0.22 mm number and does **not** transfer, so it
was re-measured the same way it was originally obtained — the far E_z on the
coarse solve against the fine near solve, on the pad-centre axis:

| solve | window | far E_z | uniformity (sd) |
|---|---|---|---|
| `potential/coarse` (0.55 mm) | z = 40–140 mm | **50.06609 V/mm** | 9.9e-06 |
| `potential/coarse` (0.55 mm) | z = 20–29 mm | 50.06609 V/mm | — |
| `potential/near` (0.1 mm) | z = 20–29 mm | **49.24328 V/mm** | 1.9e-05 |

**NEW FLOOR = 0.82281 V/mm = 1.6456 % of the 50.0 V/mm design field.**

That is **8.3× the old 0.199 %**, and it confirms — with a measurement, not an
assumption — the expectation recorded in the configs and the geometry note: the
−11.1 % coarse pad-area snap and the 0.45 mm pad-top offset made the two grids
disagree far more than they did at 0.22 mm.

The pad-top offset alone does not explain it. The naive chords are
6982.5/(149.6 − 10.45) = 50.180 V/mm coarse against 6982.5/(149.6 − 10.00) =
50.018 fine, a spread of only 0.32 %. The measured spread is 1.65 %, so **most
of it is the pad-area snap, not the laminate thickness.**

### So, against that measured floor

**The kink is 0.305647 V/mm = 0.371× the floor — it is comfortably BELOW it.**

The seam is therefore *not* the limiting error on this geometry, and 15 sweeps
is more than enough: extrapolating the 0.9361 rate backwards, the kink was
already below the floor **from sweep 0**. No extra sweeps are needed and none
are recommended.

But that verdict deserves its plain-language version, because "converged" here
is not good news:

> **The seam stopped being the limiting error only because the coarse grid got
> so much worse that its systematic pad error now dwarfs it.** At 8 cm the
> stitch had to work to reach a 0.199 % floor; here it clears a 1.65 % floor
> without trying. Relaxing the bulk to 0.55 mm bought a cheap far solve and
> paid for it in far-field accuracy — 1.65 % of the design field is a real
> error in the stitched output, smooth, and invisible to the Laplacian residual
> metric. Whether that is an acceptable trade is a physics judgement, not a
> seam-quality one, and it is not this step's to make.

## 3. WHAT DOES THE COARSER BULK COST OR SAVE?

Total **496 s**, against the 8 cm band-5 run's 666 s.

| stage | 8 cm (band 5) | 15 cm (band 2) | change |
|---|---|---|---|
| geometry generation ×3 | 45.5 s | **50.2 s** | +10 % |
| **coarse solve** | 31.7 s | **22.3 s** | **−30 %** |
| refine + `near_bc` | 0.4 s | 0.1 s | |
| near solve (sweep 0) | 10.5 s | 12.9 s | +23 % |
| **Schwarz sweep** | 575 s (20 sw) | **407.4 s** (15 sw) | |
| **per sweep** | **28.76 s** | **27.14 s** | **−5.6 %** |
| stitch | 0.19 s | 0.3 s | |
| store size | 68 MB | 105 MB | +54 % |

**The 16× smaller coarse grid bought only 30 % off the coarse solve, and the
per-sweep cost barely moved (−5.6 %).** This is exactly the launch-latency
effect Phase 3 identified, now confirmed from the other direction: the coarse
grid went from 144 k nodes to **17 k** — 8×8×273, smaller than many test
fixtures — and the solve did not get 16× faster because at that size the GPU is
not arithmetic-bound at all. **Shrinking an already-small grid buys almost
nothing.**

The near solve got *slower* (+23 %), which is the honest counterweight: its z
extent grew 50 % (298 vs 199 nodes). The sweep still dominates at 82 % of the
run.

Net: the respacing is roughly cost-neutral per sweep. The 170 s saving over the
8 cm run is almost entirely the 5 fewer sweeps, not the coarser bulk.

## Transverse corrugation at the seam

| z [mm] | index | max − min of phi [V] |
|---|---|---|
| 10.5 | 105 | 18.92 |
| 15.0 | 150 | 0.0300847 |
| 19.8 | 198 | 3.211e-05 |
| **29.7 (seam)** | **297** | **2.547e-11** |
| 30.0 | 300 | 1.796e-11 |
| 40.0 | 400 | 1.137e-12 |

**2.5e-11 V on a 988 V potential** — six orders below the old 19.8 mm value,
which was itself a non-factor. Moving the interface out to 29.7 mm has made
corrugation utterly irrelevant *for the drift field*, as expected. Whether it
did the same for the weighting probe — the field that actually motivated the
move — is Phase 4/Step 7.

Nothing was changed: no runner, no config, no `pochoir/` default.
