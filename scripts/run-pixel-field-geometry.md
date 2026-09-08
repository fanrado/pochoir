# `run-pixel-field.sh` — chosen geometry and domain

Beads: `pochoir-honm` (Phase 4/Step 1). This note records the geometry
`run-pixel-field.sh` and the four `example_*_task13_*.json` configs target, and
**why** each value is what it is.

> **Phase 4 retarget.** This replaces the 8 cm / 0.22 mm geometry documented
> here previously (Phase 0/Step 3). The drift length moves to 15 cm, the
> near/far interface moves out from 19.8 mm to **29.7 mm**, and the coarse grid
> relaxes from 0.22 mm to **0.55 mm**.
>
> Two measurements drove it, both in `NOTES-run-pixel-field-seam.md`:
> * 0.22 mm is far too fine for the bulk. The far field is essentially 1-D
>   linear out there, and **linear fields are exact on any spacing**, so the
>   resolution bought nothing.
> * the interface at 19.8 mm was measured sitting *inside* the corrugated zone
>   for the weighting field — transverse corrugation was **40 % of the local W**
>   on the seam plane (`pochoir-h92v`) — and more sweeps cannot fix that.
>
> **Steps 2–4 have not run yet: the configs and the runner still carry the 8 cm
> geometry.** This note is the derivation, verified ahead of those edits.

## ⚠ ONE BLOCKER FOUND BY THIS VERIFICATION — read before Step 2

**`--interface '29.7*mm'` is REJECTED at runtime as things stand.**
`_check_interface` compares the requested interface against the near grid's
implied split using exact float `!=`, and:

```
(298-1) * 0.1  ==  29.700000000000002842...   (not 29.7)
unitify('29.7*mm')  ==  29.7
```

so the check raises:

```
--interface 29.7*mm (29.7) disagrees with the split implied by domain/near
(29.700000000000003).
```

This is verified for **both** fields (`domain/near` and `domain/w_near`).

It is pure floating-point luck that the current geometry works: `198 * 0.1`
round-trips to *exactly* the double `19.8`, while `297 * 0.1` does not. Nothing
about 29.7 mm is wrong as a *geometry* — it is 54 coarse cells and 297 fine
cells exactly, in integer terms — the failure is only in the equality test.

**This is not fixed here** (this step is derivation and verification only, and
the fix belongs in `pochoir/hybrid_iterate.py`, not in a config or the runner).
Step 2 must not begin by working around it with a hand-tuned interface string.
The right fix is to compare with a tolerance, or to compare integer cell counts
rather than reconstructed millimetres.

## The decided geometry

| quantity | value | node arithmetic |
|---|---|---|
| coarse spacing | **0.55 mm** | same for both fields |
| fine spacing | **0.1 mm** | unchanged |
| pitch | 4.4 mm | = **8** coarse = **44** fine cells |
| pad plane | 9.9 mm | = **18** coarse = **99** fine |
| **interface** | **29.7 mm** | = **54** coarse = **297** fine → near z = **298** nodes |
| cathode | 149.6 mm | = **272** coarse = **1496** fine → full z = **1497** nodes |
| `driftZDepth` | 149.55 mm | the electron **launch** node |
| `CathodePotential` | **−6982.5 V** | = −50.0 V/mm × (149.55 − 9.9) mm |

### Why 1.1 mm matters everywhere

**1.1 mm is the shared period of the 0.55 mm and 0.1 mm grids** — the two grids
share nodes only every 1.1 mm. Every aligned plane above is a multiple of it:

```
  4.4 =   4 × 1.1        9.9 =   9 × 1.1
 29.7 =  27 × 1.1      149.6 = 136 × 1.1
```

That single rule is why all four planes land on nodes of both grids, and it is
the constraint to respect if any of them is ever moved.

### The launch node and the cathode

`driftZDepth` 149.55 mm is the **launch** node, deliberately not rounded.
`_extents` puts the cathode at `ceil(149.55/0.55) = 272` coarse cells =
**149.6 mm** = 1496 fine cells, the last plane of both grids — so there is **no
dead space** above the cathode, and the launch node does not sit on the grounded
Dirichlet plane. (`MIN_CATHODE_CLEARANCE` is 0.0, so the `ceil` alone provides
the clearance; it works here only because 149.55/0.55 = 271.9̄ is not already a
whole number.)

### The design field is 50 V/mm — a deliberate change

```
CathodePotential = -BulkField * (driftZDepth - pixelPlaneLowEdgePosition)
                 = -50.0 * (149.55 - 9.9) = -6982.5 V
```

**The bulk field is now 50.0 V/mm, down from the 8 cm geometry's 56.0 V/mm.**
This is a deliberate change of the design field, **not** a rescaling to preserve
it — unlike the Phase 0 retarget, which held 56.0 V/mm fixed precisely so drift
velocities stayed comparable. Any comparison of drift paths or induced current
against 8 cm results must account for it: `velo` derives mobility from E through
a nonlinear LAr parameterisation, so a 10.7 % lower field is not a 10.7 % slower
drift.

## COARSE PAD SNAP at 0.55 mm — the price of coarsening

Recorded plainly, because it is the real cost of relaxing the far grid:

| quantity | fine (0.1 mm) | coarse (0.55 mm) | cost |
|---|---|---|---|
| `pixelSize` | 3.5 mm (35 cells) | **3.30 mm** (6 cells) | |
| `pixelGap` | 0.9 mm (9 cells) | **1.10 mm** (2 cells) | |
| pitch | 4.4 mm (44) | 4.4 mm (6+2 = 8) | exact — `_cells()` requires it |
| pad **area** | (3.5/4.4)² = 0.6332 | (3.3/4.4)² = 0.5625 | **−11.2 %** |
| `chamfer_r` | 0.7 mm | **0.55 mm** (1 cell) | −21 % undercut |
| `padThickness` | | 0.55 (→ 1 cell) | |
| `FR4Thickness` | | 0.55 (→ 1 cell) | |
| pad **top** plane | 10.0 mm | **(18+1)×0.55 = 10.45 mm** | the grids model different pads |

The −11.2 % pad-area error is **much worse than the 8 cm geometry's +1.1 %** at
0.22 mm, and it matters for the same reason recorded there: the far field in the
stitched output *is* the upsampled coarse solution, so a coarse pad-area error
biases the far weighting field by roughly the same percentage, and it is a
*smooth* error that the Laplacian residual metric cannot detect.

**Consequence to expect in Step 4:** the coarse/fine far-field disagreement — the
0.199 % "geometry floor" the 8 cm seam was judged against — will be
substantially larger here. That floor was measured, not assumed, and it must be
re-measured for this geometry rather than carried over.

## Domain table (the SIZES block)

| grid | spacing | drift (1 pitch) | weighting (5 pitches) |
|---|---|---|---|
| coarse, full depth | 0.55 | `8,8,273` | `40,40,273` |
| near, to interface | 0.1 | `44,44,298` | `220,220,298` |
| fine, full depth | 0.1 | `44,44,1497` | `220,220,1497` |
| single (`--hybrid no`) | 0.1 | `44,44,1497` | `220,220,1497` |

Transverse counts are **open** (N = extent/spacing — the far node is the wrap of
the near one); z counts are **closed** (N = extent/spacing + 1, both faces are
real planes). Drift is one periodic 4.4 mm tile (`npixels=1`); the weighting
probe spans `Npixels` 5 = 22.0 mm, and 22.0/0.55 = 40 coarse, 22.0/0.1 = 220
fine.

The single row and the fine row are the same grid **on purpose** — that is what
makes the two modes comparable on identical output lattices.

### Verified by calling the code, not by hand

All eight shapes above were confirmed by exercising
`pochoir.hybrid_iterate._extents` and `_cells` directly on the retargeted values
for both field profiles. `_extents` returns
`transverse` 4.4 / 22.0 mm, `full` 149.6 mm, `near` 29.7 mm, and every
`_cells()` call reproduced the table exactly.

**This verification matters because `--domain no` BYPASSES `_cells()`.** The
runner supplies the shapes itself and the only check that would otherwise refuse
a geometry the spacings cannot represent is skipped — so this table, and this
check of it, is the only thing standing between a typo in the SIZES block and a
silently wrong lattice.

`_check_interface` was exercised too: it correctly **rejects** 19.8 mm and 30 mm
against a 298-node near grid, and it also rejects the intended 29.7 mm — see the
blocker section at the top.

## Caveats

1. **The coarse grid is genuinely coarse now.** `padThicknessCells` and the
   laminate are one cell each, so the coarse pad block is 0.55 mm against the
   fine grid's 0.30 mm, and the coarse pad *top* is at 10.45 mm against 10.00 mm.
   The two grids model measurably different pads — more so than at 0.22 mm.
2. **The 0.199 % far-field floor does not transfer.** It was measured on the
   8 cm / 0.22 mm geometry (55.996 coarse vs 55.885 fine V/mm). Both the field
   (50 vs 56 V/mm) and the coarse pad snap have changed; re-measure it.
3. **The interface moved to escape the weighting corrugation, and that has not
   yet been confirmed.** 29.7 mm was chosen because 19.8 mm sat where corrugation
   was 40 % of local W. Whether 29.7 mm is far enough out is a *measurement for a
   later step* — the 8 cm data had corrugation still at ~3 % of W at 30 mm, and
   this is a different depth and a different coarse grid.
4. **The `test/` copies are NOT retargeted.** `test/` holds hand-maintained
   copies of these four basenames, and `test/run-task13-hybrid.sh` still solves an
   older geometry. Nothing in the repo checks that the two sets agree — see the
   DUPLICATED CONFIGS header in `run-pixel-field.sh`. The `cmp` recipe there will
   report all four pairs as differing; that is expected, not a mistake to "fix"
   by copying either way.
