# `run-pixel-field.sh` — chosen geometry and domain

This note records the values in `run-pixel-field.sh` and the four
`example_*_task13_*.json` configs beside it, and **why** they are what they are.
Everything here is carried over from the configs' own `_description` blocks and
the `SIZES` block of the script; it is not re-derived.

> The geometry documented in the *previous* version of this note — 3.7 mm pitch,
> `pixelSize` 2.2 + `pixelGap` 1.5, `Npixels` 9, `driftZDepth` 159.9,
> `CathodePotential` −8400, spacings 0.37 / 0.0925 — is **retired** (commit
> `532e923`). Commit `be8343e` moved the script and all four configs to the
> geometry below. Nothing in `scripts/` uses the old values any more.

## Physical geometry (the inputs)

| quantity | value | note |
|---|---|---|
| pitch | **4.4 mm** | the true LArPix v2a pitch |
| `pixelSize` / `pixelGap` | 3.5 / 0.9 (fine) | the design values, unsnapped |
| | 3.52 / 0.88 (coarse) | snapped to the 0.22 mm grid |
| `Npixels` | 5 | 5×5 tile = 22.0 mm transverse for the weighting probe |
| `driftZDepth` | 79.15 mm | electron **launch** node, deliberately not rounded |
| `pixelPlaneLowEdgePosition` | 9.9 mm | a node of *both* grids |
| `CathodePotential` | **−3878 V** | 56.0 V/mm over 79.15 − 9.9 mm |
| `chamfer_r` | 0.7 (fine) / 0.66 (coarse) | |
| `GridHoleShape` | **"None"** | the shield grid is **OFF** — this is a WOGRID geometry |
| interface | **19.8 mm** | near/far seam |

`gcd(35, 9) = 1`, so **no spacing coarser than 0.1 mm divides both the pad and
the gap**: some snapping on the coarse grid is unavoidable, which is why the
coarse configs carry a different split from the fine ones.

## Spacings: 0.22 / 0.1 mm

0.22 mm was chosen over 0.2, 0.4 and 0.44 because it is the only candidate that
does both of these at once:

* it snaps the pad to **3.52 mm**, only **+1.1 %** in area — against **+5.8 %**
  for both 0.2 mm and 0.4 mm (each of which gives a 3.6 mm pad). That error is
  not cosmetic: the far field in the final stitched output *is* the upsampled
  coarse solution, so a coarse pad-area error biases the far weighting field by
  about the same percentage, and it is a *smooth* error the Laplacian residual
  metric cannot detect.
* it puts the **pad plane 9.9 mm on a node of both grids**: `9.9/0.22 = 45.0`
  and `9.9/0.1 = 99.0`, both exact in binary float. 0.44 mm would force the pad
  plane onto a multiple of 2.2 mm — 8.8 or 11.0 — moving it.

### Consequence: the ratio is 2.2, not an integer

**Coarse nodes coincide with fine nodes only every 1.1 mm.** The coarse grid is
*not* a subset of the fine grid, so the near→coarse restriction
**interpolates** rather than subsamples, and the coarse-cell staircase is
non-commensurate with the fine grid.

This is supported — `pochoir/nearfar.py` is explicitly coordinate-based
interpolation and works for both up- and downsampling, and `_cells()` only
checks each grid against the geometry, never the ratio. But it is a **real loss
relative to the retired geometry**, which had picked a 4:1 ratio *explicitly*
because it matched the validated configuration. That property is gone here, and
it is why the Schwarz band's innermost pin plane lands off-node (see the banded
seam section of `run-pixel-field.md`).

Coarse being only 2.2× coarser costs little: 100×100×361 is about 3.6 M nodes.

## Depth and interface

`driftZDepth` **79.15 mm is the electron launch node** and is deliberately not
rounded. The key is overloaded — it sets both the grounded cathode node and the
path start z in continuous mm. `hybrid_iterate._extents` puts the cathode at
`ceil(79.15/0.22) = 360` coarse cells = **792 fine** = **79.2 mm**, the last
plane of *both* grids, so there is **no dead space** above the cathode; but the
launch node 79.15 does not land on that grounded plane, which is the point.
Snapping `driftZDepth` would put the drift start exactly on the Dirichlet node.

Interface **19.8 mm = 90 coarse = 198 fine cells**, a plane of both grids; the
near grid is therefore **199 nodes**. `_check_interface`
(`pochoir/hybrid_iterate.py:458`) enforces that `(nz-1) × fine_spacing` equals
`--interface`.

## Domain (the SIZES block)

| grid | spacing | drift (1 pitch) | weighting (5 pitches) |
|---|---|---|---|
| coarse, full depth | 0.22 | `20,20,361` | `100,100,361` |
| near, to interface | 0.1 | `44,44,199` | `220,220,199` |
| fine, full depth | 0.1 | `44,44,793` | `220,220,793` |
| single (`--hybrid no`) | 0.1 | `44,44,793` | `220,220,793` |

Transverse counts are **open** (N = extent/spacing — the far node is the wrap of
the near one, not duplicated); z counts are **closed** (N = extent/spacing + 1,
both faces are real planes). That is the convention `_cells()` implements with
its `closed` flag.

The single row and the fine row are the same grid **on purpose**: that is what
makes the two modes comparable on identical output lattices.

## Cathode potential

```
CathodePotential = -BulkField * (driftZDepth - pixelPlaneLowEdgePosition)
```

Maintained **by hand** on purpose, so the bulk field stays a visible, editable
number rather than a derived one. **−3878 V preserves the design *field*, not
the old voltage**: the original −8400 V over 159.9 − 9.9 = 150.0 mm is exactly
56.0 V/mm (560 V/cm), and 56.0 × (79.15 − 9.9) = 3878.0 V reproduces it here.
Leaving −8400 would have given 121.3 V/mm.

The *field* is what is preserved because `velo` derives mobility from E through
a nonlinear LAr parameterisation — at 121 V/mm the drift velocity is not simply
doubled and the paths stop being comparable to production. The same voltage is
applied on both grids, deliberately: it is a boundary condition.

## Shield grid: OFF

`GridHoleShape` is `"None"`. It was inherited from the older with-grid task13
work and was **never intended here** — the LArPix v2a 5×5 tile is a WOGRID
geometry in production. While it was on, a perforated −900 V electrode sat
between the pad and the cathode and held the far field down to 46.374 V/mm.
`GridPotential`, `PcbWidth` and `HoleRadius` are now **inert** and are left in
place: with `"None"` neither the circular nor the square branch runs.

## Caveats

1. **`padThicknessCells` is a CELL count, so the two grids model different
   pads.** At 3 cells the pad block is **0.66 mm** on the coarse grid against
   **0.30 mm** on the fine grid. Likewise `chamfer_r` is 0.66 vs 0.7.
2. **The two sides cannot agree perfectly at the seam.** Measured far E_z is
   **55.996 V/mm coarse** against **55.885 V/mm fine**, both uniform to four
   decimals from z = 30 mm to 70 mm — a **0.199 % spread**. That spread is a
   *floor* on how well the near and far solutions can match across the
   interface; do not chase seam residuals below it, and do **not** re-tune
   `CathodePotential` to close it.
3. **The `test/` copies are NOT retargeted.** `test/` holds hand-maintained
   copies of these same four basenames, and `test/run-task13-hybrid.sh` still
   solves the **old** geometry. Only the `scripts/` side was changed. Nothing in
   the repo checks that the two sets agree — see the DUPLICATED CONFIGS header in
   `run-pixel-field.sh`. The `cmp` recipe there will report all four pairs as
   differing; that is expected, not a mistake to "fix" by copying either way.
