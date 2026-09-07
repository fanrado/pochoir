# `run-pixel-field.sh` with `--hybrid no`

What the script does, step by step, when it is run in single-grid mode:

```
./run-pixel-field.sh --hybrid no [STORE_DIR]
```

This document covers **only** the `--hybrid no` path. The `--hybrid yes` path
solves the same two fields on a coarse/near/fine stitched hierarchy; everything
in PART B and PART D below is shared by both modes.

---

## 0. Invocation and setup

| Step | Effect |
|---|---|
| `set -e` | any failing step aborts the whole run |
| `cd "$(dirname "$0")"` | switch to `scripts/`, so all config names below resolve as bare filenames |
| arg parse | `--hybrid no` is consumed (`shift 2`); only `yes`/`no` are accepted, anything else exits 1 |
| `POCHOIR_STORE` | the remaining positional arg, else the default **`store_pixel_field_no`** (the mode is baked into the default name, so a `yes` run and a `no` run cannot collide) |
| `source helpers.sh` | brings in `want`/`want_file` |
| local `want` override | replaces `helpers.sh`'s single-target `want` with a **multi-key** version: it skips the step if every `$POCHOIR_STORE/<key>.npz` already exists, otherwise runs the command and *hard-fails* if any expected key is still missing afterwards |

The `want` override is what makes a re-run resumable: interrupt the script and
start it again with the same store, and completed stages print `have <key>` and
are skipped.

## 1. Geometry and configs actually used in this mode

`--hybrid no` uses only the **fine** members of each config pair — the coarse
JSONs are read by nobody in this mode:

* drift field: `example_gen_pcb_drift_pixel_task13_fine.json` (`$dcfg_fine`)
* weighting field: `example_gen_pixel_with_grid_task13_fine.json` (`$wcfg_fine`)

Both describe the same physical validation geometry — the real LArPix v2a tile
at its true design values:

* `driftZDepth` **79.15 mm** — the electron *launch* node, deliberately not
  rounded. The pad plane low edge is at 9.9 mm, and the domain closes one plane
  further out at **79.2 mm** (`ceil(79.15/0.22) = 360` coarse `= 792` fine),
  which is the cathode plane on both grids, so there is no dead space and the
  launch node does not sit on the grounded plane
* 4.4 mm pixel pitch, split as `pixelSize` 3.5 + `pixelGap` 0.9 in the **fine**
  configs — the split that divides 0.1 mm exactly (35 + 9 cells). The coarse
  configs use **3.52 / 0.88**, the snapped split at 0.22 mm; `gcd(35, 9) = 1`, so
  no spacing coarser than 0.1 mm divides both and some coarse snapping is
  unavoidable
* `Npixels` 5 → a 5×5 pad tile, 22 mm transverse
* `GridHoleShape` **"None"** → **the shield grid is OFF**; this is a *WOGRID*
  geometry (commits `a3d5e2c`, `413460d` — the grid was never intended here).
  With-grid vs without-grid is not a flag in this script; it is entirely a
  property of the JSON the SIZES block names
* `chamfer_r` **0.7 mm** fine / **0.66 mm** coarse, `chamferMode` dynamic,
  `padThicknessCells` 3
* `enableFR4` false / `enableInsulatorFR4` true — the correct pairing.
  `enableFR4` is inert on this branch; the laminate z-layout is gated on
  `enableInsulatorFR4` alone
* drift config only: `CathodePotential` **−3878 V**, which is the design field
  **56.0 V/mm** across 79.15 − 9.9 mm. The weighting config deliberately
  carries no cathode potential — the weighting potential is a dimensionless unit
  probe in [0, 1]

The shape variables consumed in this mode are the `*_single_shape` pair, at the
uniform `spacing=0.1` mm, `precision=2e-8`:

| field | single-grid shape | meaning |
|---|---|---|
| drift | `44,44,793` | one pitch transverse, full 79.2 mm depth |
| weighting | `220,220,793` | 5 pitches (22 mm) transverse, full depth |

Those shapes are **numerically identical** to the `*_fine_shape` values used by
`--hybrid yes`. That is deliberate: both modes end up on the same output
lattice, which is what makes their results directly comparable. `interface`,
`coarse_spacing` and `fine_spacing` are set but unused here.

## 2. PART A — drift field solve (one full-depth 0.1 mm solve)

Log goes to `$POCHOIR_STORE/pochoir_driftfield.log`.

```
pochoir field-solve --hybrid no --field drift \
   --domain no \
   --config example_gen_pcb_drift_pixel_task13_fine.json \
   --shape 44,44,793 \
   --spacing 0.1 \
   --precision 0.00000002
```

Produces **`potential/drift3d`** — the electrostatic potential over the entire
79.2 mm depth in one shot, rather than the coarse+near+fine stitch of hybrid mode.

`--domain no` means the shape above is taken as authoritative: the SIZES table
is the only source of the grid, so a typo there cannot be silently "corrected"
into a different lattice.

`--field drift` also selects the edge condition **`per,per,fix`** — transversely
periodic (one 4.4 mm pixel tile, `npixels=1`), fixed in z at the pad plane and
the cathode. Like the weighting case in PART C, this is set by the profile table,
not by any flag here; see [Where the edge condition comes from](#where-the-edge-condition-comes-from).

The node-centered no-flux Neumann insulator BC is applied by `field-solve`
here — **and only here**. Nothing downstream sees `--insulator`.

## 3. PART B — velocity and drift paths (mode-independent)

These three commands are identical in both modes and are copied verbatim from
`run-for-larpix-v2a-wogrid.sh`. They implement the *enforcement-free contract*:
no surface ever clamps or terminates a trajectory, because doing so would break
the curl-free E field.

1. **Velocity** → `velocity/drift3d`
   ```
   pochoir velo --temperature 87.0*K --potential potential/drift3d \
                --velocity velocity/drift3d
   ```
   No `--boundary`, no `--insulator`: the velocity is pure `mu * grad(phi)`.
   `velo` runs mainly to attach the potential + temperature metadata that
   `drift` consumes.

2. **Starts** → `starts/drift3d` (plus plots)
   ```
   pochoir starts --starts starts/drift3d -m yes -c $dcfg_fine --plot
   ```
   Config-driven start points (`-m yes`), using the fine drift config's
   `nGridPoints: 10`.

3. **Paths** → `paths/drift3d` (plus plots)
   ```
   pochoir drift --starts starts/drift3d --velocity velocity/drift3d \
                 --interp-order linear \
                 --paths paths/drift3d '0*us,200*us,0.05*us' --plot
   ```
   No `--insulator`, so paths follow pure `grad(phi)`. Drift is integrated from
   0 to 200 µs in 0.05 µs steps. `--interp-order linear` is **required**: cubic
   interpolation rings and overshoots at the pad-plane kink and over-focuses
   paths onto the pads; linear is monotone-safe.

## 4. PART C — weighting field solve (one full-depth 0.1 mm solve)

Log switches to `$POCHOIR_STORE/pochoir_weightingfield.log`.

```
pochoir field-solve --hybrid no --field weighting \
   --domain no \
   --config example_gen_pixel_with_grid_task13_fine.json \
   --shape 220,220,793 \
   --spacing 0.1 \
   --precision 0.00000002
```

Produces **`potential/weight3d`**. Unlike the drift solve, this one spans the
full 5×5 tile transversely (220 cells = 22 mm) and is solved **non-periodic**.

### Where the edge condition comes from

The boundary condition is *not* a flag in this script — nothing on the
`field-solve` command line sets it. It is selected entirely by `--field`, via
the profile table `FIELDS` in `pochoir/hybrid_iterate.py:125-152`:

| `--field` | `edges` | generator | transverse extent |
|---|---|---|---|
| `drift` | `per,per,fix` | `pcb_drift_pixel_with_grid` | `npixels=1` → one 4.4 mm periodic tile |
| `weighting` | `fix,fix,fix` | `pcb_pixel_with_grid` | `npixels=None` → `Npixels` from the config |

Despite the name, that table is reached in **both** modes. The `--hybrid no`
path runs `pochoir/single_field.py`, which deliberately imports
`_profile`/`_solve`/`_key`/`_extents` from `hybrid_iterate` rather than
redefining them — its docstring states the intent: *"same field profiles, same
generator, same edges, same fdm flags"*, so the two modes cannot diverge in
behaviour. `single_field()` does `prof = _profile(field)` and echoes
`edges={prof["edges"]}` into the log, so the value used is visible in
`pochoir_weightingfield.log`.

Consequence: `fix,fix,fix` means the unit-probe potential decays to ~0 at the
tile edges instead of wrapping. A weighting field cannot be solved on the single
periodic pixel tile the drift field uses — hence the 220-cell transverse extent
here versus 44 for drift. There is no way to change this from the script short
of editing `FIELDS`.

## 5. PART D — induced current (mode-independent)

```
pochoir induce-pixel --weighting potential/weight3d \
   --paths paths/drift3d \
   --output current/induced_current \
   --npixels 2 \
   --config example_gen_pixel_with_grid_task13_fine.json --plot
```

Ramo's theorem applied to the weighting field and the drift paths, giving
**`current/induced_current`** plus plots. `--npixels 2` sums the target pixel
and its first ring of neighbours; `--config` supplies the pad-collection map
geometry, and is the same file used for the weighting solve so the two cannot
disagree.

Note that the weighting field is used at its native 0.1 mm spacing throughout —
coarsening it before the Ramo calculation is what produces spurious current
spikes.

## 6. Outputs

In `$POCHOIR_STORE` (default `store_pixel_field_no`):

```
potential/drift3d.npz        PART A
velocity/drift3d.npz         PART B
starts/drift3d.npz           PART B
paths/drift3d.npz            PART B
potential/weight3d.npz       PART C
current/induced_current.npz  PART D
pochoir_driftfield.log
pochoir_weightingfield.log
```

`date` is printed after the drift solve, after the weighting solve, and after
the induce step; the run ends with `=== DONE ===`.

## 7. What differs from `--hybrid yes`

Only PART A and PART C differ. Hybrid mode replaces each single
`field-solve --hybrid no ... --config ... --shape ... --spacing ...` call with a
`--hybrid yes` call taking `--coarse-config`/`--fine-config`, three shapes
(coarse `20,20,361` / `100,100,361`, near `44,44,199` / `220,220,199`, fine
`44,44,793` / `220,220,793`), `--interface 19.8*mm`, and the two spacings
0.22/0.1.
The store keys written are the same names, and at this geometry the final
lattice is the same — which is the point.

## 8. The banded near/far Schwarz seam (`--hybrid yes` only)

Hybrid mode does **not** simply pin the near solution once and stop. It runs an
alternating overlapping-Schwarz iteration (`pochoir/nearfar.py:schwarz_solve`):

* each sweep solves the **far** domain with an interior plane pinned
  `band_cells` coarse cells *below* the interface to the current near solution
  (the near grid already covers that depth), then re-solves the **near** domain
  with its interface plane pinned to that fresh far solution
* **ending each sweep on the near solve** is what makes the returned stitch
  exactly **C0**: the near is pinned to the far it was just handed. A *wider*
  band drives the seam toward gradient continuity (**C1**) as well, so the
  induced current `i(t) = q·v·∇W` has no spurious glitch where a drift electron
  crosses the seam. It also converges much faster — the ~0.95/sweep rate of the
  single-cell coupling improves strongly with width
* `max_sweeps = 0` is the **one-shot-pin fallback** (byte-identical to the old
  behaviour), not the default path

The script now states the parameters explicitly at both `--hybrid yes` call
sites rather than inheriting `hybrid_iterate.py`'s defaults:

```
--band-cells 3 --max-sweeps 4 --schwarz-tol 2e-8
```

* **band 3** is 3 *coarse* cells = `band_cells + 1` = **4 nodes**. At
  `--interface 19.8 mm` with coarse 0.22 mm those are coarse nodes 90 / 89 / 88 /
  87 = z **19.80 / 19.58 / 19.36 / 19.14 mm**.
* the innermost plane 19.14 mm is **not** a fine node. At the 2.2 coarse:fine
  ratio the two grids share nodes only every 1.1 mm, so the far Dirichlet pin
  there is **interpolated** onto the near grid rather than exact; only
  `band_cells` that are multiples of 5 give an exact pin.
* **tol 2e-8** is undimensioned *on purpose*, so a single value serves both the
  volt-valued drift potential and the dimensionless [0, 1] weighting probe.
* at 2e-8 the tolerance **never gates**: measured near deltas on this 8 cm
  grid-free geometry run 0.34 (band 2) to 3.17 (band 20), about seven orders of
  magnitude above it. **`--max-sweeps 4` is therefore the binding limit**, and
  the sweep count alone decides seam quality.

## 9. Caveat carried from the script header

The four `*task13*.json` configs in `scripts/` are hand-maintained **copies** of
same-named files in `../test/`. Nothing in the repo checks that the two sets
agree, and nothing regenerates one from the other. Before trusting a run, after
editing either side:

```
orig=../test ; for f in *task13*.json ; do cmp "$f" "$orig/$f" ; done
```

**As of this geometry the four pairs are known to differ.** The `scripts/`
copies have been retargeted to 4.4 mm / 5×5 / 79.15 mm / 0.22-0.1; the `test/`
copies have not, and `test/run-task13-hybrid.sh` still solves the old geometry.
The `cmp` recipe will report all four pairs as differing — that is expected, not
a mistake to "fix" by copying either way.
