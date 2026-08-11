# Pixelated readout — full 3D pipeline

This document is a focused companion to the main `README.org`. It describes
what the script `scripts/run-pixel-field.sh` does, what it needs, and how to
run it. Use the main README for the package philosophy and the generic
`pochoir` CLI; use this one when you want to reproduce or modify the
pixelated-readout field-response simulation.

`scripts/run-pixel-field.sh` is the main pixelated-readout runner on this
branch. For a command-by-command walkthrough of the single-grid path see
`scripts/run-pixel-field.md`.

## What the script does

The script runs the full chain that produces the field response of a pixelated
LArTPC anode with a PCB shield grid. Each step is wrapped in a multi-key `want`
helper, so a re-run prints `have <key>` for finished stages and only redoes
missing artifacts — an interrupted run resumes cleanly with the same store.

1. **Drift potential** (PART A) — `pochoir field-solve --field drift`, either
   as a coarse/near/fine hybrid stitch or as one full-depth 0.1 mm solve
   (see *Modes* below). Writes `potential/drift3d`.
2. **Velocity** (PART B) — `pochoir velo` at `87.0*K` from the full-depth drift
   potential. Writes `velocity/drift3d`.
3. **Starting points** (PART B) — `pochoir starts -m yes`, i.e. config-driven
   starts using `nGridPoints: 10` from the fine drift config.
4. **Drift paths** (PART B) — `pochoir drift` over `0*us,200*us,0.05*us` with
   `--interp-order linear`. Writes `paths/drift3d`.
5. **Weighting potential** (PART C) — `pochoir field-solve --field weighting`,
   in the same mode as PART A but on the wider 5×5-pad transverse domain.
   Writes `potential/weight3d`.
6. **Induced currents** (PART C) — `pochoir induce-pixel` over a 2-pixel window
   (target pad plus its first ring of neighbours), applying Ramo's theorem to
   the weighting potential and the drift paths. Writes
   `current/induced_current`.

`date` is printed after the drift solve, after the weighting solve, and after
the induce step; the run ends with `=== DONE ===`.

### Modes

```bash
./run-pixel-field.sh [--hybrid yes|no] [STORE_DIR]
```

`--hybrid yes` (the default) solves each field on a coarse + near + fine grid
hierarchy stitched at an interface; `--hybrid no` does ONE 0.1 mm full-depth
solve. Only PARTS A and C differ between the modes — PART B and the induce step
are identical. Both modes write the same store keys and, at this geometry, land
on the **same final lattice**, which is what makes the two directly comparable.
Anything other than `yes`/`no` exits 1.

### The enforcement-free contract

This is non-negotiable and is the reason PART B's three commands are copied
verbatim from `run-for-larpix-v2a-wogrid.sh`:

- `--insulator` (the node-centered no-flux Neumann BC) is applied to the
  **field solve only**.
- `velo` gets neither `--boundary` nor `--insulator`, and `drift` gets no
  `--insulator`, so trajectories are never clamped or terminated at a surface.
  Doing so would break the curl-free E field.
- `--interp-order linear` is **required**: cubic interpolation overshoots at the
  seam / pad-plane kink and over-focuses paths onto the pads.

The weighting field is also used at its native 0.1 mm spacing throughout;
coarsening it before the Ramo calculation produces spurious current spikes.

## Requirements

### System

- Linux (the script has been used on Debian-class hosts).
- Python 3.9+ with a working virtual environment.
- A CUDA-capable GPU is **strongly** recommended — `field-solve` uses the
  `torch` engine and the weighting domain in particular is large
  (`220,220,701` ≈ 34 M voxels at 0.1 mm, per solve).
- `bash` and standard coreutils.

### Python

Install pochoir with the extras needed by this pipeline:

```bash
python3 -m venv env
source env/bin/activate
pip install -e .[torch,plots,hdf5]
```

(See `README.org` for the canonical install, including optional `numba`,
`cupy`, and `vtk` extras.)

### Configuration files

The script `cd`s to its own directory and names its configs as bare filenames,
so all four live in `scripts/`:

| variable | file | role |
| --- | --- | --- |
| `dcfg_coarse` | `example_gen_pcb_drift_pixel_task13_coarse.json` | drift, 0.4 mm far field |
| `dcfg_fine` | `example_gen_pcb_drift_pixel_task13_fine.json` | drift, 0.1 mm near field + final volume |
| `wcfg_coarse` | `example_gen_pixel_with_grid_task13_coarse.json` | weighting, 0.4 mm |
| `wcfg_fine` | `example_gen_pixel_with_grid_task13_fine.json` | weighting, 0.1 mm |

`--hybrid no` uses only the two `_fine` files; the coarse pair is read by nobody
in that mode.

**These four files are hand-maintained COPIES of same-named files in `test/`.**
There is no symlink, no generator, and nothing in the repo that checks the two
sets agree. Retargeting a geometry means editing **both** copies — change only
one side and the other runner keeps solving the old geometry, silently. Check
the pairs by hand after editing either side:

```bash
cd scripts
orig=../test ; for f in *task13*.json ; do cmp "$f" "$orig/$f" ; done
```

Common config fields (units: mm except where noted):

| field | meaning |
| --- | --- |
| `pixelSize` | pad side length |
| `pixelGap` | gap between adjacent pads |
| `chamfer_r` | rounded-corner radius, in **mm** |
| `chamferMode` | `"dynamic"` scales the corner cut with the spacing |
| `Npixels` | number of pixels along one axis |
| `nGridPoints` | starts per pixel axis, consumed by `pochoir starts -m yes` |
| `pixelPlaneLowEdgePosition` | z of the bottom of the pixel plane |
| `pixelPlaneWidth` | thickness of the pixel plane |
| `padThickness`, `padThicknessCells` | pad thickness; the pad needs ≥3 grounded cells or the field leaks through it |
| `PcbWidth`, `HoleRadius` | shield-PCB thickness and hole size |
| `GridHoleShape` | `"None"`, `"circular"`, or `"square"` |
| `GridPotential` | shield-plane potential (V), drift configs only |
| `CathodePotential` | cathode potential (V), drift configs only |
| `driftZDepth` | full domain depth |
| `enableInsulatorFR4` | gates the pad/FR4 laminate z-layout — leave **true** |
| `enableFR4` | **inert on this branch**; nothing in `pochoir/` reads it |

Notes on the geometry these configs describe:

- The drift and weighting pairs must agree on `pixelSize`, `pixelGap`,
  `chamfer_r`, `Npixels`, and `driftZDepth`, or drift and weighting describe
  different detectors.
- `pixelSize`/`pixelGap` differ **on purpose** between the coarse and fine files
  (3.6/0.8 vs 3.5/0.9): both are the same 4.4 mm pitch, and each file uses the
  split that divides its own spacing exactly.
- The weighting configs carry no `GridPotential` and no `CathodePotential`,
  deliberately — the weighting potential is a dimensionless unit probe in
  `[0, 1]`, not volts.
- **With-grid vs without-grid is not a flag.** It follows entirely from
  `GridHoleShape` in the JSON the SIZES block names (`"None"` = no shield grid);
  that is exactly how `run-for-largepix-wgrid.sh` and
  `run-for-larpix-v2a-wogrid.sh` differ.
- The transverse/z edge conditions are likewise not flags: `--field drift`
  selects `per,per,fix` on one periodic 4.4 mm tile, `--field weighting` selects
  `fix,fix,fix` over `Npixels`, via the `FIELDS` profile table in
  `pochoir/hybrid_iterate.py` — reached in both modes.

### Shapes

The `SIZES` block in the script is the authority: `field-solve` is invoked with
`--domain no`, so a typo there cannot be silently corrected into a different
grid. For the validation geometry (`driftZDepth` 69.9 → 70.0 mm of whole 0.4 mm
cells, 4.4 mm pitch, `Npixels` 5, interface 40 mm):

| grid | spacing | drift | weighting |
| --- | --- | --- | --- |
| coarse, full depth | 0.4 mm | `11,11,176` | `55,55,176` |
| near, to interface | 0.1 mm | `44,44,401` | `220,220,401` |
| fine, full depth | 0.1 mm | `44,44,701` | `220,220,701` |
| single (`--hybrid no`) | 0.1 mm | `44,44,701` | `220,220,701` |

The single and fine rows are the same grid on purpose, and are kept as two
labelled variables rather than deduplicated so that their agreement reads as a
deliberate choice. `interface`, `coarse_spacing`, and `fine_spacing` are set but
unused in `--hybrid no`. `precision` is `2e-8` in both modes.

## Usage

From the `scripts/` directory, after activating the venv:

```bash
cd scripts
./run-pixel-field.sh                       # hybrid, store_pixel_field_yes
./run-pixel-field.sh --hybrid no            # single grid, store_pixel_field_no
./run-pixel-field.sh --hybrid yes my_store  # explicit store
```

`STORE_DIR` defaults to `store_pixel_field_<mode>`, so a `yes` run and a `no`
run cannot collide. Shapes are not overridable from the environment — edit the
`SIZES` block.

## Outputs

All artifacts land under `$POCHOIR_STORE`:

- `potential/drift3d` — drift Laplace solution (PART A).
- `velocity/drift3d` — drift velocity field (PART B).
- `starts/drift3d`, `paths/drift3d` — drift starts and paths, plus plots
  (PART B).
- `potential/weight3d` — weighting Laplace solution (PART C).
- `current/induced_current` — induced current on the 2-pixel window, plus plots
  (PART C).
- `pochoir_driftfield.log`, `pochoir_weightingfield.log` — solver logs. The
  edge condition actually used is echoed into these.

The outer store directory may be renamed freely; the inner key structure must
not change.

## Tweaking the geometry

The rounded-corner shape is implemented in `trimCorner` in
`pochoir/gen_pcb_drift_pixel_with_grid.py` and
`pochoir/gen_pcb_pixel_with_grid.py`. Both carve a quarter-disk of radius
`chamfer_r` at each inner corner. `chamfer_r` is in **mm** and is converted to
cells against the grid spacing, so the same value means the same physical corner
on the coarse and fine grids. Increase it to round the corners more strongly;
set it to `0` for sharp 90° corners. Remember to edit both the `scripts/` copy
and the `test/` copy of every config you touch.
