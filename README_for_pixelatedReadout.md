# Pixelated readout — full 3D pipeline

This document is a focused companion to the main `README.org`. It describes
what the script `test/test-full-3d-pixel.sh` does, what it needs, and how to
run it. Use the main README for the package philosophy and the generic
`pochoir` CLI; use this one when you want to reproduce or modify the
pixelated-readout field-response simulation.

## What the script does

`test/test-full-3d-pixel.sh` runs the full chain that produces the field
response of a pixelated LArTPC anode with a PCB shield grid. It chains the
following `pochoir` subcommands (each step is cached by the `want` helper, so
re-runs only redo missing artifacts):

1. **Drift domain & boundary** — build a 3D domain
   (`POCHOIR_DRIFT_SHAPE`, spacing `0.1 mm`), generate the boundary/initial
   value arrays from `example_gen_pcb_drift_pixel_with_grid.json` using the
   `pcb_drift_pixel_with_grid` generator (PCB shield plane with rounded
   square holes + pixel pads with rounded corners).
2. **Drift potential** — solve Laplace via `pochoir fdm` (torch engine,
   periodic in x/y, fixed in z).
3. **Weighting domain & boundary** — larger transverse domain
   (`POCHOIR_WEIGHT_SHAPE`) using `example_gen_pixel_with_grid.json` and
   the `pcb_pixel_with_grid` generator.
4. **Weighting potential** — `pochoir fdm` for the weighting field of the
   central pixel.
5. **Velocity field** — `pochoir velo` at 87 K from the drift potential.
6. **Starting points & drift paths** — a 10×10 grid of starts per pixel
   (100 points), drifted with `pochoir drift` over `0–210 µs` at `0.05 µs`.
7. **Induced currents** — `pochoir induce-pixel` over a 4-pixel window using
   the weighting potential and the drift paths.

Convergence is summarised by `parse_maxerr.py`, which produces
`store/maxerr_*.png` and `store/summary_log_*.pdf`.

## Requirements

### System

- Linux (the script has been used on Debian-class hosts).
- Python 3.9+ with a working virtual environment.
- A CUDA-capable GPU is **strongly** recommended — the `fdm` step uses the
  `torch` engine and the weighting domain in particular is large
  (a typical run uses `396×396×1500` ≈ 235 M voxels).
- `bash`, `make`, and standard coreutils.

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

Two JSON configs in `test/` drive the geometry generators. The fields most
relevant to the pixelated readout are:

- `example_gen_pcb_drift_pixel_with_grid.json` — drift-field geometry
  (cathode, PCB shield plane, pixel plane).
- `example_gen_pixel_with_grid.json` — weighting-field geometry.

Common fields (units: mm except where noted):

| field | meaning |
| --- | --- |
| `pixelSize` | pixel side length |
| `pixelGap` | gap between adjacent pixels |
| `chamfer_r` | rounded-corner radius (passed to `trimCorner`) |
| `Npixels` | number of pixels along one axis |
| `pixelPlaneLowEdgePosition` | z of the bottom of the pixel plane |
| `pixelPlaneWidth` | thickness of the pixel plane |
| `PcbWidth` | thickness of the PCB shield |
| `GridHoleShape` | `"None"`, `"circular"`, or `"square"` |
| `GridPotential` | shield-plane potential (V) |
| `CathodePotential` | cathode potential (V) |
| `LArPermittivity`, `FR4Permittivity` | dielectric constants |

The two configs should agree on `pixelSize`, `pixelGap`, `chamfer_r`, and
`Npixels`, otherwise the drift and weighting geometries describe different
detectors.

## Usage

From the `test/` directory, after activating the venv:

```bash
cd test
bash test-full-3d-pixel.sh [STORE_DIR]
```

`STORE_DIR` defaults to `store/`. The two domain shapes are defined at the
top of the script and may be overridden via environment variables. A
typical full-z-extent run uses:

```bash
POCHOIR_DRIFT_SHAPE=44,44,1500 \
POCHOIR_WEIGHT_SHAPE=396,396,1500 \
bash test-full-3d-pixel.sh store
```

Defaults in the script (`44,44,300` / `396,396,300`) are tuned for fast
iteration; use the larger z-extent (`1500`) for production runs.

## Outputs

All artifacts land under `STORE_DIR` (default `test/store/`):

- `domain/{drift3d,weight3d}` — domain definitions.
- `initial/*`, `boundary/*` — generator outputs.
- `potential/{drift3d,weight3d}` — Laplace solutions.
- `velocity/drift3d` — drift velocity field.
- `starts/drift3d`, `paths/drift3d_tight` — drift starts and paths.
- `current/induced_current` — induced currents on the 4-pixel window.
- `pochoir_driftfield.log`, `pochoir_weightingfield.log` — solver logs.
- `maxerr_*.png`, `summary_log_*.pdf` — convergence diagnostics.

## Tweaking the geometry

The rounded-corner shape is implemented in `trimCorner` in
`pochoir/gen_pcb_drift_pixel_with_grid.py` and
`pochoir/gen_pcb_pixel_with_grid.py`. Both carve a quarter-disk of radius
`chamfer_r` (in grid-index units after dividing by the domain spacing) at
each inner corner. Increase `chamfer_r` in the JSON configs to round the
corners more strongly; set it to `0` for sharp 90° corners.
