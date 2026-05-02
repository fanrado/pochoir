# Structure of `test-full-3d-pixel.sh`

End-to-end pipeline: builds drift field, weighting field, drift velocity, drift paths, and induced currents on pixel readout.

## 0. Setup

- `set -e` — abort on any error
- `POCHOIR_STORE` = first CLI arg, default `store` (all outputs go here)
- `source helpers.sh` — provides the `want` wrapper
  - `want <key> <cmd...>` runs `<cmd>` only if `<key>` is not already in `$POCHOIR_STORE` (caching/skip-if-exists)

---

## 1. Drift field

- `POCHOIR_LOG = $POCHOIR_STORE/pochoir_driftfield.log`

### 1a. Domain — `do_domain drift3d 44,44,500 '0.1*mm'`
- Calls `pochoir domain`
- Grid shape: 44 x 44 x 500 cells
- Spacing: 0.1 mm in each dim
- Output key: `domain/drift3d`

### 1b. Initial / boundary arrays — `do_gen drift3d quarter`
- Calls `pochoir gen`
- Generator: `pcb_drift_pixel_with_grid`
- Config file: `example_gen_pcb_drift_pixel_with_grid.json`
- Outputs: `initial/drift3d`, `boundary/drift3d`, plus `initial/drift3d_epsilon` (permittivity map)

### 1c. FDM solve — `do_fdm drift3d 20 1000000 0.00000002 per,per,fix`
- Calls `pochoir fdm` with engine = `torch`
- `nepochs = 20`
- `epoch = 1,000,000` steps per epoch
- `precision = 2e-8`
- `edges = per,per,fix` — periodic in x,y, fixed in z
- `--multisteps no`
- Inputs: `initial/drift3d`, `boundary/drift3d`
- Outputs: `potential/drift3d`, `increment/drift3d`

### 1d. Log parsing
- `python parse_maxerr.py <log> maxerr_driftfield.png summary_log_driftfield.pdf`
- Plots convergence (max error per epoch)

---

## 2. Weighting potential

- `POCHOIR_LOG = $POCHOIR_STORE/pochoir_weightingfield.log`
- `do_domain` and `do_gen` are **redefined** (shadow the drift-field versions)

### 2a. Domain — `do_domain weight3d 220,220,500 '0.1*mm'`
- 220 x 220 x 500 cells, 0.1 mm spacing

### 2b. Initial / boundary — `do_gen weight3d 3D`
- Generator: `pcb_pixel_with_grid` (note: weighting variant, not drift)
- Config: `example_gen_pixel_with_grid.json`

### 2c. FDM solve — `do_fdm weight3d 10 5000000 0.00000002 per,per,fix`
- `nepochs = 10`, `epoch = 5,000,000`, `precision = 2e-8`
- `edges = per,per,fix`
- Engine: torch, `multisteps no`
- Output: `potential/weight3d`

### 2d. Log parsing
- `parse_maxerr.py` on `store/pochoir_weightingfield.log` → `maxerr_weightingfield.png`, `summary_log_weightingfield.pdf`
- NOTE: hard-coded `store/...` (not `$POCHOIR_STORE/...`) — bug if store ≠ `store`

---

## 3. Drift velocity

- `pochoir velo --temperature '87.0*K'` (LAr temperature)
- Input: `potential/drift3d`
- Output: `velocity/drift3d`

---

## 4. Drift paths

### 4a. Starting points
- Grid configured: **10 x 10 = 100 starts per pixel**
- `dist = (0.22 0.66 1.1 1.54 1.98 2.42 2.86 3.3 3.74 4.18)` mm
  - Spacing: 0.44 mm (pixel pitch / 10)
  - First point centred at 0.22 mm (half-cell)
- Cartesian product → `points = (d_i*mm, d_j*mm, 48*mm)` for all i,j
  - z = 48 mm (drift start height)
- Alternative 5x5 (25 pts, 0.88 mm spacing, z = 29.6 mm) is commented out

### 4b. Generate starts — `pochoir starts -m yes ...`
- `-m yes` flag set
- Output: `starts/drift3d`

### 4c. Drift the paths — `pochoir drift`
- Inputs: `starts/drift3d`, `velocity/drift3d`
- Time grid: `0*us, 210*us, 0.05*us` (0–210 µs in 50 ns steps)
- Output: `paths/drift3d_tight`

---

## 5. Induced currents

- `pochoir induce-pixel`
- Inputs:
  - `--weighting potential/weight3d`
  - `--paths paths/drift3d_tight`
  - `--npixels 4` (induced on a 4-pixel neighbourhood)
- Output: `current/induced_current`

---

## Pipeline summary (data flow)

```
domain/drift3d ─┐
                ├─> initial,boundary/drift3d ─> potential/drift3d ─> velocity/drift3d ─┐
gen config ─────┘                                                                      │
                                                                            starts ────┤
                                                                                       ├─> paths/drift3d_tight ─┐
domain/weight3d ┐                                                                      │                        │
                ├─> initial,boundary/weight3d ─> potential/weight3d ───────────────────┴──────────────────────  ├─> current/induced_current
gen config ─────┘
```
