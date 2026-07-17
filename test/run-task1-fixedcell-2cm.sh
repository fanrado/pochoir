#!/bin/bash
# ===========================================================================
# TASK 1 : 2 cm drift, 0.05 mm spacing, FIXED-CELL chamfer (verbatim for_pix).
#          Drift FIELD + drift PATHS only (no weighting, no induced current).
#          FR4/PCB disabled (GridHoleShape="None" -> pure Laplace).
#
# Geometry: total z = 30 mm, pixel plane at z = 10 mm, drift gap 20 mm,
#           cathode -1000 V at the top plane -> -50 V/mm.  Expected: CLEAN
#           (fixed-cell footprint shrinks at 0.05 mm -> no overshoot).
#
# Output folder: store_task1_fixedcell_2cm/  (paths + GEOMETRY_AND_CHAMFER.md)
# Run from the test/ directory:  ./run-task1-fixedcell-2cm.sh
# ===========================================================================
set -e
cd "$(dirname "$0")"
ROOT="$(cd .. && pwd)"
export PATH="$ROOT/env/bin:$PATH"
PY="$ROOT/env/bin/python"
STORE=store_task1_fixedcell_2cm
CFG=example_gen_pcb_drift_pixel_padz10_fixedcell.json

echo "### [1/3] geometry + actual-removed-material report -> $STORE/GEOMETRY_AND_CHAMFER.md"
$PY compute_chamfer_removed.py --store "$STORE" --mode fixed_cell --spacing 0.05 \
    --pad-z 10 --drift 20 --total 30 --cathode -1000 \
    --title "Task 1 -- 2 cm drift, 0.05 mm, fixed-cell chamfer (verbatim for_pix)"

echo "### [2/3] drift field + drift paths (near-field-only, 0.05 mm, z=0..30 mm)"
NSHAPE=88,88,601 NSPACING='0.05*mm' NLAUNCH=29.8 NTWIN=30 \
    ./run-near2cm-drift.sh "$STORE" "$CFG"

echo "### [3/3] overshoot check (pad plane z = 10 mm)"
{ echo; echo "## Drift-path result (overshoot check, pad z = 10 mm)"; echo '```'
  $PY OVERSHOOT_STUDY/check_paths.py "$STORE/paths/near2cm.npz" 10; echo '```'
} | tee -a "$STORE/GEOMETRY_AND_CHAMFER.md"
echo "=== TASK 1 DONE: $STORE (paths/near2cm.npz, drift_paths_3d.png, GEOMETRY_AND_CHAMFER.md) ==="
