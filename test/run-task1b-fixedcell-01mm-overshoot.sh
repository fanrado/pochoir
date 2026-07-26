#!/bin/bash
# ===========================================================================
# TASK 1b (companion): SAME fixed-cell chamfer, but at 0.1 mm spacing.
#          This is the case where the for_pix fixed-cell chamfer OVERSHOOTS,
#          because its physical corner footprint is 2x larger than at 0.05 mm.
#          Together with Task 1 (0.05 mm, clean) this shows "overshoot with /
#          without" the shrinking fixed-cell footprint, using the REAL copied
#          for_pix trimCorner.
#
# Geometry: total z = 30 mm, pixel plane at z = 10 mm, drift gap 20 mm,
#           cathode -1000 V -> -50 V/mm.  Expected: OVERSHOOT (z_min < pad).
#
# Output folder: store_task1_fixedcell_01mm/
# Run from test/:  ./run-task1b-fixedcell-01mm-overshoot.sh
# ===========================================================================
set -e
cd "$(dirname "$0")"
ROOT="$(cd .. && pwd)"
export PATH="$ROOT/env/bin:$PATH"
PY="$ROOT/env/bin/python"
STORE=store_task1_fixedcell_01mm
CFG=example_gen_pcb_drift_pixel_padz10_fixedcell.json

echo "### [1/3] geometry + actual-removed-material report -> $STORE/GEOMETRY_AND_CHAMFER.md"
$PY compute_chamfer_removed.py --store "$STORE" --mode fixed_cell --spacing 0.10 \
    --pad-z 10 --drift 20 --total 30 --cathode -1000 \
    --title "Task 1b -- 2 cm drift, 0.1 mm, fixed-cell chamfer (OVERSHOOT case)"

echo "### [2/3] drift field + drift paths (near-field-only, 0.1 mm, z=0..30 mm)"
NSHAPE=44,44,301 NSPACING='0.1*mm' NLAUNCH=29.8 NTWIN=30 \
    ./run-near2cm-drift.sh "$STORE" "$CFG"

echo "### [3/3] overshoot check (pad plane z = 10 mm)"
{ echo; echo "## Drift-path result (overshoot check, pad z = 10 mm)"; echo '```'
  $PY OVERSHOOT_STUDY/check_paths.py "$STORE/paths/near2cm.npz" 10; echo '```'
} | tee -a "$STORE/GEOMETRY_AND_CHAMFER.md"
echo "=== TASK 1b DONE: $STORE ==="
