#!/bin/bash
# ===========================================================================
# TASK 2 : 10 cm drift, HYBRID solver (coarse 0.4 mm + fine 0.05 mm),
#          FIXED-CELL chamfer.  Drift FIELD + drift PATHS only (no weighting,
#          no induced current).  FR4/PCB disabled (pure Laplace).
#
# Wraps run-hybrid-10cm-drift.sh and adds the geometry/removed-material report
# and the overshoot check, matching the Task 1/3 runners.
#
# Geometry: pixel plane at z = 10 mm, drift gap 100 mm, cathode -5000 V ->
#           -50 V/mm.  Coarse 11x11x276 @0.4 mm (z=0..110), fine near
#           88x88x401 @0.05 mm (z=0..20), stitched 88x88x2201 @0.05 mm.
#
# Output folder: store_task2_hybrid_10cm/
# Run from test/:  ./run-task2-hybrid-10cm.sh
# ===========================================================================
set -e
cd "$(dirname "$0")"
ROOT="$(cd .. && pwd)"
export PATH="$ROOT/env/bin:$PATH"
PY="$ROOT/env/bin/python"
STORE=store_task2_hybrid_10cm
CFG=example_gen_pcb_drift_pixel_hybrid10cm_fixedcell.json

echo "### [1/3] geometry + actual-removed-material report -> $STORE/GEOMETRY_AND_CHAMFER.md"
$PY compute_chamfer_removed.py --store "$STORE" --mode fixed_cell --spacing 0.05 \
    --pad-z 10 --drift 100 --total 110 --cathode -5000 --hybrid --coarse-spacing 0.4 \
    --title "Task 2 -- 10 cm drift, hybrid (0.4+0.05 mm), fixed-cell chamfer"

echo "### [2/3] hybrid drift field + drift paths"
./run-hybrid-10cm-drift.sh "$STORE" "$CFG"

echo "### [3/3] overshoot check (pad plane z = 10 mm)"
{ echo; echo "## Drift-path result (overshoot check, pad z = 10 mm)"; echo '```'
  $PY OVERSHOOT_STUDY/check_paths.py "$STORE/paths/drift3d.npz" 10; echo '```'
} | tee -a "$STORE/GEOMETRY_AND_CHAMFER.md"
echo "=== TASK 2 DONE: $STORE ==="
