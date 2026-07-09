#!/bin/bash
# ===========================================================================
# TASK 5 : 30 cm drift, HYBRID solver (coarse 0.4 mm + fine 0.05 mm),
#          DYNAMIC (physical-unit) chamfer 0.4 mm, Npixels = 9.  Drift FIELD +
#          drift PATHS only (no weighting, no induced current).  FR4/PCB
#          disabled (GridHoleShape="None" -> pure Laplace).
#
# Wraps run-hybrid-30cm-drift.sh and adds the geometry/removed-material report
# and the overshoot check, matching the Task 1/2/3/4 runners.
#
# Geometry: pixel plane at z = 10 mm, drift gap 300 mm, cathode -15000 V ->
#           -50 V/mm.  Coarse 11x11x776 @0.4 mm (z=0..310), fine near
#           88x88x401 @0.05 mm (z=0..20), stitched 88x88x6201 @0.05 mm.
#           Npixels=9 is carried in the config; the transverse domain is a
#           single periodic pixel tile (4.4 mm), so it does not change shapes.
#
# Output folder: store_task5_hybrid_30cm/
# Run from test/:  ./run-task5-hybrid-30cm.sh
# ===========================================================================
set -e
cd "$(dirname "$0")"
ROOT="$(cd .. && pwd)"
export PATH="$ROOT/env/bin:$PATH"
PY="$ROOT/env/bin/python"
STORE=store_task5_hybrid_30cm
CFG=example_gen_pcb_drift_pixel_hybrid30cm_chamf04.json

echo "### [1/3] geometry + actual-removed-material report -> $STORE/GEOMETRY_AND_CHAMFER.md"
$PY compute_chamfer_removed.py --store "$STORE" --mode dynamic --spacing 0.05 --chamfer-mm 0.4 \
    --pad-z 10 --drift 300 --total 310 --cathode -15000 --hybrid --coarse-spacing 0.4 \
    --title "Task 5 -- 30 cm drift, hybrid (0.4+0.05 mm), dynamic 0.4 mm chamfer, Npixels=9"

echo "### [2/3] hybrid drift field + drift paths"
./run-hybrid-30cm-drift.sh "$STORE" "$CFG"

echo "### [3/3] overshoot check (pad plane z = 10 mm)"
{ echo; echo "## Drift-path result (overshoot check, pad z = 10 mm)"; echo '```'
  $PY OVERSHOOT_STUDY/check_paths.py "$STORE/paths/drift3d.npz" 10; echo '```'
} | tee -a "$STORE/GEOMETRY_AND_CHAMFER.md"
echo "=== TASK 5 DONE: $STORE ==="
