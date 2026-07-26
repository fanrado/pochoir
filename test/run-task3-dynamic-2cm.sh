#!/bin/bash
# ===========================================================================
# TASK 3 : 2 cm drift, 0.05 mm spacing, DYNAMIC (physical-unit) chamfer 0.4 mm.
#          Drift FIELD + drift PATHS only.  FR4/PCB disabled (pure Laplace).
#
# Purpose (confirmed): EQUIVALENCE PROOF.  At 0.05 mm the dynamic 0.4 mm
#          chamfer removes the byte-identical footprint of the fixed-cell
#          chamfer (Task 1) -- 0 differing boundary cells, 5744 conductor
#          cells -- so the drift paths are the SAME and CLEAN (no overshoot).
#          It expresses the same corner in physical units, making the footprint
#          spacing-independent.  (The 0.7 mm chamfer is the one that overshoots
#          at 0.05 mm; see OVERSHOOT_STUDY/chamfer_spacing_study.md.)
#
# Geometry: total z = 30 mm, pixel plane at z = 10 mm, drift gap 20 mm,
#           cathode -1000 V -> -50 V/mm.
#
# Output folder: store_task3_dynamic_2cm/
# Run from test/:  ./run-task3-dynamic-2cm.sh
# ===========================================================================
set -e
cd "$(dirname "$0")"
ROOT="$(cd .. && pwd)"
export PATH="$ROOT/env/bin:$PATH"
PY="$ROOT/env/bin/python"
STORE=store_task3_dynamic_2cm
CFG=example_gen_pcb_drift_pixel_padz10_chamf04.json

echo "### [1/3] geometry + actual-removed-material report -> $STORE/GEOMETRY_AND_CHAMFER.md"
$PY compute_chamfer_removed.py --store "$STORE" --mode dynamic --spacing 0.05 --chamfer-mm 0.4 \
    --pad-z 10 --drift 20 --total 30 --cathode -1000 \
    --title "Task 3 -- 2 cm drift, 0.05 mm, dynamic 0.4 mm chamfer (equivalence proof)"

echo "### [2/3] drift field + drift paths (near-field-only, 0.05 mm, z=0..30 mm)"
NSHAPE=88,88,601 NSPACING='0.05*mm' NLAUNCH=29.8 NTWIN=30 \
    ./run-near2cm-drift.sh "$STORE" "$CFG"

echo "### [3/3] overshoot check (pad plane z = 10 mm)"
{ echo; echo "## Drift-path result (overshoot check, pad z = 10 mm)"; echo '```'
  $PY OVERSHOOT_STUDY/check_paths.py "$STORE/paths/near2cm.npz" 10; echo '```'
} | tee -a "$STORE/GEOMETRY_AND_CHAMFER.md"
echo "=== TASK 3 DONE: $STORE ==="
