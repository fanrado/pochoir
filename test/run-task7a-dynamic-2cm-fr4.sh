#!/bin/bash
# ===========================================================================
# TASK 7a : 2 cm drift, 0.05 mm spacing, DYNAMIC chamfer 0.7 mm, FR4 ACTIVE.
#           Drift FIELD + drift PATHS only.  Shield grid DISABLED
#           (GridHoleShape "None").
#
# Purpose: Same geometry as Task 6 (dynamic 0.7 mm chamfer, 2 cm drift) but
#          with the FR4 dielectric of the pixel-plane laminate ACTIVATED.
#          Within the 0.1 mm pixel-plane layer: top ~0.05 mm = PCB pad
#          conductor (as Task 6), bottom ~0.05 mm = FR4 insulator
#          (eps = FR4Permittivity).  LAr (eps = LArPermittivity) fills the
#          drift gap above the pad and the region below the FR4.
#
#          EPSILON DEACTIVATED (pochoir-44j2): the dielectric/displacement
#          formulation did not fix the gap overshoot (it worsened it -- the
#          high-eps slab pulls field lines into the FR4).  The FDM solve is
#          back to pure Laplace on E (uniform LAr): NEPS is left UNSET so no
#          --epsilon is passed to the solver.  gen still stores the epsilon
#          array (unused); re-add NEPS=1 below to reactivate for comparison.
#
# Geometry: total z = 30 mm, pixel plane at z = 10 mm, drift gap 20 mm,
#           cathode -1000 V -> -50 V/mm.
#
# Output folder: store_task7a_dynamic_2cm_fr4/
# Run from test/:  ./run-task7a-dynamic-2cm-fr4.sh
# ===========================================================================
set -e
cd "$(dirname "$0")"
ROOT="$(cd .. && pwd)"
export PATH="$ROOT/env/bin:$PATH"
PY="$ROOT/env/bin/python"
STORE=store_task7a_dynamic_2cm_fr4
CFG=example_gen_pcb_drift_pixel_padz10_chamf07_fr4.json

echo "### [1/3] geometry + actual-removed-material report -> $STORE/GEOMETRY_AND_CHAMFER.md"
$PY compute_chamfer_removed.py --store "$STORE" --mode dynamic --spacing 0.05 --chamfer-mm 0.7 \
    --pad-z 10 --drift 20 --total 30 --cathode -1000 \
    --title "Task 7a -- 2 cm drift, 0.05 mm, dynamic 0.7 mm chamfer, FR4 laminate active"

echo "### [2/3] drift field (plain Laplace, epsilon deactivated) + drift paths (near-field-only, 0.05 mm, z=0..30 mm)"
NSHAPE=88,88,601 NSPACING='0.05*mm' NLAUNCH=29.8 NTWIN=30 \
    ./run-near2cm-drift.sh "$STORE" "$CFG"

echo "### [3/3] overshoot check (pad plane z = 10 mm)"
{ echo; echo "## Drift-path result (overshoot check, pad z = 10 mm)"; echo '```'
  $PY OVERSHOOT_STUDY/check_paths.py "$STORE/paths/near2cm.npz" 10; echo '```'
} | tee -a "$STORE/GEOMETRY_AND_CHAMFER.md"
echo "=== TASK 7a DONE: $STORE ==="
