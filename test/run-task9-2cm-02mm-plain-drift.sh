#!/bin/bash
#
# Task9: 2cm fine-spacing plain-Laplace drift control (pochoir-5hrx).
# ---------------------------------------------------------------------------
# Purpose
# -------
# A fine-resolution PLAIN-LAPLACE drift control (NO FR4, NO no-flux/Neumann
# insulator boundary) to see the bare drift-path behaviour near the pixel gap
# and, crucially, BELOW the pad.  With no insulator surface to terminate paths
# at z=10mm, electrons launched over the inter-pad gap are free to dip below the
# pad plane; this run resolves that dip at fine (0.02mm) field spacing.
#
# Contrast with the Neumann runs (run-validate-neumann-fulldepth-01mm.sh,
# run-task8-nodegrid-drift.sh): those activate the no-flux FR4 insulator so all
# paths terminate at the FR4 surface.  Here enableFR4=false and
# enableInsulatorFR4=false => pure Laplace, no termination, below-pad visible.
#
# Geometry (example_gen_pcb_drift_pixel_2cm_02mmspacing.json):
#   * Pad low edge at z=10.0mm (pixelPlaneWidth 0.1 -> pad occupies 10.0-10.1mm).
#   * ~10mm below-pad region (z=0..10mm) so paths dipping below the pad show.
#   * 2cm (20mm) drift gap above the pad to the cathode (pad top 10.1 -> 30.1mm).
#   * Cathode & grid at -1000V, pad grounded -> 1000V/20mm = 50 V/mm bulk field.
#
# Spacing: 0.02mm.  (0.01mm was requested but a 30mm domain at 0.01mm is
# 440x440x3000 ~= 33GB > 24GB GPU; coarsened to 0.02mm per user =>
# 220x220x1506 ~= 4GB, fits one card.)  Single full-depth torch FDM solve on
# ONE GPU -- NO coarse/near/far/stitch/coarsen hybrid machinery.
#
# Drift only: domain -> gen -> fdm -> velo -> starts -> drift.
#   NO weighting field, NO induced current.
#
# This runs in parallel with the CPU task8 node-grid drift run
# (run-task8-nodegrid-drift.sh) -- it must NOT be stopped; this driver uses the
# GPU (torch) for the FDM solve and its own store, so the two do not collide.
#
# Usage: ./run-task9-2cm-02mm-plain-drift.sh [STORE_DIR]
#   STORE_DIR defaults to store_task9_2cm_02mm_plain.

set -e

export POCHOIR_STORE="${1:-store_task9_2cm_02mm_plain}"

source helpers.sh

# ---------------------------------------------------------------------------
# want: run a step only when its output(s) are missing (same resume-clean guard
# as the validation / near2cm drivers).
# ---------------------------------------------------------------------------
want () {
    local targets="$1" ; shift
    local t miss=0
    for t in $targets ; do
        [ -f "$POCHOIR_STORE/${t}.npz" ] || miss=1
    done
    if [ "$miss" -eq 0 ] ; then
        echo "have $targets"
        return
    fi
    echo "$@"
    "$@"
    for t in $targets ; do
        if [ ! -f "$POCHOIR_STORE/${t}.npz" ] ; then
            echo "ERROR: step did not produce expected output $t" >&2
            exit 1
        fi
    done
    echo "made $targets"
}

date
export POCHOIR_LOG="${POCHOIR_STORE}/pochoir_task9_2cm_02mm.log"

gen="pcb_drift_pixel_with_grid"
cfg="example_gen_pcb_drift_pixel_2cm_02mmspacing.json"
echo "=== Task9 2cm plain-Laplace drift: store=$POCHOIR_STORE cfg=$cfg ==="

############################################################################
## DRIFT FIELD  (single full-depth 0.02mm plain-Laplace solve, one GPU)
############################################################################
## One periodic pixel tile, 4.4mm pitch @0.02mm = 220x220, 30.1mm deep = 1506
## (z = 0..30.1mm: 10mm below pad + pad + 20mm gap to cathode).
want domain/drift_2cm \
     pochoir domain --domain domain/drift_2cm \
     --shape=220,220,1506 --spacing '0.02*mm'

want "initial/drift_2cm boundary/drift_2cm" \
     pochoir gen --generator $gen --domain domain/drift_2cm \
     --initial initial/drift_2cm --boundary boundary/drift_2cm \
     $cfg

## Plain Laplace: NO --insulator, NO --epsilon (config enableFR4=false,
## enableInsulatorFR4=false).  per,per,fix matches the periodic pixel tile.
## Tight precision (2e-11) for a clean E-field gradient.
want "potential/drift3d increment/drift_2cm" \
     pochoir fdm \
     --nepochs 10 --epoch 130000000 --precision 0.00000000002 \
     --edges per,per,fix \
     --engine torch \
     --initial initial/drift_2cm --boundary boundary/drift_2cm \
     --potential potential/drift3d \
     --increment increment/drift_2cm
date

############################################################################
## DRIFT VELOCITY
############################################################################
echo "=== Velocity (from potential/drift3d) ==="
want velocity/drift3d \
     pochoir velo --temperature '87.0*K' \
     --potential potential/drift3d \
     --boundary boundary/drift_2cm \
     --velocity velocity/drift3d

############################################################################
## STARTS  (launch grid spanning one pitch, just below the cathode)
############################################################################
## 45x45 grid at 0.1mm over the full 4.4mm pitch (x,y = 0.0..4.4mm), launched at
## z=30.0mm (just below the cathode top at 30.1mm, still full drift velocity).
## Spanning the whole pitch samples pad-centre, pad-edge and inter-pad gap so
## the below-pad dip over the gap is resolved.  Matches the task8 launch grid
## for a direct plain-Laplace-vs-Neumann comparison.
echo "=== Starts: 45x45 launch grid at z=30.0mm ==="
dist=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.0 3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9 4.0 4.1 4.2 4.3 4.4)
points=()
for d in "${dist[@]}"; do
     for d2 in "${dist[@]}"; do
         points+=("${d}*mm,${d2}*mm,30.0*mm")
     done
done

want starts/drift3d \
     pochoir starts --starts starts/drift3d \
     -m no \
     -c $cfg "${points[@]}" \
     --plot

############################################################################
## DRIFT PATHS  (linear interp; NO insulator termination -> below-pad visible)
############################################################################
## 40us window covers the ~19us bulk transit plus below-pad excursions.
echo "=== Drift paths (plain Laplace, below-pad free) ==="
want paths/drift3d \
     pochoir drift --starts starts/drift3d \
     --velocity velocity/drift3d \
     --interp-order linear \
     --paths paths/drift3d '0*us,40*us,0.05*us' \
     --plot
# --interp-order linear: cubic rings/overshoots near the pad plane; linear is
# monotone-safe (see cubic-drift-overshoot diagnosis).

date
echo "=== DONE: Task9 2cm plain-Laplace drift-path run complete ==="
echo "    potential: $POCHOIR_STORE/potential/drift3d.npz"
echo "    paths:     $POCHOIR_STORE/paths/drift3d.npz"
