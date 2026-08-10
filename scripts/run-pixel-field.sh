#!/bin/bash

# Usage: ./run-pixel-field.sh [--hybrid yes|no] [STORE_DIR]
#
# Pixel drift + weighting fields, then the velo/starts/drift/induce chain.
# --hybrid yes runs the near/far hybrid solve (coarse + near + fine grids);
# --hybrid no runs ONE 0.1mm full-depth solve.  Both write the same store keys
# and, at this geometry, the same final lattice, so the two are comparable.
#
# WITH-GRID vs WITHOUT-GRID IS NOT A FLAG.  It follows entirely from which
# configs the SIZES block points at -- GridHoleShape in the JSON ("None" = no
# shield grid) is what the generators branch on.  That is exactly how
# run-for-largepix-wgrid.sh and run-for-larpix-v2a-wogrid.sh differ: same
# commands, different JSON filenames.
#
# ENFORCEMENT-FREE CONTRACT (non-negotiable, see PART B).  --insulator is
# applied to the FIELD SOLVE only.  velo gets neither --boundary nor
# --insulator, and drift gets no --insulator, so paths are never clamped or
# terminated at a surface -- doing so would break the curl-free E field.
# --interp-order linear is REQUIRED: cubic overshoots at the seam/pad-plane
# kink and over-focuses paths onto the pads.

set -e

# This script lives in scripts/ but the task13 configs live in test/, so cd to
# our own directory first and reach them as ../test/... below.  Same structure
# as run-for-larpix-v2a-wogrid.sh; without the cd, the relative config paths
# and helpers.sh would only resolve when invoked from scripts/.
cd "$(dirname "$0")"

HYBRID="yes"
if [ "$1" = "--hybrid" ] ; then
    HYBRID="$2" ; shift 2
fi
case "$HYBRID" in
    yes|no) ;;
    *) echo "ERROR: --hybrid must be yes or no, got '$HYBRID'" >&2 ; exit 1 ;;
esac

export POCHOIR_STORE="${1:-store_pixel_field_$HYBRID}"

source helpers.sh

# ---------------------------------------------------------------------------
# want: run a step only when its output(s) are missing (multi-key; identical to
# the override in run-task10-drift-2cm-01mm.sh so a re-run resumes cleanly).
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

############################################################################
## SIZES
############################################################################
## CONFIGS.  ../test/ because this script lives in scripts/ and we cd'd here:
## these are the Phase 1 files themselves, deliberately NOT copied into
## scripts/ -- a copy would fork and silently drift from the originals.
dcfg_coarse="../test/example_gen_pcb_drift_pixel_task13_coarse.json"
dcfg_fine="../test/example_gen_pcb_drift_pixel_task13_fine.json"
wcfg_coarse="../test/example_gen_pixel_with_grid_task13_coarse.json"
wcfg_fine="../test/example_gen_pixel_with_grid_task13_fine.json"

## SHAPES.  Explicit values, not derived: this table is the authority and
## field-solve is invoked with --domain no, so a typo here cannot be silently
## "corrected" into a different grid.
##
## Validation geometry: driftZDepth 69.9 rounds up to a whole 0.4mm coarse cell
## at 70.0mm full depth; 4.4mm pitch; Npixels 5; interface 40mm.
##
##   depth    70.0mm   -> 175 coarse cells (176 nodes), 700 fine (701 nodes)
##   near    0..40mm   -> 400 fine cells (401 nodes)
##   pitch     4.4mm   ->  11 coarse /  44 fine cells
##   probe  5*4.4=22mm ->  55 coarse / 220 fine cells
##
##                              drift        weighting
##   coarse 0.4mm, full depth   11,11,176    55,55,176
##   near   0.1mm, to interface 44,44,401    220,220,401
##   fine   0.1mm, full depth   44,44,701    220,220,701
##   single 0.1mm (--hybrid no) 44,44,701    220,220,701
##
## The single row and the fine row are the SAME grid ON PURPOSE -- that is what
## makes the two modes comparable on identical output lattices.  They are kept
## as two separate, labelled variables rather than deduplicated into one: they
## answer different questions, and collapsing them would hide the fact that
## their agreement is a deliberate choice rather than a coincidence.
d_coarse_shape="11,11,176"
d_near_shape="44,44,401"
d_fine_shape="44,44,701"
d_single_shape="44,44,701"

w_coarse_shape="55,55,176"
w_near_shape="220,220,401"
w_fine_shape="220,220,701"
w_single_shape="220,220,701"

interface='40*mm'
coarse_spacing=0.4
fine_spacing=0.1
spacing=0.1
precision=0.00000002

date
echo "=== mode: --hybrid $HYBRID, store $POCHOIR_STORE ==="

############################################################################
## PART A: DRIFT FIELD
############################################################################
export POCHOIR_LOG="${POCHOIR_STORE}/pochoir_driftfield.log"

## --insulator (the node-centered no-flux Neumann BC) is applied by field-solve
## to the SOLVE only -- never to velo or drift below.
if [ "$HYBRID" = yes ] ; then
    want potential/drift3d \
         pochoir field-solve --hybrid yes --field drift \
         --domain no \
         --coarse-config "$dcfg_coarse" --fine-config "$dcfg_fine" \
         --coarse-shape "$d_coarse_shape" \
         --near-shape "$d_near_shape" \
         --fine-shape "$d_fine_shape" \
         --interface "$interface" \
         --coarse-spacing "$coarse_spacing" --fine-spacing "$fine_spacing" \
         --precision "$precision"
else
    want potential/drift3d \
         pochoir field-solve --hybrid no --field drift \
         --domain no \
         --config "$dcfg_fine" \
         --shape "$d_single_shape" \
         --spacing "$spacing" \
         --precision "$precision"
fi

date

############################################################################
## PART B: VELOCITY, PATHS
############################################################################
## Uses the full-depth drift potential (potential/drift3d).  These three
## commands are copied VERBATIM from run-for-larpix-v2a-wogrid.sh -- do not
## "improve" them; the enforcement-free contract lives in exactly these flags.

echo "=== Velocities ==="
## Drift velocity from the full-depth drift potential.  ENFORCEMENT-FREE:
## NO --boundary and NO --insulator -> velocity is pure mu * grad(phi); velo
## runs only to carry the potential + temperature metadata drift consumes.
want velocity/drift3d \
     pochoir velo --temperature '87.0*K' \
     --potential potential/drift3d \
     --velocity velocity/drift3d

echo "=== Paths ==="
## CONFIG-DRIVEN starts (-m yes): nGridPoints=10
## (_shift_paths_pixel_grid, npaths=10) expects.
want starts/drift3d \
     pochoir starts --starts starts/drift3d \
     -m yes \
     -c $dcfg_fine \
     --plot

## ENFORCEMENT-FREE: NO --insulator on drift -> pure grad(phi) paths.
## --interp-order linear: cubic rings/overshoots near the pixel plane and
## over-focuses paths onto the pads; linear is monotone-safe.
want paths/drift3d \
     pochoir drift --starts starts/drift3d \
     --velocity velocity/drift3d \
     --interp-order linear \
     --paths paths/drift3d '0*us,200*us,0.05*us' \
     --plot

############################################################################
## PART C: WEIGHTING FIELD
############################################################################
export POCHOIR_LOG="${POCHOIR_STORE}/pochoir_weightingfield.log"

if [ "$HYBRID" = yes ] ; then
    want potential/weight3d \
         pochoir field-solve --hybrid yes --field weighting \
         --domain no \
         --coarse-config "$wcfg_coarse" --fine-config "$wcfg_fine" \
         --coarse-shape "$w_coarse_shape" \
         --near-shape "$w_near_shape" \
         --fine-shape "$w_fine_shape" \
         --interface "$interface" \
         --coarse-spacing "$coarse_spacing" --fine-spacing "$fine_spacing" \
         --precision "$precision"
else
    want potential/weight3d \
         pochoir field-solve --hybrid no --field weighting \
         --domain no \
         --config "$wcfg_fine" \
         --shape "$w_single_shape" \
         --spacing "$spacing" \
         --precision "$precision"
fi

date

echo "=== Induced currents ==="
## Induced current on the collecting pixel via Ramo (weighting field x drift
## paths).  --npixels 2 sums the target pixel + its first ring; --config supplies
## the pixel geometry for the pad-collection map (same geometry as $wcfg_fine).
want current/induced_current \
     pochoir induce-pixel --weighting potential/weight3d \
     --paths paths/drift3d \
     --output current/induced_current \
     --npixels 2 \
     --config "$wcfg_fine" \
     --plot

date

echo "=== DONE ==="
