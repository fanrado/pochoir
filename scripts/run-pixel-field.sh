#!/bin/bash

# Pixel drift + weighting fields, then the velo/starts/drift/induce chain.
#
# Usage: ./run-pixel-field.sh --hybrid yes|no [options]
#
#   --hybrid yes|no        REQUIRED.  yes: near/far hybrid solve (coarse+near+
#                          fine grids).  no: one 0.1mm full-depth solve.
#   --store DIR            store directory (def: store_pixel_field_<mode>)
#   --drift-config F       fine/single drift config JSON
#   --drift-coarse-config F   coarse drift config JSON      (--hybrid yes only)
#   --weight-config F      fine/single weighting config JSON
#   --weight-coarse-config F  coarse weighting config JSON  (--hybrid yes only)
#
# WITH-GRID vs WITHOUT-GRID IS NOT A FLAG.  It follows entirely from which
# configs you point this script at -- GridHoleShape in the JSON ("None" = no
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

HYBRID=""
STORE=""
DCFG_FINE="test/example_gen_pcb_drift_pixel_task13_fine.json"
DCFG_COARSE="test/example_gen_pcb_drift_pixel_task13_coarse.json"
WCFG_FINE="test/example_gen_pixel_with_grid_task13_fine.json"
WCFG_COARSE="test/example_gen_pixel_with_grid_task13_coarse.json"

while [ $# -gt 0 ] ; do
    case "$1" in
        --hybrid)               HYBRID="$2" ; shift 2 ;;
        --store)                STORE="$2" ; shift 2 ;;
        --drift-config)         DCFG_FINE="$2" ; shift 2 ;;
        --drift-coarse-config)  DCFG_COARSE="$2" ; shift 2 ;;
        --weight-config)        WCFG_FINE="$2" ; shift 2 ;;
        --weight-coarse-config) WCFG_COARSE="$2" ; shift 2 ;;
        -h|--help)              sed -n '3,27p' "$0" ; exit 0 ;;
        *) echo "ERROR: unknown option $1" >&2 ; exit 1 ;;
    esac
done

case "$HYBRID" in
    yes|no) ;;
    "") echo "ERROR: --hybrid yes|no is required" >&2 ; exit 1 ;;
    *)  echo "ERROR: --hybrid must be yes or no, got '$HYBRID'" >&2 ; exit 1 ;;
esac

export POCHOIR_STORE="${STORE:-store_pixel_field_$([ "$HYBRID" = yes ] && echo hybrid || echo single)}"

source "$(dirname "$0")/helpers.sh"

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
## Explicit shapes, not derived: this table is the authority and field-solve is
## invoked with --domain no so a typo here cannot be silently "corrected".
##
## Validation geometry: driftZDepth 69.9 rounds up to a whole 0.4mm coarse cell
## at 70.0mm full depth; 4.4mm pitch; Npixels 5; interface 40mm.
##
##   depth   70.0mm  -> 175 coarse cells (176 nodes), 700 fine cells (701 nodes)
##   near   0..40mm  -> 400 fine cells (401 nodes)
##   pitch    4.4mm  ->  11 coarse /  44 fine cells
##   probe 5*4.4=22mm ->  55 coarse / 220 fine cells
##
##                             drift        weighting
##   coarse 0.4mm, full depth  11,11,176    55,55,176
##   near   0.1mm, to interface 44,44,401   220,220,401
##   fine   0.1mm, full depth   44,44,701   220,220,701
##   single 0.1mm (--hybrid no) 44,44,701   220,220,701
##
## The single shape and the hybrid FINE shape are deliberately the SAME grid --
## that is what makes the two modes comparable on identical output lattices.
D_COARSE_SHAPE="11,11,176"
D_NEAR_SHAPE="44,44,401"
D_FINE_SHAPE="44,44,701"
D_SINGLE_SHAPE="44,44,701"

W_COARSE_SHAPE="55,55,176"
W_NEAR_SHAPE="220,220,401"
W_FINE_SHAPE="220,220,701"
W_SINGLE_SHAPE="220,220,701"

INTERFACE='40*mm'
COARSE_SPACING=0.4
FINE_SPACING=0.1
SPACING=0.1
PRECISION=0.00000002

date
echo "=== mode: --hybrid $HYBRID, store $POCHOIR_STORE ==="

############################################################################
## PART A: DRIFT FIELD
############################################################################
export POCHOIR_LOG="${POCHOIR_STORE}/pochoir_driftfield.log"

## --insulator (Neumann BC) is applied by field-solve to the SOLVE only.
if [ "$HYBRID" = yes ] ; then
    want potential/drift3d \
         pochoir field-solve --hybrid yes --field drift \
         --domain no \
         --coarse-config "$DCFG_COARSE" --fine-config "$DCFG_FINE" \
         --coarse-shape "$D_COARSE_SHAPE" \
         --near-shape "$D_NEAR_SHAPE" \
         --fine-shape "$D_FINE_SHAPE" \
         --interface "$INTERFACE" \
         --coarse-spacing "$COARSE_SPACING" --fine-spacing "$FINE_SPACING" \
         --precision "$PRECISION"
else
    want potential/drift3d \
         pochoir field-solve --hybrid no --field drift \
         --domain no \
         --config "$DCFG_FINE" \
         --shape "$D_SINGLE_SHAPE" \
         --spacing "$SPACING" \
         --precision "$PRECISION"
fi

date

############################################################################
## PART B: VELOCITY, PATHS
############################################################################
## Uses the full-depth drift potential (potential/drift3d).  The four commands
## below are copied VERBATIM from run-for-larpix-v2a-wogrid.sh -- do not
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
     -c $DCFG_FINE \
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
         --coarse-config "$WCFG_COARSE" --fine-config "$WCFG_FINE" \
         --coarse-shape "$W_COARSE_SHAPE" \
         --near-shape "$W_NEAR_SHAPE" \
         --fine-shape "$W_FINE_SHAPE" \
         --interface "$INTERFACE" \
         --coarse-spacing "$COARSE_SPACING" --fine-spacing "$FINE_SPACING" \
         --precision "$PRECISION"
else
    want potential/weight3d \
         pochoir field-solve --hybrid no --field weighting \
         --domain no \
         --config "$WCFG_FINE" \
         --shape "$W_SINGLE_SHAPE" \
         --spacing "$SPACING" \
         --precision "$PRECISION"
fi

date

echo "=== Induced currents ==="
## Induced current on the collecting pixel via Ramo (weighting field x drift
## paths).  --npixels 2 sums the target pixel + its first ring; --config supplies
## the pixel geometry for the pad-collection map (same geometry as $WCFG_FINE).
want current/induced_current \
     pochoir induce-pixel --weighting potential/weight3d \
     --paths paths/drift3d \
     --output current/induced_current \
     --npixels 2 \
     --config "$WCFG_FINE" \
     --plot

date

echo "=== DONE ==="
