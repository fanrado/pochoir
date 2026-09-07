#!/bin/bash

# Usage: ./run-task10b-drift-2cm-gap09-chamf07.sh [STORE_DIR]
export TMPDIR="/nfs/data/1/rrazakami/tmp"
set -e

export POCHOIR_STORE="${1:-edge_bc_validation/store_edge_bc_true_fixed_0p1mm}"

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

date

############################################################################
## PART A: DRIFT FIELD  (full-depth single 0.1mm solve, 2cm drift / 3cm volume)
############################################################################
export POCHOIR_LOG="${POCHOIR_STORE}/pochoir_driftfield.log"

gen="pcb_drift_pixel_with_grid"
dcfg="example_gen_pcb_drift_pixel_larpix_v2a_wogrid.json"


## Single full-depth drift domain: 44x44 periodic tile = EXACTLY the 4.4mm pitch
## @0.1mm.  At this spacing the true LArPix lengths need no snapping:
## pixelSize 3.5 (35 cells) + pixelGap 0.9 (9 cells) = 44 cells = 4.4mm.
## 301 planes -> 300 cells x 0.1mm = 30.0mm; the cathode is on the last plane
## (index 300, z=30mm) and the path start sits at driftZDepth=29.9mm (index
## 299), just below it.  Pixel plane low edge at 9.9mm -> 2cm drift length.
want domain/drift \
     pochoir domain --domain domain/drift \
     --shape=44,44,301 --spacing '0.1*mm'

want "initial/drift boundary/drift" \
     pochoir gen --generator $gen --domain domain/drift \
     --initial initial/drift --boundary boundary/drift \
     $dcfg

## One full-depth Laplace solve, no interface anywhere.  per,per,fix matches the
## periodic pixel tile.  Tight precision (2e-11) for a clean E-field gradient.
## --insulator (Neumann BC) applied HERE ONLY (enforcement-free: not on velo/drift).
want "potential/drift3d increment/drift" \
     pochoir fdm \
     --nepochs 10 --epoch 130000000 --precision 0.000000000001 \
     --edges per,per,fix \
     --engine torch \
     --initial initial/drift --boundary boundary/drift \
     --insulator initial/drift_insulator \
     --potential potential/drift3d \
     --increment increment/drift

date

############################################################################
## PART B: VELOCITY, PATHS, INDUCED CURRENT
############################################################################
## Uses the full-depth drift potential (potential/drift3d) and weighting field

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
     -c $dcfg \
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
## PART C: WEIGHTING FIELD  (full-depth single 0.1mm solve, 5x5 pixel grid)
############################################################################
export POCHOIR_LOG="${POCHOIR_STORE}/pochoir_weightingfield.log"

wgen="pcb_pixel_with_grid"
wcfg="example_gen_pixel_larpix_v2a_wogrid.json"


## Weighting domain: exactly 5x the drift domain transversely --
## 5 x 44 = 220 planes @0.1mm, i.e. 5 pixels x 4.4mm pitch = 22.0mm, the full
## 5x5 pad array with no padding.  Same z extent as the drift domain (301).
want domain/weight \
     pochoir domain --domain domain/weight \
     --shape=220,220,301 --spacing '0.1*mm'

want "initial/weight boundary/weight" \
     pochoir gen --generator $wgen --domain domain/weight \
     --initial initial/weight --boundary boundary/weight \
     $wcfg

## Unit-probe solve: collecting (center) pixel = 1, all other electrodes = 0.
## fix,fix,fix + --multisteps no.  --insulator applies the SAME node-centered
## no-flux FR4 BC as the drift solve.
##
## PRECISION 1e-12 (was 2e-8).  The stopping test is on the PER-ITERATION change,
## which goes small for the long-wavelength (transversely uniform) mode long
## before that mode has actually relaxed, so 2e-8 stops with the far-field W
## under-converged.  The earlier converged 0.1mm/16cm solve (13.46M iterations, 30.1h) proved
## this: its far tail rose 8x at z=88mm and landed within a few percent of the
## coarse run.  Keep 1e-12 here too so the long-wavelength mode is fully relaxed.
## Revert to 0.00000002 only to reproduce the old (early-stopping) runs.
want "potential/weight3d increment/weight" \
     pochoir fdm \
     --nepochs 10 --epoch 130000000 --precision 0.000000000001 \
     --edges fix,fix,fix \
     --true-fixed-edges \
     --engine torch \
     --initial initial/weight --boundary boundary/weight \
     --insulator initial/weight_insulator \
     --potential potential/weight3d \
     --increment increment/weight \
     --multisteps no

date



echo "=== Induced currents ==="
## Induced current on the collecting pixel via Ramo (weighting field x drift
## paths).  --npixels 2 sums the target pixel + its first ring; --config supplies
## the pixel geometry for the pad-collection map (same geometry as $wcfg).
want current/induced_current \
     pochoir induce-pixel --weighting potential/weight3d \
     --paths paths/drift3d \
     --output current/induced_current \
     --npixels 2 \
     --config "$wcfg" \
     --plot

date

echo "=== DONE ==="
