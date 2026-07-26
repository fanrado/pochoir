#!/bin/bash
#
# Task10b: 2cm drift + weighting + induced-current run, NO shield grid,
# 3.5mm pixel / 0.9mm gap / 0.7mm chamfer, full-depth single-pass, uniform 0.1mm.
# ---------------------------------------------------------------------------
# Same wiring as run-task10a-drift-2cm-chamf07.sh but with the wider 0.9mm gap
# (3.5mm pad); the pixel pitch stays 4.4mm so the domains are unchanged.
# GridHoleShape "None" -> no shield grid.  chamferMode "dynamic": chamfer is
# specified in mm (0.7), converted to cells by the generator.
#
# ENFORCEMENT-FREE wiring: the ONLY insulator condition is the solver's Neumann
# BC, applied via --insulator on the fdm solve ONLY.  velo and drift get NO
# --insulator and velo gets NO --boundary, so the drift paths are pure
# v = mu * grad(phi) of the solved potential.
#
#   drift     -> potential/drift3d   (44x44x301   @0.1mm, per,per,fix)
#   weighting -> potential/weight3d  (220x220x301 @0.1mm, fix,fix,fix, 5x5 pixels)
#                (consumed by induce-pixel)
#
# Geometry at 0.1mm: pixel pitch 4.4mm (3.5mm pad + 0.9mm gap), pp_loweredge=99
# (z=9.9mm), FR4 no-flux slab 1 cell @z=9.9mm (gap cells only), copper pad 3
# cells @z=9.8-10.0mm (padThicknessCells=3), cathode @z=30mm (node 300).
# Cathode=Grid=-1000V over the 20mm LAr gap -> 50 V/mm bulk.  Total volume 3cm
# (z=0..30mm), drift length 2cm (pixel plane z=10mm .. cathode z=30mm).
#
# Starting points are CONFIG-DRIVEN: `pochoir starts -m yes` reads nGridPoints
# (10) and driftZDepth (29.9mm, last full-velocity node before the cathode) from
# the drift JSON -> a cell-centred 10x10 launch grid over the 4.4mm x 4.4mm
# pixel tile (spacing 0.44mm, first point at 0.22mm) = 100 paths, which
# induce-pixel's tiler replicates across the 5x5-pixel weighting field.
#
# Usage: ./run-task10b-drift-2cm-gap09-chamf07.sh [STORE_DIR]

set -e

export POCHOIR_STORE="${1:-store_task10b_drift_2cm_gap09_chamf07}"

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
dcfg="example_gen_pcb_drift_pixel_task10b_insul.json"

echo "=== drift field: no shield grid, 3.5mm pixel / 0.9mm gap / 0.7mm chamfer, no-flux FR4 insulator (NO epsilon) ==="

## Single full-depth drift domain: 44x44 periodic tile (4.4mm pitch @0.1mm),
## 301 deep (z=0..30mm).
want domain/drift_2cm \
     pochoir domain --domain domain/drift_2cm \
     --shape=44,44,301 --spacing '0.1*mm'

want "initial/drift_2cm boundary/drift_2cm" \
     pochoir gen --generator $gen --domain domain/drift_2cm \
     --initial initial/drift_2cm --boundary boundary/drift_2cm \
     $dcfg

## One full-depth Laplace solve, no interface anywhere.  per,per,fix matches the
## periodic pixel tile.  Tight precision (2e-11) for a clean E-field gradient.
## --insulator (Neumann BC) applied HERE ONLY (enforcement-free: not on velo/drift).
want "potential/drift3d increment/drift_2cm" \
     pochoir fdm \
     --nepochs 10 --epoch 130000000 --precision 0.00000002 \
     --edges per,per,fix \
     --engine torch \
     --initial initial/drift_2cm --boundary boundary/drift_2cm \
     --insulator initial/drift_2cm_insulator \
     --potential potential/drift3d \
     --increment increment/drift_2cm

date

############################################################################
## PART B: WEIGHTING FIELD  (full-depth single 0.1mm solve, 5x5 pixel grid)
############################################################################
export POCHOIR_LOG="${POCHOIR_STORE}/pochoir_weightingfield.log"

wgen="pcb_pixel_with_grid"
wcfg="example_gen_pixel_with_grid_task10b_insul.json"

echo "=== weighting field: 5x5 pixel grid, no-flux FR4 insulator + 3-cell grounded pad (NO epsilon) ==="

## Weighting domain: 220x220 (5 pixels x 4.4mm pitch @0.1mm, non-periodic so
## phi_w decays to 0 at the tile edges), 301 deep (z=0..30mm, cathode node 300).
want domain/weight_2cm \
     pochoir domain --domain domain/weight_2cm \
     --shape=220,220,301 --spacing '0.1*mm'

want "initial/weight_2cm boundary/weight_2cm" \
     pochoir gen --generator $wgen --domain domain/weight_2cm \
     --initial initial/weight_2cm --boundary boundary/weight_2cm \
     $wcfg

## Unit-probe solve: collecting (center) pixel = 1, all other electrodes = 0.
## fix,fix,fix + --multisteps no.  --insulator applies the SAME node-centered
## no-flux FR4 BC as the drift solve.  Precision 2e-10 (unit 0..1 range).
want "potential/weight3d increment/weight_2cm" \
     pochoir fdm \
     --nepochs 10 --epoch 130000000 --precision 0.0000000002 \
     --edges fix,fix,fix \
     --engine torch \
     --initial initial/weight_2cm --boundary boundary/weight_2cm \
     --insulator initial/weight_2cm_insulator \
     --potential potential/weight3d \
     --increment increment/weight_2cm \
     --multisteps no

date

############################################################################
## PART C: VELOCITY, PATHS, INDUCED CURRENT
############################################################################
## Uses the full-depth drift potential (potential/drift3d) and weighting field
## (potential/weight3d) solved above -- both native 0.1mm, no stitch.

echo "=== Velocities ==="
## Drift velocity from the full-depth drift potential.  ENFORCEMENT-FREE:
## NO --boundary and NO --insulator -> velocity is pure mu * grad(phi); velo
## runs only to carry the potential + temperature metadata drift consumes.
want velocity/drift3d \
     pochoir velo --temperature '87.0*K' \
     --potential potential/drift3d \
     --velocity velocity/drift3d

echo "=== Paths ==="
## CONFIG-DRIVEN starts (-m yes): nGridPoints=10 and driftZDepth=29.9 come from
## $dcfg -> cell-centred 10x10 grid over the 4.4mm tile (spacing 0.44mm), 100
## launch points at z=29.9mm, the last full-velocity node before the cathode
## (z=30mm, node 300).  This 10x10-per-pixel layout is what induce-pixel's tiler
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
     --paths paths/drift3d '0*us,40*us,0.05*us' \
     --plot

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
echo "=== DONE: Task10b 2cm 0.1mm drift + weighting + induced-current run complete (3.5mm/0.9mm/0.7mm chamfer, no shield grid) ==="
