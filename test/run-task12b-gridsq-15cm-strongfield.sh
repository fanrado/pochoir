#!/bin/bash
#
# Task12b: 15cm drift + weighting + induced-current run WITH the square-hole
# shield grid, V2 = STRONG extraction-field grid bias, full-depth, uniform 0.1mm.
# ---------------------------------------------------------------------------
# Identical to Task12a (run-task12a-gridsq-15cm-transparent.sh) except for the
# grid bias / cathode pair:
#   * V2 grid bias: the pad->grid region carries TWICE the bulk field
#     (100 V/mm over the 1.5mm pad-to-grid gap) while the bulk (grid->cathode,
#     148.5mm) stays a uniform 50 V/mm:
#         GridPotential    = -100 V/mm * 1.5mm             =  -150V  (grid @ z=11.5mm)
#         CathodePotential = -150V - 50 V/mm * 148.5mm     = -7575V  (cathode @ z=160mm)
#     -> uniform 50 V/mm in the bulk, strong (2x) field between grid and pixel,
#     the standard operating condition for electron transmission through the grid.
#
# Geometry, wiring, and all other settings are as Task12a: 3.5mm pixel / 0.9mm
# gap / 0.7mm chamfer (pitch 4.4mm), square shield grid @ z=11.5mm (node 115),
# cathode @ z=160mm (node 1600), no-flux FR4 insulator via --insulator on the
# fdm solves ONLY (enforcement-free), NO dielectric constant anywhere.
#
#   drift     -> potential/drift3d   (44x44x1601   @0.1mm, per,per,fix)
#   weighting -> potential/weight3d  (220x220x1601 @0.1mm, fix,fix,fix, 5x5 pixels)
#
# The weighting field is BIAS-INDEPENDENT and thus identical to Task12a's; it is
# re-solved here so the store is self-contained (copy potential/weight3d.npz +
# increment/weight_15cm.npz from the Task12a store to skip it).
#
# GPU note: pochoir's torch engine hardcodes cuda:0 -- select the physical GPU
# with CUDA_VISIBLE_DEVICES (e.g. CUDA_VISIBLE_DEVICES=1 ./run-task12b-...).
#
# Usage: ./run-task12b-gridsq-15cm-strongfield.sh [STORE_DIR]

set -e

export POCHOIR_STORE="${1:-store_task12b_gridsq_15cm_strongfield}"

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
## PART A: DRIFT FIELD  (full-depth single 0.1mm solve, 15cm drift, shield grid)
############################################################################
export POCHOIR_LOG="${POCHOIR_STORE}/pochoir_driftfield.log"

gen="pcb_drift_pixel_with_grid"
dcfg="example_gen_pcb_drift_pixel_task12b_gridsq_insul.json"

echo "=== drift field: square shield grid @-150V (strong extraction), cathode -7575V, 3.5mm/0.9mm/0.7mm chamfer, no-flux FR4 insulator ==="

## Single full-depth drift domain: 44x44 periodic tile (4.4mm pitch @0.1mm),
## 1601 deep (z=0..160mm).
want domain/drift_15cm \
     pochoir domain --domain domain/drift_15cm \
     --shape=44,44,1601 --spacing '0.1*mm'

want "initial/drift_15cm boundary/drift_15cm" \
     pochoir gen --generator $gen --domain domain/drift_15cm \
     --initial initial/drift_15cm --boundary boundary/drift_15cm \
     $dcfg

## One full-depth Laplace solve.  per,per,fix matches the periodic pixel tile.
## --insulator (Neumann BC) applied HERE ONLY (enforcement-free: not on velo/drift).
want "potential/drift3d increment/drift_15cm" \
     pochoir fdm \
     --nepochs 10 --epoch 130000000 --precision 0.00000002 \
     --edges per,per,fix \
     --engine torch \
     --initial initial/drift_15cm --boundary boundary/drift_15cm \
     --insulator initial/drift_15cm_insulator \
     --potential potential/drift3d \
     --increment increment/drift_15cm

date

############################################################################
## PART B: WEIGHTING FIELD  (full-depth single 0.1mm solve, 5x5 pixels, shield grid)
############################################################################
export POCHOIR_LOG="${POCHOIR_STORE}/pochoir_weightingfield.log"

wgen="pcb_pixel_with_grid"
wcfg="example_gen_pixel_with_grid_task12_gridsq_insul.json"

echo "=== weighting field: 5x5 pixel grid, square shield grid @W=0, no-flux FR4 insulator + 3-cell grounded pad ==="

## Weighting domain: 220x220 (5 pixels x 4.4mm pitch @0.1mm, non-periodic so
## phi_w decays to 0 at the tile edges), 1601 deep (z=0..160mm, cathode node 1600).
want domain/weight_15cm \
     pochoir domain --domain domain/weight_15cm \
     --shape=220,220,1601 --spacing '0.1*mm'

want "initial/weight_15cm boundary/weight_15cm" \
     pochoir gen --generator $wgen --domain domain/weight_15cm \
     --initial initial/weight_15cm --boundary boundary/weight_15cm \
     $wcfg

## Unit-probe solve: fix,fix,fix + --multisteps no as in Task11.  --insulator
## applies the SAME node-centered no-flux FR4 BC as the drift solve.
want "potential/weight3d increment/weight_15cm" \
     pochoir fdm \
     --nepochs 10 --epoch 130000000 --precision 0.0000000002 \
     --edges fix,fix,fix \
     --engine torch \
     --initial initial/weight_15cm --boundary boundary/weight_15cm \
     --insulator initial/weight_15cm_insulator \
     --potential potential/weight3d \
     --increment increment/weight_15cm \
     --multisteps no

date

############################################################################
## PART C: VELOCITY, PATHS, INDUCED CURRENT
############################################################################

echo "=== Velocities ==="
## ENFORCEMENT-FREE: NO --boundary and NO --insulator -> velocity is pure
## mu * grad(phi).
want velocity/drift3d \
     pochoir velo --temperature '87.0*K' \
     --potential potential/drift3d \
     --velocity velocity/drift3d

echo "=== Paths ==="
## CONFIG-DRIVEN starts (-m yes): nGridPoints=10 and driftZDepth=159.9 come from
## $dcfg -> cell-centred 10x10 grid over the 4.4mm tile (spacing 0.44mm), 100
## launch points at z=159.9mm, the last full-velocity node before the cathode
## (z=160mm, node 1600).  This 10x10-per-pixel layout is what induce-pixel's
## tiler (_shift_paths_pixel_grid, npaths=10) expects.
want starts/drift3d \
     pochoir starts --starts starts/drift3d \
     -m yes \
     -c $dcfg \
     --plot

## ENFORCEMENT-FREE: NO --insulator on drift -> pure grad(phi) paths.
## --interp-order linear (cubic rings/overshoots near the pixel plane).
want paths/drift3d \
     pochoir drift --starts starts/drift3d \
     --velocity velocity/drift3d \
     --interp-order linear \
     --paths paths/drift3d '0*us,200*us,0.05*us' \
     --plot

echo "=== Induced currents ==="
## Induced current on the collecting pixel via Ramo.  --npixels 2 sums the
## target pixel + its first ring; --config supplies the pixel geometry.
want current/induced_current \
     pochoir induce-pixel --weighting potential/weight3d \
     --paths paths/drift3d \
     --output current/induced_current \
     --npixels 2 \
     --config "$wcfg" \
     --plot

date
echo "=== DONE: Task12b 15cm gridsq run complete (V2 strong extraction: grid -150V, cathode -7575V) ==="
