#!/bin/bash
#
# Task12d: 30cm drift + weighting + induced-current run WITH the square-hole
# shield grid, V4 = very strong grid bias at a LONG (30cm) drift, uniform 0.1mm.
# ---------------------------------------------------------------------------
# Extends the grid-bias scan (12a -75V, 12b -150V, 12c -500V) to a 30cm drift
# and a very strong grid bias:
#   * grid bias: GridPotential = -1000V (grid @ z=11.5mm) -> the pad->grid region
#     carries ~667 V/mm over the 1.5mm pad-to-grid gap (~13x the bulk field);
#   * bulk kept uniform 50 V/mm over the grid->cathode span (298.5mm):
#         CathodePotential = -1000V - 50 V/mm * 298.5mm = -15925V  (cathode @ z=310mm)
#
# Geometry as Task12a/b/c: 3.5mm pixel / 0.9mm gap / 0.7mm chamfer (pitch 4.4mm),
# square shield grid @ z=11.5mm (node 115), 5x5 pixel weighting grid (Npixels=5),
# no-flux FR4 insulator via --insulator on the fdm solves ONLY (enforcement-free),
# NO dielectric constant.  The ONLY difference vs 12c is the deeper domain and the
# grid/cathode pair:
#         driftZDepth = 309.9mm      (launch node, one cell below the cathode)
#         cathode     @ z=310mm      (domain last plane, node 3100)
#
#   drift     -> potential/drift3d   (44x44x3101   @0.1mm, per,per,fix)
#   weighting -> potential/weight3d  (220x220x3101 @0.1mm, fix,fix,fix, 5x5 pixels)
#
# The weighting field is BIAS-INDEPENDENT (grid held at W=0), but its DOMAIN is
# deeper here, so it is solved fresh (cannot reuse the 15cm store).
#
# NOTE: this is a HEAVY run -- the 3101-deep drift and weighting FDM solves plus
# the 30cm (0..400us) path integration take substantially longer than the 15cm
# tasks.  It relies on the pochoir-9rjv periodic-seam wrap fix: the 667 V/mm
# extraction field would otherwise strand near-pad-center paths at the tile seam.
#
# GPU note: pochoir's torch engine hardcodes cuda:0 -- select the physical GPU
# with CUDA_VISIBLE_DEVICES (e.g. CUDA_VISIBLE_DEVICES=1 ./run-task12d-...).
#
# Usage: ./run-task12d-gridsq-30cm.sh [STORE_DIR]

set -e

export POCHOIR_STORE="${1:-store_task12d_gridsq_30cm}"

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
## PART A: DRIFT FIELD  (full-depth single 0.1mm solve, 30cm drift, shield grid)
############################################################################
export POCHOIR_LOG="${POCHOIR_STORE}/pochoir_driftfield.log"

gen="pcb_drift_pixel_with_grid"
dcfg="example_gen_pcb_drift_pixel_task12d_gridsq_insul.json"

echo "=== drift field: square shield grid @-1000V (~667 V/mm pad-grid), cathode -15925V, 30cm drift, 3.5mm/0.9mm/0.7mm chamfer, no-flux FR4 insulator ==="

## Single full-depth drift domain: 44x44 periodic tile (4.4mm pitch @0.1mm),
## 3101 deep (z=0..310mm).
want domain/drift_30cm \
     pochoir domain --domain domain/drift_30cm \
     --shape=44,44,3101 --spacing '0.1*mm'

want "initial/drift_30cm boundary/drift_30cm" \
     pochoir gen --generator $gen --domain domain/drift_30cm \
     --initial initial/drift_30cm --boundary boundary/drift_30cm \
     $dcfg

## One full-depth Laplace solve.  per,per,fix matches the periodic pixel tile.
## --insulator (Neumann BC) applied HERE ONLY (enforcement-free: not on velo/drift).
want "potential/drift3d increment/drift_30cm" \
     pochoir fdm \
     --nepochs 10 --epoch 130000000 --precision 0.00000002 \
     --edges per,per,fix \
     --engine torch \
     --initial initial/drift_30cm --boundary boundary/drift_30cm \
     --insulator initial/drift_30cm_insulator \
     --potential potential/drift3d \
     --increment increment/drift_30cm

date

############################################################################
## PART B: WEIGHTING FIELD  (full-depth single 0.1mm solve, 5x5 pixels, shield grid)
############################################################################
export POCHOIR_LOG="${POCHOIR_STORE}/pochoir_weightingfield.log"

wgen="pcb_pixel_with_grid"
wcfg="example_gen_pixel_with_grid_task12d_gridsq_insul.json"

echo "=== weighting field: 5x5 pixel grid, square shield grid @W=0, no-flux FR4 insulator + 3-cell grounded pad (30cm) ==="

## Weighting domain: 220x220 (5 pixels x 4.4mm pitch @0.1mm, non-periodic so
## phi_w decays to 0 at the tile edges), 3101 deep (z=0..310mm, cathode node 3100).
want domain/weight_30cm \
     pochoir domain --domain domain/weight_30cm \
     --shape=220,220,3101 --spacing '0.1*mm'

want "initial/weight_30cm boundary/weight_30cm" \
     pochoir gen --generator $wgen --domain domain/weight_30cm \
     --initial initial/weight_30cm --boundary boundary/weight_30cm \
     $wcfg

## Unit-probe solve: fix,fix,fix + --multisteps no as in Task12a/b/c.  --insulator
## applies the SAME node-centered no-flux FR4 BC as the drift solve.
want "potential/weight3d increment/weight_30cm" \
     pochoir fdm \
     --nepochs 10 --epoch 130000000 --precision 0.0000000002 \
     --edges fix,fix,fix \
     --engine torch \
     --initial initial/weight_30cm --boundary boundary/weight_30cm \
     --insulator initial/weight_30cm_insulator \
     --potential potential/weight3d \
     --increment increment/weight_30cm \
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
## CONFIG-DRIVEN starts (-m yes): nGridPoints=10 and driftZDepth=309.9 come from
## $dcfg -> cell-centred 10x10 grid over the 4.4mm tile (spacing 0.44mm), 100
## launch points at z=309.9mm, the last full-velocity node before the cathode
## (z=310mm, node 3100).  This 10x10-per-pixel layout is what induce-pixel's
## tiler (_shift_paths_pixel_grid, npaths=10) expects.
want starts/drift3d \
     pochoir starts --starts starts/drift3d \
     -m yes \
     -c $dcfg \
     --plot

## ENFORCEMENT-FREE: NO --insulator on drift -> pure grad(phi) paths.
## --interp-order linear (cubic rings/overshoots near the pixel plane).
## 30cm transit ~188us at 50 V/mm -> 0..400us window (~2x transit) for margin.
want paths/drift3d \
     pochoir drift --starts starts/drift3d \
     --velocity velocity/drift3d \
     --interp-order linear \
     --paths paths/drift3d '0*us,400*us,0.05*us' \
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
echo "=== DONE: Task12d 30cm gridsq run complete (V4 very strong extraction: grid -1000V, cathode -15925V) ==="
