#!/bin/bash
#
# Task11: 2cm drift + weighting + induced-current run WITH a square-hole shield
# grid, full-depth, single-pass, uniform 0.1mm.
# ---------------------------------------------------------------------------
# Purpose
# -------
# Task11 is Task10 (run-task10-drift-2cm-01mm.sh) PLUS a shield grid: the
# generators' GridHoleShape is flipped from "None" to "square", so a solid PCB
# shield plane with rounded-square apertures is drawn one PcbWidth (1.6mm) above
# the pixel pads, at z = pp_loweredge + pcb_width = 9.9mm + 1.6mm = 11.5mm
# (grid index 115).  Everything else is inherited from Task10:
#
#   * DRIFT field:  the shield plane is a Dirichlet electrode held at
#     GridPotential = -75V -- the value on the undisturbed 50 V/mm linear ramp
#     at z=11.5mm (pad top 0V @10mm -> cathode -1000V @30mm), so the grid is
#     field-transparent and the bulk drift field stays ~uniform 50 V/mm.
#   * WEIGHTING field: the shield plane is held at W = 0 (Ramo: every
#     non-collecting electrode is a weighting-0 electrode), the collecting pixel
#     at W = 1 through the full 3-cell pad depth.
#
# NO DIELECTRIC CONSTANT anywhere: the square-hole drift generator would build
# an epsilon permittivity array if LArPermittivity/FR4Permittivity were present,
# so those keys are STRIPPED from the Task11 DRIFT configs -> epsilon stays None
# and the solve is a plain Laplace with the no-flux FR4 Neumann BC only.  (The
# weighting generator reads the permittivity keys unconditionally but never
# builds epsilon on the square path, so they are kept there, unused.)
#
# ENFORCEMENT-FREE wiring (as Task10 / pochoir-w3x9): the ONLY insulator
# condition is the solver's Neumann BC, applied via --insulator on the fdm solve
# ONLY.  velo and drift get NO --insulator and velo gets NO --boundary, so the
# drift paths are pure v = mu * grad(phi) of the solved potential.
#
#   drift     -> potential/drift3d   (44x44x301   @0.1mm, per,per,fix)
#   weighting -> potential/weight3d  (220x220x301 @0.1mm, fix,fix,fix)
#                (consumed by induce-pixel)
#
# Geometry at 0.1mm: pp_loweredge=99 (z=9.9mm), FR4 no-flux slab 1 cell @z=9.9mm
# (gap cells only), copper pad 3 cells @z=9.8-10.0mm (padThicknessCells=3,
# grounded through its depth), square shield grid @z=11.5mm (node 115), cathode
# @z=30mm (node 300).  Cathode=-1000V, Grid=-75V over the 20mm LAr gap -> 50 V/mm
# bulk.  Induced current via Ramo (induce-pixel).
#
# Gates: DINS/WINS default ON (insulator active -- the point of the run).  Set
# DINS=0 / WINS=0 for a plain-Laplace (no-insulator, 1-cell-pad) control using the
# plain 2cm gridsq configs (shield grid still present).
#
# Usage: ./run-task11-drift-2cm-01mm-gridsq.sh [STORE_DIR]

set -e

export POCHOIR_STORE="${1:-store_task11_drift_2cm_01mm_gridsq}"

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
## PART A: DRIFT FIELD  (full-depth single 0.1mm solve, 2cm deep, shield grid)
############################################################################
export POCHOIR_LOG="${POCHOIR_STORE}/pochoir_driftfield.log"

gen="pcb_drift_pixel_with_grid"
cfg="example_gen_pcb_drift_pixel_with_grid_2cm_gridsq.json"

## No-flux FR4 insulator for the DRIFT field (Neumann, NO epsilon).  DINS on
## (default) => insulator config + --insulator initial/drift_2cm_insulator into
## the fdm ONLY.  DINS=0 => plain-Laplace control (plain 2cm gridsq config).
dcfg="$cfg"; dins_mask=()
if [ "${DINS:-1}" != "0" ] ; then
    dcfg="example_gen_pcb_drift_pixel_with_grid_2cm_gridsq_insul.json"
    dins_mask=(--insulator initial/drift_2cm_insulator)
    echo "=== DINS on: drift field uses no-flux FR4 insulator (NO epsilon) ==="
fi

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
     "${dins_mask[@]}" \
     --potential potential/drift3d \
     --increment increment/drift_2cm

date

############################################################################
## PART B: WEIGHTING FIELD  (full-depth single 0.1mm solve, 2cm deep, shield grid)
############################################################################
export POCHOIR_LOG="${POCHOIR_STORE}/pochoir_weightingfield.log"

wgen="pcb_pixel_with_grid"
wcfg="example_gen_pixel_with_grid_2cm_gridsq.json"

## No-flux FR4 insulator + 3-cell grounded pad for the WEIGHTING field (SAME BC
## as the drift field -- the weighting field is an electric field too).  The
## square shield grid is held at W=0.  WINS on (default) => insul config
## (enableInsulatorFR4 + padThicknessCells:3) + --insulator initial/weight_2cm_insulator
## into the fdm.  WINS=0 => plain-Laplace 1-cell-pad control (shield grid kept).
wins_arg=()
if [ "${WINS:-1}" != "0" ] ; then
    wcfg="example_gen_pixel_with_grid_2cm_gridsq_insul.json"
    wins_arg=(--insulator initial/weight_2cm_insulator)
    echo "=== WINS on: weighting field uses no-flux FR4 insulator + 3-cell grounded pad (NO epsilon) ==="
fi

## Weighting domain: 220x220 (5 pixels @0.1mm, non-periodic so phi_w decays to 0
## at the tile edges), 301 deep (z=0..30mm, cathode node 300).
want domain/weight_2cm \
     pochoir domain --domain domain/weight_2cm \
     --shape=220,220,301 --spacing '0.1*mm'

want "initial/weight_2cm boundary/weight_2cm" \
     pochoir gen --generator $wgen --domain domain/weight_2cm \
     --initial initial/weight_2cm --boundary boundary/weight_2cm \
     $wcfg

## Unit-probe solve: collecting (center) pixel = 1, all other electrodes = 0
## (shield grid included).  fix,fix,fix + --multisteps no as in Task10.
## --insulator applies the SAME node-centered no-flux FR4 BC as the drift solve.
## Precision 2e-10 (unit 0..1 range).
want "potential/weight3d increment/weight_2cm" \
     pochoir fdm \
     --nepochs 10 --epoch 130000000 --precision 0.0000000002 \
     --edges fix,fix,fix \
     --engine torch \
     --initial initial/weight_2cm --boundary boundary/weight_2cm \
     "${wins_arg[@]}" \
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
## 10x10 grid of 100 launch points spanning ONE pixel tile (x,y equally spaced
## 0..4.3mm, the full 4.4mm pitch), launched at z=29.9mm -- the last full-velocity
## node before the cathode (z=30mm, node 300).  This 10x10-per-pixel layout is
## what induce-pixel's tiler (_shift_paths_pixel_grid, npaths=10) expects: it
## replicates these single-pixel paths across the weighting field's pixel grid.
dist=(0.0000 0.4778 0.9556 1.4333 1.9111 2.3889 2.8667 3.3444 3.8222 4.3000)
points=()
for d in "${dist[@]}"; do
     for d2 in "${dist[@]}"; do
         points+=("${d}*mm,${d2}*mm,29.9*mm")
     done
done

want starts/drift3d \
     pochoir starts --starts starts/drift3d \
     -m yes \
     -c example_gen_pcb_drift_pixel_with_grid_2cm_gridsq.json \
     "${points[@]}" \
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
echo "=== DONE: Task11 2cm 0.1mm drift + weighting + induced-current (square shield grid) run complete ==="
