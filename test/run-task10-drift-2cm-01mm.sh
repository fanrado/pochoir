#!/bin/bash
#
# Task10: 2cm drift-only run, full-depth, single-pass, uniform 0.1mm.
# ---------------------------------------------------------------------------
# Purpose
# -------
# A short (2cm) drift companion to run-validate-neumann-fulldepth-01mm.sh: solve
# the whole 2cm (z=0..30mm) drift domain in ONE full-depth FDM pass at a uniform
# 0.1mm spacing, with the no-flux FR4 insulator boundary active (Neumann, NO
# epsilon), then drift 100 electrons launched over the 0.6mm inter-pixel gap.
#
# ENFORCEMENT-FREE wiring (pochoir-w3x9): the ONLY insulator condition is the
# solver's Neumann BC, applied via --insulator on the fdm solve ONLY.  velo and
# drift get NO --insulator and velo gets NO --boundary, so the drift paths are
# pure v = mu * grad(phi) of the solved potential (velo re-derives E from the
# potential; it still runs to carry the potential + temperature metadata that
# drift consumes).
#
#   drift     -> potential/drift3d   (44x44x301   @0.1mm, per,per,fix)
#   weighting -> potential/weight3d  (220x220x301 @0.1mm, fix,fix,fix)
#                (consumed by induce-pixel)
#
# Geometry at 0.1mm: pp_loweredge=99 (z=9.9mm), FR4 no-flux slab 1 cell @z=9.9mm
# (gap cells only), copper pad 3 cells @z=9.8-10.0mm (padThicknessCells=3, grounded
# through its depth so it has a field-free interior node), cathode @z=30mm (node
# 300).  Cathode=Grid=-1000V over the 20mm LAr gap -> 50 V/mm bulk.
#
# The WEIGHTING field carries the SAME conditions as the drift field: node-centered
# no-flux FR4 BC (--insulator) AND the 3-cell pad (the collecting pixel holds W=1
# through the full pad depth, so W read at a drift-path endpoint inside the pad is
# a true 1 -> correct collected charge).  Induced current via Ramo (induce-pixel).
#
# Gates: DINS/WINS default ON (insulator active -- the point of the run).  Set
# DINS=0 / WINS=0 for a plain-Laplace (no-insulator, 1-cell-pad) control using the
# plain 2cm configs.
#
# Usage: ./run-task10-drift-2cm-01mm.sh [STORE_DIR]

set -e

export POCHOIR_STORE="${1:-store_task10_drift_2cm_01mm}"

source helpers.sh

# ---------------------------------------------------------------------------
# want: run a step only when its output(s) are missing (multi-key; identical to
# the override in run-validate-neumann-fulldepth-01mm.sh so a re-run resumes
# cleanly).
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
## PART A: DRIFT FIELD  (full-depth single 0.1mm solve, 2cm deep)
############################################################################
export POCHOIR_LOG="${POCHOIR_STORE}/pochoir_driftfield.log"

gen="pcb_drift_pixel_with_grid"
cfg="example_gen_pcb_drift_pixel_with_grid_2cm.json"

## No-flux FR4 insulator for the DRIFT field (Neumann, NO epsilon).  DINS on
## (default) => insulator config + --insulator initial/drift_2cm_insulator into
## the fdm ONLY.  DINS=0 => plain-Laplace control (plain 2cm config).
dcfg="$cfg"; dins_mask=()
if [ "${DINS:-1}" != "0" ] ; then
    dcfg="example_gen_pcb_drift_pixel_with_grid_2cm_insul.json"
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
## PART B: WEIGHTING FIELD  (full-depth single 0.1mm solve, 2cm deep)
############################################################################
export POCHOIR_LOG="${POCHOIR_STORE}/pochoir_weightingfield.log"

wgen="pcb_pixel_with_grid"
wcfg="example_gen_pixel_with_grid_2cm.json"

## No-flux FR4 insulator + 3-cell grounded pad for the WEIGHTING field (SAME BC
## as the drift field -- the weighting field is an electric field too).  WINS on
## (default) => insul config (enableInsulatorFR4 + padThicknessCells:3) +
## --insulator initial/weight_2cm_insulator into the fdm.  WINS=0 => plain-Laplace
## 1-cell-pad control.
wins_arg=()
if [ "${WINS:-1}" != "0" ] ; then
    wcfg="example_gen_pixel_with_grid_2cm_insul.json"
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

## Unit-probe solve: collecting (center) pixel = 1, all other electrodes = 0.
## fix,fix,fix + --multisteps no as in the full-depth validation weighting fdm.
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
## 10x10 grid of 100 launch points straddling the 0.6mm inter-pixel gap
## (x,y in [1.5,3.0]mm), launched at z=29.9mm -- the last full-velocity node
## before the cathode (z=30mm, node 300).
dist=(1.5000 1.6667 1.8333 2.0000 2.1667 2.3333 2.5000 2.6667 2.8333 3.0000)
dist=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.0 3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9 4.0 4.1 4.2)
points=()
for d in "${dist[@]}"; do
     for d2 in "${dist[@]}"; do
         points+=("${d}*mm,${d2}*mm,29.9*mm")
     done
done

want starts/drift3d \
     pochoir starts --starts starts/drift3d \
     -m no \
     -c example_gen_pcb_drift_pixel_with_grid_2cm.json \
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
echo "=== DONE: Task10 2cm 0.1mm drift + weighting + induced-current run complete ==="
