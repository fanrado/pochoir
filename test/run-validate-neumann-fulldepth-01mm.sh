#!/bin/bash
#
# Neumann-BC VALIDATION run: full-depth, single-pass, uniform 0.1mm.
# ---------------------------------------------------------------------------
# Purpose
# -------
# The collected charge Q<1 at the pixel edges persists even for a larger pixel.
# Before attributing it to physics (drift-watershed charge sharing + weighting-
# field decay) we must rule out the near/far HYBRID machinery: the Dirichlet
# interface pin, the overlapping-Schwarz seam at z=20mm, and the 0.05->0.1mm
# coarsen/stitch of the weighting field.  This script removes ALL of that: it
# solves the ENTIRE 5cm (z=0..60mm) drift domain in ONE full-depth FDM pass at
# a uniform 0.1mm spacing, for BOTH the drift and the weighting field, with the
# no-flux FR4 insulator boundary active (Neumann, NO epsilon).
#
# If Q<1 at the edges survives here, it is real physics, not a hybrid artifact.
#
# Contrast with run-full-3d-pixel.sh (hybrid): that driver does coarse(0.4mm)->
# near(0.05mm)->Schwarz->stitch.  Here there is NO coarse, NO near, NO stitch,
# NO coarsen -- one domain, one gen, one fdm per field.
#
#   drift     -> potential/drift3d      (44x44x601   @0.1mm, per,per,fix)
#   weighting -> potential/weight3d     (220x220x601 @0.1mm, fix,fix,fix)
#                (consumed by induce-pixel)
#
# GPU memory (float64, ~9 peak arrays): drift 44x44x601=1.2M cells (trivial);
# weighting 220x220x601=29M cells ~2.1GB peak -> fits a 24GB card.  Geometry at
# 0.1mm: pp_loweredge=99 (z=9.9mm), FR4 no-flux slab 1 cell @z=9.9mm, copper pad
# 1 cell @z=10.0mm, cathode @z=60mm (node 600).
#
# Gates: DINS/WINS default ON (insulator active -- the point of the run).  Set
# DINS=0 / WINS=0 for a plain-Laplace (no-insulator) control.
#
# Usage: ./run-validate-neumann-fulldepth-01mm.sh [STORE_DIR]

set -e

export POCHOIR_STORE="${1:-store_validate_neumann_fulldepth_01mm}"

source helpers.sh

# ---------------------------------------------------------------------------
# want: run a step only when its output(s) are missing (multi-key; identical to
# the override in run-full-3d-pixel.sh so a re-run resumes cleanly).
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
## PART A: DRIFT FIELD  (full-depth single 0.1mm solve)
############################################################################
export POCHOIR_LOG="${POCHOIR_STORE}/pochoir_driftfield.log"

gen="pcb_drift_pixel_with_grid"
cfg="example_gen_pcb_drift_pixel_with_grid.json"

## No-flux FR4 insulator for the DRIFT field (Neumann, NO epsilon).  DINS on
## (default) => insulator config + --insulator initial/drift_full_insulator into
## the fdm (and later velo/drift).  DINS=0 => plain-Laplace control.
dcfg="$cfg"; dins_mask=(); dins_fine=()
if [ "${DINS:-1}" != "0" ] ; then
    dcfg="example_gen_pcb_drift_pixel_with_grid_insul.json"
    dins_mask=(--insulator initial/drift_full_insulator)
    dins_fine=(--insulator initial/drift_full_insulator)
    echo "=== DINS on: drift field uses no-flux FR4 insulator (NO epsilon) ==="
fi

## Single full-depth drift domain: 44x44 periodic tile (4.4mm pitch @0.1mm),
## 601 deep (z=0..60mm).
want domain/drift_full \
     pochoir domain --domain domain/drift_full \
     --shape=44,44,601 --spacing '0.1*mm'

want "initial/drift_full boundary/drift_full" \
     pochoir gen --generator $gen --domain domain/drift_full \
     --initial initial/drift_full --boundary boundary/drift_full \
     $dcfg

## One full-depth Laplace solve, no interface anywhere.  per,per,fix matches the
## periodic pixel tile.  Tight precision (2e-11) for a clean E-field gradient.
want "potential/drift3d increment/drift_full" \
     pochoir fdm \
     --nepochs 10 --epoch 130000000 --precision 0.00000000002 \
     --edges per,per,fix \
     --engine torch \
     --initial initial/drift_full --boundary boundary/drift_full \
     "${dins_mask[@]}" \
     --potential potential/drift3d \
     --increment increment/drift_full

date

############################################################################
## PART B: WEIGHTING FIELD  (full-depth single 0.1mm solve)
############################################################################
export POCHOIR_LOG="${POCHOIR_STORE}/pochoir_weightingfield.log"

gen="pcb_pixel_with_grid"
cfg="example_gen_pixel_with_grid.json"

## No-flux FR4 insulator for the WEIGHTING field (same BC as drift).  WINS on
## (default) => insulator config + --insulator initial/weight_full_insulator.
wcfg="$cfg"
wins_arg=()
if [ "${WINS:-1}" != "0" ] ; then
    wcfg="example_gen_pixel_with_grid_insul.json"
    wins_arg=(--insulator initial/weight_full_insulator)
    echo "=== WINS on: weighting field uses no-flux FR4 insulator (NO epsilon) ==="
fi

## Single full-depth weighting domain: 220x220 (5 pixels @0.1mm, non-periodic so
## phi_w decays to 0 at the edges), 601 deep.  This is the SAME domain the hybrid
## driver stitches into (weight_full) -- here it is solved directly.
want domain/weight_full \
     pochoir domain --domain domain/weight_full \
     --shape=220,220,601 --spacing '0.1*mm'

want "initial/weight_full boundary/weight_full" \
     pochoir gen --generator $gen --domain domain/weight_full \
     --initial initial/weight_full --boundary boundary/weight_full \
     $wcfg

## One full-depth unit-probe solve.  fix,fix,fix + --multisteps no as in the
## hybrid weighting fdm.  Precision 2e-10 (unit 0..1 range).
want "potential/weight3d increment/weight_full" \
     pochoir fdm \
     --nepochs 10 --epoch 130000000 --precision 0.0000000002 \
     --edges fix,fix,fix \
     --engine torch \
     --initial initial/weight_full --boundary boundary/weight_full \
     "${wins_arg[@]}" \
     --potential potential/weight3d \
     --increment increment/weight_full \
     --multisteps no

date

############################################################################
## PART C: VELOCITY, PATHS, INDUCED CURRENT
############################################################################
## Uses the full-depth drift potential (potential/drift3d) and weighting field
## (potential/weight3d) solved above -- both native 0.1mm, no stitch.

echo "=== Velocities ==="
## Drift velocity from the full-depth drift potential.  The drift boundary and
## insulator mask are the SAME full-depth 0.1mm arrays used in the solve.
want velocity/drift3d \
     pochoir velo --temperature '87.0*K' \
     --potential potential/drift3d \
     --boundary boundary/drift_full \
     "${dins_fine[@]}" \
     --velocity velocity/drift3d

echo "=== Paths ==="
## 10x10 grid per pixel (0.44mm spacing), 100 starting points, launched near the
## cathode where the weighting potential ~0 (Ramo start baseline vanishes ->
## collected Q~1).  At 0.1mm the last full-velocity node before the cathode
## (z=60mm, v=0) is z=59.9mm.
dist=(0.22 0.66 1.1 1.54 1.98 2.42 2.86 3.3 3.74 4.18)
dist=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.0 3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9 4.0 4.1 4.2 4.3 4.4)
points=()
for d in "${dist[@]}"; do
     for d2 in "${dist[@]}"; do
         points+=("${d}*mm,${d2}*mm,59.9*mm")
     done
done

want starts/drift3d \
     pochoir starts --starts starts/drift3d \
     -m no \
     -c example_gen_pixel_with_grid.json \
     "${points[@]}" \
     --plot

want paths/drift3d_tight \
     pochoir drift --starts starts/drift3d \
     --velocity velocity/drift3d \
     "${dins_fine[@]}" \
     --interp-order linear \
     --paths paths/drift3d_tight '0*us,90*us,0.01*us' \
     --plot
# --interp-order linear: cubic rings/overshoots near the pixel plane and
# over-focuses paths onto the pads; linear is monotone-safe.

echo "=== Induced currents ==="
## Induced current on the pixel via Ramo (weighting field x drift paths).
want current/induced_current \
     pochoir induce-pixel --weighting potential/weight3d \
     --paths paths/drift3d_tight \
     --output current/induced_current \
     --npixels 2 \
     --config example_gen_pixel_with_grid.json \
     --plot

date
echo "=== DONE: full-depth 0.1mm Neumann-BC validation run complete ==="
