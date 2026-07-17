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
#   drift -> potential/drift3d   (44x44x301 @0.1mm, per,per,fix)
#
# Geometry at 0.1mm: pp_loweredge=99 (z=9.9mm), FR4 no-flux slab 1 cell @z=9.9mm,
# copper pad 1 cell @z=10.0mm, cathode @z=30mm (node 300).  Cathode=Grid=-1000V
# over the 20mm LAr gap -> 50 V/mm bulk.
#
# NO weighting, NO induced current -- drift only.
#
# Gates: DINS default ON (insulator active -- the point of the run).  Set DINS=0
# for a plain-Laplace (no-insulator) control using the plain 2cm config.
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
     --nepochs 10 --epoch 130000000 --precision 0.00000000002 \
     --edges per,per,fix \
     --engine torch \
     --initial initial/drift_2cm --boundary boundary/drift_2cm \
     "${dins_mask[@]}" \
     --potential potential/drift3d \
     --increment increment/drift_2cm

date

############################################################################
## PART C: VELOCITY, PATHS  (drift only -- NO weighting, NO induce)
############################################################################
## Uses the full-depth drift potential (potential/drift3d) solved above.

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

date
echo "=== DONE: Task10 2cm 0.1mm enforcement-free drift run complete ==="
