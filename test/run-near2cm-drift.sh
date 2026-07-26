#!/bin/bash
#
# NEAR-FIELD-ONLY drift experiment (pochoir-0oum).
#
# Goal: reproduce the clean reference drift paths (branch for_pix,
# test/store_0.05mmSpacing_0.05usTimeStep_0.1mmPixPlaneWidth) which were made
# by solving the drift potential DIRECTLY in the near-field region at 0.05mm --
# NO coarse solve, NO far field, NO stitch, NO near-far Schwarz.  Isolates the
# drift potential / drift path from every hybrid step.
#
#   * Domain: one pixel tile, 88x88x401 @ 0.05mm -> z = 0..20mm (2cm drift).
#   * Cathode at the top z-plane (z=20mm) held at -900V so the bulk field is
#     -900V/18mm = -50 V/mm, matching the reference/30cm run.
#   * Electrons launched at z=19.5mm (end of near field), drift down to the
#     pad at z=2mm.  EXPECTATION: every path is collected AT the pad, none
#     drifts below z=2mm.  Drift only -- no weighting / induced current here.
#
# Weighting field and induced-current calculation are intentionally NOT run
# here: the discontinuity first shows up in the drift paths (drift potential
# only), so these experiments isolate the drift field/paths.
#
# Usage: ./run-near2cm-drift.sh [STORE_DIR] [DRIFT_CONFIG]
#   default STORE_DIR      = store_near2cm
#   default DRIFT_CONFIG   = example_gen_pcb_drift_pixel_near2cm.json
# Env overrides (to vary the near-field spatial resolution):
#   NSHAPE    domain shape (default 88,88,401  -> 0.05mm, 4.4mm x 20mm)
#   NSPACING  domain spacing (default '0.05*mm')
# For 0.1mm: NSHAPE=44,44,201 NSPACING='0.1*mm'.  Both give a 4.4mm tile,
# z=0..20mm (2cm drift); only the near-pad resolution changes.

set -e
export POCHOIR_STORE="${1:-store_near2cm}"
NSHAPE="${NSHAPE:-88,88,401}"
NSPACING="${NSPACING:-0.05*mm}"
NLAUNCH="${NLAUNCH:-19.5}"   # launch-plane z in mm (just below the cathode top)
NTWIN="${NTWIN:-20}"         # drift time window upper bound in us
source helpers.sh

want () {
    local targets="$1" ; shift
    local t miss=0
    for t in $targets ; do [ -f "$POCHOIR_STORE/${t}.npz" ] || miss=1 ; done
    if [ "$miss" -eq 0 ] ; then echo "have $targets"; return; fi
    echo "$@"; "$@"
    for t in $targets ; do
        [ -f "$POCHOIR_STORE/${t}.npz" ] || { echo "ERROR: missing output $t" >&2; exit 1; }
    done
    echo "made $targets"
}

date
export POCHOIR_LOG="${POCHOIR_STORE}/pochoir.log"
gen="pcb_drift_pixel_with_grid"
cfg="${2:-example_gen_pcb_drift_pixel_near2cm.json}"
echo "=== near-only 2cm drift: store=$POCHOIR_STORE cfg=$cfg ==="

## Step 1: near-field-only domain + geometry (0.05mm, z=0..20mm)
want domain/near2cm \
     pochoir domain --domain domain/near2cm \
     --shape=${NSHAPE} --spacing "${NSPACING}"

want "initial/near2cm boundary/near2cm" \
     pochoir gen --generator $gen --domain domain/near2cm \
     --initial initial/near2cm --boundary boundary/near2cm \
     $cfg

## Step 2: direct fine FDM solve (cathode at top plane, periodic transverse)
## Optional spatially-varying permittivity (Task7a FR4 laminate): set NEPS=1 to
## pass the generator epsilon (stored by gen as initial/near2cm_epsilon) to the
## Poisson solve.  Default off -> pure Laplace, Task1-6 unchanged.
eps_arg=()
if [ -n "${NEPS:-}" ] ; then
    eps_arg=(--epsilon initial/near2cm_epsilon)
    echo "=== NEPS set: solving Poisson with epsilon=initial/near2cm_epsilon ==="
fi
## Optional no-flux insulator (Task7a insulating-surface boundary, EPIC
## pochoir-ktj0): set NINS=1 to thread the FR4 insulator mask (stored by gen as
## initial/near2cm_insulator) through the fdm solve, the velo field, and the
## drift paths.  This is the epsilon-free replacement for NEPS: the FR4 slab is
## a reflecting (Neumann) body, drift velocity is zeroed inside it, and paths
## terminate at the FR4 surface as surface charge.  Default off -> unchanged.
## NINS takes precedence over NEPS (NO epsilon in the insulating-surface model).
ins_arg=()
if [ -n "${NINS:-}" ] ; then
    ins_arg=(--insulator initial/near2cm_insulator)
    echo "=== NINS set: insulating-surface (no-flux) boundary active (NO epsilon) ==="
fi
want "potential/drift_near2cm increment/near2cm" \
     pochoir fdm \
     --nepochs 10 --epoch 130000000 --precision 0.00000000002 \
     --edges per,per,fix \
     --engine torch \
     --initial initial/near2cm --boundary boundary/near2cm \
     "${eps_arg[@]}" "${ins_arg[@]}" \
     --potential potential/drift_near2cm \
     --increment increment/near2cm
date

## Step 3: velocity field (0.05mm)
want velocity/near2cm \
     pochoir velo --temperature '87.0*K' \
     --potential potential/drift_near2cm \
     --boundary boundary/near2cm \
     "${ins_arg[@]}" \
     --velocity velocity/near2cm

## Step 4: launch 10x10 grid across one pixel at z=19.5mm (end of near field)
dist=(0.22 0.66 1.1 1.54 1.98 2.42 2.86 3.3 3.74 4.18)
points=()
for d in "${dist[@]}"; do
     for d2 in "${dist[@]}"; do
         points+=("${d}*mm,${d2}*mm,${NLAUNCH}*mm")
     done
done

want starts/near2cm \
     pochoir starts --starts starts/near2cm -m yes \
     -c $cfg "${points[@]}" --plot

## Step 5: drift paths (linear interp on the 0.05mm field)
want paths/near2cm \
     pochoir drift --starts starts/near2cm \
     --velocity velocity/near2cm \
     "${ins_arg[@]}" \
     --interp-order linear \
     --paths paths/near2cm "0*us,${NTWIN}*us,0.05*us" \
     --plot
date
echo "=== DONE: paths in $POCHOIR_STORE/paths/near2cm.npz, plot drift_paths_3d.png ==="
