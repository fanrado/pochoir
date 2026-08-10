#!/bin/bash
#
# Task13: hybrid near/far field solve.  PART A drift field | PART B velocity +
#         paths | PART C weighting field | PART D induced current (Ramo).
#
# Produces, in $POCHOIR_STORE:  potential/drift3d (A), paths/drift3d (B),
# potential/weight3d (C), current/induced_current (D).
#
# Edit the SIZES block below and nothing else.  `pochoir hybrid-iterate`
# internally runs coarse -> near -> far -> near and then stitches; none of that
# is configured from here.  For the method, the band and the store keys see
# `pochoir hybrid-iterate --help` and the module docstring in
# pochoir/hybrid_iterate.py -- deliberately NOT restated here, because a copy
# in this header is how it went stale before.
#
# WHAT DOES LIVE HERE -- flags on the commands below:
#   * --interp-order linear is REQUIRED: the stitched field may still have a
#     derivative kink at the seam, and cubic overshoots at a kink.
#   * ENFORCEMENT-FREE (task10b contract): --insulator goes to the FDM solves
#     only.  velo gets neither --boundary nor --insulator and drift gets no
#     --insulator, so paths are never clamped or terminated at a surface --
#     that would break the curl-free E field.
#   * the drift time window is a literal on the `pochoir drift` line.
#
# RESUMABLE: every step is guarded on its store key, so a re-run resumes.
# GPU: the torch engine hardcodes cuda:0 -- pick the physical GPU with
# CUDA_VISIBLE_DEVICES (e.g. CUDA_VISIBLE_DEVICES=1 ./run-task13-hybrid.sh).
#
# Usage: ./run-task13-hybrid.sh [STORE_DIR]

set -e
cd "$(dirname "$0")"
ROOT="$(cd .. && pwd)"
export PATH="$ROOT/env/bin:$PATH"

## ---------------------------------------------------------------------------
## SIZES -- the only knobs you are meant to edit.  Everything about HOW the
## near/far solve is carried out lives inside `pochoir hybrid-iterate`, not
## here.
## ---------------------------------------------------------------------------
STORE="${1:-store_task13_hybrid_15cm_9x9_testnewBC}"
INTERFACE="40*mm"       # near/far split depth
COARSE_SPACING=0.4      # mm, far-field grid
FINE_SPACING=0.1        # mm, near-field and final grid

export POCHOIR_STORE="$STORE"
export POCHOIR_LOG="${POCHOIR_STORE}/pochoir_driftfield.log"

source helpers.sh

## ---------------------------------------------------------------------------
## One field, one line.  The drift and weighting solves differ only in --field
## and their two configs; every other option is a SIZE from the block above.
## ---------------------------------------------------------------------------
hybrid_field () {   # <field> <coarse-config> <fine-config>
    pochoir hybrid-iterate --field "$1" \
        --coarse-config "$2" --fine-config "$3" \
        --interface "$INTERFACE" \
        --coarse-spacing "$COARSE_SPACING" \
        --fine-spacing "$FINE_SPACING"
}

## ---------------------------------------------------------------------------
## want: run a step only when its output is missing, so a re-run resumes
## cleanly.  helpers.sh already provides this; the drift-chain steps below use
## it exactly as task10b does.
## ---------------------------------------------------------------------------

dcfg=example_gen_pcb_drift_pixel_task13_fine.json
ccfg=example_gen_pcb_drift_pixel_task13_coarse.json

date

############################################################################
## PART A: DRIFT FIELD  (hybrid, Python-driven)
############################################################################
echo "=== Task13 PART A: hybrid 15cm drift field (interface z=${INTERFACE}) -> $POCHOIR_STORE ==="

## POCHOIR_STORE reaches the command through the `cli` group's envvar, so no
## --store is needed.  --precision and the near/far scheme are the command's
## own defaults.  Result: potential/drift3d, and it STOPS there -- the drift
## chain is PART B below.
hybrid_field drift "$ccfg" "$dcfg"

date

############################################################################
## PART B: VELOCITY + PATHS   (task10b PART C, copied)
##
## Deliberately run HERE rather than inside the Python driver, so every
## parameter below -- temperature, starts mode/config, --interp-order, the drift
## time window, --plot -- can be edited or overridden by hand without touching
## pochoir/hybrid_iterate.py.  Uses potential/drift3d from PART A.
############################################################################

echo "=== Velocities ==="
## Drift velocity from the hybrid drift potential.  ENFORCEMENT-FREE:
## NO --boundary and NO --insulator -> velocity is pure mu * grad(phi); velo
## runs only to carry the potential + temperature metadata drift consumes.
want velocity/drift3d \
     pochoir velo --temperature '87.0*K' \
     --potential potential/drift3d \
     --velocity velocity/drift3d

echo "=== Paths ==="
## CONFIG-DRIVEN starts (-m yes): nGridPoints and driftZDepth come from $dcfg
## -> a cell-centred 10x10 grid over the pixel tile, 100 launch points one node
## below the cathode.  Both numbers live in the config, not here.
want starts/drift3d \
     pochoir starts --starts starts/drift3d \
     -m yes \
     -c $dcfg \
     --plot

## ENFORCEMENT-FREE: NO --insulator on drift -> pure grad(phi) paths.
## --interp-order linear: cubic rings/overshoots near the pixel plane and
## over-focuses paths onto the pads; linear is monotone-safe.
## Window: the literal on the --paths line below.  It must cover the full
## transit at the configured drift length and bulk field; there is margin in the
## current value.  Shrink it here if a shorter run is wanted.
want paths/drift3d \
     pochoir drift --starts starts/drift3d \
     --velocity velocity/drift3d \
     --interp-order linear \
     --paths paths/drift3d '0*us,200*us,0.05*us' \
     --plot

date

############################################################################
## PART C: WEIGHTING FIELD  (same hybrid scheme, --field weighting)
##
## Identical scheme and precision as PART A -- only the grids, generator, edges
## and store-key names differ, all carried by the 'weighting' profile in
## pochoir/hybrid_iterate.py (which is where the shapes are derived; they are
## not restated here).
##
## A MULTI-PIXEL probe with NON-periodic edges (fix,fix,fix), so the unit probe
## phi_w decays to ~0 at the tile edges instead of wrapping -- a weighting field
## cannot be solved on the single periodic pixel tile PART A uses.
##
## The weighting config is a UNIT PROBE: collecting pixel = 1, every other
## electrode (including the cathode) = 0.  It therefore carries no
## GridPotential/CathodePotential -- the drift bias scaling does not apply.
##
## SAME STORE, on purpose: PART D's induce-pixel needs paths/drift3d (PART B) and
## potential/weight3d (here) together.  The weighting keys are w_-prefixed so they
## cannot collide with the drift keys -- a collision would make `_want` skip every
## grid/gen step as "have" and silently solve the weighting field on the DRIFT
## geometry.
##
## RUNTIME: much the longest PART here -- the weighting grids are far larger
## than the drift ones, though the full volume is only a stitch target and is
## never solved.
############################################################################
echo "=== Task13 PART C: hybrid weighting field (unit probe, fix,fix,fix) ==="

wcfg=example_gen_pixel_with_grid_task13_fine.json
wccfg=example_gen_pixel_with_grid_task13_coarse.json

hybrid_field weighting "$wccfg" "$wcfg"

date

############################################################################
## PART D: INDUCED CURRENT   (task10b's final step, copied)
##
## Ramo: i(t) = q * v(x(t)) . E_w(x(t)), using the PART B drift paths and the
## PART C weighting field.  Run here rather than in the Python driver so
## --npixels and --config stay hand-settable.
############################################################################
echo "=== Induced currents ==="
## --npixels 2 sums the target pixel + its first ring; --config supplies the
## pixel geometry for the pad-collection map (same geometry as $wcfg).
want current/induced_current \
     pochoir induce-pixel --weighting potential/weight3d \
     --paths paths/drift3d \
     --output current/induced_current \
     --npixels 2 \
     --config "$wcfg" \
     --plot

date
echo "=== DONE: Task13 drift + weighting + induced current -> $POCHOIR_STORE"
echo "         (potential/drift3d, paths/drift3d, potential/weight3d, current/induced_current) ==="
