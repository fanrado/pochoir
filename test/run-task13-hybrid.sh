#!/bin/bash
#
# Task13: reorganized ITERATIVE hybrid solver -- 5 cm drift, Python-driven,
#         node-centered Neumann (no-flux) PCB insulator BC.
#         PART A drift field | PART B velocity + paths |
#         PART C weighting field (same scheme, --field weighting) |
#         PART D induced current (Ramo).
# ---------------------------------------------------------------------------
# STRUCTURE.  The near/far ITERATION lives in Python (pochoir/hybrid_iterate.py,
# driven by `pochoir hybrid-iterate`) -- PART A below is a one-line call to it.
# The earlier bash hybrids (run-hybrid-30cm-drift.sh, run-task8-hybrid-5cm-insul.sh)
# hard-coded a fixed near/far sequence, so the iteration structure was invisible
# from the script and the near->far coupling was one-way.
#
# The DRIFT CHAIN (velo + starts + drift) is deliberately NOT inside the Python
# driver: PART B runs it as explicit `pochoir` commands, copied from task10b
# PART C, so every parameter -- temperature, starts mode/config, --interp-order,
# the drift time window, --plot -- can be edited or overridden by hand here
# without touching the module.
#
# METHOD (near/far separation at z = 20 mm, axis 2; see the module docstring):
#   1. coarse 0.4 mm full-volume solve
#   2. refine the coarse potential onto the 0.05 mm near grid
#   3. near-bc pins the z = 20 mm plane to the coarse value
#   4. near fine solve at 0.05 mm (no-flux FR4 BC active)
#   5. coarsen the near solution back to 0.4 mm (exact stride 8)
#   6. stitch-near the coarsened near field onto the coarse far field
#   7. re-impose exact coarse boundary values, then re-solve the FULL coarse
#      volume with the stitched array as INITIAL VALUES ONLY -- the near region
#      FLOATS freely (this is what distinguishes the scheme from the pinned
#      overlapping-Schwarz `near-far-solve`)
#   8. converged when max|phi_k - phi_(k-1)| over the whole coarse volume
#      < 2e-8 V; otherwise back to 2
# then refine the converged field onto the 0.1 mm full grid and re-solve ->
# potential/drift3d, where PART A ends.  PART B runs velo + starts + drift on
# THAT 0.1 mm field.  The 0.05 mm near solve exists only to sharpen the coarse
# iteration; it is never stitched into the final field.
#
# GRIDS (same 4.4 mm periodic pixel tile throughout; z = 0..60 mm = 10 mm of
# PCB/pad region below the plane + 50 mm drift, cathode at z = 60 mm).  The 8:1
# coarse/near stride is exact on every axis: 88 -> 11, 401 -> 51.
#
#   grid                            shape            spacing   extent        config
#   coarse full                     11 x 11 x 151    0.4 mm    z = 0..60 mm  coarse
#   near fine                       88 x 88 x 401    0.05 mm   z = 0..20 mm  fine
#   near coarse (coarsen target)    11 x 11 x 51     0.4 mm    z = 0..20 mm  --
#   final full                      44 x 44 x 601    0.1 mm    z = 0..60 mm  fine
#
# CONFIGS (both from Phase 1, commit 99fbd00):
#   * example_gen_pcb_drift_pixel_task13_fine.json -- a verbatim copy of
#     example_gen_pcb_drift_pixel_task10b_insul.json with exactly three values
#     changed for the 5 cm gap: GridPotential and CathodePotential -1000 ->
#     -2500 and driftZDepth 29.9 -> 59.9.  -2500 V over the 50 mm gap keeps
#     task10b's 50 V/mm bulk field.  Every length in it is an exact multiple of
#     both 0.1 mm and 0.05 mm, so this ONE config drives both the 0.05 mm near
#     solve and the 0.1 mm final solve with no rounding -- the final solve
#     therefore uses the exact 3.5 mm pixel (35 cells, not 36).
#   * example_gen_pcb_drift_pixel_task13_coarse.json -- the same geometry
#     transcribed onto 0.4 mm, which the hybrid scheme needs because the
#     generator derives cells from dom.spacing and several task10b lengths are
#     sub-0.4 mm.  pixelGap snaps DOWN to 0.8 (not up to 1.2) so the pitch stays
#     exactly 4.4 mm against the 88-cell fine tile and the pad plane keeps free
#     gap cells -- padplane_noflux_geom raises unless the plane is only
#     partially Dirichlet.
#
# THE DRIFT STEPS ARE TASK10b's, COPIED VERBATIM.  run-task10b-drift-2cm-gap09-
# chamf07.sh is the validated reference: same generator, same config values, same
# fdm flags (--nepochs 10 --epoch 130000000 --edges per,per,fix --engine torch),
# same enforcement-free velo/starts/drift wiring.  Task13 changes only what the
# hybrid scheme and the 5 cm gap require.  Task10b's PART B (weighting field) and
# PART C's induce-pixel are carried through here as PART C and PART D, both put
# on the same iterative hybrid footing as the drift field.
#
# The ONE number not inherited directly is the drift window: task10b used
# '0*us,40*us,0.05*us' for 20 mm, so 50 mm at the same 50 V/mm needs ~2.5x that.
# PART B uses '0*us,120*us,0.05*us'; measured transit is ~32 us, so there is
# ample margin.  It is a literal on the `pochoir drift` line -- edit it there.
#
# ENFORCEMENT-FREE (task10b contract): --insulator goes to the FDM solves only.
# velo gets neither --boundary nor --insulator (velocity is pure mu*grad(phi))
# and drift gets no --insulator (paths are never clamped or terminated at a
# surface, which would break the curl-free E field).  --insulator is an enable
# signal only: the no-flux interface is auto-derived from the Dirichlet geometry
# by padplane_noflux_geom, and `gen` writes the mask under <initial>_insulator.
#
# RESUMABLE: every step is guarded on its store key, and the outer-iteration keys
# carry the pass index (potential/full_k00, _k01, ...), so a re-run picks up where
# it stopped and each pass stays separately inspectable.
#
# GPU note: pochoir's torch engine hardcodes cuda:0 -- select the physical GPU
# with CUDA_VISIBLE_DEVICES (e.g. CUDA_VISIBLE_DEVICES=1 ./run-task13-hybrid.sh).
#
# Usage: ./run-task13-hybrid.sh [STORE_DIR]

set -e
cd "$(dirname "$0")"
ROOT="$(cd .. && pwd)"
export PATH="$ROOT/env/bin:$PATH"

export POCHOIR_STORE="${1:-store_task13_hybrid_5cm}"
export POCHOIR_LOG="${POCHOIR_STORE}/pochoir_driftfield.log"

source helpers.sh

## ---------------------------------------------------------------------------
## want: run a step only when its output is missing, so a re-run resumes
## cleanly.  helpers.sh already provides this; the drift-chain steps below use
## it exactly as task10b does.
## ---------------------------------------------------------------------------

dcfg=example_gen_pcb_drift_pixel_task13_fine.json
ccfg=example_gen_pcb_drift_pixel_task13_coarse.json

date

############################################################################
## PART A: DRIFT FIELD  (iterative hybrid, Python-driven)
############################################################################
echo "=== Task13 PART A: iterative hybrid 5cm drift field (interface z=20mm, tol 2e-8 V) -> $POCHOIR_STORE ==="

## POCHOIR_STORE reaches the command through the `cli` group's envvar, so no
## --store is needed.  Defaults carried by the command: --interface '20*mm',
## --tol 2e-8, --max-iters 20.  This produces potential/drift3d (44x44x601
## @0.1mm) and STOPS there -- the drift chain is PART B below.
pochoir hybrid-iterate \
    --coarse-config "$ccfg" \
    --fine-config   "$dcfg"

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
## CONFIG-DRIVEN starts (-m yes): nGridPoints=10 and driftZDepth=59.9 come from
## $dcfg -> cell-centred 10x10 grid over the 4.4mm tile (spacing 0.44mm), 100
## launch points at z=59.9mm, the last full-velocity node before the cathode
## (z=60mm, node 600).
want starts/drift3d \
     pochoir starts --starts starts/drift3d \
     -m yes \
     -c $dcfg \
     --plot

## ENFORCEMENT-FREE: NO --insulator on drift -> pure grad(phi) paths.
## --interp-order linear: cubic rings/overshoots near the pixel plane and
## over-focuses paths onto the pads; linear is monotone-safe.
## Window: task10b used '0*us,40*us,0.05*us' for its 2cm gap; this 5cm gap needs
## ~2.5x, so '0*us,120*us,0.05*us'.  Measured transit is ~32us (~1.6mm/us at
## 50 V/mm), so there is ample margin -- shrink the window here if a shorter run
## is wanted.
want paths/drift3d \
     pochoir drift --starts starts/drift3d \
     --velocity velocity/drift3d \
     --interp-order linear \
     --paths paths/drift3d '0*us,120*us,0.05*us' \
     --plot

date

############################################################################
## PART C: WEIGHTING FIELD  (same iterative hybrid, --field weighting)
##
## Identical scheme, tolerance and final 0.1mm grid as PART A -- only the grids,
## generator, edges and store-key names differ, all carried by the 'weighting'
## profile in pochoir/hybrid_iterate.py:
##
##   grid                     shape             spacing   config
##   coarse full              55 x 55 x 151     0.4 mm    w coarse
##   near fine                440 x 440 x 401   0.05 mm   w fine
##   near coarse (coarsen)    55 x 55 x 51      0.4 mm    --
##   final full               220 x 220 x 601   0.1 mm    w fine
##
## 5x5 pixels = 22mm transverse and NON-periodic edges (fix,fix,fix) so the unit
## probe phi_w decays to ~0 at the tile edges instead of wrapping -- a weighting
## field cannot be solved on the single periodic 4.4mm tile PART A uses.
##
## The weighting config is a UNIT PROBE: collecting pixel = 1, every other
## electrode (including the cathode) = 0.  It therefore carries no
## GridPotential/CathodePotential -- the -2500V drift scaling does not apply.
##
## SAME STORE, on purpose: PART D's induce-pixel needs paths/drift3d (PART B) and
## potential/weight3d (here) together.  The weighting keys are w_-prefixed so they
## cannot collide with the drift keys -- a collision would make `_want` skip every
## grid/gen step as "have" and silently solve the weighting field on the DRIFT
## geometry.
##
## RUNTIME: the near grid is 77.6M nodes (25x the drift near grid, ~0.62 GB per
## f64 array) and is re-solved once per outer iteration, and the final 0.1mm
## weighting solve is 29.1M nodes against the drift's 1.2M.  Expect this PART to
## take substantially longer than PART A's ~17 minutes.
############################################################################
echo "=== Task13 PART C: iterative hybrid weighting field (5x5 unit probe, fix,fix,fix) ==="

wcfg=example_gen_pixel_with_grid_task13_fine.json
wccfg=example_gen_pixel_with_grid_task13_coarse.json

pochoir hybrid-iterate \
    --field weighting \
    --coarse-config "$wccfg" \
    --fine-config   "$wcfg"

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
