#!/bin/bash
#
# Task13: ONE-SHOT hybrid solver -- 5 cm drift, Python-driven,
#         node-centered Neumann (no-flux) PCB insulator BC.
#         PART A drift field | PART B velocity + paths |
#         PART C weighting field (same scheme, --field weighting) |
#         PART D induced current (Ramo).
# ---------------------------------------------------------------------------
# STRUCTURE.  The near/far scheme lives in Python (pochoir/hybrid_iterate.py,
# driven by `pochoir hybrid-iterate`) -- PART A below is a one-line call to it.
# The earlier bash hybrids (run-hybrid-30cm-drift.sh, run-task8-hybrid-5cm-insul.sh)
# hard-coded the near/far sequence, so the structure was invisible from the
# script.  There is NO outer iteration any more: one coarse solve, one near
# solve, one stitch (beads pochoir-uy3c).
#
# The DRIFT CHAIN (velo + starts + drift) is deliberately NOT inside the Python
# driver: PART B runs it as explicit `pochoir` commands, copied from task10b
# PART C, so every parameter -- temperature, starts mode/config, --interp-order,
# the drift time window, --plot -- can be edited or overridden by hand here
# without touching the module.
#
# METHOD (single pass, no iteration; near/far separation at z = 20 mm, axis 2;
# see the module docstring):
#   1. coarse 0.4 mm full-volume solve
#   2. refine the coarse potential onto the 0.1 mm near grid, then near-bc pins
#      the near grid's last z plane (z = 20 mm) to the coarse value there --
#      a FIXED Dirichlet BC, never revisited
#   3. near solve at 0.1 mm over z = 0..20 mm (no-flux FR4 BC active)
#   4. stitch-near multi-linearly upsamples the coarse far field onto the full
#      0.1 mm grid and overwrites the first 201 z planes with the fine near
#      solution.  Continuous at the seam because step 2 pinned the near top
#      plane to the same coarse values.
#   5. that stitched array IS the final field -- written straight to
#      potential/drift3d.  No full-volume re-solve, no coarsen-back, no
#      convergence loop.
# PART A ends there; PART B runs velo + starts + drift on that 0.1 mm field.
#
# SEAM CAVEAT.  The far field is a piecewise-linear upsample of the 0.4 mm solve
# and is never re-solved at 0.1 mm, so E_z there is piecewise-constant across
# 0.4 mm blocks and phi has a derivative kink at z = 20 mm.  PART B therefore
# MUST keep --interp-order linear: cubic overshoots at a kink.
#
# GRIDS (same 4.4 mm periodic pixel tile throughout; z = 0..60 mm = 10 mm of
# PCB/pad region below the plane + 50 mm drift, cathode at z = 60 mm).  TWO
# spacings, three grids; the near solve and the final volume share the 0.1 mm
# lattice, so the stitch is a plane-for-plane overwrite, not a resample.
#
#   grid                            shape            spacing   extent        config
#   coarse full                     11 x 11 x 151    0.4 mm    z = 0..60 mm  coarse
#   near fine                       44 x 44 x 201    0.1 mm    z = 0..20 mm  fine
#   final full (stitch target)      44 x 44 x 601    0.1 mm    z = 0..60 mm  fine
#
# CONFIGS (both from Phase 1, commit 99fbd00):
#   * example_gen_pcb_drift_pixel_task13_fine.json -- a verbatim copy of
#     example_gen_pcb_drift_pixel_task10b_insul.json with exactly three values
#     changed for the 5 cm gap: GridPotential and CathodePotential -1000 ->
#     -2500 and driftZDepth 29.9 -> 59.9.  -2500 V over the 50 mm gap keeps
#     task10b's 50 V/mm bulk field.  Every length in it is an exact multiple of
#     both 0.1 mm and 0.05 mm, so this ONE config drives both the 0.1 mm near
#     solve and the 0.1 mm final volume with no rounding -- both use the exact
#     3.5 mm pixel (35 cells, not 36).
#   * example_gen_pcb_drift_pixel_task13_coarse.json -- the same geometry
#     transcribed onto 0.4 mm, which the hybrid scheme needs because the
#     generator derives cells from dom.spacing and several task10b lengths are
#     sub-0.4 mm.  pixelGap snaps DOWN to 0.8 (not up to 1.2) so the pitch stays
#     exactly 4.4 mm against the 88-cell fine tile and the pad plane keeps free
#     gap cells -- padplane_noflux_geom raises unless the plane is only
#     partially Dirichlet.
#
#     pixelPlaneLowEdgePosition: the generator TRUNCATES this
#     (gen_pcb_drift_pixel_with_grid.py, int(value/spacing)) rather than
#     rounding, so the value interacts with the spacing.  9.6 looks exact but is
#     not -- 9.6/0.4 = 23.999999999999996 in binary float, so int() gives 23 and
#     the pad top lands at 9.6 mm.
#
#     *** OPEN, beads pochoir-f3it ***  Both task13 configs now carry 10.0 (was
#     9.9).  At 10.0 the two grids DISAGREE on where the pad is: the 0.4 mm
#     coarse grid truncates to 25 -> z_pad 26 -> pad top 10.4 mm, while the
#     0.1 mm grids give 100 -> z_pad 101 -> pad top 10.1 mm.  At the former 9.9
#     both landed on 10.0 mm (coarse 24/25, fine 99/100).  A 0.3 mm electrode
#     offset between the coarse far field and the fine near field is not
#     something the one-shot stitch can absorb, so the pad-index table below is
#     deliberately NOT restated as correct until pochoir-f3it is resolved
#     (revert to 9.9, or snap the value per spacing).
#
#   PAD-PLANE INDICES.  Two different indices get conflated here; both are
#   listed deliberately.  pp_loweredge is the derived FR4-slab low edge, and is
#   NOT what the solver uses.  With padThicknessCells=3 the pad is a 3-cell
#   grounded block and padplane_noflux_geom (fdm_generic.py:634) takes its
#   DRIFT-FACING FACE, z_pad = z_top = pp_loweredge + pp_width, for drift_sign
#   > 0.  The no-flux mirror is applied at z_pad, never at pp_loweredge:
#
#   Indices as they stand TODAY (pixelPlaneLowEdgePosition = 10.0), i.e. the
#   pochoir-f3it mismatch, NOT a validated configuration:
#
#     grid                spacing    pp_loweredge   z_pad   pad top
#     coarse full         0.4 mm         25          26     10.4 mm   <-- offset
#     near fine           0.1 mm        100         101     10.1 mm
#     final full          0.1 mm        100         101     10.1 mm
#
#   The validated task10b reference has the pad top at 10.0 mm on every grid,
#   which is what 9.9 produced (coarse 24/25, fine 99/100).  Quote z_pad when
#   talking about the solver's interface; quote pp_loweredge only about the FR4
#   slab.
#
# THE DRIFT STEPS ARE TASK10b's, COPIED VERBATIM.  run-task10b-drift-2cm-gap09-
# chamf07.sh is the validated reference: same generator, same config values, same
# fdm flags (--nepochs 10 --epoch 130000000 --edges per,per,fix --engine torch),
# same enforcement-free velo/starts/drift wiring.  Task13 changes only what the
# hybrid scheme and the 5 cm gap require.  Task10b's PART B (weighting field) and
# PART C's induce-pixel are carried through here as PART C and PART D, both put
# on the same one-shot hybrid footing as the drift field.
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
# RESUMABLE: every step is guarded on its store key, so a re-run picks up where
# it stopped.  There are no k-indexed pass keys any more -- each step runs once
# and writes one key (potential/coarse, potential/near, then the output).
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
## PART A: DRIFT FIELD  (one-shot hybrid, Python-driven)
############################################################################
echo "=== Task13 PART A: one-shot hybrid 5cm drift field (interface z=20mm) -> $POCHOIR_STORE ==="

## POCHOIR_STORE reaches the command through the `cli` group's envvar, so no
## --store is needed.  Defaults carried by the command: --interface '20*mm',
## --precision 2e-8, --coarse-spacing 0.4, --fine-spacing 0.1.  One coarse
## solve, one near solve, one stitch -> potential/drift3d (44x44x601 @0.1mm),
## and it STOPS there -- the drift chain is PART B below.
pochoir hybrid-iterate \
    --coarse-config "$ccfg" \
    --fine-config   "$dcfg" \
    --interface "40*mm" \
    --coarse-spacing 0.4 \
    --fine-spacing 0.1

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
     --paths paths/drift3d '0*us,200*us,0.05*us' \
     --plot

date

############################################################################
## PART C: WEIGHTING FIELD  (same one-shot hybrid, --field weighting)
##
## Identical scheme, precision and 0.1mm fine grid as PART A -- only the grids,
## generator, edges and store-key names differ, all carried by the 'weighting'
## profile in pochoir/hybrid_iterate.py:
##
##   grid                        shape             spacing   config
##   coarse full                 55 x 55 x 151     0.4 mm    w coarse
##   near fine                   220 x 220 x 201   0.1 mm    w fine
##   final full (stitch target)  220 x 220 x 601   0.1 mm    w fine
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
## RUNTIME: the near grid is 9.7M nodes (25x the drift near grid) and is solved
## ONCE, and there is no final full-volume weighting solve at all -- the 29.1M
## node full grid is only a stitch target.  This is much cheaper than the old
## iterated scheme, but still the longest PART here.
############################################################################
echo "=== Task13 PART C: one-shot hybrid weighting field (5x5 unit probe, fix,fix,fix) ==="

wcfg=example_gen_pixel_with_grid_task13_fine.json
wccfg=example_gen_pixel_with_grid_task13_coarse.json

pochoir hybrid-iterate \
    --field weighting \
    --coarse-config "$wccfg" \
    --fine-config   "$wcfg" \
    --interface "40*mm" \
    --coarse-spacing 0.4 \
    --fine-spacing 0.1

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
