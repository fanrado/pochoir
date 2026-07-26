#!/bin/bash
#
# Task13: reorganized ITERATIVE hybrid drift solver -- 5 cm drift, Python-driven,
#         node-centered Neumann (no-flux) PCB insulator BC.
#         Drift field + drift paths ONLY: NO weighting field, NO induced current.
# ---------------------------------------------------------------------------
# THIN CONFIG WRAPPER.  All of the loop logic lives in Python
# (pochoir/hybrid_iterate.py, driven by `pochoir hybrid-iterate`); this script
# only supplies the two configs and the store path.  The earlier bash hybrids
# (run-hybrid-30cm-drift.sh, run-task8-hybrid-5cm-insul.sh) hard-coded a fixed
# near/far sequence, so the iteration structure was invisible from the script and
# the near->far coupling was one-way.  Nothing is orchestrated here.
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
# then refine the converged field onto the 0.1 mm full grid, re-solve, and run
# velo + starts + drift on THAT 0.1 mm field.  The 0.05 mm near solve exists only
# to sharpen the coarse iteration; it is never stitched into the final field.
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
# PART C's induce-pixel are dropped -- drift field and drift paths only.
#
# The ONE number not inherited directly is the drift window: task10b used
# '0*us,40*us,0.05*us' for 20 mm, so 50 mm at the same 50 V/mm needs ~2.5x that.
# The module uses '0*us,120*us,0.05*us'; confirm from the paths plot that every
# path lands before the window closes.
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

date
echo "=== Task13: iterative hybrid 5cm drift solve (interface z=20mm, tol 2e-8 V) -> $POCHOIR_STORE ==="

## POCHOIR_STORE reaches the command through the `cli` group's envvar, so no
## --store is needed.  Defaults carried by the command: --interface '20*mm',
## --tol 2e-8, --max-iters 20.
pochoir hybrid-iterate \
    --coarse-config example_gen_pcb_drift_pixel_task13_coarse.json \
    --fine-config   example_gen_pcb_drift_pixel_task13_fine.json

date
echo "=== DONE: Task13 drift field + paths -> $POCHOIR_STORE (potential/drift3d, paths/drift3d) ==="
