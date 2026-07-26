#!/bin/bash
#
# Task13: reorganized ITERATIVE hybrid drift solver -- 5cm drift, 0.4mm coarse
#         far field + 0.05mm fine near field, node-centered Neumann (no-flux)
#         PCB insulator BC.  Drift field + drift paths ONLY (no weighting field,
#         no induced current).
# ---------------------------------------------------------------------------
# WHAT IS "REORGANIZED" HERE
#
# The earlier 5cm hybrid (Task8, run-task8-hybrid-5cm-insul.sh) stitched the
# near solve onto the coarse far field ONCE: a single-shot Dirichlet pin of the
# near top plane (`near-bc`) taken from the coarse bulk, then `stitch-near`.
# That makes the stitched potential continuous in VALUE at the seam but not in
# GRADIENT, and the coarse far field never learns anything from the fine near
# solve -- so the interface error is whatever the one-shot coarse guess had.
#
# Task13 replaces that single shot with the ITERATIVE Python driver already in
# the codebase: `pochoir near-far-solve` (pochoir/nearfar.py:schwarz_solve).
# This shell script is a THIN CONFIG WRAPPER -- all loop logic lives in Python.
# Each Schwarz sweep does exactly the cycle asked for in pochoir-nm59:
#
#   coarse 0.4mm full-volume solve            (seed, step 1 below)
#     |
#     +-> split at z=2cm, upsample near to 0.05mm   (refine + near-bc, step 2)
#     +-> re-solve near with Dirichlet interface     (fdm --insulator, step 3)
#     |
#     `-> LOOP (near-far-solve, step 4):
#           downsample near -> 0.4mm and graft onto the far domain one coarse
#           cell below the interface   (nearfar.graft_plane, RGI resample)
#           re-solve the full coarse volume
#           re-pin the near interface plane from the fresh far solution
#           re-solve the near
#           until max|near_new - near_prev| < 2e-8 V   (--tol '0.00000002*V')
#
# Iterating to convergence drives the seam toward C1 (gradient continuity), so
# E = -grad(phi) has no kink where a drift electron crosses z=2cm.  Ending each
# sweep with the NEAR solve leaves near exactly pinned to far at the interface,
# so the final `stitch-near` seam is exact in value as well.
#
# GEOMETRY (single periodic pixel tile; matches the Task10 2cm reference so the
# hybrid can be compared against a full-resolution single-domain solve):
#   * 3.8mm pixel / 0.6mm gap -> 4.4mm pitch; dynamic 0.4mm (physical-unit)
#     chamfer; GridHoleShape "None" (NO shield grid); NO dielectric constant
#     (no LArPermittivity/FR4Permittivity -> the epsilon path stays off).
#   * pad low edge z=9.9mm, 0.1mm pixel plane -> pad top z=10.0mm, grounded
#     3 cells deep (padThicknessCells=3) so the field cannot leak through a
#     1-cell sheet.
#   * FR4Thickness 0.1mm under the pad is the no-flux insulator mask
#     (enableInsulatorFR4=true) -> node-centered Neumann BC at the PCB surface.
#   * drift gap 50mm: pad top z=10mm -> cathode at the domain top plane z=60mm,
#     held at -2500V => -2500/50 = -50 V/mm bulk.
#   * coarse: 11x11x151   @0.4mm  -> z=0..60mm (full volume, far field)
#   * near  : 88x88x401   @0.05mm -> z=0..20mm (pad + FR4 + 10mm of bulk)
#   * fine  : 88x88x1201  @0.05mm -> z=0..60mm (stitched, for velo/drift)
#   * electrons launched at z=59.95mm (driftZDepth, one fine cell below the
#     cathode), drifting ~49.95mm down to the pad.
#
# ENFORCEMENT-FREE (Task10/Task12 contract): --insulator is passed to the FDM
# solves ONLY -- never to velo or drift.  Velocity is pure mu*grad(phi) and
# paths are never clamped or terminated at a surface; clamping would break the
# curl-free E field.  endtag is therefore always DRIFT_NONE.
#
# GPU note: pochoir's torch engine hardcodes cuda:0 -- select the physical GPU
# with CUDA_VISIBLE_DEVICES (e.g. CUDA_VISIBLE_DEVICES=1 ./run-task13-hybrid.sh).
#
# Usage: ./run-task13-hybrid.sh [STORE_DIR]

set -e
cd "$(dirname "$0")"
ROOT="$(cd .. && pwd)"
export PATH="$ROOT/env/bin:$PATH"
PY="$ROOT/env/bin/python"

export POCHOIR_STORE="${1:-store_task13_hybrid_5cm}"
export POCHOIR_LOG="${POCHOIR_STORE}/pochoir_driftfield.log"

source helpers.sh

gen="pcb_drift_pixel_with_grid"
dcfg="example_gen_pcb_drift_pixel_task13_hybrid5cm_insul.json"

# ---------------------------------------------------------------------------
# want: run a step only when its output(s) are missing, so a re-run resumes
# cleanly (identical to the override in run-task10-drift-2cm-01mm.sh).
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

echo "### [1/6] geometry + actual-removed-material report -> $POCHOIR_STORE/GEOMETRY_AND_CHAMFER.md"
$PY compute_chamfer_removed.py --store "$POCHOIR_STORE" --mode dynamic --spacing 0.05 --chamfer-mm 0.4 \
    --pad-z 10 --drift 50 --total 60 --cathode -2500 --hybrid --coarse-spacing 0.4 \
    --title "Task 13 -- 5 cm drift, ITERATIVE hybrid (0.4+0.05 mm Schwarz, interface z=2cm), dynamic 0.4 mm chamfer, node-centered Neumann FR4 insulator (NO epsilon)"

############################################################################
## STEP 1: coarse full-volume solve (0.4mm, 11x11x151, z=0..60mm).
##
## This is the Schwarz seed / far field.  The insulator is NOT applied here:
## the 0.1mm FR4 slab and the 3-cell pad are unresolved at 0.4mm pitch, and the
## whole near region z<20mm is replaced by the fine solve at stitch time.
############################################################################
echo "### [2/6] coarse 0.4mm full-volume drift solve (far-field seed)"

want domain/coarse \
     pochoir domain --domain domain/coarse --shape=11,11,151 --spacing '0.4*mm'

want "initial/coarse boundary/coarse" \
     pochoir gen --generator $gen --domain domain/coarse \
     --initial initial/coarse --boundary boundary/coarse $dcfg

want "potential/coarse increment/coarse" \
     pochoir fdm --nepochs 10 --epoch 130000000 --precision 0.00000002 \
     --edges per,per,fix --engine torch \
     --initial initial/coarse --boundary boundary/coarse \
     --potential potential/coarse --increment increment/coarse

date

############################################################################
## STEP 2: split at z=2cm and upsample the near region to 0.05mm.
##
## `gen` on the near domain stores the node-centered no-flux mask as
## initial/near_insulator.  `refine` upsamples the coarse solution onto the
## fine near grid as the starting interior; `near-bc` pins the z=20mm top plane
## Dirichlet from the coarse bulk (Schwarz sweep-0 interface condition).
############################################################################
echo "### [3/6] near-field split at z=2cm + 0.05mm upsample of the coarse seed"

want domain/near \
     pochoir domain --domain domain/near --shape=88,88,401 --spacing '0.05*mm'

want "initial/near boundary/near" \
     pochoir gen --generator $gen --domain domain/near \
     --initial initial/near --boundary boundary/near $dcfg

want initial/near_refined \
     pochoir refine --coarse potential/coarse \
     --initial initial/near --boundary boundary/near --output initial/near_refined

want "initial/near_bc boundary/near_bc" \
     pochoir near-bc --initial initial/near_refined --boundary boundary/near \
     --coarse potential/coarse \
     --initial-out initial/near_bc --boundary-out boundary/near_bc

############################################################################
## STEP 3: near fine solve with the node-centered Neumann insulator BC.
##
## Sweep 0 of the Schwarz iteration, done as discrete store steps so its
## intermediate arrays are kept; handed to near-far-solve as --near-potential.
############################################################################
echo "### [4/6] near 0.05mm solve, Dirichlet interface + no-flux FR4 (sweep 0)"

want "potential/near increment/near" \
     pochoir fdm --nepochs 10 --epoch 130000000 --precision 0.00000000002 \
     --edges per,per,fix --engine torch \
     --initial initial/near_bc --boundary boundary/near_bc \
     --insulator initial/near_insulator \
     --potential potential/near --increment increment/near

date

############################################################################
## STEP 4: THE ITERATIVE LOOP (the reorganization).
##
## near-far-solve alternates far and near solves, resampling across the 8x pitch
## ratio each way (downsample near->coarse for the far pin, upsample far->near
## for the near pin), until the near solution stops changing by more than
## 2e-8 V.  --overlap 3 pins the far 3 coarse cells (1.2mm) below the interface
## instead of 1: the near already covers that depth, and the wider overlap both
## converges much faster than the ~0.95/sweep single-cell rate and pushes the
## seam toward gradient continuity.
##
## --insulator applies the node-centered no-flux FR4 BC to every NEAR re-solve
## (without it the loop would DISCARD the Neumann field computed in step 3).
## The far re-solves never see it -- the FR4 is unresolved at 0.4mm.
##
## Outputs overwrite potential/near and add potential/far.
############################################################################
echo "### [5/6] iterative near<->far Schwarz sweeps to 2e-8 V (interface z=2cm)"

want "potential/near_converged potential/far" \
     pochoir near-far-solve \
     --coarse-potential potential/coarse \
     --coarse-initial initial/coarse --coarse-boundary boundary/coarse \
     --near-initial initial/near --near-boundary boundary/near \
     --near-potential potential/near \
     --insulator initial/near_insulator \
     --interface '20*mm' --axis 2 \
     --edges per,per,fix --engine torch \
     --epoch 130000000 --nepochs 10 \
     --near-precision 2e-11 --far-precision 2e-8 \
     --tol '0.00000002*V' --max-iters 12 --overlap 3 \
     --near-out potential/near_converged --far-out potential/far

date

############################################################################
## STEP 5: stitch the converged near onto the converged far.
##
## The fine full-volume gen also stores initial/fine_insulator on the 1201-deep
## grid (kept for reference/plots; NOT passed to velo or drift -- see the
## enforcement-free note in the header).
############################################################################
echo "### [6/6] stitch converged near onto converged far -> potential/drift3d"

want domain/fine \
     pochoir domain --domain domain/fine --shape=88,88,1201 --spacing '0.05*mm'

want "initial/fine boundary/fine" \
     pochoir gen --generator $gen --domain domain/fine \
     --initial initial/fine --boundary boundary/fine $dcfg

want potential/drift3d \
     pochoir stitch-near --near potential/near_converged --coarse potential/far \
     --domain domain/fine --output potential/drift3d

date

############################################################################
## VELOCITY + DRIFT PATHS  (drift only -- no weighting field, no induced current)
############################################################################
echo "=== Velocities (enforcement-free: pure mu*grad(phi), no --boundary/--insulator) ==="

want velocity/drift3d \
     pochoir velo --temperature '87.0*K' \
     --potential potential/drift3d \
     --velocity velocity/drift3d

echo "=== Starts (config-driven 10x10 cell-centred tile grid at z=59.95mm) ==="

want starts/drift3d \
     pochoir starts --starts starts/drift3d \
     -m yes -c $dcfg --plot

echo "=== Paths ==="
## ~1.5 us/mm at 50 V/mm x 49.95mm transit ~= 75us; 0..120us gives ~1.6x margin.
## --interp-order linear: cubic RGI overshoots/rings across the potential kink at
## the pad plane and drags paths toward pixel centre.
want paths/drift3d \
     pochoir drift --starts starts/drift3d \
     --velocity velocity/drift3d \
     --interp-order linear \
     --paths paths/drift3d '0*us,120*us,0.05*us' --plot

date

echo "### overshoot check (pad plane z = 10 mm)"
{ echo; echo "## Drift-path result (overshoot check, pad z = 10 mm)"; echo '```'
  $PY OVERSHOOT_STUDY/check_paths.py "$POCHOIR_STORE/paths/drift3d.npz" 10; echo '```'
} | tee -a "$POCHOIR_STORE/GEOMETRY_AND_CHAMFER.md"

echo "=== DONE: Task13 iterative hybrid 5cm drift field + paths -> $POCHOIR_STORE ==="
