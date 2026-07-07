#!/bin/bash
#
# Full pixel field calculation via near-field-only refinement.
#
# Combines two near-field workflows into one driver:
#   PART A: drift field      (was test-full-3d-drift-pixel-nearfield.sh)
#   PART B: weighting field  (was test-full-3d-pixel-weight-nearfield.sh)
#
# Each part runs: coarse 0.4mm full-depth solve -> fine near-field
# refine (z=0..20mm, ~7.5x fewer cells) with a Dirichlet interface at
# z=20mm (`near-bc`) pinned to the coarse bulk, then `stitch-near`
# combines the fine near-field with the upsampled coarse far-field into
# the full 0.1mm grid.
#
#   drift     -> potential/drift3d
#   weighting -> potential/weight3d   (consumed by induce-pixel)
#
# Store keys are non-overlapping between the two parts (drift uses
# coarse/near/fine/full; weighting uses weight_*).
#
# Usage: ./run-full-3d-pixel.sh [STORE_DIR]

set -e

export POCHOIR_STORE="${1:-store}"

source helpers.sh

# ---------------------------------------------------------------------------
# want: run a step only when its output(s) are missing.
#
# Overrides the single-key `want` from helpers.sh.  The first argument is a
# space-separated list of store keys (no .npz suffix).  The step is SKIPPED
# only when EVERY listed output already exists under $POCHOIR_STORE; if any
# is missing the command is run and all outputs are verified afterwards.
# This lets a re-run reuse existing results and resume an interrupted step
# (e.g. an fdm that wrote potential/ but not increment/).
# Backward compatible with a single key (the loop just runs once).
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
## PART A: DRIFT FIELD  (near-field refinement)
############################################################################
export POCHOIR_LOG="${POCHOIR_STORE}/pochoir_driftfield.log"

gen="pcb_drift_pixel_with_grid"
cfg="example_gen_pcb_drift_pixel_with_grid.json"

## ---------------------------------------------------------------------------
## Step 1: coarse solve (0.4mm, 11x11x375), full drift region
## ---------------------------------------------------------------------------

want domain/coarse \
     pochoir domain --domain domain/coarse \
     --shape=11,11,775 --spacing '0.4*mm'

want "initial/coarse boundary/coarse" \
     pochoir gen --generator $gen --domain domain/coarse \
     --initial initial/coarse --boundary boundary/coarse \
     $cfg

want "potential/coarse increment/coarse" \
     pochoir fdm \
     --nepochs 10 --epoch 130000000 --precision 0.0000002 \
     --edges per,per,fix \
     --engine torch \
     --initial initial/coarse --boundary boundary/coarse \
     --potential potential/coarse \
     --increment increment/coarse

date

## ---------------------------------------------------------------------------
## Step 2: near-field gen + refined coarse seed (0.05mm, 88x88x401, z=0..20mm)
## ---------------------------------------------------------------------------

want domain/near \
     pochoir domain --domain domain/near \
     --shape=88,88,401 --spacing '0.05*mm'
     #--shape=44,44,201 --spacing '0.05*mm'

want "initial/near boundary/near" \
     pochoir gen --generator $gen --domain domain/near \
     --initial initial/near --boundary boundary/near \
     $cfg

# Seed the near-field interior with the upsampled coarse solution.
want initial/near_refined \
     pochoir refine \
     --coarse potential/coarse \
     --initial initial/near \
     --boundary boundary/near \
     --output initial/near_refined

## ---------------------------------------------------------------------------
## Step 3: Dirichlet interface plane at z=20mm from the coarse bulk
## ---------------------------------------------------------------------------

want "initial/near_bc boundary/near_bc" \
     pochoir near-bc \
     --initial initial/near_refined \
     --boundary boundary/near \
     --coarse potential/coarse \
     --initial-out initial/near_bc \
     --boundary-out boundary/near_bc

## ---------------------------------------------------------------------------
## Step 4: near-field fine solve (0.05mm), seeded + pinned interface (sweep 0)
## ---------------------------------------------------------------------------

want "potential/near increment/near" \
     pochoir fdm \
     --nepochs 10 --epoch 130000000 --precision 0.00000000002 \
     --edges per,per,fix \
     --engine torch \
     --initial initial/near_bc --boundary boundary/near_bc \
     --potential potential/near \
     --increment increment/near

date

## ---------------------------------------------------------------------------
## Step 4b: overlapping-Schwarz refinement for a continuous interface
## ---------------------------------------------------------------------------
# Steps 2-4 above are the original single-pin pipeline and keep producing all
# their intermediate store keys (initial/near_refined, initial/near_bc,
# boundary/near_bc, potential/near, increment/near) so a later run/test can
# resume from them.  That pin gives only value (C0) continuity at z=20mm, so
# E_z = -dphi/dz is discontinuous at the seam.  This step refines it: starting
# from the sweep-0 near solve, it alternately re-solves the full coarse far
# domain (interior plane one coarse cell below the interface pinned to the near
# solution) and the fine near domain (interface pinned to the far solution),
# iterating to convergence.  The result is continuous in value AND gradient
# across z=20mm.  It updates potential/near and writes the consistent far field.
want "potential/near potential/far" \
     pochoir near-far-solve \
     --coarse-potential potential/coarse \
     --coarse-initial initial/coarse --coarse-boundary boundary/coarse \
     --near-initial initial/near --near-boundary boundary/near \
     --near-potential potential/near \
     --interface '20*mm' --axis 2 \
     --edges per,per,fix --engine torch \
     --epoch 130000000 --nepochs 10 \
     --near-precision 2e-11 --far-precision 2e-7 \
     --tol '1*V' --max-iters 6 \
     --near-out potential/near --far-out potential/far

date

## ---------------------------------------------------------------------------
## Step 5: stitch the 0.05mm near solve onto the Schwarz-updated far field
##         (potential/far), keeping the whole stitched drift potential at
##         0.05mm by linearly upsampling the far field (no near coarsen).
## ---------------------------------------------------------------------------

# The near solve stays at its native 0.05mm (88x88x401).  The full stitched
# domain is 0.05mm (88x88x6200): stitch-near upsamples the coarse/far field
# (potential/far) to 0.05mm with linear RGI and overwrites the first 401 z
# planes (z 0..20mm) with the near solve.  Transverse shapes match (88x88).
want domain/fine \
     pochoir domain --domain domain/fine \
     --shape=88,88,6200 --spacing '0.05*mm'

# Generate the full-fine electrode geometry (boundary mask) for the
# stitched domain.  The FDM solve is NOT run here (that is the whole
# point of the near-field workflow); this only builds the boundary array
# so PART C's velo can zero the E-field at electrode cells.
want "boundary/fine initial/fine" \
     pochoir gen --generator $gen --domain domain/fine \
     --initial initial/fine --boundary boundary/fine \
     $cfg

want potential/drift3d \
     pochoir stitch-near \
     --near potential/near \
     --coarse potential/far \
     --domain domain/fine \
     --output potential/drift3d

date

# ############################################################################
# ## PART B: WEIGHTING FIELD  (near-field refinement)
# ############################################################################
# export POCHOIR_LOG="${POCHOIR_STORE}/pochoir_weightingfield.log"
# 
# gen="pcb_pixel_with_grid"
# cfg="example_gen_pixel_with_grid.json"
# 
# ## ---------------------------------------------------------------------------
# ## Step 1: coarse weighting solve (0.4mm, 99x99x375), full drift depth
# ## ---------------------------------------------------------------------------
# 
# want domain/weight_coarse \
#      pochoir domain --domain domain/weight_coarse \
#      --shape=55,55,775 --spacing '0.4*mm'
#     # --shape=99,99,775 --spacing '0.4*mm'
# 
# want "initial/weight_coarse boundary/weight_coarse" \
#      pochoir gen --generator $gen --domain domain/weight_coarse \
#      --initial initial/weight_coarse --boundary boundary/weight_coarse \
#      $cfg
# 
# want "potential/weight_coarse increment/weight_coarse" \
#      pochoir fdm \
#      --nepochs 10 --epoch 130000000 --precision 0.0000002 \
#      --edges fix,fix,fix \
#      --engine torch \
#      --initial initial/weight_coarse --boundary boundary/weight_coarse \
#      --potential potential/weight_coarse \
#      --increment increment/weight_coarse \
#      --multisteps no
# 
# date
# 
# ## ---------------------------------------------------------------------------
# ## Step 2: near-field gen + refined coarse seed (0.1mm, 396x396x201, z=0..20mm)
# ## ---------------------------------------------------------------------------
# 
# want domain/weight_near \
#      pochoir domain --domain domain/weight_near \
#      --shape=440,440,401 --spacing '0.05*mm'
#      #--shape=220,220,201 --spacing '0.05*mm'
#      #--shape=396,396,201 --spacing '0.1*mm'
# 
# want "initial/weight_near boundary/weight_near" \
#      pochoir gen --generator $gen --domain domain/weight_near \
#      --initial initial/weight_near --boundary boundary/weight_near \
#      $cfg
# 
# # Seed the near-field interior with the upsampled coarse weighting solution.
# want initial/weight_near_refined \
#      pochoir refine \
#      --coarse potential/weight_coarse \
#      --initial initial/weight_near \
#      --boundary boundary/weight_near \
#      --output initial/weight_near_refined
# 
# ## ---------------------------------------------------------------------------
# ## Step 3: Dirichlet interface plane at z=20mm from the coarse bulk
# ## ---------------------------------------------------------------------------
# 
# want "initial/weight_near_bc boundary/weight_near_bc" \
#      pochoir near-bc \
#      --initial initial/weight_near_refined \
#      --boundary boundary/weight_near \
#      --coarse potential/weight_coarse \
#      --initial-out initial/weight_near_bc \
#      --boundary-out boundary/weight_near_bc
# 
# ## ---------------------------------------------------------------------------
# ## Step 4: near-field fine weighting solve (0.1mm), seeded + pinned interface
# ## ---------------------------------------------------------------------------
# 
# want "potential/weight_near increment/weight_near" \
#      pochoir fdm \
#      --nepochs 10 --epoch 130000000 --precision 0.0000000002 \
#      --edges fix,fix,fix \
#      --engine torch \
#      --initial initial/weight_near_bc --boundary boundary/weight_near_bc \
#      --potential potential/weight_near \
#      --increment increment/weight_near \
#      --multisteps no
# 
# date
# 
# ## ---------------------------------------------------------------------------
# ## Step 5: coarsen the 0.05mm near weighting solve back to 0.1mm, then stitch.
# ##         Stored as potential/weight3d for downstream induce-pixel.
# ## ---------------------------------------------------------------------------
# 
# # As in PART A, the near weighting solve ran at 0.05mm (440x440x401) for a
# # symmetric pixel tile.  Stride-downsample by 2 to 0.1mm (220x220x201) before
# # stitching, avoiding the 440x440x6200 full-fine domain (OOM).
# want domain/weight_near_01 \
#      pochoir domain --domain domain/weight_near_01 \
#      --shape=220,220,201 --spacing '0.1*mm'
# 
# want potential/weight_near_01 \
#      pochoir coarsen \
#      --input potential/weight_near \
#      --domain domain/weight_near_01 \
#      --output potential/weight_near_01
# 
# want domain/weight_full \
#      pochoir domain --domain domain/weight_full \
#      --shape=220,220,3100 --spacing '0.1*mm'
# 
# want potential/weight3d \
#      pochoir stitch-near \
#      --near potential/weight_near_01 \
#      --coarse potential/weight_coarse \
#      --domain domain/weight_full \
#      --output potential/weight3d
# 
# date

############################################################################
## PART C: VELOCITY, PATHS, INDUCED CURRENT
############################################################################
## Uses the drift potential (potential/drift3d) and the weighting field
## (potential/weight3d) produced above.

echo "=== Velocities ==="
## Drift velocity field from the stitched drift potential.
want velocity/drift3d \
     pochoir velo --temperature '87.0*K' \
     --potential potential/drift3d \
     --boundary boundary/fine \
     --velocity velocity/drift3d

echo "=== Paths ==="
## 10x10 grid per pixel (0.44 mm spacing), 100 starting points total,
## launched from the cathode plane (z=148 mm).
dist=(0.22 0.66 1.1 1.54 1.98 2.42 2.86 3.3 3.74 4.18)
points=()
for d in "${dist[@]}"; do
     for d2 in "${dist[@]}"; do
         points+=("${d}*mm,${d2}*mm,308*mm")
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
     --interp-order linear \
     --paths paths/drift3d_tight '0*us,210*us,0.05*us' \
     --plot
# --interp-order linear: cubic rings/overshoots near the pixel plane (strong
# geometry) and over-focuses paths onto the pads; linear is monotone-safe.
# Evidence: test/interp_overshoot_test.py, docs/drift/drift-path-ode-integration.md.

# echo "=== Induced currents ==="
# ## Induced current on the pixel via Ramo (weighting field x drift paths).
# want current/induced_current \
#      pochoir induce-pixel --weighting potential/weight3d \
#      --paths paths/drift3d_tight \
#      --output current/induced_current \
#      --npixels 2 \
#      --config example_gen_pixel_with_grid.json \
#      --plot

date
