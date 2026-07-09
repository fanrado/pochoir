#!/bin/bash
#
# DEBUG harness: short 10cm drift variant of run-full-3d-pixel.sh.
#
# Purpose (pochoir-0oum): fast-iteration reproduction of the hybrid
# near-far drift-solver current DISCONTINUITY.  The reference driver
# run-full-3d-pixel.sh drifts the full 30.8cm (driftZDepth=308mm); at that
# depth a single full pipeline run is slow, which makes debugging the
# near-pad overshoot (paths slipping past the pad edge -> bipolar Ramo
# current) painfully slow to iterate on.
#
# This script drifts only 10cm (DRIFT_MM below) so the coarse/far solves,
# the stitched full domain, and the drift-path integration all shrink,
# while the *near-field* domains (z=0..20mm, the suspect strong-geometry
# region we are debugging) are kept BYTE-FOR-BYTE identical to the
# reference driver so the overshoot behaviour is preserved.
#
# What changes vs run-full-3d-pixel.sh:
#   * full-depth z cell counts scale with DRIFT_MM (coarse/fine/weight_*)
#   * drift launch plane and drift time window scale with DRIFT_MM
#   * the weighting generator's grounded cathode plane (driftZDepth) is
#     pinned to the shorter depth via example_gen_pixel_with_grid_debug.json
# What is UNCHANGED (so the bug still reproduces):
#   * near/weight_near domains (88x88x401, 440x440x401) and all solver
#     parameters (epochs/precision/edges), the Schwarz near-far-solve, the
#     stitch, and the --interp-order linear drift.
#
# The reference driver run-full-3d-pixel.sh is intentionally left untouched.
#
# Usage: ./run-full-3d-pixel-debug.sh [STORE_DIR]   (default: store_debug)

set -e

export POCHOIR_STORE="${1:-store_debug10cm}"

source helpers.sh

# ---------------------------------------------------------------------------
# Short-drift geometry.  Everything full-depth is derived from DRIFT_MM so
# there is a single knob to lengthen/shorten the debug drift.
#
#   DRIFT_MM      physical drift length (cathode-to-launch), 100mm = 10cm.
#   DOMAIN_MM     full z extent = DRIFT_MM + 2mm cathode margin (mirrors the
#                 reference driver's 310mm domain for a 308mm drift).
#
# z cell counts (transverse shapes are unchanged from the reference):
#   coarse / weight_coarse : 0.4mm  -> DOMAIN_MM/0.4  = 255
#   fine (stitched drift)  : 0.05mm -> DOMAIN_MM/0.05 = 2040
#   weight_full            : 0.1mm  -> DOMAIN_MM/0.1  = 1020
#   near / weight_near     : z=0..20mm -> 401 (UNCHANGED)
#   weight_near_01         : z=0..20mm -> 201 (UNCHANGED)
# ---------------------------------------------------------------------------
DRIFT_MM=100
Z_COARSE=255      # DOMAIN_MM(102) / 0.4mm
Z_FINE=2040       # DOMAIN_MM(102) / 0.05mm
Z_WFULL=1020      # DOMAIN_MM(102) / 0.1mm

# ---------------------------------------------------------------------------
# want: run a step only when its output(s) are missing.  (copied verbatim
# from run-full-3d-pixel.sh; see there for the full rationale.)
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
echo "=== DEBUG short-drift harness: DRIFT_MM=${DRIFT_MM} (10cm), store=${POCHOIR_STORE} ==="

############################################################################
## PART A: DRIFT FIELD  (near-field refinement)
############################################################################
export POCHOIR_LOG="${POCHOIR_STORE}/pochoir_driftfield.log"

gen="pcb_drift_pixel_with_grid"
# The drift generator places the cathode at the last z-plane, so the physical
# drift field = CathodePotential / gap.  Over the short 100mm gap the reference
# -15400V would give ~150 V/mm (3x too strong) and change the near-pad physics
# under debug.  The debug config scales CathodePotential to -5000V so the bulk
# field stays ~-50 V/mm (5000V/99.6mm), matching the reference 30cm run
# (15400V/307.6mm).  Laplace streamlines are voltage-scale-invariant so the
# overshoot still reproduces; this just keeps the drift speed/physics faithful.
cfg="example_gen_pcb_drift_pixel_with_grid_debug.json"

## ---------------------------------------------------------------------------
## Step 1: coarse solve (0.4mm, 11x11xZ_COARSE), short drift region
## ---------------------------------------------------------------------------

want domain/coarse \
     pochoir domain --domain domain/coarse \
     --shape=11,11,${Z_COARSE} --spacing '0.4*mm'

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
##         (UNCHANGED from reference driver -- suspect region under debug)
## ---------------------------------------------------------------------------

want domain/near \
     pochoir domain --domain domain/near \
     --shape=88,88,401 --spacing '0.05*mm'

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
##          (identical params to reference driver)
## ---------------------------------------------------------------------------
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
## Step 5: stitch the 0.05mm near solve onto the Schwarz-updated far field.
## ---------------------------------------------------------------------------

want domain/fine \
     pochoir domain --domain domain/fine \
     --shape=88,88,${Z_FINE} --spacing '0.05*mm'

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

############################################################################
## PART B: WEIGHTING FIELD  (near-field refinement, overlapping-Schwarz)
############################################################################
export POCHOIR_LOG="${POCHOIR_STORE}/pochoir_weightingfield.log"

gen="pcb_pixel_with_grid"
# Debug weighting config: identical to example_gen_pixel_with_grid.json but
# with driftZDepth=100 so the grounded cathode plane lands at the shortened
# domain depth (weight_coarse/weight_full).
cfg="example_gen_pixel_with_grid_debug.json"

## ---------------------------------------------------------------------------
## Step 1: coarse weighting solve (0.4mm, 55x55xZ_COARSE), short drift depth
## ---------------------------------------------------------------------------

want domain/weight_coarse \
     pochoir domain --domain domain/weight_coarse \
     --shape=55,55,${Z_COARSE} --spacing '0.4*mm'

want "initial/weight_coarse boundary/weight_coarse" \
     pochoir gen --generator $gen --domain domain/weight_coarse \
     --initial initial/weight_coarse --boundary boundary/weight_coarse \
     $cfg

want "potential/weight_coarse increment/weight_coarse" \
     pochoir fdm \
     --nepochs 10 --epoch 130000000 --precision 0.0000002 \
     --edges fix,fix,fix \
     --engine torch \
     --initial initial/weight_coarse --boundary boundary/weight_coarse \
     --potential potential/weight_coarse \
     --increment increment/weight_coarse \
     --multisteps no

date

## ---------------------------------------------------------------------------
## Step 2: near-field gen + refined coarse seed (0.05mm, 440x440x401, z=0..20mm)
##         (UNCHANGED from reference driver -- suspect region under debug)
## ---------------------------------------------------------------------------

want domain/weight_near \
     pochoir domain --domain domain/weight_near \
     --shape=440,440,401 --spacing '0.05*mm'

want "initial/weight_near boundary/weight_near" \
     pochoir gen --generator $gen --domain domain/weight_near \
     --initial initial/weight_near --boundary boundary/weight_near \
     $cfg

# Seed the near-field interior with the upsampled coarse weighting solution.
want initial/weight_near_refined \
     pochoir refine \
     --coarse potential/weight_coarse \
     --initial initial/weight_near \
     --boundary boundary/weight_near \
     --output initial/weight_near_refined

## ---------------------------------------------------------------------------
## Step 3: Dirichlet interface plane at z=20mm from the coarse bulk (sweep-0 seed)
## ---------------------------------------------------------------------------

want "initial/weight_near_bc boundary/weight_near_bc" \
     pochoir near-bc \
     --initial initial/weight_near_refined \
     --boundary boundary/weight_near \
     --coarse potential/weight_coarse \
     --initial-out initial/weight_near_bc \
     --boundary-out boundary/weight_near_bc

## ---------------------------------------------------------------------------
## Step 4: near-field fine weighting solve (0.05mm), seeded + pinned interface
## ---------------------------------------------------------------------------

want "potential/weight_near increment/weight_near" \
     pochoir fdm \
     --nepochs 10 --epoch 130000000 --precision 0.0000000002 \
     --edges fix,fix,fix \
     --engine torch \
     --initial initial/weight_near_bc --boundary boundary/weight_near_bc \
     --potential potential/weight_near \
     --increment increment/weight_near \
     --multisteps no

date

## ---------------------------------------------------------------------------
## Step 4b: overlapping-Schwarz refinement for a continuous interface
##          (identical params to reference driver)
## ---------------------------------------------------------------------------
want "potential/weight_near potential/weight_far" \
     pochoir near-far-solve \
     --coarse-potential potential/weight_coarse \
     --coarse-initial initial/weight_coarse --coarse-boundary boundary/weight_coarse \
     --near-initial initial/weight_near --near-boundary boundary/weight_near \
     --near-potential potential/weight_near \
     --interface '20*mm' --axis 2 \
     --edges fix,fix,fix --engine torch \
     --epoch 130000000 --nepochs 10 \
     --near-precision 2e-10 --far-precision 2e-7 \
     --tol '0.001*V' --max-iters 6 \
     --near-out potential/weight_near --far-out potential/weight_far

date

## ---------------------------------------------------------------------------
## Step 5: coarsen the 0.05mm near weighting solve to 0.1mm, then stitch.
## ---------------------------------------------------------------------------

want domain/weight_near_01 \
     pochoir domain --domain domain/weight_near_01 \
     --shape=220,220,201 --spacing '0.1*mm'

want potential/weight_near_01 \
     pochoir coarsen \
     --input potential/weight_near \
     --domain domain/weight_near_01 \
     --output potential/weight_near_01

want domain/weight_full \
     pochoir domain --domain domain/weight_full \
     --shape=220,220,${Z_WFULL} --spacing '0.1*mm'

want potential/weight3d \
     pochoir stitch-near \
     --near potential/weight_near_01 \
     --coarse potential/weight_far \
     --domain domain/weight_full \
     --output potential/weight3d

date

############################################################################
## PART C: VELOCITY, PATHS, INDUCED CURRENT
############################################################################

echo "=== Velocities ==="
want velocity/drift3d \
     pochoir velo --temperature '87.0*K' \
     --potential potential/drift3d \
     --boundary boundary/fine \
     --velocity velocity/drift3d

echo "=== Paths ==="
## 10x10 grid per pixel (0.44 mm spacing), 100 starting points total,
## launched from the short-drift cathode plane (z=DRIFT_MM=100mm).
dist=(0.22 0.66 1.1 1.54 1.98 2.42 2.86 3.3 3.74 4.18)
points=()
for d in "${dist[@]}"; do
     for d2 in "${dist[@]}"; do
         points+=("${d}*mm,${d2}*mm,${DRIFT_MM}*mm")
     done
done

want starts/drift3d \
     pochoir starts --starts starts/drift3d \
     -m no \
     -c example_gen_pixel_with_grid_debug.json \
     "${points[@]}" \
     --plot

# Drift time window scaled for the 10cm drift.  A shorter drift with the same
# electrode voltages gives a stronger (and hence faster) field, so the full
# 210us reference window is unnecessary; 90us with the same 0.05us step leaves
# generous margin for all paths to reach and be collected at the pad.
want paths/drift3d_tight \
     pochoir drift --starts starts/drift3d \
     --velocity velocity/drift3d \
     --interp-order linear \
     --paths paths/drift3d_tight '0*us,90*us,0.05*us' \
     --plot
# --interp-order linear: cubic rings/overshoots near the pixel plane (strong
# geometry) and over-focuses paths onto the pads; linear is monotone-safe.

echo "=== Induced currents ==="
want current/induced_current \
     pochoir induce-pixel --weighting potential/weight3d \
     --paths paths/drift3d_tight \
     --output current/induced_current \
     --npixels 2 \
     --config example_gen_pixel_with_grid_debug.json \
     --plot

date
