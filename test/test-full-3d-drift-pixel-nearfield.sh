#!/bin/bash
#
# Near-field-only refinement of the pixel drift field.
#
# Instead of solving the full 150mm drift region at 0.1mm (44x44x1500),
# only the near-field z=0..20mm is refined to fine resolution
# (44x44x201, ~7.5x fewer cells).  The bulk z>20mm is a near-linear
# drift region already captured by the coarse 0.4mm solve.
#
# Continuity at the z=20mm interface is enforced as a Dirichlet
# condition (`near-bc`): the near-field top plane is pinned to the
# coarse bulk potential there.  After the near solve, `stitch-near`
# combines the fine near-field with the upsampled coarse far-field into
# a full 44x44x1500 potential.
#
# Steps:
#   1. coarse solve            11x11x375  @0.4mm -> prec 2e-6
#   2. near gen + refine seed  44x44x201  @0.1mm
#   3. near-bc interface plane (Dirichlet at z=20mm from coarse)
#   4. near fine solve         44x44x201  @0.1mm -> prec 2e-8
#   5. stitch-near             -> full 44x44x1500 potential
#
# Usage: ./test-full-3d-drift-pixel-nearfield.sh [STORE_DIR]

set -e

export POCHOIR_STORE="${1:-store}"

source helpers.sh

date

gen="pcb_drift_pixel_with_grid"
cfg="example_gen_pcb_drift_pixel_with_grid.json"

## ---------------------------------------------------------------------------
## Step 1: coarse solve (0.4mm, 11x11x375), full drift region
## ---------------------------------------------------------------------------

want domain/coarse \
     pochoir domain --domain domain/coarse \
     --shape=11,11,375 --spacing '0.4*mm'

want initial/coarse \
     pochoir gen --generator $gen --domain domain/coarse \
     --initial initial/coarse --boundary boundary/coarse \
     $cfg

want potential/coarse \
     pochoir fdm \
     --nepochs 10 --epoch 130000000 --precision 0.000000002 \
     --edges per,per,fix \
     --engine torch \
     --initial initial/coarse --boundary boundary/coarse \
     --potential potential/coarse \
     --increment increment/coarse

date

## ---------------------------------------------------------------------------
## Step 2: near-field gen + refined coarse seed (0.1mm, 44x44x201, z=0..20mm)
## ---------------------------------------------------------------------------

want domain/near \
     pochoir domain --domain domain/near \
     --shape=44,44,201 --spacing '0.1*mm'

want initial/near \
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

want initial/near_bc \
     pochoir near-bc \
     --initial initial/near_refined \
     --boundary boundary/near \
     --coarse potential/coarse \
     --initial-out initial/near_bc \
     --boundary-out boundary/near_bc

## ---------------------------------------------------------------------------
## Step 4: near-field fine solve (0.1mm), seeded + pinned interface
## ---------------------------------------------------------------------------

want potential/near \
     pochoir fdm \
     --nepochs 10 --epoch 130000000 --precision 0.0000000002 \
     --edges per,per,fix \
     --engine torch \
     --initial initial/near_bc --boundary boundary/near_bc \
     --potential potential/near \
     --increment increment/near

date

## ---------------------------------------------------------------------------
## Step 5: stitch near fine + coarse far into the full fine grid
## ---------------------------------------------------------------------------

want domain/fine \
     pochoir domain --domain domain/fine \
     --shape=44,44,1500 --spacing '0.1*mm'

want potential/full \
     pochoir stitch-near \
     --near potential/near \
     --coarse potential/coarse \
     --domain domain/fine \
     --output potential/full

date
