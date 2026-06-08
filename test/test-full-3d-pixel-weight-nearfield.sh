#!/bin/bash
#
# Near-field-only refinement of the pixel WEIGHTING field.
#
# Applies the pochoir-3jo near-field method (coarse full-depth solve +
# fine near-field refine + Dirichlet interface at z=20mm + stitch) to
# the weighting potential.  The full fine weighting solve is
# 396x396x1500 ~= 235M cells (very slow); refining only z=0..20mm
# (396x396x201 ~= 31.5M cells, ~7.5x smaller) plus a coarse 0.4mm
# full-depth solve is far cheaper.
#
# The refine / near-bc / stitch-near commands are generator-agnostic and
# are reused UNCHANGED from the drift near-field driver
# (test-full-3d-drift-pixel-nearfield.sh); only the parameters differ:
#   - generator pcb_pixel_with_grid, config example_gen_pixel_with_grid.json
#   - edges fix,fix,fix (transverse weighting -> 0 at the 39.6mm edges)
#   - --multisteps no
#
# The stitched full 396x396x1500 field is stored as potential/weight3d so
# downstream `induce-pixel` consumes it unchanged.
#
# Domains / index alignment (39.6mm x 39.6mm x 150mm):
#   coarse weight  99,99,375   @0.4mm  (39.6/0.4=99, 150/0.4=375; z=20mm -> idx 50)
#   full fine      396,396,1500 @0.1mm (396=4*99, 1500=4*375)  [gen geometry]
#   near fine      396,396,201  @0.1mm (interface at z-idx 200 = 20.0mm)
#
# Usage: ./test-full-3d-pixel-weight-nearfield.sh [STORE_DIR]

set -e

export POCHOIR_STORE="${1:-store}"
export POCHOIR_LOG="${POCHOIR_STORE}/pochoir_weightingfield.log"

source helpers.sh

date

gen="pcb_pixel_with_grid"
cfg="example_gen_pixel_with_grid.json"

## ---------------------------------------------------------------------------
## Step 1: coarse weighting solve (0.4mm, 99x99x375), full drift depth
## ---------------------------------------------------------------------------

want domain/weight_coarse \
     pochoir domain --domain domain/weight_coarse \
     --shape=99,99,375 --spacing '0.4*mm'

want initial/weight_coarse \
     pochoir gen --generator $gen --domain domain/weight_coarse \
     --initial initial/weight_coarse --boundary boundary/weight_coarse \
     $cfg

want potential/weight_coarse \
     pochoir fdm \
     --nepochs 1 --epoch 130000000 --precision 0.000002 \
     --edges fix,fix,fix \
     --engine torch \
     --initial initial/weight_coarse --boundary boundary/weight_coarse \
     --potential potential/weight_coarse \
     --increment increment/weight_coarse \
     --multisteps no

date

## ---------------------------------------------------------------------------
## Step 2: near-field gen + refined coarse seed (0.1mm, 396x396x201, z=0..20mm)
## ---------------------------------------------------------------------------

want domain/weight_near \
     pochoir domain --domain domain/weight_near \
     --shape=396,396,201 --spacing '0.1*mm'

want initial/weight_near \
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
## Step 3: Dirichlet interface plane at z=20mm from the coarse bulk
## ---------------------------------------------------------------------------

want initial/weight_near_bc \
     pochoir near-bc \
     --initial initial/weight_near_refined \
     --boundary boundary/weight_near \
     --coarse potential/weight_coarse \
     --initial-out initial/weight_near_bc \
     --boundary-out boundary/weight_near_bc

## ---------------------------------------------------------------------------
## Step 4: near-field fine weighting solve (0.1mm), seeded + pinned interface
## ---------------------------------------------------------------------------

want potential/weight_near \
     pochoir fdm \
     --nepochs 1 --epoch 130000000 --precision 0.00000002 \
     --edges fix,fix,fix \
     --engine torch \
     --initial initial/weight_near_bc --boundary boundary/weight_near_bc \
     --potential potential/weight_near \
     --increment increment/weight_near \
     --multisteps no

date

## ---------------------------------------------------------------------------
## Step 5: stitch near fine + coarse far into the full fine grid.
##         Stored as potential/weight3d for downstream induce-pixel.
## ---------------------------------------------------------------------------

want domain/weight_full \
     pochoir domain --domain domain/weight_full \
     --shape=396,396,1500 --spacing '0.1*mm'

want potential/weight3d \
     pochoir stitch-near \
     --near potential/weight_near \
     --coarse potential/weight_coarse \
     --domain domain/weight_full \
     --output potential/weight3d

date
