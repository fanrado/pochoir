#!/bin/bash
#
# Two-step coarse->fine drift-field FDM.
#
# Step 1: solve the pixel drift potential on a coarse 11x11x375 grid at
#         0.4mm spacing to a loose precision (2e-6).  This converges
#         quickly and captures the large-scale field.
#
# Step 2: refine (4x upsample + exact fine-boundary merge) the coarse
#         potential onto the fine 44x44x1500 grid at 0.1mm spacing and use
#         it as the FDM initial guess, then solve to tight precision (2e-8).
#         Starting from the refined coarse solution greatly reduces the
#         number of fine-grid iterations needed.
#
# Usage: ./test-full-3d-drift-pixel-coarse2fine.sh [STORE_DIR]

set -e

export POCHOIR_STORE="${1:-store}"

source helpers.sh

date

gen="pcb_drift_pixel_with_grid"
cfg="example_gen_pcb_drift_pixel_with_grid.json"

## ---------------------------------------------------------------------------
## Step 1: coarse solve (0.4mm, 11x11x375)
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
     --nepochs 1 --epoch 130000000 --precision 0.000002 \
     --edges per,per,fix \
     --engine torch \
     --initial initial/coarse --boundary boundary/coarse \
     --potential potential/coarse \
     --increment increment/coarse

date

## ---------------------------------------------------------------------------
## Step 2: fine solve (0.1mm, 44x44x1500), seeded by the refined coarse field
## ---------------------------------------------------------------------------

want domain/fine \
     pochoir domain --domain domain/fine \
     --shape=44,44,1500 --spacing '0.1*mm'

# Generate the fine initial/boundary arrays (exact fine boundary values).
want initial/fine \
     pochoir gen --generator $gen --domain domain/fine \
     --initial initial/fine --boundary boundary/fine \
     $cfg

# Upsample the coarse potential onto the fine grid and merge the exact
# fine boundary values, producing the FDM initial guess.
want initial/fine_refined \
     pochoir refine \
     --coarse potential/coarse \
     --initial initial/fine \
     --boundary boundary/fine \
     --output initial/fine_refined

want potential/fine \
     pochoir fdm \
     --nepochs 1 --epoch 130000000 --precision 0.00000002 \
     --edges per,per,fix \
     --engine torch \
     --initial initial/fine_refined --boundary boundary/fine \
     --potential potential/fine \
     --increment increment/fine

date
