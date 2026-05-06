#!/bin/bash
# Full 3-D pixel pipeline using the multi-resolution FDM command.
#
# Usage:
#   cd test/
#   bash test-full-3d-pixel-multires.sh [store_dir]
#
# This script mirrors test-full-3d-pixel.sh but replaces the `pochoir fdm`
# calls with `pochoir fdm-multires`, which solves on a coarse-to-fine grid
# sequence before reaching the target resolution.  The drift-field pipeline
# uses 5 stages (5 → 3 → 1 → 0.5 → 0.1 mm) and the weighting-field pipeline
# uses 3 stages (1 → 0.5 → 0.1 mm).
#
# Downstream steps (velocity, paths, induced currents) are identical to the
# original script so outputs are directly comparable.

set -e
export POCHOIR_STORE="${1:-store_multires}"

## Domain shapes (Nx,Ny,Nz). Override via env if needed.
POCHOIR_DRIFT_SHAPE="${POCHOIR_DRIFT_SHAPE:-44,44,300}"
POCHOIR_WEIGHT_SHAPE="${POCHOIR_WEIGHT_SHAPE:-396,396,300}"

source helpers.sh

# ============================================================================
# Drift field
# ============================================================================
export POCHOIR_LOG="${POCHOIR_STORE}/pochoir_driftfield.log"

echo "=== Drift: multi-resolution FDM ==="

# fdm-multires handles domain construction and gen calls internally, so we do
# NOT need separate `pochoir domain` / `pochoir gen` steps for the drift field.
# The target domain matches POCHOIR_DRIFT_SHAPE at 0.1 mm spacing.

want potential/drift3d \
     pochoir fdm-multires \
         --target-shape   "$POCHOIR_DRIFT_SHAPE" \
         --target-spacing 0.1 \
         --origin         0,0,0 \
         --stages         5 \
         --coarsest-spacing 5.0 \
         --generator      pcb_drift_pixel_with_grid \
         --gen-config     example_gen_pcb_drift_pixel_with_grid.json \
         --engine         torch \
         --precision      0.00000002 \
         --epoch          1000000 \
         --edges          per,per,fix \
         --epoch-base     200 \
         --prec-scale-alpha 1.0 \
         --potential      potential/drift3d

python parse_maxerr.py "${POCHOIR_STORE}/pochoir_driftfield.log" \
                       "${POCHOIR_STORE}/maxerr_driftfield.png" \
                       "${POCHOIR_STORE}/summary_log_driftfield.pdf"

# ============================================================================
# Weighting field
# ============================================================================
export POCHOIR_LOG="${POCHOIR_STORE}/pochoir_weightingfield.log"

echo "=== Weighting: multi-resolution FDM ==="

want potential/weight3d \
     pochoir fdm-multires \
         --target-shape   "$POCHOIR_WEIGHT_SHAPE" \
         --target-spacing 0.1 \
         --origin         0,0,0 \
         --stages         3 \
         --coarsest-spacing 1.0 \
         --generator      pcb_pixel_with_grid \
         --gen-config     example_gen_pixel_with_grid.json \
         --engine         torch \
         --precision      0.00000002 \
         --epoch          5000000 \
         --edges          per,per,fix \
         --epoch-base     200 \
         --prec-scale-alpha 1.0 \
         --potential      potential/weight3d

python parse_maxerr.py "${POCHOIR_STORE}/pochoir_weightingfield.log" \
                       "${POCHOIR_STORE}/maxerr_weightingfield.png" \
                       "${POCHOIR_STORE}/summary_log_weightingfield.pdf"

# ============================================================================
# Velocities  (identical to original script)
# ============================================================================
echo "=== Velocities ==="

want velocity/drift3d \
     pochoir velo --temperature '87.0*K' \
         --potential potential/drift3d \
         --velocity  velocity/drift3d

# ============================================================================
# Drift paths  (identical to original script)
# ============================================================================
echo "=== Paths ==="

# 10×10 grid per pixel (0.44 mm spacing) — 100 starting points
dist=(0.22 0.66 1.1 1.54 1.98 2.42 2.86 3.3 3.74 4.18)
points=()
for d in "${dist[@]}"; do
    for d2 in "${dist[@]}"; do
        points+=("${d}*mm,${d2}*mm,28*mm")
    done
done

want starts/drift3d \
     pochoir starts --starts starts/drift3d \
         -m no \
         "${points[@]}"

want paths/drift3d_tight \
     pochoir drift --starts starts/drift3d \
         --velocity velocity/drift3d \
         --paths paths/drift3d_tight '0*us,210*us,0.05*us' \
         --plot

# ============================================================================
# Induced currents  (identical to original script)
# ============================================================================
echo "=== Induced currents ==="

want current/induced_current \
     pochoir induce-pixel \
         --weighting potential/weight3d \
         --paths     paths/drift3d_tight \
         --output    current/induced_current \
         --npixels   4 \
         --config    example_gen_pixel_with_grid.json \
         --plot
