#!/bin/bash
#
# TASK 2: 10 cm drift, HYBRID (coarse 0.4mm + fine 0.05mm) drift-field solve.
#
# Drift FIELD + drift PATHS only -- NO weighting field, NO induced current.
# Reproduces the near-field-refinement workflow of run-full-3d-pixel.sh PART A
# (coarse full-depth solve -> refine seed -> near-bc Dirichlet interface at
# z=20mm -> fine near fdm -> near-far-solve overlapping Schwarz -> stitch-near),
# then velocity + paths.
#
# Geometry (fixed-cell chamfer, FR4/PCB disabled = pure Laplace):
#   * Pixel plane low edge at z=10mm; cathode at the top z-plane held at
#     -5000V so the bulk field is -5000V/100mm = -50 V/mm over the 10cm gap.
#   * Coarse: 11x11x276 @ 0.4mm  -> z = 0..110mm (full depth).
#   * Near : 88x88x401 @ 0.05mm -> z = 0..20mm (contains the pixel plane).
#   * Fine (stitched full): 88x88x2201 @ 0.05mm -> z = 0..110mm.
#   * Electrons launched at z=109.5mm, drift down to the pad at z=10mm
#     (99.5mm drift). No drift below the pad is expected physically.
#
# Usage: ./run-hybrid-10cm-drift.sh [STORE_DIR] [DRIFT_CONFIG]

set -e
export POCHOIR_STORE="${1:-store_task2_hybrid_10cm}"
cfg="${2:-example_gen_pcb_drift_pixel_hybrid10cm_fixedcell.json}"
source helpers.sh

want () {
    local targets="$1" ; shift
    local t miss=0
    for t in $targets ; do [ -f "$POCHOIR_STORE/${t}.npz" ] || miss=1 ; done
    if [ "$miss" -eq 0 ] ; then echo "have $targets"; return; fi
    echo "$@"; "$@"
    for t in $targets ; do
        [ -f "$POCHOIR_STORE/${t}.npz" ] || { echo "ERROR: missing output $t" >&2; exit 1; }
    done
    echo "made $targets"
}

date
export POCHOIR_LOG="${POCHOIR_STORE}/pochoir_driftfield.log"
gen="pcb_drift_pixel_with_grid"
echo "=== TASK2 hybrid 10cm drift: store=$POCHOIR_STORE cfg=$cfg ==="

## Step 1: coarse solve (0.4mm, 11x11x276), full 110mm depth
want domain/coarse \
     pochoir domain --domain domain/coarse \
     --shape=11,11,276 --spacing '0.4*mm'

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

## Step 2: near-field gen + refined coarse seed (0.05mm, 88x88x401, z=0..20mm)
want domain/near \
     pochoir domain --domain domain/near \
     --shape=88,88,401 --spacing '0.05*mm'

want "initial/near boundary/near" \
     pochoir gen --generator $gen --domain domain/near \
     --initial initial/near --boundary boundary/near \
     $cfg

want initial/near_refined \
     pochoir refine \
     --coarse potential/coarse \
     --initial initial/near \
     --boundary boundary/near \
     --output initial/near_refined

## Step 3: Dirichlet interface plane at z=20mm from the coarse bulk
want "initial/near_bc boundary/near_bc" \
     pochoir near-bc \
     --initial initial/near_refined \
     --boundary boundary/near \
     --coarse potential/coarse \
     --initial-out initial/near_bc \
     --boundary-out boundary/near_bc

## Step 4: near-field fine solve (0.05mm), seeded + pinned interface (sweep 0)
want "potential/near increment/near" \
     pochoir fdm \
     --nepochs 10 --epoch 130000000 --precision 0.00000000002 \
     --edges per,per,fix \
     --engine torch \
     --initial initial/near_bc --boundary boundary/near_bc \
     --potential potential/near \
     --increment increment/near
date

## Step 4b: overlapping-Schwarz refinement for a continuous interface
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

## Step 5: stitch the 0.05mm near solve onto the Schwarz-updated far field
want domain/fine \
     pochoir domain --domain domain/fine \
     --shape=88,88,2201 --spacing '0.05*mm'

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
## VELOCITY + PATHS  (drift only -- no weighting, no induced current)
############################################################################
echo "=== Velocity ==="
want velocity/drift3d \
     pochoir velo --temperature '87.0*K' \
     --potential potential/drift3d \
     --boundary boundary/fine \
     --velocity velocity/drift3d

echo "=== Paths ==="
## 10x10 grid per pixel, launched near the cathode plane (z=109.5mm).
dist=(0.22 0.66 1.1 1.54 1.98 2.42 2.86 3.3 3.74 4.18)
points=()
for d in "${dist[@]}"; do
     for d2 in "${dist[@]}"; do
         points+=("${d}*mm,${d2}*mm,109.5*mm")
     done
done

want starts/drift3d \
     pochoir starts --starts starts/drift3d -m no \
     -c $cfg "${points[@]}" --plot

want paths/drift3d \
     pochoir drift --starts starts/drift3d \
     --velocity velocity/drift3d \
     --interp-order linear \
     --paths paths/drift3d '0*us,150*us,0.05*us' \
     --plot
date
echo "=== DONE: paths in $POCHOIR_STORE/paths/drift3d.npz, plot drift_paths_3d.png ==="
