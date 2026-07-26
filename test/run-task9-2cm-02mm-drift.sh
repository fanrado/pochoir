#!/bin/bash
#
# Task9: 2 cm drift, fine 0.02 mm spacing, PLAIN LAPLACE (no Neumann/insulator).
# ---------------------------------------------------------------------------
# Purpose
# -------
# A fine-spacing, plain-Laplace (NO no-flux FR4 insulator, NO epsilon) drift
# control.  The pixel pad sits at z=10 mm with a ~10 mm region BELOW it, so drift
# paths that dip below the pad plane are inside the domain and visible.  Above the
# pad is a full 2 cm (20 mm) drift gap to the cathode.
#
# Resolution note: 0.01 mm was requested, but pad@10mm + 10mm-below + 20mm-gap is
# a ~30 mm domain; at 0.01 mm that is 440x440x3001 ~= 5.8e8 cells (~33 GB peak) >
# the 24 GB GPU.  Coarsened to 0.02 mm (still 5x finer than the 0.1 mm reference):
# 220x220x1506 ~= 7.3e7 cells (~4 GB) -> fits with headroom.
#
# Scope: drift potential -> velocity -> paths ONLY.  NO weighting field, NO
# induced current.  The FDM solve runs on ONE GPU (torch engine).
#
# Geometry (0.02 mm):
#   pitch 4.4 mm            -> 220 cells (periodic x,y)
#   pad plane z=10.0-10.1mm -> pp_loweredge node 500, pp_width 5 cells
#   below-pad region z=0..10mm (500 cells)   -> paths below the pad are captured
#   drift gap  z=10.1..30.1mm (20 mm)        -> 2 cm drift
#   cathode    z=30.1 mm  (node 1505)
#   field = (0 - (-1000 V)) / 20 mm = 50 V/mm  (pad grounded, cathode -1000 V)
#
# Usage: ./run-task9-2cm-02mm-drift.sh [STORE_DIR] [GPU_INDEX]
#   STORE_DIR defaults to store_task9_2cm_02mm_nobc ; GPU_INDEX defaults to 0.

set -e

export POCHOIR_STORE="${1:-store_task9_2cm_02mm_nobc}"
export CUDA_VISIBLE_DEVICES="${2:-0}"

# Redirect torch-inductor compile cache + TMPDIR off the tiny /tmp partition
# (fills mid-compile with Errno 28) onto /nfs/data/1 (ample free space).
_cachedir="$(pwd)/.torchinductor_cache_task9"
_tmpdir="$(pwd)/.tmp_task9"
mkdir -p "$_cachedir" "$_tmpdir"
export TORCHINDUCTOR_CACHE_DIR="$_cachedir"
export TMPDIR="$_tmpdir"

source helpers.sh

# ---------------------------------------------------------------------------
# want: run a step only when its output(s) are missing (multi-key; resume-safe).
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

echo "=== Task9: 2cm drift, 0.02mm, PLAIN LAPLACE (no Neumann), GPU $CUDA_VISIBLE_DEVICES ==="
date

gen="pcb_drift_pixel_with_grid"
cfg="example_gen_pcb_drift_pixel_2cm_02mmspacing.json"

############################################################################
## PART A: DRIFT FIELD  (single full-depth 0.02mm plain-Laplace solve)
############################################################################
export POCHOIR_LOG="${POCHOIR_STORE}/pochoir_task9_driftfield.log"

## Single drift domain: 220x220 periodic tile (4.4mm pitch @0.02mm), 1506 deep
## (z=0..30.1mm).
want domain/drift_full \
     pochoir domain --domain domain/drift_full \
     --shape=220,220,1506 --spacing '0.02*mm'

## PLAIN Laplace: no --insulator (no Neumann), no epsilon.  Config has
## enableFR4=false and enableInsulatorFR4=false so gen returns the legacy
## 3-tuple and no insulator mask is produced.
want "initial/drift_full boundary/drift_full" \
     pochoir gen --generator $gen --domain domain/drift_full \
     --initial initial/drift_full --boundary boundary/drift_full \
     $cfg

## One full-depth Laplace solve, per,per,fix (periodic pixel tile, fixed z).
## Tight precision (2e-11) matching the reference drift solves.  NO --insulator.
want "potential/drift3d increment/drift_full" \
     pochoir fdm \
     --nepochs 10 --epoch 130000000 --precision 0.00000000002 \
     --edges per,per,fix \
     --engine torch \
     --initial initial/drift_full --boundary boundary/drift_full \
     --potential potential/drift3d \
     --increment increment/drift_full

date

############################################################################
## PART B: DRIFT VELOCITY
############################################################################
export POCHOIR_LOG="${POCHOIR_STORE}/pochoir_task9_paths.log"

echo "=== Velocity (plain, no insulator) ==="
want velocity/drift3d \
     pochoir velo --temperature '87.0*K' \
     --potential potential/drift3d \
     --boundary boundary/drift_full \
     --velocity velocity/drift3d

############################################################################
## PART C: STARTS + DRIFT PATHS  (no weighting, no induced current)
############################################################################
## Launch on a 0.1mm grid across the 4.4mm pitch (45x45 = 2025), just below the
## cathode (z=30.0 mm, one 0.1mm step below the cathode node at 30.1 mm).
echo "=== Starts (0.1mm launch grid, near cathode z=30.0mm) ==="
dist=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.0 3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9 4.0 4.1 4.2 4.3 4.4)
points=()
for d in "${dist[@]}"; do
     for d2 in "${dist[@]}"; do
         points+=("${d}*mm,${d2}*mm,30.0*mm")
     done
done

want starts/drift3d_nodes \
     pochoir starts --starts starts/drift3d_nodes \
     -m no \
     -c $cfg \
     "${points[@]}" \
     --plot

## Drift down through the pad plane (z=10mm) into the below-pad region.  Window
## 0..20us covers ~20mm at 50 V/mm (~1.6 mm/us -> ~12.5us) with margin.  10 ns
## output step, linear interp (monotone-safe near the pad).  NO --insulator, so
## there is NO surface-termination event -- paths continue below the pad plane
## until the velocity field drops to zero, which is exactly what we want to see.
echo "=== Drift paths (through and below the pad plane) ==="
want paths/drift3d_nodes \
     pochoir drift --starts starts/drift3d_nodes \
     --velocity velocity/drift3d \
     --interp-order linear \
     --paths paths/drift3d_nodes '0*us,20*us,0.01*us' \
     --plot

date
echo "=== DONE: Task9 2cm 0.02mm plain-Laplace drift-path run complete ==="
echo "    potential: $POCHOIR_STORE/potential/drift3d.npz"
echo "    paths:     $POCHOIR_STORE/paths/drift3d_nodes.npz"
