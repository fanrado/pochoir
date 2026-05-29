#!/bin/bash
# Phase 2 (pochoir-uty): drift-length sweep at 3 cm and 9 cm.
# Reuses the SHARED shallow weighting field + drift potential + velocity already
# in store/ (Option B).  Only starts -> drift -> induce are re-run per length,
# with constant-edge W extrapolation (clamp) so the literal start-z works on the
# shallow weight domain.  Per-stage wall-clock is logged to phase2_runtime.csv.
set -e
export POCHOIR_STORE=store
CFG=example_gen_pixel_with_grid.json
source helpers.sh
export POCHOIR_LOG="${POCHOIR_STORE}/phase2.log"

CSV=store/phase2_runtime.csv
echo "drift_cm,start_mm,window_us,nticks,starts_s,drift_s,induce_s,total_s" > "$CSV"

# 10x10 quarter-pixel grid (0.44 mm spacing), as in the main script
dist=(0.22 0.66 1.1 1.54 1.98 2.42 2.86 3.3 3.74 4.18)

run_one () {
    local Lcm=$1 ; local zmm=$2 ; local Tus=$3
    echo "===== drift length ${Lcm} cm (start z=${zmm} mm, window 0..${Tus} us) ====="
    # fresh starts/paths/current for this length
    rm -f store/starts/drift3d.* store/paths/drift3d_tight.* store/current/induced_current.* \
          store/startpoints.npy store/endpoints.npy 2>/dev/null

    local points=()
    for d in "${dist[@]}"; do for d2 in "${dist[@]}"; do
        points+=("${d}*mm,${d2}*mm,${zmm}*mm")
    done; done

    local t0 t1 ts td ti
    t0=$(date +%s.%N)
    pochoir starts --starts starts/drift3d -m no -c "$CFG" "${points[@]}" >/dev/null 2>&1
    t1=$(date +%s.%N); ts=$(echo "$t1-$t0"|bc)

    t0=$(date +%s.%N)
    pochoir drift --starts starts/drift3d --velocity velocity/drift3d \
        --paths paths/drift3d_tight "0*us,${Tus}*us,0.05*us" >/dev/null 2>&1
    t1=$(date +%s.%N); td=$(echo "$t1-$t0"|bc)

    t0=$(date +%s.%N)
    pochoir induce-pixel --weighting potential/weight3d --paths paths/drift3d_tight \
        --output current/induced_current --npixels 4 --config "$CFG" >/dev/null 2>&1
    t1=$(date +%s.%N); ti=$(echo "$t1-$t0"|bc)

    cp store/fr_4p4pitch_3.8pix_nogrid_10pathsperpixel.npy "store/fr_${Lcm}cm.npy"
    local nticks=$(python -c "print(int(${Tus}/0.05))")
    local tot=$(echo "$ts+$td+$ti"|bc)
    echo "${Lcm},${zmm},${Tus},${nticks},${ts},${td},${ti},${tot}" >> "$CSV"
    echo "  starts=${ts}s drift=${td}s induce=${ti}s total=${tot}s -> store/fr_${Lcm}cm.npy"
}

run_one 3 30 40
run_one 9 90 90

echo "=== runtime table ==="; column -s, -t "$CSV"
echo "DONE_PHASE2_SWEEP"
