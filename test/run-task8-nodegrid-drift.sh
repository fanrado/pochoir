#!/bin/bash
#
# Task8: drift paths launched from EVERY grid node (node-resolution map).
# ---------------------------------------------------------------------------
# Purpose
# -------
# To see the drift / charge-sharing behaviour near the pixel gap and chamfer
# clearly, launch ONE drift path from each 0.1mm grid node in the (x,y) plane
# (45x45 = 2025 launches at x,y = 0.0..4.4 mm) near the cathode, and integrate
# down to the pixel plane.  This is the "better solution" than a coarse launch
# grid: every node gets its own path, so the landing pattern (pad vs FR4-gap vs
# chamfer notch) is resolved at the solve grid itself.
#
# This run REUSES the existing full-depth 0.1mm drift potential solved by
# run-validate-neumann-fulldepth-01mm.sh (store_validate_neumann_fulldepth_01mm):
# same spacing, same no-flux FR4 insulator BC, same geometry.  It therefore does
# NOT re-solve any field:
#   * NO drift domain/gen/fdm  (reuse potential/drift3d)
#   * NO weighting field at all
#   * NO induced current
# It runs only: velo (from the reused potential) -> starts (grid nodes) -> drift.
#
# Time step: 10 ns (0.01*us) output sampling, matching the validation run.
# (The integrator itself is adaptive; the step is only the t_eval spacing.)
#
# Usage: ./run-task8-nodegrid-drift.sh [STORE_DIR]
#   STORE_DIR defaults to store_validate_neumann_fulldepth_01mm so the existing
#   potential/drift3d, boundary/drift_full and initial/drift_full_insulator are
#   reused in place.  New keys written: velocity/drift3d, starts/drift3d_nodes,
#   paths/drift3d_nodes (existing keys are not overwritten).

set -e

export POCHOIR_STORE="${1:-store_validate_neumann_fulldepth_01mm}"

source helpers.sh

# ---------------------------------------------------------------------------
# want: run a step only when its output(s) are missing (same override as the
# validation driver so a re-run resumes cleanly).
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

# Sanity: the reused drift potential must already exist in the store.
if [ ! -f "$POCHOIR_STORE/potential/drift3d.npz" ] ; then
    echo "ERROR: $POCHOIR_STORE/potential/drift3d.npz not found." >&2
    echo "       Run run-validate-neumann-fulldepth-01mm.sh first (PART A)." >&2
    exit 1
fi

export POCHOIR_LOG="${POCHOIR_STORE}/pochoir_task8_nodegrid.log"

## No-flux FR4 insulator (Neumann, NO epsilon) -- SAME mask used to solve the
## reused potential.  DINS on (default) => paths terminate at the FR4 surface.
dins_fine=()
if [ "${DINS:-1}" != "0" ] ; then
    dins_fine=(--insulator initial/drift_full_insulator)
    echo "=== DINS on: drift paths use the no-flux FR4 insulator (NO epsilon) ==="
fi

date

############################################################################
## DRIFT VELOCITY  (from the REUSED full-depth 0.1mm drift potential)
############################################################################
echo "=== Velocity (from existing potential/drift3d) ==="
want velocity/drift3d \
     pochoir velo --temperature '87.0*K' \
     --potential potential/drift3d \
     --boundary boundary/drift_full \
     "${dins_fine[@]}" \
     --velocity velocity/drift3d

############################################################################
## STARTS  (one launch per grid node)
############################################################################
## Grid nodes at 0.1mm: x,y = 0.0 .. 4.4 mm (45 nodes each -> 45x45 = 2025
## launches).  Launched at z=59.9mm, the last full-velocity node before the
## cathode (z=60mm, v=0).  This is the grid-node dist list copied from
## run-validate-neumann-fulldepth-01mm.sh.
echo "=== Starts: one launch per 0.1mm grid node (45x45) ==="
dist=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.0 3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9 4.0 4.1 4.2 4.3 4.4)
points=()
for d in "${dist[@]}"; do
     for d2 in "${dist[@]}"; do
         points+=("${d}*mm,${d2}*mm,59.9*mm")
     done
done

want starts/drift3d_nodes \
     pochoir starts --starts starts/drift3d_nodes \
     -m no \
     -c example_gen_pixel_with_grid.json \
     "${points[@]}" \
     --plot

############################################################################
## DRIFT PATHS  (10 ns output step, linear interp, insulator termination)
############################################################################
echo "=== Drift paths (grid-node launches) ==="
want paths/drift3d_nodes \
     pochoir drift --starts starts/drift3d_nodes \
     --velocity velocity/drift3d \
     "${dins_fine[@]}" \
     --interp-order linear \
     --paths paths/drift3d_nodes '0*us,90*us,0.01*us' \
     --plot
# --interp-order linear: cubic rings/overshoots near the pixel plane and
# over-focuses paths onto the pads; linear is monotone-safe.

############################################################################
## DRIFT VELOCITY + E-FIELD ALONG THE PATHS
############################################################################
## Sample the drift velocity and drift E-field at every point ALONG each path,
## using the SAME PotentialField the drift integrated (interpolate potential ->
## finite-diff E -> mobility -> velocity).  This gives the dynamics AT the landing
## instant (and just before), so we can argue about what happens at t+dt -- i.e.
## whether a gap electron is still being driven (v != 0, funnelled toward the pad)
## rather than sitting in equilibrium on the FR4 surface.  Analysis helper only,
## no production code.  Writes alongpath/velocity.npz + alongpath/efield.npz (V/mm).
## Guarded: set ALONGPATH=0 to skip; want() makes a re-run with the arrays a no-op.
if [ "${ALONGPATH:-1}" != "0" ] ; then
    echo "=== Sample velocity + E-field along the drift paths ==="
    want "alongpath/velocity alongpath/efield" \
         python sample_field_along_paths.py "$POCHOIR_STORE" \
         --potential potential/drift3d --paths paths/drift3d_nodes \
         "${dins_fine[@]}" --domain domain/drift_full
fi

date
echo "=== DONE: Task8 node-grid drift-path run complete ==="
echo "    starts:   $POCHOIR_STORE/starts/drift3d_nodes.npz"
echo "    paths:    $POCHOIR_STORE/paths/drift3d_nodes.npz  (+ _endtag)"
echo "    alongpath:$POCHOIR_STORE/alongpath/velocity.npz , efield.npz (V/mm)"
