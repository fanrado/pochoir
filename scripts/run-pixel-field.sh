#!/bin/bash

# Usage: ./run-pixel-field.sh [--hybrid yes|no] [STORE_DIR]
#
# Pixel drift + weighting fields, then the velo/starts/drift/induce chain.
# --hybrid yes runs the near/far hybrid solve (coarse + near + fine grids);
# --hybrid no runs ONE 0.1mm full-depth solve.  Both write the same store keys
# and, at this geometry, the same final lattice, so the two are comparable.
#
# =========================================================================
# DUPLICATED CONFIGS -- READ BEFORE CHANGING ANY GEOMETRY
# =========================================================================
# The four task13 JSON configs in scripts/ are COPIES of the originals of the
# same basename in the sibling "test" directory.  scripts/ is the production
# folder and deliberately reads nothing from there, so both sets exist and
# BOTH ARE MAINTAINED BY HAND -- there is no symlink and no generator keeping
# them in step.
#
# Therefore: RETARGETING A GEOMETRY MEANS EDITING BOTH COPIES.  Change only
# scripts/ and run-task13-hybrid.sh over there keeps solving the old geometry;
# change only that side and this script does.  Either way both runners still
# call themselves "task13" and neither one errors at run time -- the
# divergence is SILENT, and surfaces only as two sets of results that will
# not reconcile.
#
# NOTHING IN THIS REPO CHECKS THAT THE TWO SETS AGREE.  There is no committed
# test asserting byte equality across the pairs, so if you cloned this branch,
# the hand-maintenance rule above is the ONLY protection you have -- discipline
# and the recipe below, nothing automated.
#
# (Most of test/ IS tracked -- this is not a repo that omits its tests.  But
# the specific guard covering these four pairs is held back from the repo on
# purpose, so it is not yours and cannot protect you.  Assume it is absent,
# because for you it is.)
#
# This is not hypothetical.  This repo has already shipped stale config facts
# twice, both caught only by reading the files:
#   * a weighting _description claiming "Npixels 17 ... = 74.8mm transverse"
#     while the Npixels key next to it said 25;
#   * driftZDepth left at 159.9 in the weighting pair after the drift pair had
#     already moved to 69.9.
# Both are fixed now, and both are why this warning is here: each was one
# file disagreeing with another that nothing compared.  Two full copies of
# four configs is that same failure mode with more surface area.
#
# The pairs -- same basename on each side:
#   example_gen_pcb_drift_pixel_task13_coarse.json
#   example_gen_pcb_drift_pixel_task13_fine.json
#   example_gen_pixel_with_grid_task13_coarse.json
#   example_gen_pixel_with_grid_task13_fine.json
#
# CHECK THE PAIRS BY HAND before trusting a run, after editing either side:
#   orig=../test ; for f in *task13*.json ; do cmp "$f" "$orig/$f" ; done
# =========================================================================
#
# WITH-GRID vs WITHOUT-GRID IS NOT A FLAG.  It follows entirely from which
# configs the SIZES block points at -- GridHoleShape in the JSON ("None" = no
# shield grid) is what the generators branch on.  That is exactly how
# run-for-largepix-wgrid.sh and run-for-larpix-v2a-wogrid.sh differ: same
# commands, different JSON filenames.
#
# enableFR4 IS INERT ON THIS BRANCH -- DO NOT REACH FOR IT.  The drift configs
# still carry the key, but NOTHING reads it: grep it in pochoir/ and you get
# zero hits.  The pad/FR4 laminate z-layout is gated on enableInsulatorFR4
# ALONE (gen_pcb_drift_pixel_with_grid.py:507).  So flipping enableFR4 to true
# does NOT move the pad and does NOT create a laminate -- it does nothing at
# all.  enableInsulatorFR4 is the key that matters; leave it TRUE.
#
# This is worth stating because the failure is silent: a config with
# enableFR4 true and enableInsulatorFR4 false looks like it asked for a
# laminate, gets none, and drops the pad top by ~1.5mm with no error and no
# warning -- a live defect already found in the task7a config.  The four
# task13 drift configs and the largepix wgrid config all currently set
# enableFR4 false / enableInsulatorFR4 true, which is the correct pairing.
#
# ENFORCEMENT-FREE CONTRACT (non-negotiable, see PART B).  --insulator is
# applied to the FIELD SOLVE only.  velo gets neither --boundary nor
# --insulator, and drift gets no --insulator, so paths are never clamped or
# terminated at a surface -- doing so would break the curl-free E field.
# --interp-order linear is REQUIRED: cubic overshoots at the seam/pad-plane
# kink and over-focuses paths onto the pads.

set -e

# This script lives in scripts/ and its configs sit beside it, so cd to our own
# directory first and reach them by bare filename below.  Same structure
# as run-for-larpix-v2a-wogrid.sh; without the cd, the relative config paths
# and helpers.sh would only resolve when invoked from scripts/.
cd "$(dirname "$0")"

HYBRID="yes"
if [ "$1" = "--hybrid" ] ; then
    HYBRID="$2" ; shift 2
fi
case "$HYBRID" in
    yes|no) ;;
    *) echo "ERROR: --hybrid must be yes or no, got '$HYBRID'" >&2 ; exit 1 ;;
esac

export POCHOIR_STORE="${1:-store_pixel_field_$HYBRID}"

source helpers.sh

# ---------------------------------------------------------------------------
# want: run a step only when its output(s) are missing (multi-key; identical to
# the override in run-task10-drift-2cm-01mm.sh so a re-run resumes cleanly).
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

############################################################################
## SIZES
############################################################################
## CONFIGS.  Bare filenames, resolved against scripts/ by the cd above.  See
## the DUPLICATED CONFIGS warning in the header before editing any of them.
dcfg_coarse="example_gen_pcb_drift_pixel_task13_coarse.json"
dcfg_fine="example_gen_pcb_drift_pixel_task13_fine.json"
wcfg_coarse="example_gen_pixel_with_grid_task13_coarse.json"
wcfg_fine="example_gen_pixel_with_grid_task13_fine.json"

## SHAPES.  Explicit values, not derived: this table is the authority and
## field-solve is invoked with --domain no, so a typo here cannot be silently
## "corrected" into a different grid.  The flip side is that --domain no ALSO
## skips hybrid_iterate.py's _cells() check, which is the only thing that would
## otherwise refuse a geometry the spacings cannot represent -- so this table
## must be re-derived BY HAND whenever the configs' pitch or depth moves.  See
## the SPACINGS note below for what that derivation is.
##
## Geometry: pitch 4.4mm (pixelSize 3.5 + pixelGap 0.9 on the fine grid),
## Npixels 5, driftZDepth 149.55, pad plane 9.9mm, interface 29.7mm.  This is
## the real LArPix v2a tile at a 15cm drift length.
##
## PHASE 4 RETARGET.  This was the 8cm / 0.22mm-coarse / 19.8mm-interface
## geometry.  Two measurements moved it, both in NOTES-run-pixel-field-seam.md:
##   * 0.22mm was far too fine for the BULK.  The far field out there is
##     essentially 1-D linear, and linear fields are exact on ANY spacing, so
##     the resolution bought nothing.  Coarse relaxes to 0.55mm.
##   * the interface at 19.8mm was measured sitting INSIDE the corrugated zone
##     for the WEIGHTING field -- transverse corrugation 40% of the local W on
##     the seam plane, against 7e-08 for drift -- which no number of Schwarz
##     sweeps can fix, because the sweep converges the two domains to each
##     other, not to the truth.  The interface moves out to 29.7mm.
##
## THE BULK FIELD IS NOW 50.0 V/mm.  It REPLACES the old 56.0 V/mm rather than
## being preserved: CathodePotential -6982.5 = -50.0 * (149.55 - 9.9).  The 8cm
## geometry's -3878 V existed specifically to hold 56.0 V/mm, and that choice
## has been given up.  Drift velocities and induced currents from this geometry
## are therefore NOT comparable to the 8cm results -- velo derives mobility
## from E through a nonlinear LAr parameterisation, so a 10.7% lower field is
## not a 10.7% slower drift.  Do not "restore" 56.0 V/mm.
##
## SPACINGS ARE 0.55 / 0.1.  ONE coarse spacing serves BOTH fields.
## DERIVATION -- this table is the authority and must be re-derived BY HAND if
## the configs move:
##
##   pitch      4.4mm   =   8 coarse cells   =  44 fine cells
##   probe    5*4.4mm   =  40 coarse         = 220 fine     (weighting, 22mm)
##   depth    149.6mm   = 272 coarse cells   = 1496 fine    -> 273 / 1497 nodes
##   interface 29.7mm   =  54 coarse cells   =  297 fine    -> near 298 nodes
##   pad plane  9.9mm   =  18 coarse         =   99 fine
##
## Transverse counts use N = extent/spacing (the far node is the wrap of the
## near one, not duplicated); z uses N = extent/spacing + 1 (both faces are
## real planes).  That is the convention _cells() implements with its `closed`
## flag.  All eight shapes below were verified in Phase 4/Step 1 by calling
## _extents() and _cells() directly on the retargeted configs.
##
## 1.1mm IS THE SHARED PERIOD of the 0.55 and 0.1mm grids -- they share nodes
## ONLY every 1.1mm -- and every plane that must land on both is a whole
## multiple of it: 4.4 = 4x1.1, 9.9 = 9x1.1, 29.7 = 27x1.1, 149.6 = 136x1.1.
## That rule is the constraint to respect if any plane is ever moved, and it is
## also what makes the --band-cells choice load-bearing (see the SCHWARZ block
## above PART A).
##
## CONSEQUENCE: the coarse:fine ratio is now 5.5 -- still NOT an integer, so
## coarse nodes are not a subset of fine nodes; they coincide only every 1.1mm.
## This is supported: pochoir/nearfar.py is explicitly coordinate-based
## interpolation and works for both up- and downsampling, and _cells() only
## checks each grid against the geometry, never the ratio.  But the near->coarse
## restriction INTERPOLATES rather than subsamples, and the coarse-cell
## staircase is non-commensurate with the fine grid.
##
## THE PRICE OF COARSENING TO 0.55mm: the coarse pad snaps to 3.30mm + 1.10mm
## gap (6 + 2 = 8 cells), which is a pad area of (3.3/4.4)^2 = 0.5625 against
## the fine grid's (3.5/4.4)^2 = 0.6327 -- i.e. -11.1%, much worse than the
## +1.1% at 0.22mm.  The chamfer is undercut 21% (0.55 vs 0.7) and the coarse
## pad top lands at 10.45mm against the fine grid's 10.00mm.  The far field in
## the stitched output IS the upsampled coarse solution, so that error biases
## the far amplitude by roughly the same fraction, smoothly, where the
## Laplacian residual metric cannot see it.  This is the accepted cost; it is
## MEASURED in Phase 4/Steps 6-7, not assumed here.  In particular the 0.199%
## coarse/fine far-field "geometry floor" quoted for the 8cm geometry does NOT
## transfer and must be re-measured.
##
## NO DEAD SPACE: depth 149.6mm equals driftZDepth's cathode node
## (ceil(149.55/0.55) = 272 coarse = 1496 fine), so the cathode is the last
## plane on both grids, while the launch node 149.55 does not land on it.
##
##                                    drift          weighting
##   coarse 0.55mm,  full depth       8,8,273        40,40,273
##   near   0.1mm,   to interface     44,44,298      220,220,298
##   fine   0.1mm,   full depth       44,44,1497     220,220,1497
##   single 0.1mm    (--hybrid no)    44,44,1497     220,220,1497
##
## COST: the weighting fine grid is ~72.5 M nodes (was ~38 M at 8cm) and the
## drift fine grid ~2.9 M; the coarse grids are tiny -- 17 k drift and 0.44 M
## weighting.  Note hybrid_iterate does NOT solve the full-depth fine grid:
## _stitch upsamples the coarse solution onto that lattice and overwrites the
## first 298 planes with the near solution.
##
## The single row and the fine row are the SAME grid ON PURPOSE -- that is what
## makes the two modes comparable on identical output lattices.  They are kept
## as two separate, labelled variables rather than deduplicated into one: they
## answer different questions, and collapsing them would hide the fact that
## their agreement is a deliberate choice rather than a coincidence.
##
## THE test/ COPIES NOW DIVERGE.  The four task13 configs under scripts/ have
## been retargeted to this geometry; their same-named copies in test/ have NOT,
## and test/run-task13-hybrid.sh is OUT OF SCOPE for this work and still solves
## an older geometry.  The DUPLICATED CONFIGS rule in the header -- edit both
## copies -- was therefore KNOWINGLY NOT APPLIED here.  The cmp recipe in that
## header will report all four pairs as differing; that is expected, not a
## mistake to "fix" by copying either way.
d_coarse_shape="8,8,273"
d_near_shape="44,44,298"
d_fine_shape="44,44,1497"
d_single_shape="44,44,1497"

w_coarse_shape="40,40,273"
w_near_shape="220,220,298"
w_fine_shape="220,220,1497"
w_single_shape="220,220,1497"

interface='29.7*mm'
coarse_spacing=0.55
fine_spacing=0.1
spacing=0.1
precision=0.00000002

date
echo "=== mode: --hybrid $HYBRID, store $POCHOIR_STORE ==="

############################################################################
## PART A: DRIFT FIELD
############################################################################
export POCHOIR_LOG="${POCHOIR_STORE}/pochoir_driftfield.log"

## --insulator (the node-centered no-flux Neumann BC) is applied by field-solve
## to the SOLVE only -- never to velo or drift below.
## SCHWARZ PARAMETERS (stated explicitly below rather than inherited from
## hybrid_iterate.py's DEFAULT_BAND_CELLS / DEFAULT_MAX_SWEEPS / DEFAULT_TOL).
## The full study behind them is scripts/NOTES-run-pixel-field-seam.md.
##
##   --band-cells 2 --max-sweeps 15 --schwarz-tol 2e-8   (BOTH fields)
##
## THE BAND WIDTH IN MM MUST BE A MULTIPLE OF 1.1mm.  THIS IS THE RULE; THE
## CELL COUNT IS ONLY ITS CONSEQUENCE, AND IT CHANGES WHEN THE COARSE SPACING
## CHANGES.  1.1mm is the shared period of the 0.55 and 0.1mm grids, so a band
## whose inner plane sits a multiple of 1.1mm below the interface lands on a
## REAL FINE NODE and its far Dirichlet pin is EXACT; any other width leaves
## the pin INTERPOLATED onto the near grid.
##
##   --band-cells 2 at 0.55mm = 1.10mm  -> inner plane 29.7 - 1.1 = 28.6mm
##                                      = coarse node 52 = fine node 286 EXACTLY
##
## The PHYSICAL band is therefore UNCHANGED from the validated configuration --
## only the cell count moved, because the cells are 2.5x bigger.  At the
## previous 0.22mm coarse spacing the same 1.1mm band was --band-cells 5.
##
## DO NOT CARRY --band-cells 5 OVER FROM THE 8cm GEOMETRY: 5 x 0.55 = 2.75mm,
## which is 2.5 x 1.1mm and NOT a whole multiple, so the pin would silently go
## back to being interpolated.  There is NO error message for this -- the run
## succeeds and just converges more slowly.  Phase 3 measured what that costs:
## the Schwarz contraction rate was 0.9237/sweep with an interpolated pin
## against 0.8737/sweep with an exact one, a difference worth ~10 sweeps to
## reach the same seam quality.
##
## THE SWEEP COUNT IS THE ONLY OTHER KNOB.  --schwarz-tol 2e-8 NEVER GATES: on
## the 8cm drift field the sweep stopped on max_sweeps with the final near delta
## ~6.7 ORDERS OF MAGNITUDE above the tol.  Every run measured so far ended on
## max_sweeps, never on the tolerance.
##
## --max-sweeps 15 is CARRIED OVER, NOT RE-DERIVED.  It came from the band-5
## exact-pin rate at the 8cm / 0.22mm geometry, where 15 sweeps brought the
## drift kink to the then-measured 0.199% coarse/fine floor (against 1.0045% at
## the 4 sweeps originally shipped).  WHETHER 15 STILL SUFFICES HERE IS AN OPEN
## QUESTION measured in Phase 4/Step 6 -- the geometry, the coarse spacing, the
## bulk field and the pad snap have all changed, and that 0.199% floor itself
## does not transfer and must be re-measured.  Do not treat 15 as validated for
## this geometry.
##
## WIDER BANDS ARE DELIBERATELY NOT USED.  Pinning the coarse far solve to fine
## data over a large fraction of its depth converges the seam by turning the
## hybrid into the single-spacing solve, which defeats the point of the method.
## Do not "improve" the seam by widening the band.
##
## --schwarz-tol 2e-8 is undimensioned ON PURPOSE so one value serves both the
## volt-valued drift potential and the dimensionless [0,1] weighting probe.
##
## KNOWN BLOCKER (Phase 4/Step 1, pochoir-honm): --interface '29.7*mm' is
## REJECTED by hybrid_iterate._check_interface as of this writing.  That check
## compares the requested interface against the near grid's implied split with
## an exact float !=, and (298-1)*0.1 == 29.700000000000003, not 29.7.  The
## geometry is integer-exact -- 54 coarse and 297 fine cells -- so this is
## purely a float-equality defect; the old 19.8mm worked only because 198*0.1
## round-trips to exactly 19.8.  Until that check compares with a tolerance (or
## compares integer cell counts), BOTH --hybrid yes branches below will abort
## immediately.  Do NOT work around it by perturbing the interface string.
##
## WHAT THE SWEEP COUNT DOES NOT FIX.  The weighting seam is not sweep-limited:
## at the 8cm geometry its kink was 4.20% of the local |E_z| against the drift
## field's 0.093%, because the interface sat where transverse corrugation was
## still 40% of the local W.  Moving the interface to 29.7mm is the Phase 4
## response to exactly that, and whether it worked is measured in Phase 4/Step
## 7.  Separately, the weighting far tail is a linear ramp rather than a decay,
## because the fix,fix,fix transverse edges are implemented as a NEUMANN MIRROR
## rather than a Dirichlet zero (fdm_generic.py:8-34); see
## scripts/NOTES-weighting-farfield.md.  That is a known open defect and no
## sweep count addresses it.

if [ "$HYBRID" = yes ] ; then
    want potential/drift3d \
         pochoir field-solve --hybrid yes --field drift \
         --domain no \
         --coarse-config "$dcfg_coarse" --fine-config "$dcfg_fine" \
         --coarse-shape "$d_coarse_shape" \
         --near-shape "$d_near_shape" \
         --fine-shape "$d_fine_shape" \
         --interface "$interface" \
         --coarse-spacing "$coarse_spacing" --fine-spacing "$fine_spacing" \
         --band-cells 2 --max-sweeps 15 --schwarz-tol 2e-8 \
         --precision "$precision"
else
    want potential/drift3d \
         pochoir field-solve --hybrid no --field drift \
         --domain no \
         --config "$dcfg_fine" \
         --shape "$d_single_shape" \
         --spacing "$spacing" \
         --precision "$precision"
fi

date

############################################################################
## PART B: VELOCITY, PATHS
############################################################################
## Uses the full-depth drift potential (potential/drift3d).  These three
## commands are copied VERBATIM from run-for-larpix-v2a-wogrid.sh -- do not
## "improve" them; the enforcement-free contract lives in exactly these flags.

echo "=== Velocities ==="
## Drift velocity from the full-depth drift potential.  ENFORCEMENT-FREE:
## NO --boundary and NO --insulator -> velocity is pure mu * grad(phi); velo
## runs only to carry the potential + temperature metadata drift consumes.
want velocity/drift3d \
     pochoir velo --temperature '87.0*K' \
     --potential potential/drift3d \
     --velocity velocity/drift3d

echo "=== Paths ==="
## CONFIG-DRIVEN starts (-m yes): nGridPoints=10
## (_shift_paths_pixel_grid, npaths=10) expects.
want starts/drift3d \
     pochoir starts --starts starts/drift3d \
     -m yes \
     -c $dcfg_fine \
     --plot

## ENFORCEMENT-FREE: NO --insulator on drift -> pure grad(phi) paths.
## --interp-order linear: cubic rings/overshoots near the pixel plane and
## over-focuses paths onto the pads; linear is monotone-safe.
want paths/drift3d \
     pochoir drift --starts starts/drift3d \
     --velocity velocity/drift3d \
     --interp-order linear \
     --paths paths/drift3d '0*us,200*us,0.05*us' \
     --plot

############################################################################
## PART C: WEIGHTING FIELD
############################################################################
export POCHOIR_LOG="${POCHOIR_STORE}/pochoir_weightingfield.log"

if [ "$HYBRID" = yes ] ; then
    want potential/weight3d \
         pochoir field-solve --hybrid yes --field weighting \
         --domain no \
         --coarse-config "$wcfg_coarse" --fine-config "$wcfg_fine" \
         --coarse-shape "$w_coarse_shape" \
         --near-shape "$w_near_shape" \
         --fine-shape "$w_fine_shape" \
         --interface "$interface" \
         --coarse-spacing "$coarse_spacing" --fine-spacing "$fine_spacing" \
         --band-cells 2 --max-sweeps 15 --schwarz-tol 2e-8 \
         --precision "$precision"
else
    want potential/weight3d \
         pochoir field-solve --hybrid no --field weighting \
         --domain no \
         --config "$wcfg_fine" \
         --shape "$w_single_shape" \
         --spacing "$spacing" \
         --precision "$precision"
fi

date

echo "=== Induced currents ==="
## Induced current on the collecting pixel via Ramo (weighting field x drift
## paths).  --npixels 2 sums the target pixel + its first ring; --config supplies
## the pixel geometry for the pad-collection map (same geometry as $wcfg_fine).
want current/induced_current \
     pochoir induce-pixel --weighting potential/weight3d \
     --paths paths/drift3d \
     --output current/induced_current \
     --npixels 2 \
     --config "$wcfg_fine" \
     --plot

date

echo "=== DONE ==="
