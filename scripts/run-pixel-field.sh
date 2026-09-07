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
## Geometry: pitch 4.4mm (pixelSize 3.5 + pixelGap 0.9), Npixels 5,
## driftZDepth 79.15, pad plane 9.9mm, interface 19.8mm.  This is the 8cm /
## 2cm method-confirmation geometry -- the real LArPix v2a tile, at the true
## design values.
##
## SPACINGS ARE 0.22 / 0.1.  DERIVATION -- this table is the authority and must
## be re-derived BY HAND if the configs move:
##
##   pitch      4.4mm   =  20 coarse cells   =  44 fine cells
##   probe    5*4.4mm   = 100 coarse         = 220 fine     (weighting, 22mm)
##   depth     79.2mm   = 360 coarse cells   = 792 fine cells  -> 361 / 793 nodes
##   interface 19.8mm   =  90 coarse cells   = 198 fine cells  -> near 199 nodes
##   pad plane  9.9mm   =  45 coarse         =  99 fine
##
## Transverse counts use N = extent/spacing (the far node is the wrap of the
## near one, not duplicated); z uses N = extent/spacing + 1 (both faces are real
## planes).  That is the convention _cells() implements with its `closed` flag.
##
## WHY 0.22 AND NOT 0.4.  Searching every coarse spacing that divides the 4.4mm
## pitch, 0.22 is the only one that both snaps the pad to within +1.1% of area
## (3.52mm; 0.2 and 0.4 give 3.6mm and +5.8%) and puts the pad plane on a node
## of BOTH grids -- shared nodes come every 1.1mm and 9.9 = 9 x 1.1, whereas
## 0.44 forces multiples of 2.2mm, which would move the pad plane to 8.8 or
## 11.0.  Coarse being only 2.2x coarser costs nothing: 100x100x361 is about
## 3.6M nodes.
##
## CONSEQUENCE: the coarse:fine ratio is 2.2, NOT an integer, so coarse nodes
## are not a subset of fine nodes -- they coincide only every 1.1mm.  This is
## supported: pochoir/nearfar.py is explicitly coordinate-based interpolation
## and works for both up- and downsampling, and _cells() only checks each grid
## against the geometry, never the ratio.  But the near->coarse restriction now
## INTERPOLATES rather than subsamples, and the coarse-cell staircase is
## non-commensurate with the fine grid.
##
## NO DEAD SPACE: depth 79.2mm equals driftZDepth's cathode node
## (ceil(79.15/0.22) = 360 coarse = 792 fine), so the cathode is the last plane
## on both grids, while the launch node 79.15 does not land on it.
##
##                                    drift          weighting
##   coarse 0.22mm,  full depth       20,20,361      100,100,361
##   near   0.1mm,   to interface     44,44,199      220,220,199
##   fine   0.1mm,   full depth       44,44,793      220,220,793
##   single 0.1mm    (--hybrid no)    44,44,793      220,220,793
##
## The weighting fine grid is ~38 M nodes, well down from the ~225 M of the
## retired 9x9 / 0.0925mm transcription.
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
## the old 3.7mm / 9x9 / 159.9mm geometry.  The DUPLICATED CONFIGS rule in the
## header -- edit both copies -- was therefore KNOWINGLY NOT APPLIED here.  The
## cmp recipe in that header will report all four pairs as differing; that is
## expected, not a mistake to "fix" by copying either way.
d_coarse_shape="20,20,361"
d_near_shape="44,44,199"
d_fine_shape="44,44,793"
d_single_shape="44,44,793"

w_coarse_shape="100,100,361"
w_near_shape="220,220,199"
w_fine_shape="220,220,793"
w_single_shape="220,220,793"

interface='19.8*mm'
coarse_spacing=0.22
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
## hybrid_iterate.py's DEFAULT_BAND_CELLS / DEFAULT_MAX_SWEEPS / DEFAULT_TOL):
##
##   * --band-cells 3 is 3 COARSE cells = band_cells+1 = 4 nodes.  At
##     --interface 19.8mm with coarse 0.22mm those are coarse nodes
##     90/89/88/87 = z 19.80/19.58/19.36/19.14mm (spelled out at
##     hybrid_iterate.py:319).
##   * the innermost plane 19.14mm is NOT a fine node: coarse and fine nodes
##     coincide only every 1.1mm at the 2.2 ratio, so the far Dirichlet pin
##     there is INTERPOLATED onto the near grid, not exact.  Only band_cells
##     that are multiples of 5 give an exact pin.  Measured in Phase 2, not
##     fixed here.
##   * --schwarz-tol 2e-8 is undimensioned ON PURPOSE so one value serves both
##     the volt-valued drift potential and the dimensionless [0,1] weighting
##     probe.
##   * at 2e-8 the tolerance NEVER GATES -- measured near deltas on this 8cm
##     grid-free geometry run 0.34 (band 2) to 3.17 (band 20), about seven
##     orders above it -- so --max-sweeps 4 is the BINDING limit and the sweep
##     count alone decides seam quality.

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
         --band-cells 3 --max-sweeps 4 --schwarz-tol 2e-8 \
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
         --band-cells 3 --max-sweeps 4 --schwarz-tol 2e-8 \
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
