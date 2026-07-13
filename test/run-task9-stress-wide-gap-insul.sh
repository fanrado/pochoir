#!/bin/bash
# ===========================================================================
# TASK 9 : STRESS TEST of the insulating-surface (no-flux Neumann) FR4 boundary
#          at a DRASTICALLY WIDER inter-pixel gap.
#
#          Identical pipeline + geometry to Task 8 (5 cm hybrid drift, dynamic
#          chamfer, no-flux insulator, NO epsilon; PART A drift field + paths +
#          overshoot check, PART B weighting field, PART C induced current) but
#          with a much larger gap:
#              pixelSize  3.8 -> 3.0 mm
#              pixelGap   0.6 -> 1.4 mm   (>2x wider gap)
#              chamfer_r  0.7 -> 1.0 mm   (dynamic)
#          The PITCH is unchanged: 3.0 + 1.4 = 3.8 + 0.6 = 4.4 mm, so every
#          domain shape carries over exactly (88x88 drift tile, 440x440 = 5-pixel
#          weighting extent).  Only the pad/gap split and the chamfer change.
#
# HYPOTHESIS UNDER TEST: the no-flux Neumann BC at the FR4 gap surface forces
#          drifting electrons directly onto the pixel pad.  If that is truly
#          what the boundary does, then more than doubling the gap width (0.6 ->
#          1.4 mm, so the gap is now ~47% of the pitch vs ~14%) should STILL
#          leave zero paths below the pad plane.  A wide gap is the hard case:
#          if any physics leaked field lines into/through the FR4, a big gap is
#          where overshoot would reappear.
#
# ACCEPTANCE (same as Task 8): 0 paths below the pad plane at z = 10 mm, global
#          z_min >= ~9.95 mm.  Compare against Task 8 (0.6 mm gap) to show the
#          fix is gap-width independent.
#
# Geometry (dynamic 1.0 mm chamfer, GridHoleShape "None" -> plain Laplace on E
# plus the no-flux insulator; NO permittivity):
#   * Physical laminate: FR4 slab z = 8.4..10.0 mm (1.6 mm, 32 cells @0.05),
#     1 oz Cu pad (~0.0348 mm, 1 cell) on top at z = 10.0 mm.  The FR4 slab is
#     the no-flux insulator mask (enableInsulatorFR4=true).
#   * Pixel plane / pad collection surface at z = 10 mm; drift gap 50 mm;
#     cathode at the top z-plane held at -2500 V -> -2500/50 = -50 V/mm.
#   * Coarse: 11x11x151  @ 0.4 mm  -> z = 0..60 mm (full depth, far field).
#   * Near : 88x88x401   @ 0.05 mm -> z = 0..20 mm (contains pad + FR4).
#   * Fine (stitched full): 88x88x1201 @ 0.05 mm -> z = 0..60 mm.
#   * Electrons launched at z = 59.5 mm, drift ~49.5 mm to the pad at z = 10 mm.
#
# INSULATOR THREADING: --insulator is passed to the near fine fdm solve (where
# the FR4 is resolved), to velo (zeros velocity inside FR4), and to drift (stops
# paths at the FR4 surface = surface charge, endtag 2).  The overlapping-Schwarz
# `near-far-solve` step is OMITTED for the DRIFT field (same rationale as Task8:
# overshoot is a pad-local z=10 mm phenomenon far below the z=20 mm interface,
# already pinned Dirichlet via near-bc); it IS used for the WEIGHTING field
# (insulator-aware, pochoir-oz2l).
#
# Output folder: store_task9_stress_wide_gap_insul/
# Run from test/:  ./run-task9-stress-wide-gap-insul.sh
# ===========================================================================
set -e
cd "$(dirname "$0")"
ROOT="$(cd .. && pwd)"
export PATH="$ROOT/env/bin:$PATH"
PY="$ROOT/env/bin/python"
export POCHOIR_STORE=store_task9_stress_wide_gap_insul
export POCHOIR_LOG="${POCHOIR_STORE}/pochoir.log"
CFG=example_gen_pcb_drift_pixel_task9_stress_insul.json
gen="pcb_drift_pixel_with_grid"
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

echo "### [1/4] geometry + actual-removed-material report -> $POCHOIR_STORE/GEOMETRY_AND_CHAMFER.md"
$PY compute_chamfer_removed.py --store "$POCHOIR_STORE" --mode dynamic --spacing 0.05 --chamfer-mm 1.0 \
    --pad-z 10 --drift 50 --total 60 --cathode -2500 --hybrid --coarse-spacing 0.4 \
    --title "Task 9 -- STRESS TEST wide 1.4 mm gap, 5 cm drift, hybrid (0.4+0.05 mm), dynamic 1.0 mm chamfer, insulating-surface FR4 (NO epsilon)"

echo "### [2/4] hybrid drift field (plain Laplace + no-flux insulator, NO epsilon)"
date

## Step 1: coarse solve (0.4mm, 11x11x151), full 60mm depth.  Bulk far field;
## the insulator is NOT applied here (FR4 is unresolved at 0.4mm and the near
## region z<20mm is replaced by the fine solve after stitch).
want domain/coarse \
     pochoir domain --domain domain/coarse --shape=11,11,151 --spacing '0.4*mm'
want "initial/coarse boundary/coarse" \
     pochoir gen --generator $gen --domain domain/coarse \
     --initial initial/coarse --boundary boundary/coarse $CFG
want "potential/coarse increment/coarse" \
     pochoir fdm --nepochs 10 --epoch 130000000 --precision 0.0000002 \
     --edges per,per,fix --engine torch \
     --initial initial/coarse --boundary boundary/coarse \
     --potential potential/coarse --increment increment/coarse
date

## Step 2: near-field gen (0.05mm, 88x88x401, z=0..20mm) + refined coarse seed.
## gen stores the no-flux insulator mask as initial/near_insulator.
want domain/near \
     pochoir domain --domain domain/near --shape=88,88,401 --spacing '0.05*mm'
want "initial/near boundary/near" \
     pochoir gen --generator $gen --domain domain/near \
     --initial initial/near --boundary boundary/near $CFG
want initial/near_refined \
     pochoir refine --coarse potential/coarse \
     --initial initial/near --boundary boundary/near --output initial/near_refined

## Step 3: Dirichlet interface plane at z=20mm from the coarse bulk.
want "initial/near_bc boundary/near_bc" \
     pochoir near-bc --initial initial/near_refined --boundary boundary/near \
     --coarse potential/coarse \
     --initial-out initial/near_bc --boundary-out boundary/near_bc

## Step 4: near-field fine solve WITH the no-flux insulator BC (NO epsilon),
## seeded + pinned interface.
want "potential/near increment/near" \
     pochoir fdm --nepochs 10 --epoch 130000000 --precision 0.00000000002 \
     --edges per,per,fix --engine torch \
     --initial initial/near_bc --boundary boundary/near_bc \
     --insulator initial/near_insulator \
     --potential potential/near --increment increment/near
date

## Step 5: fine full domain gen (stores initial/fine_insulator on the 1201-cell
## grid for velo/drift) + stitch the near fine solve onto the coarse far field.
want domain/fine \
     pochoir domain --domain domain/fine --shape=88,88,1201 --spacing '0.05*mm'
want "boundary/fine initial/fine" \
     pochoir gen --generator $gen --domain domain/fine \
     --initial initial/fine --boundary boundary/fine $CFG
want potential/drift3d \
     pochoir stitch-near --near potential/near --coarse potential/coarse \
     --domain domain/fine --output potential/drift3d
date

############################################################################
## VELOCITY + PATHS  (drift only -- insulator active, NO epsilon)
############################################################################
echo "### [3/4] velocity + drift paths (insulator active: velo zeros FR4, drift stops at surface)"
want velocity/drift3d \
     pochoir velo --temperature '87.0*K' \
     --potential potential/drift3d --boundary boundary/fine \
     --insulator initial/fine_insulator \
     --velocity velocity/drift3d

## 10x10 launch grid per pixel, near the cathode plane (z=59.5mm).  Spread across
## the full 4.4 mm pitch so charges launch over BOTH the pad and the wide gap.
dist=(0.22 0.66 1.1 1.54 1.98 2.42 2.86 3.3 3.74 4.18)
points=()
for d in "${dist[@]}"; do
     for d2 in "${dist[@]}"; do
         points+=("${d}*mm,${d2}*mm,59.5*mm")
     done
done
want starts/drift3d \
     pochoir starts --starts starts/drift3d -m no -c $CFG "${points[@]}" --plot

## Drift window ~1.5 us/mm x 49.5 mm ~= 74 us; use 90 us with margin.
want paths/drift3d \
     pochoir drift --starts starts/drift3d --velocity velocity/drift3d \
     --insulator initial/fine_insulator \
     --interp-order linear \
     --paths paths/drift3d '0*us,90*us,0.05*us' --plot
date

echo "### [A/overshoot] overshoot check (pad plane z = 10 mm) -- WIDE-GAP STRESS TEST"
{ echo; echo "## Drift-path result (WIDE 1.4 mm gap overshoot check, pad z = 10 mm)"; echo '```'
  $PY OVERSHOOT_STUDY/check_paths.py "$POCHOIR_STORE/paths/drift3d.npz" 10; echo '```'
} | tee -a "$POCHOIR_STORE/GEOMETRY_AND_CHAMFER.md"

############################################################################
## PART B: WEIGHTING FIELD  (unit probe, multi-pixel, no-flux insulator)
##
## Same as Task8: UNIT probe (target pad = 1 V, all other conductors = 0 V) on a
## MULTI-PIXEL, NON-periodic (fix,fix,fix) 5-pixel (22 mm) domain so phi_w decays
## to ~0 at the edges.  SAME no-flux FR4 insulator mask as the drift field; NO
## epsilon.  Pad plane aligned to the drift geometry (z = 10 mm) via
## example_gen_pixel_with_grid_task9_stress_insul.json (3.0 mm pad / 1.4 mm gap).
############################################################################
echo "### [B] weighting field (unit probe, no-flux insulator, NO epsilon)"
export POCHOIR_LOG="${POCHOIR_STORE}/pochoir_weighting.log"
wgen="pcb_pixel_with_grid"
WCFG=example_gen_pixel_with_grid_task9_stress_insul.json
date

## B1: coarse weighting solve (0.4 mm, 55x55x151, z = 0..60 mm).
want domain/weight_coarse \
     pochoir domain --domain domain/weight_coarse --shape=55,55,151 --spacing '0.4*mm'
want "initial/weight_coarse boundary/weight_coarse" \
     pochoir gen --generator $wgen --domain domain/weight_coarse \
     --initial initial/weight_coarse --boundary boundary/weight_coarse $WCFG
want "potential/weight_coarse increment/weight_coarse" \
     pochoir fdm --nepochs 10 --epoch 130000000 --precision 0.0000002 \
     --edges fix,fix,fix --engine torch --multisteps no \
     --initial initial/weight_coarse --boundary boundary/weight_coarse \
     --potential potential/weight_coarse --increment increment/weight_coarse
date

## B2: near weighting gen (0.05 mm, 440x440x401, z = 0..20 mm) + refined seed.
## gen stores the no-flux mask as initial/weight_near_insulator.
want domain/weight_near \
     pochoir domain --domain domain/weight_near --shape=440,440,401 --spacing '0.05*mm'
want "initial/weight_near boundary/weight_near" \
     pochoir gen --generator $wgen --domain domain/weight_near \
     --initial initial/weight_near --boundary boundary/weight_near $WCFG
want initial/weight_near_refined \
     pochoir refine --coarse potential/weight_coarse \
     --initial initial/weight_near --boundary boundary/weight_near \
     --output initial/weight_near_refined

## B3: Dirichlet interface at z = 20 mm from the coarse bulk.
want "initial/weight_near_bc boundary/weight_near_bc" \
     pochoir near-bc --initial initial/weight_near_refined --boundary boundary/weight_near \
     --coarse potential/weight_coarse \
     --initial-out initial/weight_near_bc --boundary-out boundary/weight_near_bc

## B4: near fine weighting solve WITH the no-flux insulator BC (NO epsilon).
want "potential/weight_near increment/weight_near" \
     pochoir fdm --nepochs 10 --epoch 130000000 --precision 0.0000000002 \
     --edges fix,fix,fix --engine torch --multisteps no \
     --initial initial/weight_near_bc --boundary boundary/weight_near_bc \
     --insulator initial/weight_near_insulator \
     --potential potential/weight_near --increment increment/weight_near
date

## B4b: overlapping-Schwarz interface refinement, insulator-aware (pochoir-oz2l).
want "potential/weight_near potential/weight_far" \
     pochoir near-far-solve \
     --coarse-potential potential/weight_coarse \
     --coarse-initial initial/weight_coarse --coarse-boundary boundary/weight_coarse \
     --near-initial initial/weight_near --near-boundary boundary/weight_near \
     --near-potential potential/weight_near \
     --insulator initial/weight_near_insulator \
     --interface '20*mm' --axis 2 \
     --edges fix,fix,fix --engine torch \
     --epoch 130000000 --nepochs 10 \
     --near-precision 2e-10 --far-precision 2e-7 \
     --tol '0.001*V' --max-iters 6 \
     --near-out potential/weight_near --far-out potential/weight_far
date

## B5: stitch the near weighting solve onto the far field -> potential/weight3d.
## CAVEAT (memory weighting-field-must-stay-005mm): coarsening the NEAR weighting
## field 0.05mm->0.1mm before the Ramo calc has been shown to introduce spurious
## induced-current SPIKES.  The reference driver coarsens to avoid a 440x440x1201
## (0.05mm) full-volume OOM; keeping the near at 0.05mm through induce is the
## correct behaviour and is tracked by W2 (pochoir-tkbm).  Until W2 lands this
## follows the reference coarsen-and-stitch, so treat the induced-current
## MAGNITUDE as PRELIMINARY (spike-prone).  The SHAPE/timing and the drift/
## overshoot results (PART A) are unaffected.
want domain/weight_near_01 \
     pochoir domain --domain domain/weight_near_01 --shape=220,220,201 --spacing '0.1*mm'
want potential/weight_near_01 \
     pochoir coarsen --input potential/weight_near \
     --domain domain/weight_near_01 --output potential/weight_near_01
want domain/weight_full \
     pochoir domain --domain domain/weight_full --shape=220,220,601 --spacing '0.1*mm'
want potential/weight3d \
     pochoir stitch-near --near potential/weight_near_01 \
     --coarse potential/weight_far --domain domain/weight_full --output potential/weight3d
date

############################################################################
## PART C: INDUCED CURRENT  (Ramo: weighting field x drift paths)
############################################################################
echo "### [C] induced current (Ramo, insulator-consistent)"
want current/induced_current \
     pochoir induce-pixel --weighting potential/weight3d \
     --paths paths/drift3d \
     --output current/induced_current \
     --npixels 2 --config $WCFG --plot
date

echo "=== TASK 9 STRESS TEST DONE (wide 1.4 mm gap; drift + overshoot + weighting + induced current): $POCHOIR_STORE ==="
