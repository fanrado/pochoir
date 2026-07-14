#!/bin/bash
# ===========================================================================
# FULL-DEPTH ALL-AT-ONCE 0.1mm SOLVE -- Neumann (no-flux FR4) validation
# (pochoir-ytw9)
#
# PURPOSE: Q<1 at the pixel edges persists even for larger pixels.  Is that
#   REAL physics (transverse charge sharing + weighting-field decay across the
#   gap) or an artifact of the near/far HYBRID pipeline (the z=20mm stitch, the
#   0.05->0.1mm weighting coarsen, the Schwarz interface)?  This script removes
#   every hybrid moving part: it solves the ENTIRE 5cm drift domain in ONE
#   full-depth FDM pass at a single uniform 0.1mm spacing -- no coarse/near/far
#   decomposition, no near-bc pin, no near-far-solve, no coarsen, no stitch --
#   for BOTH the drift and weighting fields, with the no-flux FR4 insulator BC
#   active (enableInsulatorFR4, NO epsilon), then runs velo/starts/drift/
#   induce-pixel exactly as the hybrid does.  Comparing this edge-Q map against
#   the hybrid run (default store/) isolates hybrid interface error from real
#   charge-sharing physics.
#
# WHY 0.1mm (not 0.05mm): the WEIGHTING field is multi-pixel non-periodic
#   (5 pixels = 22mm transverse) so it cannot be a periodic tile.  Full-depth
#   fine 0.05mm weighting is 440x440x1201 = ~15GB/array -> OOM (see memory
#   pochoir-gpu-mem-drift-vs-weighting).  At 0.1mm it is 220x220x601 = ~29M
#   cells = ~0.23GB/array (~2GB peak) -> a single full-depth pass fits the 24GB
#   card.  The drift field is a single-pixel periodic tile (44x44x601 @0.1mm),
#   trivially full-depth.
#
# CAVEAT (memory weighting-field-must-stay-005mm): a 0.1mm weighting field is
#   coarser than the 0.05mm reference, so the induced-current MAGNITUDE here is
#   diagnostic, not production.  The comparison target is the EDGE-Q MAP shape
#   (does Q<1 at the gap survive with zero hybrid interfaces?), not the absolute
#   current.  This is deliberately a native 0.1mm solve, NOT a 0.05->0.1 coarsen
#   of a fine solve (which is the separate spike source).
#
# Geometry: identical to the hybrid 5cm run (3.8mm pad / 0.6mm gap / 0.4mm
#   dynamic chamfer / pad plane z=10mm / cathode z=60mm @ -2500V / no-flux FR4
#   slab just below the pad, NO epsilon).  Reuses the SAME insulator configs as
#   run-full-3d-pixel.sh (they are spacing-independent -- geometry is in mm).
#
# Output store: store_fulldepth_01mm_neumann/
# Run from test/:  ./run-fulldepth-01mm-neumann.sh [STORE_DIR]
# ===========================================================================
set -e
cd "$(dirname "$0")"
ROOT="$(cd .. && pwd)"
export PATH="$ROOT/env/bin:$PATH"
PY="$ROOT/env/bin/python"
export POCHOIR_STORE="${1:-store_fulldepth_01mm_neumann}"
export POCHOIR_LOG="${POCHOIR_STORE}/pochoir.log"

# Same no-flux FR4 insulator configs as the hybrid driver (NO epsilon).
dgen="pcb_drift_pixel_with_grid"
dcfg="example_gen_pcb_drift_pixel_with_grid_insul.json"
wgen="pcb_pixel_with_grid"
wcfg="example_gen_pixel_with_grid_insul.json"

# want: run a step only when EVERY listed output key already exists.
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
echo "### FULL-DEPTH 0.1mm Neumann validation -> $POCHOIR_STORE"

############################################################################
## PART A: DRIFT FIELD -- single full-depth 0.1mm solve (periodic tile)
############################################################################
## One 4.4mm pixel pitch (3.8+0.6) = 44 cells @0.1mm; full 60mm depth = 601
## planes.  Periodic transverse (per,per,fix), no near/far.  gen stores the
## no-flux mask initial/drift_insulator; fdm applies it (NO epsilon).
echo "### [A] drift field: 44x44x601 @0.1mm, full-depth, no-flux insulator"
want domain/drift \
     pochoir domain --domain domain/drift --shape=44,44,601 --spacing '0.1*mm'

want "initial/drift boundary/drift" \
     pochoir gen --generator $dgen --domain domain/drift \
     --initial initial/drift --boundary boundary/drift $dcfg

want "potential/drift3d increment/drift3d" \
     pochoir fdm \
     --nepochs 10 --epoch 130000000 --precision 0.000000002 \
     --edges per,per,fix --engine torch \
     --initial initial/drift --boundary boundary/drift \
     --insulator initial/drift_insulator \
     --potential potential/drift3d --increment increment/drift3d
date

############################################################################
## PART B: WEIGHTING FIELD -- single full-depth 0.1mm solve (multi-pixel)
############################################################################
## 5-pixel non-periodic extent (22mm) = 220 cells @0.1mm so phi_w decays to ~0
## at the edges; full 60mm depth = 601 planes.  Dirichlet transverse
## (fix,fix,fix), --multisteps no, SAME no-flux FR4 mask as the drift field.
echo "### [B] weighting field: 220x220x601 @0.1mm, full-depth, no-flux insulator"
export POCHOIR_LOG="${POCHOIR_STORE}/pochoir_weighting.log"

want domain/weight \
     pochoir domain --domain domain/weight --shape=220,220,601 --spacing '0.1*mm'

want "initial/weight boundary/weight" \
     pochoir gen --generator $wgen --domain domain/weight \
     --initial initial/weight --boundary boundary/weight $wcfg

want "potential/weight3d increment/weight3d" \
     pochoir fdm \
     --nepochs 10 --epoch 130000000 --precision 0.000000002 \
     --edges fix,fix,fix --engine torch --multisteps no \
     --initial initial/weight --boundary boundary/weight \
     --insulator initial/weight_insulator \
     --potential potential/weight3d --increment increment/weight3d
date

############################################################################
## PART C: VELOCITY, PATHS, INDUCED CURRENT (insulator-consistent)
############################################################################
export POCHOIR_LOG="${POCHOIR_STORE}/pochoir.log"
echo "### [C] velocity + drift paths + induced current"

## Drift velocity (velo zeros the field inside the FR4 insulator + electrodes).
want velocity/drift3d \
     pochoir velo --temperature '87.0*K' \
     --potential potential/drift3d --boundary boundary/drift \
     --insulator initial/drift_insulator \
     --velocity velocity/drift3d

## 10x10 launch grid across one 4.4mm pitch, at z=59.95mm (last full-velocity
## node below the cathode plane z=60mm), matching the hybrid run so the edge-Q
## map is directly comparable.
dist=(0.22 0.66 1.1 1.54 1.98 2.42 2.86 3.3 3.74 4.18)
points=()
for d in "${dist[@]}"; do
     for d2 in "${dist[@]}"; do
         points+=("${d}*mm,${d2}*mm,59.95*mm")
     done
done
want starts/drift3d \
     pochoir starts --starts starts/drift3d -m no -c $dcfg "${points[@]}" --plot

## Drift window ~90us (49.5mm drift @ ~1.5us/mm + margin); linear interp
## (cubic over-focuses near the pixel plane -- memory cubic-drift-overshoot).
want paths/drift3d \
     pochoir drift --starts starts/drift3d --velocity velocity/drift3d \
     --insulator initial/drift_insulator \
     --interp-order linear \
     --paths paths/drift3d '0*us,90*us,0.05*us' --plot
date

echo "### [A/overshoot] overshoot check (pad plane z = 10 mm)"
$PY OVERSHOOT_STUDY/check_paths.py "$POCHOIR_STORE/paths/drift3d.npz" 10 || true

## Induced current via Ramo (weighting field x drift paths).  See the header
## CAVEAT: the 0.1mm weighting magnitude is diagnostic; the edge-Q MAP is the
## comparison target vs the hybrid run (default store/).
echo "### [C] induced current (Ramo, full-depth 0.1mm weighting)"
want current/induced_current \
     pochoir induce-pixel --weighting potential/weight3d \
     --paths paths/drift3d \
     --output current/induced_current \
     --npixels 2 --config $wcfg --plot
date

echo "=== FULL-DEPTH 0.1mm NEUMANN VALIDATION DONE: $POCHOIR_STORE ==="
echo "Compare the edge-Q map / induced-charge-per-start against the hybrid run"
echo "(default store/) to separate real charge-sharing physics from"
echo "near/far hybrid-interface artifacts (stitch, coarsen, Schwarz seam)."
