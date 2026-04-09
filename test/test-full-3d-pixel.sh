#!/bin/bash
set -e
export POCHOIR_STORE="${1:-store}"

source helpers.sh

# date

##
# ## set the log filename for the Drift field calculation
# export POCHOIR_LOG="${POCHOIR_STORE}/pochoir_driftfield.log"
# ## Domains ##
# echo "=== Domains ==="
# do_domain () {
#     local name=$1 ; shift
#     local shape=$1; shift
#     local spacing=$1; shift

#     want domain/$name \
#          pochoir domain --domain domain/$name \
#          --shape=$shape --spacing $spacing
# }
# do_domain drift3d  44,44,700  '0.1*mm'

# # # fixme: these weight* identifiers need to split up for N planes.
# # #do_domain weight2d 1092,2000   '0.1*mm'
# # #do_domain weight3d 350,32,2000 '0.1*mm'


# ## Initial/Boundary Value Arrays ##
# do_gen () {
#     local name=$1 ; shift
#     local geom=$1; shift
#     local gen="pcb_drift_pixel_with_grid"
#     local cfg="example_gen_pcb_drift_pixel_with_grid.json"

#     want initial/$name \
#          pochoir gen --generator $gen --domain domain/$name \
#          --initial initial/$name --boundary boundary/$name \
#          $cfg

# }
# do_gen drift3d quarter
# # # do_gen weight2d 2D 
# # # do_gen weight3d 3D

# # echo "=== Fields ==="
# ## Fields
# do_fdm () {
#     local name=$1 ; shift
#     local nepochs=$1 ; shift
#     local epoch=$1 ; shift
#     local prec=$1 ; shift
#     local edges=$1 ; shift

#     want potential/$name \
#          pochoir fdm \
#          --nepochs $nepochs --epoch $epoch --precision $prec \
#          --edges $edges \
#          --engine torch \
#          --initial initial/$name --boundary boundary/$name \
#          --potential potential/$name \
#          --increment increment/$name \
#          --multisteps yes
# }
# do_fdm drift3d  20      1000000      0.00000002    per,per,fix #130,000,000
# # do_fdm drift3d  10      13000      0.00000002     per,per,fix #130,000,000
# # do_fdm weight2d 1      1200      0.00000002   fix,fix #1250000

# python parse_maxerr.py ${POCHOIR_STORE}/pochoir_driftfield.log ${POCHOIR_STORE}/maxerr_driftfield.png ${POCHOIR_STORE}/summary_log_driftfield.pdf
##
# echo "=== Weighting potentials ==="
## set the log filename for the weighting potential calculation
export POCHOIR_LOG="${POCHOIR_STORE}/pochoir_weightingfield.log"
## Weighting potentials
## Domains ##
do_domain () {
    local name=$1 ; shift
    local shape=$1; shift
    local spacing=$1; shift

    want domain/$name \
         pochoir domain --domain domain/$name \
         --shape=$shape --spacing $spacing
}

# do_domain weight3d 396,396,1500 '0.1*mm' #220,220,1500 '0.1*mm'
do_domain weight3d 220,220,700 '0.1*mm'

## Initial/Boundary Value Arrays ##
do_gen () {
    local name=$1 ; shift
    local geom=$1; shift
    local gen="pcb_pixel_with_grid"
    local cfg="example_gen_pixel_with_grid.json"

    want initial/$name \
         pochoir gen --generator $gen --domain domain/$name \
         --initial initial/$name --boundary boundary/$name \
         $cfg

}
do_gen weight3d 3D

# ## Fields
# do_fdm () {
#     local name=$1 ; shift
#     local nepochs=$1 ; shift
#     local epoch=$1 ; shift
#     local prec=$1 ; shift
#     local edges=$1 ; shift

#     want potential/$name \
#          pochoir fdm \
#          --nepochs $nepochs --epoch $epoch --precision $prec \
#          --edges $edges \
# 	 --engine torch \
#          --initial initial/$name --boundary boundary/$name \
#          --potential potential/$name \
#          --increment increment/$name
# }
# do_fdm weight3d 1      500000      0.00000002   fix,fix,fix #5000000 ### try 100 epochs, and more steps per epoch
# # python parse_maxerr.py store/pochoir.log store/maxerr_weightingfield.png store/summary_log_weightingfield.pdf
# #
# echo "=== Velocities ==="
# ## Velocities
# want velocity/drift3d \
#      pochoir velo --temperature '87.0*K' \
#      --potential potential/drift3d \
#      --velocity velocity/drift3d
# #

# # ## Need to be run separately
# echo "=== Paths ==="

# dist=(0.22 0.66 1.1 1.54 1.98 2.42 2.86 3.3 3.74 4.18)
# points=()
# for d in "${dist[@]}"; do
#      for d2 in "${dist[@]}"; do
#          points+=("${d}*mm,${d2}*mm,148*mm")
#      done
# done
# ## Paths
# want starts/drift3d \
#     pochoir starts --starts starts/drift3d \
#     -m no \
#     ${points[@]}

# #rm -r /Users/sergey/Desktop/ICARUS/LArStand/pochoir/test/store/paths

# want paths/drift3d_tight \
#      pochoir drift --starts starts/drift3d \
#      --velocity velocity/drift3d \
#      --paths paths/drift3d_tight '0*us,320*us,0.05*us'


# ##
# echo "=== Induced currents ==="
# ## Induced currents
# want current/induced_current \
#      pochoir induce-pixel --weighting potential/weight3d \
#      --paths paths/drift3d_tight \
#      --output current/induced_current \
#      --npixels 2.0