#!/bin/bash

set -e

#tdir="$(dirname $(realpath $BASH_SOURCE))"

export POCHOIR_STORE="${1:-store}"

source helpers.sh

date

## Domains ##
do_domain () {
    local name=$1 ; shift
    local shape=$1; shift
    local spacing=$1; shift

    want domain/$name \
         pochoir domain --domain domain/$name \
         --shape=$shape --spacing $spacing
}

do_domain weight3d 396,396,1500 '0.1*mm' #220,220,1500 '0.1*mm'


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

## Fields
do_fdm () {
    local name=$1 ; shift
    local nepochs=$1 ; shift
    local epoch=$1 ; shift
    local prec=$1 ; shift
    local edges=$1 ; shift

    want potential/$name \
         pochoir fdm \
         --nepochs $nepochs --epoch $epoch --precision $prec \
         --edges $edges \
	 --engine torch \
         --initial initial/$name --boundary boundary/$name \
         --potential potential/$name \
         --increment increment/$name
}
do_fdm weight3d 1      5000000      0.00000002   fix,fix,fix




#want velocity/drift3d \
#     pochoir velo --temperature '89*K' \
#     --potential potential/drift3d \
#     --velocity velocity/drift3d

#rm -r store/starts

#want starts/drift3d \
#     pochoir starts --starts starts/drift3d \
#     '1.25*mm,0.835*mm,69*mm'  

#want paths/drift3d \
#     pochoir drift --starts starts/drift3d \
#     --velocity velocity/drift3d \
#     --paths paths/drift3d '0*us,4250*us,0.1*us'


#want current/induced_current \
#     pochoir induce --weighting potential/weight3dfull \
#     --paths paths/drift3d \
#     --output current/induced_current

date
