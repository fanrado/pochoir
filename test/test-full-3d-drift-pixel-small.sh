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
do_domain drift3d  38,38,1500  '0.1*mm'

# fixme: these weight* identifiers need to split up for N planes.
#do_domain weight2d 1092,2000   '0.1*mm'
#do_domain weight3d 350,32,2000 '0.1*mm'


## Initial/Boundary Value Arrays ##
do_gen () {
    local name=$1 ; shift
    local geom=$1; shift
    local gen="pcb_drift_pixel_with_grid"
    local cfg="example_gen_pcb_drift_pixel_with_grid_small.json"

    want initial/$name \
         pochoir gen --generator $gen --domain domain/$name \
         --initial initial/$name --boundary boundary/$name \
         $cfg

}
do_gen drift3d quarter
#do_gen weight2d 2D
#do_gen weight3d 3D

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
do_fdm drift3d  1      130000000      0.00000002     per,per,fix
#do_fdm weight2d 4      1250000      0.00000002   fix,fix

date

