#!/bin/bash

set -e

#tdir="$(dirname $(realpath $BASH_SOURCE))"

export POCHOIR_STORE="${1:-store}"

source helpers.sh

want current/induced_current_avg_Ind1_ \
     pochoir induce-pixel --weighting potential/weight3d \
     --paths paths/drift3d_tight \
     --output current/pixel \
     --npixels 2.0


