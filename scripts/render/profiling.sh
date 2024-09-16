#!/bin/bash
set -e
FOLDER=$1
NERF=$2
# SSHADOW=$3
# NSHADOW=$4
ROOT=".."
CONFIG_SCRIPT="$ROOT/scripts/virtual_desc/$FOLDER.json"
EXEC="$ROOT/build/instant-ngp.exe"
INGP_PATH="$ROOT/data/nerf/$NERF.ingp"

for ((SSHADOW=1; SSHADOW<=8; SSHADOW *= 2))
do
    for ((NSHADOW=1; NSHADOW<=8; NSHADOW *= 2))
    do
        OUTPUT_NVVP="s${SSHADOW}_${NSHADOW}"
        PROFILE_PATH="$ROOT/render/profiling/${FOLDER}_${OUTPUT_NVVP}.nvvp"
        nvprof -f -o $PROFILE_PATH $EXEC --snapshot $INGP_PATH --virtual $CONFIG_SCRIPT --frag "$ROOT/scripts/virtual_desc/main.frag" --width 1280 --height 720 --sshadows $SSHADOW --nshadows $NSHADOW
    done
done