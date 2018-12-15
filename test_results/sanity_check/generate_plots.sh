#!/usr/bin/env bash

# The folder that this script is in, with pose estimates
RESULTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

# The folder with all the KITTI ground-truth poses
POSES_DIR="$RESULTS_DIR"/../../data/dataset/poses
if [ ! -d "$POSES_DIR" ]; then
    echo "ERROR: $POSES_DIR does not exist. Map your KITTI data with docker."
    exit 1
fi


for fullfile in *.csv; do
  filename=$(basename -- "$fullfile")
  filename="${filename%.*}"
  echo Processing "$filename"
  evo_traj kitti "$fullfile" "$POSES_DIR"/"$filename".txt --save_plot "$filename".pdf
done
