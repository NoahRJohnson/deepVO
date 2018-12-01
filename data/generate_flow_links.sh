#!/usr/bin/env bash

###
# Run this from the caffe flownet docker container
###

# The 'data' folder that this script is in
DATA_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

# The folder with all the KITTI sequences
SEQUENCES_FOLDER="$DATA_DIR"/dataset/sequences
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: $DATA_DIR/dataset/sequences does not exist. Map your KITTI data with docker."
    exit 1
fi

# The folder to write out flows to (will create if it doesn't exist)
FLOWS_FOLDER="$DATA_DIR"/dataset/flows
if [ ! -d "$FLOWS_FOLDER" ]; then
    echo "Creating $FLOWS_FOLDER"
    mkdir "$FLOWS_FOLDER"
fi

# The output .txt file with "x.png y.png z.flo" lines for use by flownet2
OUT_FILE="$DATA_DIR"/flow_links.txt
# Delete file if it already exists, so we can remake it
if [ -f "$OUT_FILE" ]; then
    rm "$OUT_FILE"
fi

# Loop over every sequence
for seq_path in "$SEQUENCES_FOLDER"/*/ ; do

    # Get name of corresponding flow folder for this sequence
    seq=$(basename "$seq_path")
    FLOW_FOLDER="$FLOWS_FOLDER"/"$seq"

    # Create a corresponding flow folder for this sequence if it doesn't exist
    if [ ! -d "$FLOW_FOLDER" ]; then
        echo "Creating $FLOW_FOLDER"
        mkdir "$FLOW_FOLDER"
    fi

    echo "Generating image frame pair -> flow file mappings for sequence $seq"

    # Gather all of the image frames for this sequence into an ordered list
    IMAGE_ARRAY=($(ls "$seq_path"image_2/* | sort -n))

    # Loop over every sequential pair of image frames
    for (( i = 0; i < ${#IMAGE_ARRAY[@]} - 1; ++i )); do
        IMAGE1="${IMAGE_ARRAY[i]}"
        IMAGE2="${IMAGE_ARRAY[i+1]}"
        FLO="$FLOW_FOLDER/$i".flo

        # And write out a "x.png y.png z.flo" line
        echo "$IMAGE1 $IMAGE2 $FLO" >> "$OUT_FILE"
    done
done
