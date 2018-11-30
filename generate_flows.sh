#!/usr/bin/env bash

# Generate text file of "x.png y.png z.flo" lines
# for all of the KITTI image frame pairs, and
# creates flows/ directory
bash data/generate_flow_links.sh

# Pass text file to FlowNet2, let it produce .flo files
python flownet2/scripts/run-flownet-many.py \
       flownet2/models/FlowNet2/FlowNet2_weights.caffemodel.h5 \
       flownet2/models/FlowNet2/FlowNet2_deploy.prototxt.template \
       data/flow_links.txt
