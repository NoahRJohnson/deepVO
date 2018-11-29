#!/usr/bin/env bash
python flownet2/scripts/run-flownet-many.py \
       flownet2/models/FlowNet2/FlowNet2_weights.caffemodel.h5 \
       flownet2/models/FlowNet2/FlowNet2_deploy.prototxt.template \
       data/flow_links.txt
