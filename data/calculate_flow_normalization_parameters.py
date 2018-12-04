#!/usr/bin/env python

"""
Utility to figure out normalization parameters to set in
train_model.py.
"""

import glob
import numpy as np
import os
import sys

from tqdm import tqdm

def read_flow(name):
    """Open .flo file as np array.

    Args:
        name: string path to file

    Returns:
        Flat numpy array
    """
    # Open the file in binary format
    f = open(name, 'rb')

    # Read first 4 bytes of file, it should be a header
    header = f.read(4)
    if header.decode("utf-8") != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')

    # Read width and height from file
    width = np.fromfile(f, dtype=np.int32, count=1).squeeze()
    height = np.fromfile(f, dtype=np.int32, count=1).squeeze()

    # Read optical flow data from file
    flow = np.fromfile(f, dtype=np.float32, count=width * height * 2)\
             .reshape((height, width, 2))\
             .astype(np.float32)

    return flow

script_dir = sys.path[0]
flow_pattern = os.path.join(script_dir,
                            "dataset",
                            "flows",
                            "**",
                            "*.flo")

min_pixel_val = np.inf
max_pixel_val = -np.inf
sum_pixel_vals = 0
n = 0

for flow_img_path in tqdm(glob.glob(flow_pattern, recursive=True)):
    try:
        flow_img = read_flow(flow_img_path)
    except FileNotFoundError:
        print("Error reading file \"{}\"".format(flow_img_path))
        continue
    flat = flow_img.flatten()

    min_pixel_val = min(min_pixel_val, flat.min())
    max_pixel_val = max(max_pixel_val, flat.max())
    sum_pixel_vals += flat.sum()
    n += len(flat)

mean_pixel_val = sum_pixel_vals / n

print("Minimum flow value: {}".format(min_pixel_val))
print("Maximum flow value: {}".format(max_pixel_val))
print("Mean flow value: {}".format(mean_pixel_val))
