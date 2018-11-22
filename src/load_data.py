"""
Helper functions for loading KITTI data for Keras.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from odometry import odometry
from PIL import Image

# The directory with KITTI data (docker mounts data here)
basedir = '../data/dataset'

def 

# Specify the sequence to load
sequence = '04'

# Load the data. Optionally, specify the frame range to load.
# dataset = pykitti.odometry(basedir, sequence)
dataset = odometry(basedir, sequence, frames=range(0, 20, 5))

# dataset.calib:      Calibration data are accessible as a named tuple
# dataset.timestamps: Timestamps are parsed into a list of timedelta objects
# dataset.poses:      List of ground truth poses T_w_cam0
# dataset.camN:       Generator to load individual images from camera N
# dataset.gray:       Generator to load monochrome stereo pairs (cam0, cam1)
# dataset.rgb:        Generator to load RGB stereo pairs (cam2, cam3)
# dataset.velo:       Generator to load velodyne scans as [x,y,z,reflectance]

# Grab some data
second_pose = dataset.poses[1]
#first_gray = next(iter(dataset.gray))
#first_cam1 = next(iter(dataset.cam1))
first_rgb = dataset.get_rgb(0)
first_cam2 = dataset.get_cam2(0)
#third_velo = dataset.get_velo(2)

# Display some of the data
np.set_printoptions(precision=4, suppress=True)
print('\nSequence: ' + str(dataset.sequence))
print('\nFrame range: ' + str(dataset.frames))

def dir_filter(_dir):
    splt = _dir[0].split("/")
    return len(splt) > 2 and splt[1].startswith("201")


def get_date_drive_pairs(datadir):
    sequence_dirs = filter(dir_filter, os.walk(datadir))
    date_drive_pairs = set()
    for thing in sequence_dirs:
        path_parts = thing[0].split("/")
        date = path_parts[1]
        drive = path_parts[2].split("_")[-2]
        date_drive_pairs.add((date, drive))
    return date_drive_pairs


def ground_truth(datadir):
    """Create y vectors arrays.

    Return list of arrays with shape
    [frames, 6], whose rows are the ground truth
    [lat, lng, alt, roll, pitch, yaw] values
    for the corresponding frame in the corresponding
    squence.
    """
    ground_truth = []
    date_drive_pairs = get_date_drive_pairs(datadir)
    datapath = os.path.join('./', datadir)
    for date, drive in date_drive_pairs:
        current_drive = []
        data = pykitti.raw(datapath, date, drive)
        for packet in data.oxts:
            current_drive.append(list(packet.packet[:6]))
        ground_truth.append(np.array(current_drive))
    return ground_truth


def get_frame_sequences(datadir):
    """Create x vectors."""

    x = []
    date_drive_pairs = get_date_drive_pairs(datadir)
    datapath = os.path.join('./', datadir)
    for date, drive in date_drive_pairs:
        current_drive = []
        data = pykitti.raw(datapath, date, drive)
        rgbs = [cam1 for cam1, _ in data.rgb]
        for first_frame, second_frame in zip(rgbs, rgbs[1:]):
            current_drive.append(
                np.dstack([np.array(first_frame),
                          np.array(second_frame)]))
        x.append(current_drive)
    return x

