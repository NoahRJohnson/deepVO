import pykitti
import os
import numpy as np
from PIL import Image

"""datadir = 'data'
date = '2011_09_26'
drive = '0001'
datapath = os.path.join('./', datadir)"""

# The 'frames' argument is optional - default: None, which loads the whole dataset.
# Calibration, timestamps, and IMU data are read automatically. 
# Camera and velodyne data are available via properties that create generators
# when accessed, or through getter methods that provide random access.
"""data = pykitti.raw(datadir, date, drive, frames=range(0, 50, 5))"""

# dataset.calib:         Calibration data are accessible as a named tuple
# dataset.timestamps:    Timestamps are parsed into a list of datetime objects
# dataset.oxts:          List of OXTS packets and 6-dof poses as named tuples
# dataset.camN:          Returns a generator that loads individual images from camera N
# dataset.get_camN(idx): Returns the image from camera N at idx  
# dataset.gray:          Returns a generator that loads monochrome stereo pairs (cam0, cam1)
# dataset.get_gray(idx): Returns the monochrome stereo pair at idx  
# dataset.rgb:           Returns a generator that loads RGB stereo pairs (cam2, cam3)
# dataset.get_rgb(idx):  Returns the RGB stereo pair at idx  
# dataset.velo:          Returns a generator that loads velodyne scans as [x,y,z,reflectance]
# dataset.get_velo(idx): Returns the velodyne scan at idx  



"""for thing in data.oxts:
    print(thing.packet[:6])
    break"""


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


        