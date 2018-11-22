
import matplotlib.pyplot as plt
import numpy as np
import os
from odometry import odometry
from PIL import Image
import random

# The directory with KITTI data (docker mounts data here)
basedir = '../data/dataset'

# Separate the sequences for which there is ground truth into test
# and train according to the paper's partition.
train_seqs = ['00', '02', '08', '09']

test_seqs = ['03', '04', '05', '06', '07', '10']

# Select a sequence at random.
sequence = random.choice(train_seqs)

# Load the data. Optionally, specify the frame range to load.
# dataset = pykitti.odometry(basedir, sequence)
dataset = odometry(basedir, sequence)
