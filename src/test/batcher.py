"""Test batch loader."""
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

import sys
testdir = os.path.dirname(__file__)
srcdir = '..'
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))
import batcher

basedir = '/home/noah/kitti_data/dataset'
BATCH_SIZE = 5

train_seqs = ['00', '02', '08', '09']

batch = batcher.batcher(basedir, BATCH_SIZE, train_seqs)

X = batch['x']

Y = batch['y']

print(X.shape)

print(Y.shape)


