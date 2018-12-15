"""Test batch loader."""
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

# Import epoch class
import sys
testdir = os.path.dirname(__file__)
srcdir = '..'
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))
from epoch import Epoch

datadir = 'data/dataset'
batch_size = 1
subseq_length = 4
step_size = 1

# Separate the sequences for which there is ground truth into test 
# and train according to the paper's partition. 
train_seqs = ['00', '02', '08', '09']
test_seqs = ['03', '04', '05', '06', '07', '10']

# Create a data loader to get batches one epoch at a time
epoch_data_loader = Epoch(datadir=datadir,
                          flowdir=os.path.join(datadir, "flows"),
                          train_seq_nos=train_seqs,
                          test_seq_nos=test_seqs,
                          window_size=subseq_length,
                          step_size=step_size,
                          batch_size=batch_size)

# What is the shape of the input flow images?
flow_input_shape = epoch_data_loader.get_input_shape()
print("Input shape: {}".format(flow_input_shape))

# Test a batch
X, Y = epoch_data_loader.get_training_batch()

print("[Batch] X.shape = {}".format(X.shape))
print("[Batch] Y.shape = {}".format(Y.shape))

random_sample_index = np.random.randint(0, len(X))

sample = X[random_sample_index]
label = Y[random_sample_index]

print("[Sample] X.shape = {}".format(sample.shape))
print("[Sample] Y.shape = {}".format(label.shape))

print("Example flow pixel value: {}".format(sample[0, 0, 0, 0:]))

print("Sample min: {}".format(np.min(sample)))
print("Sample max: {}".format(np.max(sample)))
print("Sample mean: {}".format(np.mean(sample)))

f, axarr = plt.subplots(2,2)
axarr[0,0].imshow(sample[0][:,:,0])
axarr[0,1].imshow(sample[0][:,:,1])
axarr[1,0].imshow(sample[1][:,:,0])
axarr[1,1].imshow(sample[1][:,:,1])

plt.savefig('sample.png')

# Split an entire test sequence
test_x, test_y = epoch_data_loader.get_testing_samples('01')

print(len(test_y))
print(test_y[:10])
