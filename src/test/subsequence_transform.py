#!/usr/bin/env python
import numpy as np
import os

# Import epoch and  class
import sys
testdir = os.path.dirname(__file__)
srcdir = '..'
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))
from epoch import Epoch
from subseq_preds_to_full_pred import subseq_preds_to_full_pred

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


pose_labels = np.array([y for x,y in epoch_data_loader.get_testing_samples('01')])

subseq_preds_to_full_pred(pose_labels, '01_predictions.txt')
