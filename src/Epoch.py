"""Generate batches for training.

A batch is of size one, meaning one sub-sequence. A sub-sequence contains
batch_size frames. Ground truth for the subsequence is modified so that
translations and rotations are relative to the first frame of the sub-sequence
rather than the first frame of the full sequence. Rotation matrices are
converted to Euler angles.
"""

import math
import numpy as np
import os
import random

from numpy.linalg import inv
from odometry import odometry
from os.path import join
from PIL import Image


def is_rotation_matrix(r):
    """Check if a matrix is a valid rotation matrix.

    referred from https://www.learnopencv.com/rotation-matrix-to-euler-angles/
    """
    rt = np.transpose(r)
    should_be_identity = np.dot(rt, r)
    i = np.identity(3, dtype=r.dtype)
    n = np.linalg.norm(i - should_be_identity)
    return n < 1e-6


def rotation_matrix_to_euler_angles(r):
    """Convert rotation matrix to euler angles.

    referred from https://www.learnopencv.com/rotation-matrix-to-euler-angles
    """
    assert(is_rotation_matrix(r))
    sy = math.sqrt(r[0, 0] * r[0, 0] + r[1, 0] * r[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(r[2, 1], r[2, 2])
        y = math.atan2(-r[2, 0], sy)
        z = math.atan2(r[1, 0], r[0, 0])
    else:
        x = math.atan2(-r[1, 2], r[1, 1])
        y = math.atan2(-r[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def rectify_poses(poses):
    """Set ground truth relative to first pose in subsequence.

    Poses are rotation-translation matrices relative to the first
    pose in the full sequence. To get meaningful output from sub-
    sequences, we need to alter them to be relative to the
    first position in the sub-sequence.

    Args:
        poses:  An iterable of 4x4 rotation-translation matrices representing
                the vehicle's pose at each step in the sequence.

    Returns:
        An iterable of rectified rotation-translation matrices
    """
    first_frame = poses[0]
    rectified_poses = [np.dot(inv(first_frame), x) for x in poses[1:]]
    return rectified_poses


def mat_to_pose_vector(pose):
    """Convert the 4x4 rotation-translation matrix into a 6-dim vector.

    Args:
        pose:  The 4x4 rotation-translation matrix representing the vehicle's
        pose.

    Returns:
        The pose represented as a vector.

        I.e. a (roll, pitch, yaw, lat, lng, alt) numpy array.
    """
    return np.concatenate((rotation_matrix_to_euler_angles(pose[:3, :3]),
                          pose[:3, 3]))


def process_poses(poses):
    """Fully convert subsequence of poses."""
    rectified_poses = rectify_poses(poses)
    return np.array([mat_to_pose_vector(pose) for pose in rectified_poses])


def get_stacked_rgbs(dataset, batch_frames):
    """Return list of dstacked rbg images."""
    rgbs = [np.array(left_cam) for left_cam, _ in dataset.rgb]
    mean_rgb = sum(rgbs) / float(batch_frames)
    rgbs = [rgb - mean_rgb for rgb in rgbs]
    return [np.dstack((frame1, frame2))
            for frame1, frame2 in zip(rgbs, rgbs[1:])]


def test_batch(basedir, seq):
    """Process images and ground truth for a test sequence.

    Args:
        basedir: The directory where KITTI data is stored.
        seq: The KITTI sequence number to test.

    Returns:
        A batch of the form

        {'x': x, 'y': y}

        for consumption by Keras, where x is data and y is labels.
    """
    dataset = odometry(basedir, seq)
    poses = dataset.poses
    x = np.array([np.vstack(get_stacked_rgbs(dataset))])
    y = process_poses(poses)
    return {'x': x, 'y': y}


class Epoch():
    """Create batches of sub-sequences.

    Divide all train sequences into subsequences
    and yield batches subsequences without repetition
    until all subsequences have been exhausted.
     """

    def __init__(self, datadir, traindir, train_seq_nos,
                 step_size, n_frames, batch_size):
        if step_size > n_frames:
            print("WARNING: step_size greater than n_frames. "
                  "This will result in unseen sequence frames.")
        self.traindir = traindir
        self.datadir = datadir
        self.train_seq_nos = train_seq_nos
        self.step_size = step_size
        self.n_frames = n_frames
        self.batch_size = batch_size

        self.window_idxs_dict = self.partition_sequences()

    def is_complete(self):
        for seq_no, idx_dict in self.window_idxs_dict.items():
            if len(idx_dict) > 0:
                return False
        else:
            return True

    def partition_sequences(self):
        idx_dict = {seq_no: None for seq_no in self.train_seq_nos}
        for seq_no in idx_dict:
            windows = []
            len_seq = len(os.listdir(join(self.traindir, seq_no)))
            for window_start in range(1, len_seq - self.n_frames + 1,
                                      self.step_size):
                window_end = window_start + self.n_frames + 1
                windows.append((window_start, window_end))
            random.shuffle(windows)
            idx_dict[seq_no] = windows
        return idx_dict

    def get_sample(self, seq, window_bounds):
        """Create one sample."""
        seq_path = join(self.traindir, seq)
        frame_nos = range(*(window_bounds))

        x = [np.array(Image.open(join(seq_path, "{i}.png".format(i=frame_no))))
             for frame_no in frame_nos]

        x = np.array(x)

        raw_poses = odometry(self.datadir, seq, frames=frame_nos).poses
        y = process_poses(raw_poses)
        return (x, y)

    def get_batch(self):
        x = []
        y = []
        for sample in range(self.batch_size):
            empty_idx_dicts = []
            seq = random.choice(self.train_seq_nos)
            while len(self.window_idxs_dict[seq]) == 0:
                empty_idx_dicts.append(seq)
                seq = random.choice([key for key in self.window_idxs_dict
                                     if key not in empty_idx_dicts])
            window_bounds = self.window_idxs_dict[seq].pop()
            sample_x, sample_y = self.get_sample(seq, window_bounds)
            x.append(sample_x)
            y.append(sample_y)
        x = np.array(x)
        y = np.array(y)
        return (x, y)
