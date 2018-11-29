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


def read_flow(name):
    """Open .flo file as np array.

    Args:
        name: string path to file

    Returns:
        Flat numpy array
    """
    if name.endswith('.pfm') or name.endswith('.PFM'):
        return readPFM(name)[0][:,:,0:2]

    f = open(name, 'rb')

    header = f.read(4)
    if header.decode("utf-8") != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')

    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()

    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))

return flow.astype(np.float32)


class Epoch():
    """Create batches of sub-sequences.

    Divide all train sequences into subsequences
    and yield batches of subsequences without repetition
    until all subsequences have been exhausted.
    """

    def __init__(self, datadir, traindir, train_seq_nos,
                 step_size, n_frames, batch_size):
        """Initialize.

        Args:
            datadir: The directory where the kitti `sequences` folder
                     is located.
            traindir: The directory where the flownet images are
            train_seq_nos: list of strings corresponding to kitti
                           sequences in the training set
            step_size: int. Step size for sliding window in sequence
                       partitioning
            n_frames: Number of frames per window in sequence
                      partitioning.
            batch_size: Number of samples (subsequences) per batch.
                        Final batch may be smaller if batch_size is
                        greater than the number of subsequences remaining
                        when get_batch() is called.
        """
        if step_size > n_frames:
            print("WARNING: step_size greater than n_frames. "
                  "This will result in unseen sequence frames.")
        self.traindir = traindir
        self.datadir = datadir
        self.train_seq_nos = train_seq_nos
        self.step_size = step_size
        self.n_frames = n_frames
        self.batch_size = batch_size
        self.window_idxs = []

        self.partition_sequences()

    def is_complete(self):
        """Stop serving batches if there are no more unused subsequences."""
        if len(self.window_idxs) > 0:
            return False
        else:
            return True

    def partition_sequences(self):
        """Partition a sequence into subsequences.

        Create subsequences of length n_frames, with starting indices
        staggered by step_size.

        NOTES: This will NOT output a short, final subsequence if the
        arithmetic doesn't work out nicely. Doing so would screw up
        the dimensions everywhere unless the final subsequence was
        zero-buffered to length n_frames, which could cause other issues.
        ALSO: self.step_size > self.n_frames will result in frames from
        the full sequence failing to appear in the epoch.
        """
        for seq_no in self.train_seq_nos:
            len_seq = len(os.listdir(join(self.traindir, seq_no)))
            for window_start in range(1, len_seq - self.n_frames + 1,
                                      self.step_size):
                window_end = window_start + self.n_frames + 1
                self.window_idxs.append((seq_no, (window_start, window_end)))
        random.shuffle(self.window_idxs)

    def get_sample(self, window_idx):
        """Create one sample.

        Create one n_frames long subsequence.

        Args:
            windox_idx: (seq_no, (start_frame, end_frame + 1))

        Returns:
            (x, y):
                x: An (n_frames, HxWx3) array of flownet image pixels
                y: An (n_frames, 6) array of ground truth poses
        """
        seq, window_bounds = window_idx
        seq_path = join(self.traindir, seq)
        frame_nos = range(*(window_bounds))

        x = [read_flow(join(seq_path,
                            "{i}.flo".format(i=frame_no)))
             for frame_no in frame_nos]

        x = np.array(x)

        raw_poses = odometry(self.datadir, seq, frames=frame_nos).poses
        y = process_poses(raw_poses)
        return (x, y)

    def get_batch(self):
        """Get a batch.

        Returns:
            (x, y):
                x: A (batch_size, n_frames, HxWx3) np array of subsequences.
                y: A (batch_size, n_frames, HxWx3) np array of ground truth
                   pose vectors.
        NOTE: See __init__ docstring note about batch_size.
        """
        x = []
        y = []
        for sample in range(self.batch_size):
            if not self.is_complete():
                window_idx = self.window_idxs.pop()
                sample_x, sample_y = self.get_sample(window_idx)
                x.append(sample_x)
                y.append(sample_y)
        x = np.array(x)
        y = np.array(y)
        return (x, y)
