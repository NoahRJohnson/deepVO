"""Generate batches for training.

A batch consists of samples, where each sample is a sub-sequence. A sub-
sequence contains batch_size frames. Ground truth for the subsequence is
modified so that translations and rotations are relative to the first
frame of the sub-sequence rather than the first frame of the full
sequence. Rotation matrices are converted to Euler angles.
"""

import math
import numpy as np
import os
import random
import operator

from odometry import odometry
from os.path import join


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


def rectify_poses(reference_pose, poses):
    """Set ground truth relative to reference pose.

    Poses are rotation-translation matrices relative to the first
    pose in the full sequence. To get meaningful output from sub-
    sequences, we need to alter them to be relative to the
    first position in the sub-sequence.

    Args:
        reference_pose: A 4x4 rotation-translation matrix representing
                        the very first pose from which a subsequence
                        starts. Note that this would be one before
                        the first pose in a subsequence label, since
                        the first time step in a subsequence has output
                        pose equal to the ground truth at the second
                        image frame.
        poses:  An iterable of 4x4 rotation-translation matrices
                representing the vehicle's pose at each time step
                in the subsequence.

    Returns:
        An iterable of rectified rotation-translation matrices
    """

    rectified_poses = [np.dot(np.linalg.inv(reference_pose), x)
                       for x in poses]
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


def convert_flow_to_feature_vector(flow, crop_shape):
    """Crop the center, and flatten."""
    # Crop image from center based on minimum size
    # https://stackoverflow.com/a/50322574
    start = tuple(map(lambda a, da: int(a // 2) - int(da // 2), flow.shape, crop_shape))
    end = tuple(map(operator.add, start, crop_shape))
    slices = tuple(map(slice, start, end))
    crop = flow[slices]

    # Flatten image to 1-D feature vector
    return crop.flatten()


class Epoch():
    """Create batches of sub-sequences.

    Divide all train sequences into subsequences
    and yield batches of subsequences without repetition.
    """

    def __init__(self, datadir, flowdir, train_seq_nos,
                 window_size, step_size, batch_size):
        """Initialize.

        Args:
            datadir: The directory where the kitti `sequences` folder
                     is located.
            flowdir: The directory where the flownet images are
            train_seq_nos: list of strings corresponding to kitti
                           sequences in the training set
            window_size: Number of flow images per window in sequence
                         partitioning, i.e. subsequence length.
            step_size: int. Step size for sliding window in sequence
                       partitioning
            batch_size: Number of samples (subsequences) per batch.
                        Final batch may be smaller if batch_size is
                        greater than the number of subsequences remaining
                        when get_batch() is called.
        """
        if step_size > window_size:
            print("WARNING: step_size greater than window size. "
                  "This will result in unseen sequence frames.")

        self.datadir = datadir
        self.flowdir = flowdir
        self.train_seq_nos = train_seq_nos
        self.window_size = window_size
        self.step_size = step_size
        self.batch_size = batch_size

        # Calculate subsequence indices
        self.partitions = self.partition_sequences()

        # Compute minimum flow image size (they can differ)
        # We'll use these dimensions to crop all flow images
        self.min_flow_shape = self.compute_min_flow_shape()

    def compute_min_flow_shape(self):
        """Compute minimum dimension of .flo images across sequences."""
        min_shape = np.full((3,), fill_value=np.inf)

        for seq_no in self.train_seq_nos:
            ex_path = join(self.flowdir, seq_no, "0.flo")

            try:
                ex_img = read_flow(ex_path)
            except FileNotFoundError:
                continue

            min_shape = np.minimum(min_shape, ex_img.shape)
            min_shape = np.array([int(thing) for thing in min_shape])
            print("################# min_shape: ", min_shape)
        return min_shape

    def get_num_features(self):
        """Number of pixels in flow images."""
        return np.prod(self.min_flow_shape)

    def is_complete(self):
        """Check if epoch is complete.

        The epoch is done if we can't completely fill up
        another batch.
        """
        if len(self.partitions) < self.batch_size:
            return True
        else:
            return False

    def reset(self):
        """Reset the Epoch instance.

        Call when an epoch is done, and you want to train
        over another epoch.
        """
        self.partitions = self.partition_sequences()

    def partition_sequences(self):
        """Partition training sequences into subsequences.

        No data is actually loaded yet, we just figure out
        the indices for the different subsequences which will
        act as the samples for training.

        Subsequences are of length window_size, with starting indices
        staggered by step_size.

        The indices will be used later to actually load the
        corresponding .flo images.

        Returns:
            A list of (sequence_number, start_index, end_index) tuples
            where each tuple corresponds to a subsequence. start_index
            is inclusive, and end_index is exclusive.

        NOTE:
            The final subsequence may need to be padded to be the same
            length as all the others, if the arithmetic doesn't work
            out nicely. Also, self.step_size > self.window_size will
            result in flow samples from the full sequence failing to
            appear in the epoch.
        """
        partitions = []

        # For every KITTI sequence
        for seq_no in self.train_seq_nos:

            # Get the length of that sequence
            len_seq = len(os.listdir(join(self.flowdir, seq_no)))

            # For every sliding window in that sequence
            for window_start in range(0, len_seq - self.window_size + 1,
                                      self.step_size):

                # Don't give window bounds with upper bound greater than
                # the number of actual frames in the sequence. Padding
                # is handled in get_sample() for short final sub-sequence.
                # End bounds are exclusive, to match range().
                window_end = min(window_start + self.window_size, len_seq)
                partitions.append((seq_no, window_start, window_end))

        # Shuffle the training data
        random.shuffle(partitions)

        return partitions

    def get_sample(self, seq_no, start_idx, end_idx):
        """Load one sample.

        Load a subsequence of optical flow images from
        sequence seq_no, and the corresponding
        subsequence of pose vectors.

        Pads the ends with zeros if necessary to ensure
        the subsequences are of length window_size.

        Args:
            seq_no: string, KITTI sequence
            start_idx: What index in the sequence to start from (inclusive)
            end_idx: What index in the sequence to end at (exclusive). May
                     be less than window_size away from start_idx, which
                     will require padding.
        Returns:
            A tuple (x, y):
                x: A (window_size, H*W*2) array of flownet image pixels
                y: A (window_size, 6) array of rectified ground truth poses
        """

        # Path to flow folder for this sequence
        flow_seq_path = join(self.flowdir, seq_no)

        # The flow indices to load. These are the
        # same as the file names (w/o extension)
        flow_indices = range(start_idx, end_idx)

        # Load raw optical flow images
        x = [read_flow(join(flow_seq_path,
                            "{}.flo".format(frame_no)))
             for frame_no in frame_nos]

        # Crop and flatten images into feature vectors
        x = [convert_flow_to_feature_vector(flow, self.min_flow_shape)
             for flow in x]

        # Convert to numpy array
        x = np.array(x)

        # The pose file containing ground truth poses for
        # this sequence
        pose_file_path = join(self.datadir,
                              'poses',
                              '{}.txt'.format(seq_no))

        # The indices of the poses are shifted one ahead of
        # the flows (flow image 0 was made from two frames,
        # and the pose we want to estimate from that is the
        # true pose at the timestamp of the second frame, at
        # index 1)
        pose_indices = set(range(start_idx + 1, end_idx + 1))

        # Read and parse the ground truth poses
        # This part comes from pykitti
        # https://github.com/utiasSTARS/pykitti/blob/0e5fd7fefa7cd10bbdfb5bd131bb58481d481116/pykitti/odometry.py#L210
        raw_poses = []
        try:
            with open(pose_file_path, 'r') as f:

                # Read just the lines needed
                for i, line in enumerate(f):
                    if i == start_idx:
                        reference_pose = np.fromstring(line,
                                                       dtype=float,
                                                       sep=' ')
                        reference_pose = reference_pose.reshape(3, 4)
                        reference_pose = np.vstack((reference_pose,
                                                    [0, 0, 0, 1]))
                    elif i in pose_indices:
                        
                        T_w_cam0 = np.fromstring(line, dtype=float, sep=' ')
                        T_w_cam0 = T_w_cam0.reshape(3, 4)
                        T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
                        raw_poses.append(T_w_cam0)

        except FileNotFoundError:
            print('Ground truth poses are not available for sequence ' +
                  seq_no + '.')

        # Rectify poses so that they're relative to the first pose
        # associated with the first image frame of the pair that made
        # the first flow image in the subsequence.
        # Also converts orientations to euler
        # angles, and returns a numpy array
        y = process_poses(reference_pose, raw_poses)

        # pad both arrays if necessary
        # (should only happen on final subsequence
        # of sequence)
        if len(frame_nos) < self.window_size:
            num_to_pad = self.window_size - len(frame_nos)
            num_features = np.prod(x[0].shape)

            for i in range(num_to_pad):
                x = np.vstack(x, np.zeros(num_features))

            for i in range(num_to_pad):
                y = np.vstack(y, np.zeros(6))

        # Return the data and labels for this sample
        return (x, y)

    def get_batch(self):
        """Get a batch.

        Returns:
            (x, y):
                x: A (batch_size, window_size, HxWx3) np array of subsequences.
                y: A (batch_size, window_size, HxWx3) np array of ground truth
                   pose vectors.

        NOTE: See __init__ docstring note about batch_size.
        """
        if self.is_complete():
            return None

        x = []
        y = []
        for sample in range(self.batch_size):

            # get and remove first element
            window_idx = self.partitions.pop()
            seq_no, window_start_idx, window_end_idx = window_idx

            # Load the sample
            sample_x, sample_y = self.get_sample(seq_no,
                                                 window_start_idx,
                                                 window_end_idx)

            # Add sample data and truth pose to batch
            x.append(sample_x)
            y.append(sample_y)

        # Convert to numpy arrays
        x = np.array(x)
        y = np.array(y)

        return (x, y)
