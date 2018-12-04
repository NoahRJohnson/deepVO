"""Convert batch of predictions to one long prediction."""

import math
import numpy as np

def subseq_preds_to_full_pred(predictions, outfile_name):
    """Convert a batch of subseq predictions to one long path.

    Args:
        predictions: (batch_size, n_frames, 6) np array

    Returns: None
    """
    # Calculates Rotation Matrix given euler angles.
    def euler_angles_to_rotation_matrix(theta):
        """Convert Euler angles to rotation matrix."""
        print(theta)
        print(math.cos(theta[0]))
        r_x = np.array([[1, 0, 0],
                        [0, math.cos(theta[0]), -math.sin(theta[0])],
                        [0, math.sin(theta[0]), math.cos(theta[0])]
                        ])

        r_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                        [0, 1, 0],
                        [-math.sin(theta[1]), 0, math.cos(theta[1])]
                        ])

        r_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                        [math.sin(theta[2]), math.cos(theta[2]), 0],
                        [0, 0, 1]
                        ])

        r = np.dot(r_z, np.dot(r_y, r_x))

        return r

    with open(outfile_name, "w+") as oFile:

        last_rot = np.eye(3)
        for subseq in predictions:
            for pose in subseq:
                position = pose[:3]
                orientation = pose[3:]

                print("Position = {}".format(position))
                print("Orientation = {}".format(orientation))

                current_rot = euler_angles_to_rotation_matrix(orientation)
                absolute_rot = np.dot(last_rot, current_rot)
                list_matrix = list(absolute_rot)
                components = list_matrix[0] + [pose[3]] + \
                    list_matrix[1] + [pose[4]] + \
                    list_matrix[2] + [pose[5]]
                oFile.write(" ".join(components))
            last_rot = current_rot

