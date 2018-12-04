import math
import numpy as np


def subseq_preds_to_full_pred(predictions, outfile_name):
    """Convert a batch of subseq predictions to one long path.

    Args:
        predictions: (batch_size, n_frames, 6) np array

    Returns: None
    """
    # Calculates Rotation Matrix given euler angles.
    def euler_angles_to_rotation_matrix(theta, rot_order):
        """Convert Euler angles to rotation matrix."""
        print("Theta = {}".format(theta))
        print("Cos(Theta) = {}".format(math.cos(theta[0])))
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
        
        x, y, z = rot_orders[rot_order]
        r = np.dot(r_z, np.dot(r_y, r_x))

        return r
    

    with open(outfile_name, "w+") as oFile:

        last_rotation = np.eye(3)
        last_position = np.zeros(3)
        for subseq in predictions:
            for pose in subseq:

                relative_position = pose[:3]
                relative_orientation = pose[3:]

                print("Position = {}".format(relative_position))
                print("Orientation = {}".format(relative_orientation))

                # Get position in original reference frame
                absolute_position = last_position + relative_position

                # Transform RPY vector to rotation matrix
                relative_rotation = euler_angles_to_rotation_matrix(relative_orientation)
                # Get rotation matrix in original reference frame
                absolute_rotation = np.dot(last_rotation, relative_rotation)

                # Put estimate in 3x4 transformation matrix, same
                # as ground truth
                transformation_matrix = np.hstack((absolute_rotation, absolute_position.reshape((3,1))))

                # Write out this matrix in row-major order as a line
                out_line = ' '.join(map(str, transformation_matrix.flatten().tolist()))
                oFile.write(out_line + '\n')

                """
                list_matrix = list(absolute_rot)
                components = list_matrix[0] + [pose[3]] + \
                    list_matrix[1] + [pose[4]] + \
                    list_matrix[2] + [pose[5]]
                oFile.write(" ".join(components))
                """

            # Keep track of last values
            # don't start at the origin start where the last
            # subsequence left off
            last_rotation = absolute_rotation
            last_position = absolute_position
