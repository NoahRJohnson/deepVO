import math
import numpy as np

# Calculates Rotation Matrix given euler angles.
def euler_angles_to_rotation_matrix(theta):#,rot_order):
    """Convert Euler angles to rotation matrix."""
    #print("Theta = {}".format(theta))
    #print("Cos(Theta) = {}".format(math.cos(theta[0])))
    r_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]])

    r_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]])

    r_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]])

    #x, y, z = rot_orders[rot_order]
    r = np.dot(r_z, np.dot(r_y, r_x))

    return r

def subseq_preds_to_full_pred(predictions, outfile_name):
    """Convert a batch of subseq predictions to one long path.

    Args:
        predictions: (batch_size, n_frames, 6) np array

    Returns: None
    """

    with open(outfile_name, "w+") as oFile:

        last_transformation_matrix = np.eye(4)
        for subseq in predictions:
            for pose in subseq:

                # Break pose vector into position and orientation vectors
                relative_position = pose[:3]
                relative_orientation = pose[3:]

                #print("Position = {}".format(relative_position))
                #print("Orientation = {}".format(relative_orientation))

                # Transform Roll-Pitch-Yaw vector to 3x3 rotation matrix
                relative_rotation_matrix = euler_angles_to_rotation_matrix(relative_orientation)

                # Put estimates in 4x4 transformation matrix, same
                # format that ground truth uses
                relative_transformation_matrix = np.hstack((relative_rotation_matrix,
                                                            relative_position.reshape((3,1))))
                relative_transformation_matrix = np.vstack((relative_transformation_matrix,
                                                            [0, 0, 0, 1]))

                # Multiply by the last transformation matrix of the previous subsequence,
                # in order to put the transformation matrix back into the original
                # reference frame
                absolute_transformation_matrix = np.dot(last_transformation_matrix,
                                                        relative_transformation_matrix)

                """
                absolute_position = last_position + relative_position

                # Get rotation matrix in original reference frame
                absolute_rotation = np.dot(last_rotation, relative_rotation)

                # Put estimates in 3x4 transformation matrix, same
                # format that ground truth uses
                transformation_matrix = np.hstack((absolute_rotation, absolute_position.reshape((3,1))))
                """

                # Write out the matrix in row-major order as a line
                # The last row of (0,0,0,1) is excluded (KITTI standard)
                out_line = ' '.join(map(str, absolute_transformation_matrix[:3,:].flatten().tolist()))
                oFile.write(out_line + '\n')

                """
                list_matrix = list(absolute_rot)
                components = list_matrix[0] + [pose[3]] + \
                    list_matrix[1] + [pose[4]] + \
                    list_matrix[2] + [pose[5]]
                oFile.write(" ".join(components))
                """

            # Keep track of last value
            # don't start at the origin start where the last
            # subsequence left off
            last_transformation_matrix = absolute_transformation_matrix
            #last_rotation = absolute_rotation
            #last_position = absolute_position
