Go to the [KITTI odometry benchmark website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) and download the odometry data set (color), odometry data set (calibration files), and odometry ground truth poses. Extract the dataset folder to a drive with ~70 GB of space, and then change the DATA path in the project root's Makefile accordingly.

The Makefile mounts the DATA folder to this data/ folder, so the data will be accessible from within your docker containers.

This data is not stored in version control, as it is too large.
