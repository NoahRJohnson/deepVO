# DeepVO
Implementation of a Recurrent CNN for monocular visual odometry from video.

Following [S. Wang, R. Clark, H. Wen, and N. Trigoni](https://www.cs.ox.ac.uk/files/9026/DeepVO.pdf).

## Usage
Download and preprocess KITTI data

```bash
python3 download_kitti_data.py
```

spin up container, which automatically mounts this repository

```bash
sudo make bash
```

train model

```bash
python3 train_model.py
```

visualize training with tensorboard. In a separate terminal, from project root, run:

```bash
./start_tensorboard.sh
```

validate and test model

```bash

```

view test results in ROS Gazebo

```bash

```

generate result statistics

```bash

```
