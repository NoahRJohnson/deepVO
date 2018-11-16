# DeepVO
Implementation of a Recurrent CNN for monocular visual odometry from video.

Following [S. Wang, R. Clark, H. Wen, and N. Trigoni](https://www.cs.ox.ac.uk/files/9026/DeepVO.pdf).

## Usage

If you don't already have the KITTI odometry benchmark data, you'll need to download it. A download script is available for your convenience.

```python
python3 download_kitti_data.py
```

If the download doesn't work anymore, go to the [source](http://www.cvlibs.net/datasets/kitti/eval_odometry.php). Manually download the color, calibration, and ground truth files. You will have to enter an email address, and will get a download link. Download the zipped file, and extract its contents. You should now have a 'dataset' folder, with 'poses' and 'sequences' folders within.

Now to avoid setting up CUDA yourself, spin up a docker container for Keras. Use the Makefile, which automatically mounts this repository and mounts your kitti data in the 'data' folder. Just modify the Makefile DATA variable to point to wherever your 'dataset' folder is.

```bash
sudo make bash
```

Now you can train the model using:

```bash
python3 train_model.py
```

To visualize the loss function while training, use TensorBoard. Run the following in a separate terminal from the project root:

```bash
./start_tensorboard.sh
```

and open localhost:6006 in a browser.

Model weights are saved in 'checkpoints'.

To validate and test the trained model

```bash

```

view test results in ROS Gazebo

```bash

```

generate result statistics

```bash

```
