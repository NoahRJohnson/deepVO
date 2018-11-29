# DeepVO
Implementation of a Recurrent CNN for monocular visual odometry from video.

Following [S. Wang, R. Clark, H. Wen, and N. Trigoni](https://www.cs.ox.ac.uk/files/9026/DeepVO.pdf).

## Usage

If you don't already have the KITTI odometry benchmark data, you'll need to download it. A download script is available for your convenience.

```python
python3 download_kitti_data.py /drive/with/space/kitti
```

If the download doesn't work anymore, go to the [source](http://www.cvlibs.net/datasets/kitti/eval_odometry.php). Manually download the color, calibration, and ground truth files. You will have to enter an email address, and will get a download link. Download the zipped file, and extract its contents. You should now have a 'dataset' folder, with 'poses' and 'sequences' folders within.

Next, convert the KITTI image frames into optical flow images using FlowNet. Make sure you've pulled submodules first, e.g.:

```bash
git submodule update --init
```

Use the download script in flownet2/models/ to download pre-trained Caffe networks. This may take a while. Once that's done, build and run a Caffe image using the provided Makefile, which mounts your kitti data into the 'data/dataset' folder:

```bash
make caffe
```

Just modify the Makefile DATA variable to point to wherever your 'dataset' folder is.


```bash
python flownet2/scripts/run-flownet.py \
        flownet2/models/FlowNet2/FlowNet2_weights.caffemodel.h5 \
        flownet2/models/FlowNet2/FlowNet2_deploy.prototxt.template \
        data/dataset/sequences/00/image_2/000000.png \
        data/dataset/sequences/00/image_2/000001.png \
        data/dataset/flows/00/1.flo

python flownet2/scripts/run-flownet-many.py \
        flownet2/models/FlowNet2/FlowNet2_weights.caffemodel.h5 \
        flownet2/models/FlowNet2/FlowNet2_deploy.prototxt.template \
        data/flow_links.txt

```


Now you should have a flow/ folder within your dataset folder, containing flow images for all of the KITTI sequences. This data, along with the poses/ ground-truth, will be used for training and testing our LSTM network.

If you want to visualize these .flo images, use the flow-code library.

As before, build and run a Keras docker container using the Makefile:

```bash
make keras
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
