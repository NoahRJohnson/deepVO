# DeepVO
Implementation of a Recurrent CNN for monocular visual odometry from video.

Following [S. Wang, R. Clark, H. Wen, and N. Trigoni](https://www.cs.ox.ac.uk/files/9026/DeepVO.pdf).

## Usage

If you don't already have the KITTI odometry benchmark data, you'll need to download it. A download script is available for your convenience.

```python
python download_kitti_data.py /drive/with/space/kitti
```

If the download doesn't work anymore, go to the [source](http://www.cvlibs.net/datasets/kitti/eval_odometry.php). Manually download the color, calibration, and ground truth files. You will have to enter an email address, and will get a download link. Download the zipped file, and extract its contents. You should now have a 'dataset' folder, with 'poses' and 'sequences' folders within.

Next, convert the KITTI image frames into optical flow images using FlowNet. Make sure you've pulled submodules first, e.g.:

```bash
git submodule update --init
```

Use the download script in flownet2/models/ to download pre-trained Caffe networks. This may take a while. Once that's done, build and run a Caffe image using the provided Makefile, which automatically mounts this repository and mounts your kitti data into the 'data' folder. Just modify the Makefile DATA variable to point to wherever your 'dataset' folder is.

```bash
make caffe
```

From within the container, you should see the project directory. Run the following to generate .flo images for all of your KITTI sequences. WARNING: This will take a very long time, you might want to remove any KITTI sequences that you don't plan to use (i.e. the training code only uses sequences 00-10). Also consider running this from within a tmux session.

```bash
./generate_flows.sh
```

Once that finishes you will have a flows/ folder within your dataset folder, containing flow images for all of the KITTI sequences. This data, along with the poses/ ground-truth, will be used for training and testing our LSTM network.

If you want to visualize these .flo images, use the flow-code library.

```bash
cd flow-code/imageLib/
make
cd ..
make

```

Now we're ready to train our network. As before, use the makefile to build and run a Keras docker container. Make sure you've exited the Caffe container first.

```bash
make keras
```

From within the container, train the model using:

```bash
python train_model.py
```

To visualize the loss function while training, use TensorBoard. Run the following in a separate terminal from the project root:

```bash
./start_tensorboard.sh
```

and open localhost:6006 in a browser. If you're on a server you'll have to forward the port.

Model weights are saved in 'checkpoints'.

To test the trained model.

```bash
python src/train_model.py --mode test
```

which will output csvs to "test_results". To view the results graphically, install [evo](https://github.com/MichaelGrupp/evo). The evo package has been installed for your convenience in the keras docker image.


Once you have evo installed, run

```bash
test_results/generate_plots.sh
```

to generate pdfs of the results.
