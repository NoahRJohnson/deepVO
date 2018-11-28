help:
	@cat Makefile

DATA?="/media/gpudata_backup/kitti/dataset/"
GPU?=0
KERAS_DOCKER_FILE=docker/Dockerfile
CAFFE_DOCKER_FILE=flownet2/docker/standalone/gpu/Dockerfile
DOCKER=GPU=$(GPU) nvidia-docker
BACKEND=tensorflow
PYTHON_VERSION?=3.6
CUDA_VERSION?=9.0
CUDNN_VERSION?=7
TEST=tests/
SRC=$(shell pwd)
CAFFE-TAG="caffe:gpu"
KERAS-TAG="keras"

build-caffe:
	docker build -t $(CAFFE-TAG) -f $(CAFFE_DOCKER_FILE) docker/

build-keras:
	docker build -t $(KERAS-TAG) --build-arg python_version=$(PYTHON_VERSION) --build-arg cuda_version=$(CUDA_VERSION) --build-arg cudnn_version=$(CUDNN_VERSION) -f $(KERAS_DOCKER_FILE) docker/

caffe: build-caffe
	$(DOCKER) run -it -v $(SRC):/workspace -v $(DATA):/workspace/data/dataset $(CAFFE-TAG) bash

keras: build-keras
	$(DOCKER) run -it -v $(SRC):/workspace -v $(DATA):/workspace/data/dataset --env KERAS_BACKEND=$(BACKEND) $(KERAS-TAG) bash

