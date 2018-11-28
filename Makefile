help:
	@cat Makefile

DATA?="/home/noah/kitti_data/dataset/"
GPU?=0
KERAS-DOCKER-FILE=docker/keras_dockerfile
CAFFE-DOCKER-FILE=docker/caffe_dockerfile
DOCKER=GPU=$(GPU) nvidia-docker
BUILD-CONTEXT="docker/"
KERAS-BACKEND=tensorflow
PYTHON-VERSION?=3.6
CUDA-VERSION?=9.0
CUDNN-VERSION?=7
TEST=tests/
SRC=$(shell pwd)
CAFFE-TAG="caffe:gpu"
KERAS-TAG="keras"

build-caffe:
	docker build -t $(CAFFE-TAG) --build-arg cuda_version=$(CUDA-VERSION) --build-arg cudnn_version=$(CUDNN-VERSION) -f $(CAFFE-DOCKER-FILE) $(BUILD-CONTEXT)

build-keras:
	docker build -t $(KERAS-TAG) --build-arg python_version=$(PYTHON-VERSION) --build-arg cuda_version=$(CUDA-VERSION) --build-arg cudnn_version=$(CUDNN-VERSION) -f $(KERAS-DOCKER-FILE) $(BUILD-CONTEXT)

caffe: build-caffe
	$(DOCKER) run -it -v $(SRC):/workspace -v $(DATA):/workspace/data/dataset $(CAFFE-TAG) bash

keras: build-keras
	$(DOCKER) run -it -v $(SRC):/workspace -v $(DATA):/workspace/data/dataset --env KERAS_BACKEND=$(KERAS-BACKEND) $(KERAS-TAG) bash

