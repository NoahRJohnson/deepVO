help:
	@cat Makefile

DATA?="/home/noah/kitti_data/dataset"
GPU?=0
DOCKER_FILE=docker/Dockerfile
DOCKER=GPU=$(GPU) nvidia-docker
BACKEND=tensorflow
PYTHON_VERSION?=3.6
CUDA_VERSION?=9.0
CUDNN_VERSION?=7
TEST=tests/
SRC=$(shell pwd)

build:
	docker build -t keras --build-arg python_version=$(PYTHON_VERSION) --build-arg cuda_version=$(CUDA_VERSION) --build-arg cudnn_version=$(CUDNN_VERSION) -f $(DOCKER_FILE) .

bash: build
	$(DOCKER) run -it -v $(SRC):/src/workspace -v $(DATA):/src/workspace/data/dataset --env KERAS_BACKEND=$(BACKEND) keras bash

ipython: build
	$(DOCKER) run -it -v $(SRC):/src/workspace -v $(DATA):/src/workspace/data/dataset --env KERAS_BACKEND=$(BACKEND) keras ipython

notebook: build
	$(DOCKER) run -it -v $(SRC):/src/workspace -v $(DATA):/src/workspace/data/dataset --net=host --env KERAS_BACKEND=$(BACKEND) keras

test: build
	$(DOCKER) run -it -v $(SRC):/src/workspace -v $(DATA):/src/workspace/data/dataset --env KERAS_BACKEND=$(BACKEND) keras py.test $(TEST)
