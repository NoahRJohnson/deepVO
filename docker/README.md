# Using Caffe/Keras via Docker

This directory contains `Dockerfiles` to make it easy to get up and running with Keras and Caffe via [Docker](http://www.docker.com/).

## Installing Docker

General installation instructions are
[on the Docker site](https://docs.docker.com/installation/), but we give some
quick links here:

* [OSX](https://docs.docker.com/installation/mac/): [docker toolbox](https://www.docker.com/toolbox)
* [ubuntu](https://docs.docker.com/installation/ubuntulinux/)

## Running the container

We are using `Makefile` to simplify docker commands within make commands. This Makefile is in the project root.

Build and run the Keras container

    $ make keras

Build and run the Caffe container

    $ make caffe

For GPU support install NVIDIA drivers (ideally latest) and
[nvidia-docker](https://github.com/NVIDIA/nvidia-docker). Run using

    $ make keras GPU=0 # or [caffe]

Mount a volume for external data sets

    $ make DATA=~/mydata

Prints all make tasks

    $ make help

