# Learning Context-free Languages with Nondeterministic Stack RNNs

This repository contains the code for the paper "Learning Context-free
Languages with Nondeterministic Stack RNNs" (DuSell and Chiang, 2020).

## Directory Structure

* `scripts`: Contains helper scripts for building container images, running
  containers, installing software, etc.
* `src`: Contains the implementations of the NS-RNN, baseline models,
  experiments, plots, and so on.
* `experiments`: Contains scripts for running all of the experiments presented
  in the paper and generating figures.
  * `train`: Experiments for training models.
  * `grid-search`: Scripts for determining the best models after all training
    experiments are finished.
  * `test-data`: Scripts for generating test sets.
  * `test`: Scripts for running trained models on the test sets.
  * `plot-train`: Scripts for plotting training curves.
  * `plot-test`: Scripts for plotting test set performance.

## Installation and Setup

In order to improve reproducibility, the code for this paper was developed and
run inside of a [Docker](https://www.docker.com/) container defined in the file
[`Dockerfile`](Dockerfile). To run this code, you can build or pull the Docker
image for the container yourself or simply use `Dockerfile` as a reference for
setting up the dependencies on your own system. You can also build or pull an
equivalent [Singularity](https://sylabs.io/docs/#singularity) image which can
be used on an HPC cluster, where Singularity is more likely to be available
than Docker.

In any case, it is highly recommended to run the experiments that use the
NS-RNN model on a machine with access to an NVidia GPU so that they finish
within a reasonable amount of time. On the other hand, the experiments for the
baseline models (LSTM, Joulin-Mikolov Stack LSTM, and Grefenstette Stack LSTM)
finish more quickly on CPU rather than GPU and should be run in CPU mode.

### Using Docker

In order to use the Docker image, you must first
[install Docker](https://www.docker.com/get-started).
If you intend to run any experiments on a GPU, you must also ensure that your
NVidia driver is set up properly and install the
[NVIDIA Container Toolkit](https://www.docker.com/get-started).

In order to automatically pull the public Docker image, start the container,
and open up a bash shell inside of it, run

    $ bash scripts/docker-shell.bash --pull

If you prefer to build the image from scratch yourself, you can run

    $ bash scripts/docker-shell.bash --build

If you know you have already pulled or built the image, you can skip those
steps and just run

    $ bash scripts/docker-shell.bash

By default, this script starts the container in GPU mode, which will fail if
you are not running on a machine with a GPU. If you only want to run things in
CPU mode, you can run

    $ bash scripts/docker-shell.bash --cpu

You can combine this with the `--pull` or `--build` options.

### Using Singularity

If you have access to an HPC cluster (e.g. at a university), you will probably
want to use Singularity instead of Docker, since it is more suitable for shared
computing environments and is often available on HPC clusters instead of
Docker.

In order to run the Singularity container, you must build or download the
`.sif` (Singularity image) file. You can pull the public Singularity image
using

    $ bash scripts/pull-singularity-image.bash

This will download the image to the file `nondeterministic-stack-rnn.sif`.

If you like, you can also build the image yourself by first building (or
pulling) the Docker image and converting it to a Singularity image.

    $ bash scripts/build-singularity-image.bash --build

You can open a shell in the Singularity container using

    $ bash scripts/singularity-shell.bash

This will work on both GPU and CPU machines, although it will output a warning
if there is no GPU.

You can find a more general tutorial on Singularity
[here](https://github.com/bdusell/singularity-tutorial).

### Installing Python Dependencies

Whatever method you use to run the code (whether in a Docker container,
Singularity container, or neither), you must also install the Python
dependencies required by this code, which will be stored in the local
directory rather than system-wide. We use the package manager
[Poetry](https://python-poetry.org/) to manage Python packages.

In order to install the required Python packages, run this *inside of the
container*:

    $ bash scripts/setup.bash

## Running Experiments

The [`experiments`](experiments) directory contains scripts for running all of
the experiments and generating all of the plots presented in the paper.
