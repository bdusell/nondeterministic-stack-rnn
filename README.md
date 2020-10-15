# Learning Context-Free Languages with Nondeterministic Stack RNNs

This repository contains the code for the paper "Learning Context-Free
Languages with Nondeterministic Stack RNNs" (DuSell and Chiang, 2020)
\[[arXiv](https://arxiv.org/abs/2010.04674)\].
It contains a PyTorch implementation of the Nondeterministic Stack RNN
(NS-RNN) model proposed in the paper, as well as implementations of the
following stack RNN models:

* The superposition-based Stack LSTM from "Inferring Algorithmic Patterns with
  Stack-Augmented Recurrent Nets" (Joulin and Mikolov, 2015)
  \[[pdf](https://papers.nips.cc/paper/5857-inferring-algorithmic-patterns-with-stack-augmented-recurrent-nets.pdf)\]
* The stratification-based Stack LSTM from "Learning to Transduce with
  Unbounded Memory" (Grefenstette et al., 2015)
  \[[pdf](https://papers.nips.cc/paper/5648-learning-to-transduce-with-unbounded-memory.pdf)\]

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
setting up the dependencies on your own system. You can also build an
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

In order to run the Singularity container, you must build the `.sif`
(Singularity image) file from the Docker image. You can pull the public Docker
image and convert it to the file `nondeterministic-stack-rnn.sif` by running

    $ bash scripts/build-singularity-image.bash --pull

Note that this will take several minutes. Since Docker is probably not
available on your HPC cluster, you may want to run this on a private
workstation where Docker and Singularity are installed, then `scp` the `.sif`
file to the HPC cluster.

Once the image is built, you can open a shell in the Singularity container
using

    $ bash scripts/singularity-shell.bash

This will work on both GPU and CPU machines (it will output a harmless warning
if there is no GPU).

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

All commands to run experiments need to be prefixed with `poetry run` so that
they have access to the Python packages managed by Poetry.

## Running Experiments

The [`experiments`](experiments) directory contains scripts for running all of
the experiments and generating all of the plots presented in the paper. The
experiments are split up into the following steps:

* Train each model multiple times on each task, testing different
  hyperparameters. Record validation perplexity and save the trained models.
* Figure out which hyperparameter setting for each model worked best on the
  validation set.
* Generate test sets.
* For each model and task, evaluate the saved models from the best
  hyperparameter setting on the test set.
* Generate the figures for training and test performance presented in the
  paper.

The script [`submit-job.bash`](experiments/submit-job.bash) is a stub that is
called by all of the experiment scripts. By default it prints its arguments,
but you can edit it to submit experiments to your batch job scheduler of
choice. You may find this a convenient way to replicate the findings of our
paper on your own computing infrastructure. All scripts in `experiments` are
meant to be run outside of the container and must be run with the top-level
directory of this repository as the current working directory.
