# Learning Hierarchical Structures with Differentiable Nondeterministic Stacks

This repository contains the code for the paper
["Learning Hierarchical Structures with Differentiable Nondeterministic Stacks" (DuSell and Chiang, 2022)](https://openreview.net/forum?id=5LXw_QplBiF).
It includes all of the code necessary to reproduce the experiments and figures
used in the paper, as well as a Docker image definition that can be used to
replicate the software environment it was developed in.

If you are looking for the code for our earlier paper
["Learning Context-free Languages with Nondeterministic Stack RNNs"](https://aclanthology.org/2020.conll-1.41/),
please see
[this release](https://github.com/bdusell/nondeterministic-stack-rnn/tree/conll2020).

This repository includes PyTorch implementations of the following models:

* The
  [Renormalizing Nondeterministic Stack RNN (RNS-RNN)](src/nsrnn/models/nondeterministic_stack.py)
  proposed in
  ["Learning Hierarchical Structures with Differentiable Nondeterministic Stacks"](https://openreview.net/forum?id=5LXw_QplBiF),
  which is based on the Nondeterministic Stack RNN (NS-RNN) proposed in our
  earlier paper
  ["Learning Context-free Languages with Nondeterministic Stack RNNs"](https://aclanthology.org/2020.conll-1.41/).
* [Memory-limited versions](src/nsrnn/models/limited_nondeterministic_stack.py)
  of the RNS-RNN and NS-RNN that can be run
  incrementally on arbitrarily long sequences, which is useful for natural
  language modeling.
* The
  [superposition-based stack RNN](src/nsrnn/models/joulin_mikolov.py)
  from
  ["Inferring Algorithmic Patterns with Stack-Augmented Recurrent Nets" (Joulin and Mikolov, 2015)](https://proceedings.neurips.cc/paper/2015/file/26657d5ff9020d2abefe558796b99584-Paper.pdf).
* The
  [stratification-based stack RNN](src/nsrnn/models/grefenstette.py)
  from
  ["Learning to Transduce with Unbounded Memory"](https://proceedings.neurips.cc/paper/2015/file/b9d487a30398d42ecff55c228ed5652b-Paper.pdf).

## Directory Structure

* `data/`: Contains data used for experiments, including Mikolov's PTB language
* `experiments/`: Contains scripts for reproducing all of the experiments and
  figures presented in the paper.
  * `train/`: Scripts for training models on the CFL tasks.
  * `grid-search/`: Scripts for determining the best models after all training
    experiments are finished on the CFL tasks.
  * `test-data/`: Scripts for generating test sets for the CFL tasks.
  * `test/`: Scripts for running trained models on the CFL test sets.
  * `plot-train/`: Scripts for plotting training curves for the CFL tasks.
  * `plot-test/`: Scripts for plotting test set performance for the CFL tasks.
  * `train-ptb/`: Scripts for training language models on the Penn Treebank
    corpus.
* `scripts/`: Contains helper scripts for setting up the software environment,
  building container images, running containers, installing Python packages,
  preprocessing data, etc. Instructions for using these scripts are below.
* `src/`: Contains source code for all models, training routines, plotting
  scripts, etc.
* `tests/`: Contains unit tests for the code under `src/`.

## Installation and Setup

In order to foster reproducibility, the code for this paper was developed and
run inside of a [Docker](https://www.docker.com/) container defined in the file
[`Dockerfile-dev`](Dockerfile-dev). To run this code, you can build or pull the
Docker image yourself and run it using Docker. Or, if you don't feel like
installing Docker, you can simply use `Dockerfile-dev` as a reference for
setting up the software environment on your own system. You can also build
an equivalent [Singularity](https://sylabs.io/docs/#singularity) image which
can be used on an HPC cluster, where it is likely that Docker is not available
but Singularity is.

In any case, it is highly recommended to run experiments that use the NS-RNN or
RNS-RNN models on a machine with access to an NVIDIA GPU so that they finish
within a reasonable amount of time. On the other hand, the experiments for the
baseline models (LSTM, Joulin-Mikolov Stack LSTM, and Grefenstette Stack LSTM)
finish more quickly on CPU rather than GPU and should be run in CPU mode.

### Using Docker

In order to use the Docker image, you must first
[install Docker](https://www.docker.com/get-started).
If you intend to run any experiments on a GPU, you must also ensure that your
NVIDIA driver is set up properly and install the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

In order to automatically pull the public Docker image, start the container,
and open up a bash shell inside of it, run

    $ bash scripts/docker-shell.bash --pull

If you prefer to build the image from scratch yourself, you can run

    $ bash scripts/docker-shell.bash --build

After you have pulled or built the image once, there is no need to do so again,
so afterwards you can simply run

    $ bash scripts/docker-shell.bash

By default, this script starts the container in GPU mode, which will fail if
you are not running on a machine with a GPU. If you only want to run things in
CPU mode, you can run

    $ bash scripts/docker-shell.bash --cpu

You can combine this with the `--pull` or `--build` options.

### Using Singularity

If you use a shared HPC cluster at your institution, it might not support
Docker, but there's a chance it does support Singularity, which is an
alternative container runtime that is more suitable for shared computing
environments.

In order to run the Singularity container, you must obtain the Docker image and
then convert it to a `.sif` (Singularity image) file. Assuming you have already
pulled or built the Docker image according to the instructions above, you can
use the following to create the `.sif` file:

    $ bash scripts/build-singularity-image.bash

This will create the file `renormalizing-ns-rnn.sif`.

You can open a shell in the Singularity container using

    $ bash scripts/singularity-shell.bash

This will work on both GPU and CPU machines, although it will output a warning
if there is no GPU.

You can find a more general tutorial on Singularity
[here](https://github.com/bdusell/singularity-tutorial).

### Additional Setup

Whatever method you use to run the code (whether in a Docker container,
Singularity container, or no container), there are some additional setup and
preprocessing steps you need to run. The following script will take care of
these for you (note that if you are using a container, you must run this
*inside the container shell*):

    $ bash scripts/setup.bash

More specifically, this script:

* Installs the Python packages required by our code, which will be stored in
  the local directory rather than system-wide. We use the package manager
  [Poetry](https://python-poetry.org/) to manage Python packages.
* Downloads and preprocesses the Penn Treebank language modeling dataset.

## Running Experiments

The [`experiments`](experiments) directory contains scripts for reproducing
all of the experiments and plots presented in the paper. These scripts are
intended to be used to submit jobs to a computing cluster. You will need to
edit the file [`experiments/submit-job.bash`](experiments/submit-job.bash)
to tailor it to your specific computing cluster.

The script `src/get_ptb_results.bash` is used to report results for the PTB
experiments. Given a directory containing multiple pre-trained random restarts,
it will select the model with the best validation perplexity, print parameter
counts, compute test perplexity, and compute the SG score.
