# The Surprising Computational Power of Nondeterministic Stack RNNs

This repository contains the code for the paper
["The Surprising Computational Power of Nondeterministic Stack RNNs"](https://openreview.net/forum?id=o58JtGDs6y)
(DuSell and Chiang, 2023).
It includes all of the code necessary to reproduce the experiments and figures
used in the paper, as well as a Docker image definition that can be used to
replicate the software environment it was developed in.

If you are looking for the code for our earlier paper
["Learning Hierarchical Structures with Differentiable Nondeterministic Stacks"](https://openreview.net/forum?id=5LXw_QplBiF)
(DuSell and Chiang, 2022), please see
[this release](https://github.com/bdusell/nondeterministic-stack-rnn/tree/iclr2022).

If you are looking for the code for our earlier paper
["Learning Context-free Languages with Nondeterministic Stack RNNs"](https://aclanthology.org/2020.conll-1.41/),
(DuSell and Chiang, 2020), please see
[this release](https://github.com/bdusell/nondeterministic-stack-rnn/tree/conll2020).

This repository includes PyTorch implementations of the following models:

* The
  [Renormalizing Nondeterministic Stack RNN (RNS-RNN)](src/stack_rnn_models/nondeterministic_stack.py),
  including the minor changes described in this paper (bottom symbol fix and
  asymptotic speedup).
* The 
  [Vector RNS-RNN (VRNS-RNN)](src/stack_rnn_models/vector_nondeterministic_stack.py)
  introduced in this paper.
* Memory-limited versions of the
  [RNS-RNN](src/stack_rnn_models/limited_nondeterministic_stack.py)
  and
  [VRNS-RNN](src/stack_rnn_models/limited_vector_nondeterministic_stack.py)
  that run in linear time, which is useful for natural language modeling.
* The
  [superposition stack RNN](src/stack_rnn_models/joulin_mikolov.py)
  from
  ["Inferring Algorithmic Patterns with Stack-Augmented Recurrent Nets"](https://proceedings.neurips.cc/paper/2015/file/26657d5ff9020d2abefe558796b99584-Paper.pdf) (Joulin and Mikolov, 2015).
* The
  [stratification stack RNN](src/stack_rnn_models/grefenstette.py)
  from
  ["Learning to Transduce with Unbounded Memory"](https://proceedings.neurips.cc/paper/2015/file/b9d487a30398d42ecff55c228ed5652b-Paper.pdf) (Grefenstette et al., 2015).

## Directory Structure

* `data/`: Contains datasets used for experiments, namely the PTB language
  modeling dataset.
* `experiments/`: Contains scripts for reproducing all of the experiments and
  figures presented in the paper.
  * `capacity/`: Scripts for the capacity experiments in Section 5.
  * `non-cfls/`: Scripts for the non-CFL experiments in Section 4.
  * `ptb/`: Scripts for the PTB language modeling experiments in Section 6.
* `scripts/`: Contains helper scripts for setting up the software environment,
  building container images, running containers, installing Python packages,
  preprocessing data, etc. Instructions for using these scripts are below.
* `src/`: Contains source code for all models, training routines, plotting
  scripts, etc.
* `tests/`: Contains unit tests for the code under `src/`.

## Installation and Setup

In order to foster reproducibility, the code for this paper was developed and
run inside of a [Docker](https://www.docker.com/) container defined in the file
[`Dockerfile-dev`](Dockerfile-dev). To run this code, you can build the
Docker image yourself and run it using Docker. Or, if you don't feel like
installing Docker, you can simply use `Dockerfile-dev` as a reference for
setting up the software environment on your own system. You can also build
an equivalent [Singularity](https://sylabs.io/docs/#singularity) image which
can be used on an HPC cluster, where it is likely that Docker is not available
but Singularity is.

In any case, it is highly recommended to run most experiments on a machine with
access to an NVIDIA GPU so that they finish within a reasonable amount of time.
The exception to this is the experiments for the baseline models (LSTM,
superposition stack LSTM, and stratification stack LSTM) on the CFL language
modeling tasks, as they finish more quickly on CPU rather than GPU and should
be run in CPU mode.

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

After you have built the image once, there is no need to do so again, so
afterwards you can simply run

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

In order to run the code in a Singularity container, you must first obtain the
Docker image and then convert it to a `.sif` (Singularity image) file on a
machine where you have root access (e.g. your personal computer or
workstation). This requires installing both Docker and
[Singularity](https://docs.sylabs.io/guides/latest/user-guide/quick_start.html)
on that machine. Assuming you have already built the Docker image according to
the instructions above, you can use the following to create the `.sif` file:

    $ bash scripts/build-singularity-image.bash

This will create the file `nondeterministic-stack-rnn-2023.sif`. It is normal
for this to take several minutes. Afterwards, you can upload the `.sif` file to
your HPC cluster and use it there.

You can open a shell in the Singularity container using

    $ bash scripts/singularity-shell.bash

This will work on machines that do and do not have an NVIDIA GPU, although it
will output a warning if there is no GPU.

You can find a more general tutorial on Singularity
[here](https://github.com/bdusell/singularity-tutorial).

### Additional Setup

Whatever method you use to run the code (whether in a Docker container,
Singularity container, or no container), there are some additional setup and
preprocessing steps you need to run. The following script will take care of
these for you (if you are using a container, you must run this *inside the
container shell*):

    $ bash scripts/setup.bash

More specifically, this script:

* Installs the Python packages required by our code, which will be stored in
  the local directory rather than system-wide. We use the package manager
  [Poetry](https://python-poetry.org/) to manage Python packages.
* Downloads and preprocesses the Penn Treebank language modeling dataset.

## Running Code

All files under `src/` should be run using `poetry` so they have access to the
Python packages provided by the Poetry package manager. This means you should
either prefix all of your commands with `poetry run` or run `poetry shell`
beforehand to enter a shell with Poetry's virtualenv enabled all the time. You
should run both Python and Bash scripts with Poetry, because the Bash scripts
might call out to Python scripts. All Bash scripts under `src/` should be run
with `src/` as the current working directory.

All scripts under `scripts/` should be run with the top-level directory as the
current working directory.

## Running Experiments

The [`experiments/`](experiments) directory contains scripts for reproducing
all of the experiments and plots presented in the paper. Some of these scripts
are intended to be used to submit jobs to a computing cluster. They should be
run outside of the container. You will need to edit the file
[`experiments/submit-job.bash`](experiments/submit-job.bash)
to tailor it to your specific computing cluster. Other scripts are for plotting
or printing tables and should be run inside the container.
