# Build a .sif file that can be used to run the code in a Singularity
# container. This simply builds the Docker image and converts it to .sif
# format. Note that this may take several minutes.
#
# Singularity is an alternative to Docker that is more suitable for
# shared HPC clusters.
# See https://sylabs.io/guides/3.9/user-guide/
#
# You will likely only need to use this script if your institution's HPC
# cluster does not support Docker but has Singularity pre-installed. In that
# case, run this script on a machine where you have root access (e.g. your
# personal computer or workstation), then upload the .sif file it generates
# to your account on your HPC cluster and use it to run Singularity containers
# there.

set -e
set -o pipefail

. scripts/variables.bash

bash scripts/get-docker-image.bash "$@"
singularity build "$IMAGE".sif docker-daemon://"$DEV_IMAGE":latest
