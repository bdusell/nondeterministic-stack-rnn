# Build the Docker image used to run the code in this repository.

set -e
set -o pipefail

. scripts/variables.bash

DOCKER_BUILDKIT=1 docker build "$@" -t "$DEV_IMAGE":latest -f Dockerfile-dev .
