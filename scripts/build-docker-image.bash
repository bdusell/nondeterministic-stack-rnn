# Build a self-contained Docker image that includes all the code under src/
# and the software environment necessary to run it. This is required when
# building Docker images to be used with SyntaxGym to compute SG scores.

set -e
set -o pipefail

. scripts/variables.bash

bash scripts/build-docker-dev-image.bash
DOCKER_BUILDKIT=1 docker build "$@" -t "$IMAGE":latest -f Dockerfile .
