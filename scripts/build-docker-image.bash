set -e
set -o pipefail

. scripts/variables.bash

DOCKER_BUILDKIT=1 docker build "$@" -t "$IMAGE":latest -f Dockerfile .
