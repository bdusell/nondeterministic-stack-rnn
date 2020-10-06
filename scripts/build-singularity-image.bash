set -e
set -o pipefail

. scripts/variables.bash

bash scripts/get-docker-image.bash "$@"
singularity build "$IMAGE".sif docker-daemon://"$IMAGE":latest
