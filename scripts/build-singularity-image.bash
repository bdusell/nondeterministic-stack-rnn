set -e
set -u

. scripts/variables.bash

bash scripts/get-docker-image.bash "$@"
singularity build "$IMAGE".sif docker-daemon://"$DEV_IMAGE":latest
