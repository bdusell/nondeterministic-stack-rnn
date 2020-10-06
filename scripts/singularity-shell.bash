set -e
set -o pipefail

. scripts/variables.bash

singularity shell --nv "$IMAGE".sif
