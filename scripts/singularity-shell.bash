set -e
set -u

. scripts/variables.bash

singularity shell --nv "$IMAGE".sif
