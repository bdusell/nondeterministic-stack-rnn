# Open a shell in the Singularity container, assuming the .sif file has already
# been created.

set -e
set -o pipefail

. scripts/variables.bash

singularity shell --nv "$IMAGE".sif
