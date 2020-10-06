set -e
set -o pipefail

. scripts/variables.bash

singularity pull library://brian/default/"$IMAGE"
