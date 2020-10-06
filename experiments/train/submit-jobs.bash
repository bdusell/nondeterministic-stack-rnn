set -e
set -o pipefail

. experiments/variables.bash

for task in "${TASKS[@]}"; do
  for model in "${MODELS[@]}"; do
    bash experiments/train/$task/$model/submit-jobs.bash
  done
done
