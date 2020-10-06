set -e
set -o pipefail

. experiments/variables.bash

for task in "${TASKS[@]}"; do
  bash experiments/test-data/$task/submit-jobs.bash
done
