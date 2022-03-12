set -e
set -o pipefail

. experiments/variables.bash

logs=$(dirname "$BASH_SOURCE")

mkdir -p $logs/{png,tex}
for task in "${TASKS[@]}"; do
  key=plot-test-$task
  bash experiments/submit-job.bash $key $logs/outputs cpu \
    poetry run bash $logs/run-job.bash \
      $logs \
      $task \
      $(
        for model in "${MODELS[@]}"; do
          echo $logs/../test/logs/$task/$model
        done
      )
done
