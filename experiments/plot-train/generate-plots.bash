set -e
set -o pipefail

. experiments/variables.bash

logs=$(dirname "$BASH_SOURCE")

for task in "${TASKS[@]}"; do
  key=plot-train-$task
  bash experiments/submit-job.bash $key $logs/outputs cpu \
    poetry run bash $logs/run-job.bash \
      $logs \
      $task \
      $(
        for model in "${MODELS[@]}"; do
          echo $logs/../grid-search/$task/$model
        done
      )
done
