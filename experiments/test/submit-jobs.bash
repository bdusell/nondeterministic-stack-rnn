set -e
set -o pipefail

. experiments/variables.bash

logs=$(dirname "$BASH_SOURCE")

for task in "${TASKS[@]}"; do
  for model in "${MODELS[@]}"; do
    key=test-$task-$model
    if [[ $model = ns* ]]; then
      mode=gpu
    else
      mode=cpu
    fi
    bash experiments/submit-job.bash $key $logs/outputs $mode \
      poetry run bash $logs/run-job.bash $logs $task $model
  done
done
