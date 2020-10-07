set -e
set -o pipefail

. experiments/variables.bash

logs=$(dirname "$BASH_SOURCE")

for task in "${TASKS[@]}"; do
  for model in "${MODELS[@]}"; do
    key=grid-$task-$model
    output_file=$logs/$task/$model
    bash experiments/submit-job.bash $key $logs/outputs cpu \
      poetry run bash $logs/run-job.bash \
        "$output_file" \
        --trials "${#TRIALS[@]}" \
        "$@" \
        "$logs"/../train/$task/$model/logs
  done
done
