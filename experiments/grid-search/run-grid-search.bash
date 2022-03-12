set -e
set -o pipefail

. experiments/variables.bash

logs=$(dirname "$BASH_SOURCE")

for task in "${TASKS[@]}"; do
  for model in "${MODELS[@]}"; do
    key=grid-$task-$model
    input_dir="$logs"/../train/"$task"/"$model"/logs
    output_file=$logs/$task/$model
    bash experiments/submit-job.bash $key $logs/outputs cpu \
      poetry run bash $logs/run-job.bash \
        "$input_dir" \
        "$output_file" \
        "$@"
  done
done
