set -e
set -u

. "$(dirname "$BASH_SOURCE")"/include.bash

for task in "${TASKS[@]}"; do
  for model in "${MODELS[@]}"; do
    if [[ $model = lstm || $model = jm* ]]; then
      device=cpu
    else
      device=gpu
    fi
    for trial_no in "${TRIALS[@]}"; do
      bash experiments/submit-job.bash \
        "$task+$model+$trial_no" \
        "$LOG_DIR"/outputs \
        "$device" \
        bash cfl_language_modeling/train_ns_variant.bash \
          "$LOG_DIR" \
          "$model" \
          "$task" \
          "$trial_no" \
          --no-progress
    done
  done
done
