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
    model_dirs=()
    for trial_no in "${TRIALS[@]}"; do
      model_dirs+=("$(get_output_directory "$task" "$model" "$trial_no")")
    done
    bash experiments/submit-job.bash \
      "test+$task+$model" \
      "$LOG_DIR"/outputs \
      "$device" \
      bash cfl_language_modeling/test_best_model.bash \
        "$(get_test_data_file "$task")" \
        "$model" \
        "${model_dirs[@]}" \
        -- \
        --no-progress
  done
done
