set -e
set -u

. "$(dirname "$BASH_SOURCE")"/include.bash

for model in "${MODELS[@]}"; do
  trial_args=()
  for trial_no in "${TRIALS[@]}"; do
    trial_args+=("$(get_output_name "$model" "$trial_no")")
  done
  bash experiments/submit-job.bash \
    "ptb-test+$model" \
    "$LOG_DIR"/outputs \
    gpu \
    bash natural_language_modeling/test_best_model_on_ptb.bash \
      "${trial_args[@]}" \
      -- \
      --no-progress
done
