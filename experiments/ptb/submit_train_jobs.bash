set -e
set -u

. "$(dirname "$BASH_SOURCE")"/include.bash

for model in "${MODELS[@]}"; do
  for trial_no in "${TRIALS[@]}"; do
    bash experiments/submit-job.bash \
      "ptb+$model+$trial_no" \
      "$LOG_DIR"/outputs \
      gpu \
      bash natural_language_modeling/train_model_on_ptb.bash \
        "$(get_output_name "$model" "$trial_no")" \
        "$model" \
        --no-progress
  done
done
