set -e
set -u

. "$(dirname "$BASH_SOURCE")"/include.bash

for task in "${TASKS[@]}"; do
  for model in "${MODELS[@]}"; do
    case $model in
      lstm|jm-*) device=cpu ;;
      *) device=gpu ;;
    esac
    for alphabet_size in "${ALPHABET_SIZES[@]}"; do
      for trial_no in "${TRIALS[@]}"; do
        bash experiments/submit-job.bash \
          "$model+$task+$alphabet_size+$trial_no" \
          "$LOG_DIR"/outputs \
          "$device" \
          bash cfl_language_modeling/capacity_train.bash \
            "$LOG_DIR" \
            "$model" \
            "$task" \
            "$alphabet_size" \
            "$trial_no"
      done
    done
  done
done
