set -e
set -o pipefail

. experiments/variables.bash

logs=$(dirname "$BASH_SOURCE")

mkdir -p $logs/{png,tex}
for task in "${TASKS[@]}"; do
  key=plot-test-$task
  bash experiments/submit-job.bash $key $logs/outputs cpu \
    poetry run python src/plot_test.py \
      --inputs $(
        for model in "${MODELS[@]}"; do
          echo $logs/../test/logs/$task/$model
        done
      ) \
      --labels "${MODEL_LABELS[@]}" \
      --title "${TASK_TITLES[$task]}" \
      --output $logs/png/$task.png \
      --pgfplots-output $logs/tex/$task.tex
done
