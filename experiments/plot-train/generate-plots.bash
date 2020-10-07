set -e
set -o pipefail

. experiments/variables.bash

logs=$(dirname "$BASH_SOURCE")

mkdir -p $logs/{png,tex}
for task in "${TASKS[@]}"; do
  task=plot-train-$task
  bash experiments/submit-job.bash $key $logs/outputs cpu \
    poetry run python src/plot_train.py \
      --inputs $(
        for model in "${MODELS[@]}"; do
          cat $logs/../grid-search/$task/$model
        done
      ) \
      --labels "${MODEL_LABELS[@]}" \
      --metric cross-entropy-diff \
      --title "${TASK_TITLES[$task]}" \
      --output $logs/png/$task.png \
      --pgfplots-output $logs/tex/$task.tex
done
