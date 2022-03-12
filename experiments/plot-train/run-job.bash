set -e
set -o pipefail

. experiments/variables.bash

logs=$1
task=$2
shift 2
inputs=("$@")

mkdir -p $logs/{png,tex}
python src/plot_train.py \
  --inputs $(for x in "${inputs[@]}"; do echo "$(< "$x")"/..; done) \
  --labels "${MODEL_LABELS[@]}" \
  --title "${TASK_TITLES[$task]}" \
  --output $logs/png/$task.png \
  --pgfplots-output $logs/tex/$task.tex
