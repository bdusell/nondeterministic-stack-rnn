set -e
set -o pipefail

. experiments/variables.bash

logs=$1
task=$2
shift 2
inputs=("$@")

mkdir -p $logs/{png,tex}
python src/plot_test.py \
  --inputs "${inputs[@]}" \
  --labels "${MODEL_LABELS[@]}" \
  --title "${TASK_TITLES[$task]}" \
  --output $logs/png/$task.png \
  --pgfplots-output $logs/tex/$task.tex
