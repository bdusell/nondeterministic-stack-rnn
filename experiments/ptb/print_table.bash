set -e
set -u
set -o pipefail

. "$(dirname "$BASH_SOURCE")"/include.bash

cd "$ROOT_DIR"/src

mkdir -p "$FIGURES_DIR"/tex/ptb

output=$FIGURES_DIR/tex/ptb/table.tex

model_args=()
for model in "${MODELS[@]}"; do
  model_args+=(--label "$(format_model_name "$model")" --inputs)
  for trial_no in "${TRIALS[@]}"; do
    model_args+=("$(get_output_name "$model" "$trial_no")")
  done
done

echo "writing $output"
python natural_language_modeling/print_table.py "${model_args[@]}" \
  | tee "$output"
