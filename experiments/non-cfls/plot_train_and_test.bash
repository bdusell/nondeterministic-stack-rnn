set -e
set -u

. "$(dirname "$BASH_SOURCE")"/include.bash

MAIN_TASKS=( \
  marked-reverse-and-copy \
  marked-copy \
  unmarked-copy-different-alphabets \
  unmarked-copy \
)
EXTRA_TASKS=( \
  count-3 \
  count-and-copy \
  unmarked-reverse-and-copy \
)

print_table() {
  for task in "$@"; do
    echo "\\scalebox{0.8}{\\input{ICLR 2023/figures/non-cfls/train/$task}}"
    echo "&\\scalebox{0.8}{\\input{ICLR 2023/figures/non-cfls/test/$task}}"
    echo "\\\\"
  done
}

write_table() {
  local output=$1
  shift 1
  echo "writing $output"
  print_table "$@" > "$output"
}

cd "$ROOT_DIR"/src

mkdir -p "$FIGURES_DIR"/{png,tex}/non-cfls/{train,test}

main_table_output=$FIGURES_DIR/tex/non-cfls/main-table.tex
extra_table_output=$FIGURES_DIR/tex/non-cfls/extra-table.tex

write_table "$main_table_output" "${MAIN_TASKS[@]}"
write_table "$extra_table_output" "${EXTRA_TASKS[@]}"

legend_png_output=$FIGURES_DIR/png/non-cfls/legend.png
legend_tex_output=$FIGURES_DIR/tex/non-cfls/legend.tex
wrote_legend=false

write_plots() {
  local tasks=("$@")
  local last_task=${tasks[${#tasks[@]} - 1]}
  for task in "${tasks[@]}"; do
    plot_args=( \
      --title "$(format_task_name "$task")" \
      --target-runs "${#TRIALS[@]}" \
      --separate-legend \
      --width 3.4 \
      --height 1.575 \
    )
    plot_train_args=()
    plot_test_args=()
    for model in "${MODELS[@]}"; do
      trial_args=()
      for trial_no in "${TRIALS[@]}"; do
        trial_args+=("$(get_output_directory "$task" "$model" "$trial_no")")
      done
      output=$(python utils/print_best.py "${trial_args[@]}")
      best_model=$(cut -f 1 <<<"$output")
      num_trials=$(cut -f 2 <<<"$output")
      plot_train_args+=(--input)
      plot_test_args+=(--input)
      if [[ $best_model ]]; then
        plot_train_args+=("$best_model")
        plot_test_args+=("$best_model"/test)
      fi
      plot_args+=(--label "$(format_model_name "$model")" --runs "$num_trials")
    done
    if ! $wrote_legend; then
      echo "writing $legend_png_output"
      echo "writing $legend_tex_output"
      plot_train_args+=( \
        --legend-output "$legend_png_output" \
        --legend-pgfplots-output "$legend_tex_output" \
      )
      wrote_legend=true
    fi
    if [[ $task = $last_task ]]; then
      plot_args+=(--show-x-label)
    fi
    train_png_output=$FIGURES_DIR/png/non-cfls/train/$task.png
    train_tex_output=$FIGURES_DIR/tex/non-cfls/train/$task.tex
    echo "writing $train_png_output"
    echo "writing $train_tex_output"
    python cfl_language_modeling/plot_train.py \
      --output "$train_png_output" \
      --pgfplots-output "$train_tex_output" \
      "${plot_args[@]}" \
      "${plot_train_args[@]}"
    test_png_output=$FIGURES_DIR/png/non-cfls/test/$task.png
    test_tex_output=$FIGURES_DIR/tex/non-cfls/test/$task.tex
    echo "writing $test_png_output"
    echo "writing $test_tex_output"
    python cfl_language_modeling/plot_test.py \
      --output "$test_png_output" \
      --pgfplots-output "$test_tex_output" \
      "${plot_args[@]}" \
      "${plot_test_args[@]}"
  done
}

write_plots "${MAIN_TASKS[@]}"
write_plots "${EXTRA_TASKS[@]}"
