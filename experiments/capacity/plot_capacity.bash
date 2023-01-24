set -e
set -u

. "$(dirname "$BASH_SOURCE")"/include.bash

MAIN_TASKS=( \
  marked-reversal \
  dyck \
)

cd "$ROOT_DIR"/src

mkdir -p "$FIGURES_DIR"/{png,tex}/capacity

legend_png_output=$FIGURES_DIR/png/capacity/legend.png
legend_tex_output=$FIGURES_DIR/tex/capacity/legend.tex

wrote_legend=false

write_plots() {
  local mode=$1
  shift 1
  local tasks=("$@")
  local last_task=${tasks[${#tasks[@]} - 1]}
  for task in "${tasks[@]}"; do
    plot_args=()
    for model in "${MODELS[@]}"; do
      for symbols in "${ALPHABET_SIZES[@]}"; do
        plot_args+=(--label "$(format_model_name "$model")" --symbols "$symbols" --inputs)
        for trial_no in "${TRIALS[@]}"; do
          plot_args+=("$(get_output_directory "$model" "$task" "$symbols" "$trial_no")")
        done
      done
    done
    if ! $wrote_legend; then
      echo "writing $legend_png_output"
      echo "writing $legend_tex_output"
      plot_args+=( \
        --best-legend-output "$legend_png_output" \
        --best-legend-pgfplots-output "$legend_tex_output" \
        --legend-columns 4 \
      )
      wrote_legend=true
    fi
    if [[ $task = $last_task ]]; then
      plot_args+=(--show-x-label)
    fi
    png_output=$FIGURES_DIR/png/capacity/$task-$mode.png
    tex_output=$FIGURES_DIR/tex/capacity/$task-$mode.tex
    if [[ $mode = mean || $mode = mean-std ]]; then
      plot_args+=( \
        --mean-output "$png_output" \
        --mean-pgfplots-output "$tex_output" \
      )
      if [[ $mode = mean-std ]]; then
        plot_args+=(--show-stddev)
      fi
    elif [[ $mode = best ]]; then
      plot_args+=( \
        --best-output "$png_output" \
        --best-pgfplots-output "$tex_output" \
      )
    else
      return 1
    fi
    echo "writing $png_output"
    echo "writing $tex_output"
    task_name=$(format_task_name "$task")
    python cfl_language_modeling/plot_capacity.py \
      --best-separate-legend \
      --mean-separate-legend \
      --best-pgfplots-strict-mode \
      --mean-pgfplots-strict-mode \
      --best-title "$task_name" \
      --mean-title "$task_name" \
      "${plot_args[@]}"
  done
}

for task in "${MAIN_TASKS[@]}"; do
  write_plots mean "$task"
done
write_plots mean-std "${TASKS[@]}"
write_plots best "${TASKS[@]}"
