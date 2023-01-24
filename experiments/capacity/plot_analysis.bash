set -e
set -u

. "$(dirname "$BASH_SOURCE")"/include.bash

cd "$ROOT_DIR"/src

mkdir -p "$FIGURES_DIR"/{png,tex}/capacity

task=marked-reversal
alphabet_size=40
model=rns-2-3

trial_args=()
for trial_no in "${TRIALS[@]}"; do
  trial_args+=("$(get_output_directory "$model" "$task" "$alphabet_size" "$trial_no")")
done
output=$(python utils/print_best.py "${trial_args[@]}")
best_model=$(cut -f 1 <<<"$output")
echo "best model: $best_model"

pca_png_output=$FIGURES_DIR/png/capacity/reading-pca-$task-$alphabet_size-$model.png
pca_tex_output=$FIGURES_DIR/tex/capacity/reading-pca-$task-$alphabet_size-$model.tex
echo "writing $pca_png_output"
echo "writing $pca_tex_output"
python cfl_language_modeling/plot_ns_pca.py \
  --data-seed 123 \
  --length 40:80 \
  --num-samples 100 \
  --batch-size 1 \
  --output "$pca_png_output" \
  --pgfplots-output "$pca_tex_output" \
  --separate-legend \
  --input "$best_model" \
  --eos-in-input \
  --task "$task" \
  --symbol-types "$alphabet_size"

plot_heatmap() {
  local input=$1
  local name=$2
  shift 2
  heatmap_png_output=$FIGURES_DIR/png/capacity/reading-heatmap-$name.png
  heatmap_tex_output=$FIGURES_DIR/tex/capacity/reading-heatmap-$name.tex
  echo "writing $heatmap_png_output"
  echo "writing $heatmap_tex_output"
  rm -f "$FIGURES_DIR"/tex/capacity/reading-heatmap-"$name"-*.png
  python cfl_language_modeling/plot_ns_sample_heatmap.py \
    --input-string <( \
      python cfl_language_modeling/print_marked_reversal.py \
        --length 41 \
        --start 0 \
        --stop 3 \
        --pattern rainbow \
      ) \
    --separate-legend \
    --width 2.5 \
    --height 11 \
    --output "$heatmap_png_output" \
    --pgfplots-output "$heatmap_tex_output" \
    --plot-type readings \
    --input "$input" \
    --task marked-reversal \
    --symbol-types 40 \
    --mean-length 60 \
    "$@"
  sed -i '
  s/\\mathtt/\\sym/g;
  s|\([^{]\+\.png\)|figures/capacity/\1|;
  ' "$heatmap_tex_output"
}

plot_heatmap_trial() {
  local trial_no=$1
  shift 1
  plot_heatmap "$(get_output_directory "$model" "$task" "$alphabet_size" "$trial_no")" "$trial_no" "$@"
}

plot_heatmap "$best_model" best --eos-in-input --show-y-label
plot_heatmap_trial 1 --eos-in-input
plot_heatmap_trial 9
