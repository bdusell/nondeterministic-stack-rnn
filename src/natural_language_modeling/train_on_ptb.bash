set -e
set -u
set -o pipefail

usage() {
  echo "$0 <output-dir> ...

Arguments:

  <output-dir>    Output directory for logs and model parameters.

Extra options:
  --device <device>
  --no-progress
  ...             All arguments for initializing and running the model.
"
}

output_dir=${1-}
if ! shift 1; then
  usage >&2
  exit 1
fi
extra_args=("$@")

DATA_DIR=../data/mikolov-ptb

learning_rate=$(python utils/random_sample.py --log 1 100)
divide_lr_by=1.5
learning_rate_scaling_factor=$(python -c "print(1./$divide_lr_by)")
gradient_clip_threshold=$(python utils/random_sample.py --log 0.0112 1.12)

python natural_language_modeling/train.py \
  --output "$output_dir" \
  --save-model \
  --train-data "$DATA_DIR"/ptb.train.txt \
  --valid-data "$DATA_DIR"/ptb.valid.txt \
  --vocab "$DATA_DIR"/vocab.txt \
  --batch-size 32 \
  --bptt-limit 35 \
  --eval-batch-size 32 \
  --eval-bptt-limit 35 \
  --optimizer SGD \
  --learning-rate "$learning_rate" \
  --learning-rate-schedule-type epochs-without-improvement \
  --learning-rate-patience 0 \
  --learning-rate-scaling-factor "$learning_rate_scaling_factor" \
  --gradient-clip-threshold "$gradient_clip_threshold" \
  --epochs 100 \
  --early-stopping-patience 2 \
  "${extra_args[@]}"
