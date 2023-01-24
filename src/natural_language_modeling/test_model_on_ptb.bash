set -e
set -u
set -o pipefail

usage() {
  echo "$0 <model-dir> ...

  <model-dir>   The directory containing the trained model to evaluate.
  ...           Any other arguments to pass to test.py.
"
}

model_dir=${1-}
if ! shift 1; then
  usage >&2
  exit 1
fi
extra_args=("$@")

DATA_DIR=../data/mikolov-ptb

python natural_language_modeling/test.py \
  --input "$model_dir" \
  --output "$model_dir"/test \
  --test-data "$DATA_DIR"/ptb.test.txt \
  --vocab "$DATA_DIR"/vocab.txt \
  --eval-batch-size 32 \
  --eval-bptt-limit 35 \
  "${extra_args[@]}"
