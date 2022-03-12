set -e
set -u
set -o pipefail

usage() {
  echo "Usage: $0 <dir>"
}

dir=${1-}
if ! shift 1; then
  usage >&2
  exit 1
fi

best_model=$(python print_best.py --metric perplexity --ignore-missing-trials "$dir"/logs/{1..10})
echo "Best model: $best_model"
num_params=$(python count_parameters.py --input "$best_model")
echo "Parameter count: $num_params"
valid_perplexity=$(python print_valid_perplexity.py "$best_model")
echo "Validation perplexity: $valid_perplexity"
python test_natural_with_context.py \
  --block-size 128 \
  --input "$best_model" \
  --test-data ../data/mikolov-ptb/ptb.test.txt \
  --eval-batch-size 32 \
  --eval-bptt-limit 35 \
  --vocab ../data/mikolov-ptb/vocab.txt
bash compute-sg-score.bash \
  "$best_model" \
  "$best_model"/sgscore \
  --dataset ptb \
  --hide-stderr \
  --no-build-docker-image \
  --block-size 128
python aggregate_sg_score_by_circuit.py \
  --circuits ../data/circuits.json \
  "$best_model"/sgscore/suite-accuracy.tsv
