set -e
set -o pipefail

. experiments/variables.bash

data=$DATA_DIR
logs=$(dirname "$BASH_SOURCE")

[[ $BASH_SOURCE =~ /([^/]+)/([^/]+)/[^/]+\.bash$ ]]
task=${BASH_REMATCH[1]}
model_type=${BASH_REMATCH[2]}
name=train-$task-$model_type

sample() {
  bash experiments/run-job.bash poetry run python src/random_sample.py "$@"
}

divide_lr_by=1.5
learning_rate_scaling_factor=$(bc <<<"scale=16; 1/$divide_lr_by")

hidden_units=256
stack_embedding_size=22

for random_search_no in {1..10}; do
  output=$logs/random-search-logs/$random_search_no
  random_search_file=$(bash experiments/replace-paths.bash echo $logs/random-search/$random_search_no.x)
  if [[ -f $random_search_file ]]; then echo "error: $random_search_file exists"; exit 1; fi
  random_search="
    --learning-rate $(sample --log 1 100)
    --gradient-clip-threshold $(sample --log 1e-5 1e-3)
  "
  mkdir -p $(dirname $random_search_file)
  echo $random_search > $random_search_file
  key=$name-$random_search_no
  bash experiments/submit-job.bash $key $logs/outputs gpu \
    poetry run python src/train_natural_with_context.py \
      --device cuda \
      --output $logs/logs/$random_search_no \
      --no-progress \
      --save-model \
      --train-data $data/mikolov-ptb/ptb.train.txt \
      --batch-size 32 \
      --bptt-limit 35 \
      --valid-data $data/mikolov-ptb/ptb.valid.txt \
      --eval-batch-size 32 \
      --eval-bptt-limit 35 \
      --vocab $data/mikolov-ptb/vocab.txt \
      --parameter-seed $RANDOM \
      --model-type lstm \
      --hidden-units $hidden_units \
      --layers 1 \
      --stack-model jm \
      --jm-push-type learned \
      --stack-embedding-size $stack_embedding_size \
      --stack-depth-limit 10 \
      --uniform-init-scale 0.05 \
      --optimizer SGD \
      --learning-rate-schedule-type epochs-without-improvement \
      --learning-rate-patience 0 \
      --learning-rate-scaling-factor $learning_rate_scaling_factor \
      --epochs 100 \
      --early-stopping-patience 2 \
      $random_search
done
