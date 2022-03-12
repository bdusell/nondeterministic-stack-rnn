set -e
set -o pipefail

. experiments/variables.bash

logs=$(dirname "$BASH_SOURCE")

[[ $BASH_SOURCE =~ /([^/]+)/([^/]+)/submit-jobs\.bash$ ]]
task=${BASH_REMATCH[1]}
model_type=${BASH_REMATCH[2]}
name=train-$task-$model_type

for stack_embedding_size in 2 $HIDDEN_UNITS $((HIDDEN_UNITS*2)); do
  for learning_rate in "${LEARNING_RATES[@]}"; do
    for trial_no in "${TRIALS[@]}"; do
      hyperparams=$stack_embedding_size-$learning_rate
      key=$name-$hyperparams-$trial_no
      bash experiments/submit-job.bash $key $logs/outputs cpu \
        poetry run python src/train_cfl.py \
          --device cpu \
          --output $logs/logs/$hyperparams/$trial_no \
          --no-progress \
          --save-model \
          --data-seed $RANDOM \
          --train-length-range $TRAIN_LENGTH:$((TRAIN_LENGTH + TRAIN_LENGTH_RANGE)) \
          --train-data-size 10000 \
          --batch-size 10 \
          --valid-length-range $TRAIN_LENGTH:$((TRAIN_LENGTH + TRAIN_LENGTH_RANGE)) \
          --valid-data-size 1000 \
          --valid-batch-size 10 \
          --task $task \
          --bracket-types 2 \
          --mean-bracket-splits 1 \
          --mean-nesting-depth 40 \
          --parameter-seed $RANDOM \
          --model-type $model_type \
          --hidden-units $HIDDEN_UNITS \
          --stack-embedding-size $stack_embedding_size \
          --init-scale 0.1 \
          --shuffle-seed $RANDOM \
          --optimizer Adam \
          --learning-rate $learning_rate \
          --learning-rate-patience 5 \
          --learning-rate-decay 0.9 \
          --gradient-clipping 5 \
          --epochs 200 \
          --early-stopping-patience 10
    done
  done
done
