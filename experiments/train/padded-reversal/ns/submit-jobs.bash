set -e
set -o pipefail

. experiments/variables.bash

logs=$(dirname "$BASH_SOURCE")

[[ $BASH_SOURCE =~ /([^/]+)/([^/]+)/submit-jobs\.bash$ ]]
task=${BASH_REMATCH[1]}
model_type=${BASH_REMATCH[2]}
name=train-$task-$model_type

for learning_rate in "${LEARNING_RATES[@]}"; do
  for trial_no in "${TRIALS[@]}"; do
    hyperparams=$learning_rate
    key=$name-$hyperparams-$trial_no
    bash experiments/submit-job.bash $key $logs/outputs gpu \
      poetry run python src/train.py \
        --device cuda \
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
        --symbol-types 2 \
        --mean-content-length $((TRAIN_LENGTH + TRAIN_LENGTH_RANGE / 2)) \
        --mean-padding-length $(( (TRAIN_LENGTH + TRAIN_LENGTH_RANGE / 2) / 2 )) \
        --parameter-seed $RANDOM \
        --model-type $model_type \
        --hidden-units $HIDDEN_UNITS \
        --num-states 3 \
        --stack-alphabet-size 2 \
        --block-size 32 \
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
