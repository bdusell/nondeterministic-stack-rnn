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
        --mean-num-commas 0.5 \
        --mean-short-filler-length 0.5 \
        --mean-long-filler-length 2 \
        --semicolon-probability 0.25 \
        --mean-bracket-splits 1.5 \
        --mean-nesting-depth 3 \
        --parameter-seed $RANDOM \
        --model-type $model_type \
        --hidden-units $HIDDEN_UNITS \
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
