set -e
set -o pipefail

. experiments/variables.bash

logs=$(dirname "$BASH_SOURCE")

[[ $BASH_SOURCE =~ /([^/]+)/submit-jobs\.bash$ ]]
task=${BASH_REMATCH[1]}
key=test-data-$task

bash experiments/submit-job.bash $key $logs/outputs cpu \
  poetry run python src/generate_test_data.py \
    --test-length-range $TRAIN_LENGTH:$((TRAIN_LENGTH + TEST_LENGTH_RANGE)) \
    --test-data-size 100 \
    --test-batch-size 128 \
    --test-data-seed 0 \
    --output $logs/$task-test-data.pt \
    --task $task \
    --mean-num-commas 0.5 \
    --mean-short-filler-length 0.5 \
    --mean-long-filler-length 2 \
    --semicolon-probability 0.25 \
    --mean-bracket-splits 1.5 \
    --mean-nesting-depth 3
