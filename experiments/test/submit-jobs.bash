set -e
set -o pipefail

. experiments/variables.bash

logs=$(dirname "$BASH_SOURCE")

for task in "${TASKS[@]}"; do
  for model in "${MODELS[@]}"; do
    for trial_no in "${TRIALS[@]}"; do
      key=$task-$model-$trial_no
      name=test-$key
      if [[ $model = ns ]]; then
        bash experiments/submit-job.bash $key $logs/outputs gpu \
          poetry run python src/test.py \
            --device cuda \
            --output $logs/logs/$task/$model/$trial_no \
            --data $logs/../test-data/$task/$task-test-data.pt \
            --block-size 32 \
            --no-progress \
            --input $(< $logs/../grid-search/$task/$model)/$trial_no
      else
        bash experiments/submit-job.bash $key $logs/outputs cpu \
          poetry run python src/test.py \
            --device cpu \
            --output $logs/logs/$task/$model/$trial_no \
            --data $logs/../test-data/$task/$task-test-data.pt \
            --no-progress \
            --input $(< $logs/../grid-search/$task/$model)/$trial_no
      fi
    done
  done
done
