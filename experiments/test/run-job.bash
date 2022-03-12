set -e
set -o pipefail

logs=$1
task=$2
model=$3
shift 3

if [[ $model = *ns* ]]; then
  python src/test.py \
    --device cuda \
    --output $logs/logs/$task/$model \
    --data $logs/../test-data/$task/$task-test-data.pt \
    --block-size 8 \
    --no-progress \
    --input $(< $logs/../grid-search/$task/$model)
else
  python src/test.py \
    --device cpu \
    --output $logs/logs/$task/$model \
    --data $logs/../test-data/$task/$task-test-data.pt \
    --no-progress \
    --input $(< $logs/../grid-search/$task/$model)
fi
