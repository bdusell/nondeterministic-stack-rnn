set -e
set -u
set -o pipefail

usage() {
  echo "Usage: $0 <output-dir> <model-str> <task-str> <trial-no> ..."
}

output_dir=${1-}
model_str=${2-}
task_str=${3-}
trial_no=${4-}
if ! shift 4; then
  usage >&2
  exit 1
fi
extra_args=("$@")

if [[ $model_str = lstm ]]; then
  model_args=(--model-type lstm)
elif [[ $model_str =~ ^rns-([0-9]+)-([0-9]+)$ ]]; then
  model_args=( \
    --model-type ns \
    --num-states "${BASH_REMATCH[1]}" \
    --stack-alphabet-size "${BASH_REMATCH[2]}" \
  )
elif [[ $model_str =~ ^emb-rns-([0-9]+)-([0-9]+)-([0-9]+)$ ]]; then
  model_args=( \
    --model-type vns \
    --num-states "${BASH_REMATCH[1]}" \
    --stack-alphabet-size "${BASH_REMATCH[2]}" \
    --stack-embedding-size "${BASH_REMATCH[3]}" \
  )
elif [[ $model_str = jm-hidden ]]; then
  model_args=( \
    --model-type jm \
    --push-hidden-state \
  )
elif [[ $model_str =~ ^jm-([0-9]+(\.[0-9]+)*)$ ]]; then
  model_args=( \
    --model-type jm \
    --stack-embedding-size $(sed 's/\./ /g' <<<"${BASH_REMATCH[1]}") \
  )
else
  usage >&2
  exit 1
fi

if [[ $task_str =~ ^(unmarked-reversal|marked-reversal)-([0-9]+)$ ]]; then
  task=${BASH_REMATCH[1]}
  symbol_types=${BASH_REMATCH[2]}
elif [[ $task_str = marked-reverse-and-copy ]]; then
  task=$task_str
  symbol_types=2
elif [[ $task_str =~ ^(count-3|marked-copy|unmarked-copy|unmarked-reverse-and-copy|count-and-copy|unmarked-copy-different-alphabets)$ ]]; then
  task=$task_str
  symbol_types=x
else
  usage >&2
  exit 1
fi

bash cfl_language_modeling/train_on_task.bash \
  "$output_dir"/"$model_str"/"$task_str" \
  "$task" \
  "$symbol_types" \
  "$trial_no" \
  "${model_args[@]}" \
  "${extra_args[@]}"
