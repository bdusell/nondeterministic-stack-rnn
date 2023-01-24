set -e
set -u
set -o pipefail

usage() {
  echo "$0 <output-dir> <model-str> <task> <alphabet-size> <trial-no> ..."
}

output_dir=${1-}
model_str=${2-}
task=${3-}
alphabet_size=${4-}
trial_no=${5-}
if ! shift 5; then
  usage >&2
  exit 1
fi
extra_args=("$@")

if [[ $model_str = lstm ]]; then
  model_args=(--model-type lstm)
elif [[ $model_str =~ ^jm-(.+)$ ]]; then
  model_args=( \
    --model-type jm \
    --stack-embedding-size "${BASH_REMATCH[1]}" \
  )
elif [[ $model_str =~ ^old-rns-(.+)-(.+)$ ]]; then
  model_args=( \
    --model-type ns \
    --original-bottom-symbol-behavior \
    --num-states "${BASH_REMATCH[1]}" \
    --stack-alphabet-size "${BASH_REMATCH[2]}" \
  )
elif [[ $model_str =~ ^rns-(.+)-(.+)$ ]]; then
  model_args=( \
    --model-type ns \
    --num-states "${BASH_REMATCH[1]}" \
    --stack-alphabet-size "${BASH_REMATCH[2]}" \
  )
elif [[ $model_str =~ ^vrns-old-(.+)-(.+)-(.+)$ ]]; then
  model_args=( \
    --model-type vns \
    --original-bottom-symbol-behavior \
    --num-states "${BASH_REMATCH[1]}" \
    --stack-alphabet-size "${BASH_REMATCH[2]}" \
    --stack-embedding-size "${BASH_REMATCH[3]}" \
  )
elif [[ $model_str =~ ^vrns-(zero|one|learned)-(.+)-(.+)-(.+)$ ]]; then
  model_args=( \
    --model-type vns \
    --bottom-vector "${BASH_REMATCH[1]}" \
    --num-states "${BASH_REMATCH[2]}" \
    --stack-alphabet-size "${BASH_REMATCH[3]}" \
    --stack-embedding-size "${BASH_REMATCH[4]}" \
  )
else
  usage >&2
  exit 1
fi

bash cfl_language_modeling/train_on_task.bash \
  "$output_dir"/"$model_str"/"$task"/"$alphabet_size" \
  "$task" \
  "$alphabet_size" \
  "$trial_no" \
  "${model_args[@]}" \
  "${extra_args[@]}"
