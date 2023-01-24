set -e
set -u
set -o pipefail

usage() {
  echo "$0 <output-dir> <model-str> ...

Arguments:

  <output-dir>    Output directory for logs and model parameters.
  <model-str>     String describing a model to train.

Extra options:
  --device <device>
  --no-progress
"
}

output_dir=${1-}
model_str=${2-}
if ! shift 2; then
  usage >&2
  exit 1
fi
extra_args=("$@")

DEFAULT_HIDDEN_UNITS=256

if [[ $model_str =~ ^lstm-([0-9]+)$ ]]; then
  hidden_units=${BASH_REMATCH[1]}
  model_args=( \
    --hidden-units "$hidden_units" \
    --stack-model none \
  )
elif [[ $model_str =~ ^jm-hidden-([0-9]+)$ ]]; then
  hidden_units=${BASH_REMATCH[1]}
  model_args=( \
    --hidden-units "$hidden_units" \
    --stack-model jm \
    --jm-push-type hidden-state \
    --stack-depth-limit 10 \
  )
elif [[ $model_str =~ ^jm-learned-([0-9]+)$ ]]; then
  stack_embedding_size=${BASH_REMATCH[1]}
  model_args=( \
    --hidden-units "$DEFAULT_HIDDEN_UNITS" \
    --stack-model jm \
    --jm-push-type learned \
    --stack-embedding-size "$stack_embedding_size" \
    --stack-depth-limit 10 \
  )
elif [[ $model_str =~ ^rns-([0-9]+)-([0-9]+)$ ]]; then
  num_states=${BASH_REMATCH[1]}
  stack_alphabet_size=${BASH_REMATCH[2]}
  model_args=( \
    --hidden-units "$DEFAULT_HIDDEN_UNITS" \
    --stack-model ns \
    --num-states "$num_states" \
    --stack-alphabet-size "$stack_alphabet_size" \
    --window-size 35 \
  )
elif [[ $model_str =~ ^vrns-([0-9]+)-([0-9]+)-([0-9]+)$ ]]; then
  num_states=${BASH_REMATCH[1]}
  stack_alphabet_size=${BASH_REMATCH[2]}
  stack_embedding_size=${BASH_REMATCH[3]}
  model_args=( \
    --hidden-units "$DEFAULT_HIDDEN_UNITS" \
    --stack-model vns \
    --num-states "$num_states" \
    --stack-alphabet-size "$stack_alphabet_size" \
    --stack-embedding-size "$stack_embedding_size" \
    --window-size 35 \
  )
else
  usage >&2
  exit 1
fi

model_args+=( \
  --model-type lstm \
  --layers 1 \
  --uniform-init-scale 0.05 \
)

bash natural_language_modeling/train_on_ptb.bash \
  "$output_dir" \
  "${model_args[@]}" \
  "${extra_args[@]}"
