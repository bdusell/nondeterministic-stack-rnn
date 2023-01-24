ROOT_DIR=$(cd "$(dirname "$BASH_SOURCE")"/../.. && pwd)
. "$ROOT_DIR"/experiments/include.bash
LOG_DIR=$HOME/Private/logs/2022-09-16/ptb
MODELS=( \
  lstm-256 \
  jm-hidden-247
  jm-learned-22 \
  rns-1-29 \
  rns-2-13 \
  rns-4-5 \
  vrns-1-1-256 \
  vrns-1-1-32 \
  vrns-1-5-20 \
  vrns-2-3-10 \
  vrns-3-3-5 \
)
TRIALS=({1..10})

format_model_name() {
  local name=$1
  local result
  if [[ $name =~ ^lstm-(.+)$ ]]; then
    local hidden_units=${BASH_REMATCH[1]}
    result="LSTM, $hidden_units units"
  elif [[ $name =~ ^jm-hidden-(.+)$ ]]; then
    local hidden_units=${BASH_REMATCH[1]}
    result="Sup. (push hidden), $hidden_units units"
  elif [[ $name =~ ^jm-learned-(.+)$ ]]; then
    local stack_embedding_size=${BASH_REMATCH[1]}
    result="Sup. (push learned), \$|\\mathbf{v}_t| = $stack_embedding_size\$"
  elif [[ $name =~ ^rns-(.+)-(.+)$ ]]; then
    local num_states=${BASH_REMATCH[1]}
    local stack_alphabet_size=${BASH_REMATCH[2]}
    result="RNS $num_states-$stack_alphabet_size"
  elif [[ $name =~ ^vrns-(.+)-(.+)-(.+)$ ]]; then
    local num_states=${BASH_REMATCH[1]}
    local stack_alphabet_size=${BASH_REMATCH[2]}
    local stack_embedding_size=${BASH_REMATCH[3]}
    result="VRNS $num_states-$stack_alphabet_size-$stack_embedding_size"
  else
    return 1
  fi
  echo -n "$result"
}

get_output_name() {
  local model=$1
  local trial_no=$2
  local result=$LOG_DIR/$model/$trial_no
  echo -n "$result"
}
