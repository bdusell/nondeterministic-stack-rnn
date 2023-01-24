ROOT_DIR=$(cd "$(dirname "$BASH_SOURCE")"/../.. && pwd)
. "$ROOT_DIR"/experiments/include.bash
LOG_DIR=$HOME/Private/logs/2022-06-30/non-cfls
TASKS=( \
  count-3 \
  marked-reverse-and-copy \
  count-and-copy \
  marked-copy \
  unmarked-copy-different-alphabets \
  unmarked-reverse-and-copy \
  unmarked-copy \
)
MODELS=( \
  lstm \
  jm-10 \
  jm-3.3.3 \
  jm-hidden \
  rns-3-3 \
)
TRIALS=({1..10})

format_model_name() {
  local name=$1
  local result
  if [[ $name = lstm ]]; then
    result='LSTM'
  elif [[ $name = gru ]]; then
    result='GRU'
  elif [[ $name = jm-hidden ]]; then
    result='Sup. h'
  elif [[ $name =~ ^jm-(.+)$ ]]; then
    result="Sup. $(sed 's/\./-/g' <<<"${BASH_REMATCH[1]}")"
  elif [[ $name =~ ^rns-(.+)-(.+)$ ]]; then
    result="RNS ${BASH_REMATCH[1]}-${BASH_REMATCH[2]}"
  else
    return 1
  fi
  echo -n "$result"
}

format_task_name() {
  local name=$1
  local result
  case $name in
    count-3) result='\CountThree{}' ;;
    count-and-copy) result='\CountAndCopy{}' ;;
    marked-reverse-and-copy) result='\MarkedReverseAndCopy{}' ;;
    marked-copy) result='\MarkedCopy{}' ;;
    unmarked-copy) result='\UnmarkedCopy{}' ;;
    unmarked-reverse-and-copy) result='\UnmarkedReverseAndCopy{}' ;;
    unmarked-copy-different-alphabets) result='\UnmarkedCopyDifferentAlphabets{}' ;;
    *) return 1 ;;
  esac
  echo -n "$result"
}

get_output_directory() {
  local task=$1
  local model=$2
  local trial_no=$3
  local result=$LOG_DIR/$model/$task/$trial_no
  echo -n "$result"
}

get_test_data_file() {
  local task=$1
  local result=$LOG_DIR/test-sets/$task-test-data.pt
  echo -n "$result"
}
