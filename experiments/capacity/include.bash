ROOT_DIR=$(cd "$(dirname "$BASH_SOURCE")"/../.. && pwd)
. "$ROOT_DIR"/experiments/include.bash
LOG_DIR=$HOME/Private/logs/2022-07-15/capacity
TASKS=( \
  marked-reversal \
  dyck \
  unmarked-reversal \
)
MODELS=( \
  lstm \
  jm-3 \
  rns-1-3 \
  rns-2-3 \
  vrns-learned-1-1-3 \
  vrns-learned-2-1-3 \
  vrns-learned-2-3-3 \
)
TRIALS=({1..10})
ALPHABET_SIZES=(2 40 80 120 160 200)

format_model_name() {
  local name=$1
  local result
  if [[ $name = lstm ]]; then
    result='LSTM'
  elif [[ $name =~ ^jm-(.+)$ ]]; then
    result="Sup. ${BASH_REMATCH[1]}"
  elif [[ $name =~ ^(old-rns|rns)-(.+)-(.*)$ ]]; then
    local type=${BASH_REMATCH[1]}
    local Q=${BASH_REMATCH[2]}
    local S=${BASH_REMATCH[3]}
    local desc
    if [[ $type = old-rns ]]; then
      desc='RNS (before bottom fix)'
    else
      desc='RNS'
    fi
    result="$desc $Q-$S"
  elif [[ $name =~ ^vrns-(.+)-(.+)-(.+)-(.+)$ ]]; then
    local type=${BASH_REMATCH[1]}
    local Q=${BASH_REMATCH[2]}
    local S=${BASH_REMATCH[3]}
    local m=${BASH_REMATCH[4]}
    local desc
    case $type in
      learned) desc='VRNS' ;;
      old) desc='VRNS (bottom is zeroes, before bottom fix)' ;;
      one) desc='VRNS (bottom is ones, after bottom fix)' ;;
      zero) desc='VRNS (bottom is zeroes, after bottom fix)' ;;
    esac
    result="$desc $Q-$S-$m"
  else
    return 1
  fi
  echo -n "$result"
}

format_task_name() {
  local name=$1
  local result
  case $name in
    marked-reversal) result='\MarkedReversal{}' ;;
    dyck) result='Dyck' ;;
    unmarked-reversal) result='\UnmarkedReversal{}' ;;
    *) return 1 ;;
  esac
  echo -n "$result"
}

get_output_directory() {
  local model=$1
  local task=$2
  local symbols=$3
  local trial_no=$4
  local result=$LOG_DIR/$model/$task/$symbols/$trial_no
  echo -n "$result"
}
