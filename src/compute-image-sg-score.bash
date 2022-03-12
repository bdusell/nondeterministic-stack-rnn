set -e
set -o pipefail

usage() {
  echo "Usage: $0 <docker-image> <log-dir> [options]

  <docker-image>  Docker image for a language model.
  <log-dir>       Directory to contain logs.
  --hide-stderr   Do not show stderr from syntaxgym.
"
}

model=
log_dir=
hide_stderr=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --hide-stderr) hide_stderr=true ;;
    *)
      if [[ ! $model ]]; then
        model=$1
      elif [[ ! $log_dir ]]; then
        log_dir=$1
      else
        usage >&2
        exit 1
      fi
      ;;
  esac
  shift
done

suite_accuracy_file="$log_dir"/suite-accuracy.tsv
sg_score_file="$log_dir"/score.txt

if $hide_stderr; then
  stderr_file=/dev/null
else
  stderr_file=/dev/stderr
fi

suite_files=($(find ../data/syntaxgym -name '*.json' | sort))
mkdir -p "$log_dir"
{
  for suite_file in "${suite_files[@]}"; do
    syntaxgym run docker://"$model" "$suite_file" 2>$stderr_file | python syntaxgym_result_to_accuracy.py
  done
} | tee "$suite_accuracy_file" >&2
cut -f 2 "$suite_accuracy_file" | python compute_average.py | tee "$sg_score_file"
