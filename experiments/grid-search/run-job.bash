set -e
set -o pipefail

. experiments/variables.bash

input_dir=$1
output_file=$2
shift 2

input_files=()
for d in "$input_dir"/*; do
  for trial in "${TRIALS[@]}"; do
    input_files+=("$d"/"$trial")
  done
done

mkdir -p "$(dirname "$output_file")"
python src/print_best.py "$@" "${input_files[@]}" > "$output_file"
