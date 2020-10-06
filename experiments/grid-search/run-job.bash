set -e
set -o pipefail

output_file=$1
shift 1
mkdir -p "$(dirname "$output_file")"
python src/print_grid_search.py "$@" > "$output_file"
