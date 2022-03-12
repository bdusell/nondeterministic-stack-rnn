# This script is used to edit paths in commands so that it is possible to
# configure results to be stored in a different directory. By default results
# are stored in the experiments directory. It accepts a list of arguments and
# executes them as a command. You can configure the output directory with the
# environment variables OUTPUT_DIR and OUTPUT_SUBDIR.

OUTPUT_DIR=${OUTPUT_DIR-experiments}
if [[ $OUTPUT_SUBDIR ]]; then
  OUTPUT_DIR+=/$OUTPUT_SUBDIR
fi

modified_args=()
for arg in "$@"; do
  if [[ $arg != *.bash && $arg =~ ^experiments/(.*)$ ]]; then
    path=${BASH_REMATCH[1]}
    arg=$OUTPUT_DIR/$path
  fi
  modified_args+=("$arg")
done

"${modified_args[@]}"
