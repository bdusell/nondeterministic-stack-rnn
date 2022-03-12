set -e
set -o pipefail

usage() {
  echo "Usage: $0 [options] <saved-model-dir> [-- <args>...]

  <saved-model-dir>     Directory containing a saved model.
  <args>                Additional arguments to be passed to get_surprisals.py.

Options:
  --version <version>   Version string.
  --name <name>         Name for this model.
  --dataset {ptb,wikitext-2}
                        The dataset used for this model.
"
}

version=
name=
dataset=
model_dir=
extra_args=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --version) shift; version=$1 ;;
    --name) shift; name=$1 ;;
    --dataset) shift; dataset=$1 ;;
    --) shift; extra_args=("$@"); break ;;
    *)
      if [[ ! $model_dir ]]; then
        model_dir=$1
      else
        usage >&2
        exit 1
      fi
      ;;
  esac
  shift
done

if [[ ! $version || ! $name || ! $model_dir || ! ( $dataset = ptb || $dataset = wikitext-2 ) ]]; then
  usage >&2
  exit 1
fi

cp -r lm_zoo_util/image/. "$model_dir"
cd "$model_dir"
echo "$dataset" > dataset.txt
case $dataset in
  ptb) vocab_file=/app/data/mikolov-ptb/vocab.txt ;;
  wikitext-2) vocab_file=/app/data/wikitext-2/vocab.txt ;;
  *) exit 1 ;;
esac
echo "$vocab_file" > vocab-file.txt
echo "${extra_args[@]}" > extra-args.txt
DOCKER_BUILDKIT=1 docker build \
  -t "$name":"$version" \
  -t "$name":"$version" \
  --build-arg VERSION="$version" \
  --build-arg NAME="$name" \
  .
