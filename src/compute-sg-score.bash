set -e
set -o pipefail

usage() {
  echo "Usage: $0 <saved-model-dir> <log-dir> [options] [<args>...]

  <saved-model-dir>     Directory containing a saved model.
  <log-dir>             Directory where logs for computing the SG score will
                        be saved.
  --dataset {ptb,wikitext-2}
                        The dataset used for this model.
  --hide-stderr         Do not show stderr from syntaxgym.
  --no-build-docker-image
                        Do not try to rebuild the base Docker image.
  <args>                Extra arguments passed to get_surprisals.py.
"
}

model_dir=
log_dir=
build_docker_image=true
build_lm_zoo_image_args=()
compute_image_sg_score_args=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --dataset) build_lm_zoo_image_args+=("$1" "$2"); shift ;;
    --hide-stderr) compute_image_sg_score_args+=("$1") ;;
    --no-build-docker-image) build_docker_image=false ;;
    *)
      if [[ ! $model_dir ]]; then
        model_dir=$1
      elif [[ ! $log_dir ]]; then
        log_dir=$1
      else
        break
      fi
      ;;
  esac
  shift
done
extra_args=("$@")

if $build_docker_image; then
  (cd .. && bash scripts/build-docker-image.bash)
fi
model_dir_hash=$(md5sum <<<"$model_dir" | cut -d ' ' -f 1)
name=sg-model
version=$model_dir_hash
image=$name:$version
bash build-lm-zoo-image.bash \
  --name "$name" \
  --version "$version" \
  "${build_lm_zoo_image_args[@]}" \
  "$model_dir" \
  -- \
  "${extra_args[@]}" \
  > /dev/null
bash compute-image-sg-score.bash \
  "$image" \
  "$log_dir" \
  "${compute_image_sg_score_args[@]}"
docker image rm -f "$image" > /dev/null
