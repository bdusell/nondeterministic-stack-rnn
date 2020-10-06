set -e
set -o pipefail

. scripts/variables.bash

usage() {
  echo "Usage: $0 [options]

Ensure that the Docker image has been created.
  
Options:
  --pull    Pull the public Docker image.
  --build   Build the Docker image from scratch.
"
}

mode=none
while [[ $# -gt 0 ]]; do
  case $1 in
    --pull) mode=pull ;;
    --build) mode=build ;;
    *) usage >&2; exit 1 ;;
  esac
  shift
done

case $mode in
  none) ;;
  pull) bash scripts/pull-docker-image.bash ;;
  build) bash scripts/build-docker-image.bash ;;
esac
