# Pull the public Docker image. This is generally faster than building it from
# scratch.

set -e
set -o pipefail

. scripts/variables.bash

docker pull bdusell/"$IMAGE":latest
docker tag bdusell/"$IMAGE":latest "$IMAGE":latest
