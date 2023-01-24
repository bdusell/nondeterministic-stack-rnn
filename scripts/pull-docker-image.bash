# Pull the public Docker image. This is generally faster than building it from
# scratch.

set -e
set -u

. scripts/variables.bash

docker pull bdusell/"$IMAGE":latest
docker tag bdusell/"$IMAGE":latest "$DEV_IMAGE":latest
