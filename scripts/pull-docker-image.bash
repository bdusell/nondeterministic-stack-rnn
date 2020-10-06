set -e
set -o pipefail

. scripts/variables.bash

docker pull bdusell/"$IMAGE":latest
docker tag bdusell/"$IMAGE":latest "$IMAGE":latest
