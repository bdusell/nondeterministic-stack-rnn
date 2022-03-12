# Perform some one-time setup tasks. This script should be run *inside the
# Docker/Singularity container*. Although the container includes the software
# environment necessary to run the code, this script is still needed to install
# some Python packages in the local directory and preprocess some data.

set -e
set -o pipefail

poetry install
bash scripts/download-ptb.bash
