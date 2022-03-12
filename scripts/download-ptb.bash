# Download and preprocess the standard Penn Treebank (PTB) language modeling
# dataset as preprocessed by Mikolov et al. (2011). This script should be run
# *inside the Docker/Singularity container*.

set -e
set -o pipefail

mkdir -p data/mikolov-ptb
cd data/mikolov-ptb
curl -O http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
tar xzf simple-examples.tgz --strip-components 3 ./simple-examples/data
cd ../../src
poetry run python build_vocab.py \
  --data ../data/mikolov-ptb/ptb.train.txt \
  --output ../data/mikolov-ptb/vocab.txt
