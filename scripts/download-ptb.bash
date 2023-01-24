set -e

mkdir -p data/mikolov-ptb
cd data/mikolov-ptb
curl -O http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
tar xzf simple-examples.tgz --strip-components 3 ./simple-examples/data
cd ../../src
python natural_language_modeling/build_vocab.py \
  --data ../data/mikolov-ptb/ptb.train.txt \
  --output ../data/mikolov-ptb/vocab.txt
