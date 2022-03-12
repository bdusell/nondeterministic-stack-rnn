MODELS=(lstm gref jm ns ns+s ns+u ns+s+u)
TASKS=(marked-reversal unmarked-reversal padded-reversal dyck hardest-cfl)
LEARNING_RATES=(0.01 0.005 0.001 0.0005)
TRIALS=({1..5})
HIDDEN_UNITS=20
TRAIN_LENGTH=40
TRAIN_LENGTH_RANGE=40
TEST_LENGTH_RANGE=60
MODEL_LABELS=(LSTM Gref JM NS NS+S NS+U NS+S+U)
declare -A TASK_TITLES=( \
  [marked-reversal]='Marked Reversal' \
  [unmarked-reversal]='Unmarked Reversal' \
  [padded-reversal]='Padded Reversal' \
  [dyck]='Dyck' \
  [hardest-cfl]='Hardest CFL' \
)
DATA_DIR=data
