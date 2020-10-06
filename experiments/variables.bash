MODELS=(lstm gref jm ns)
TASKS=(marked-reversal unmarked-reversal padded-reversal dyck hardest-cfl)
LEARNING_RATES=(0.01 0.005 0.001 0.0005)
TRIALS=($(seq 1 5))
HIDDEN_UNITS=20
TRAIN_LENGTH=40
TRAIN_LENGTH_RANGE=40
