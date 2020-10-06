import math

class EarlyStopping:

    def __init__(self, mode, patience=math.inf, goal=None):
        r"""A patience of 0 means that training will stop as soon as the score
        stops improving."""
        super().__init__()
        if mode not in ('min', 'max'):
            raise ValueError
        self.mode = mode
        if mode == 'min':
            self.is_better = lambda x, y: x < y
            self.best = math.inf
        else:
            self.is_better = lambda x, y: x > y
            self.best = -math.inf
        self.patience = patience
        self.updates_since_improvement = 0
        if goal is None:
            if mode == 'min':
                goal = -math.inf
            else:
                goal = math.inf
        self.goal = goal

    def update(self, value):
        is_best = self.is_better(value, self.best)
        if is_best:
            self.best = value
            self.updates_since_improvement = 0
        else:
            self.updates_since_improvement += 1
        # Stop if patience has run out or if this update achieved a perfect
        # score.
        should_stop = (
            self.updates_since_improvement > self.patience or
            not self.is_better(self.goal, value))
        return is_best, should_stop
