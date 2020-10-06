class Curriculum:

    def step(self):
        raise NotImplementedError

    def data(self):
        raise NotImplementedError

    def done(self):
        raise NotImplementedError

class RandomShuffling(Curriculum):

    def __init__(self, data, generator):
        super().__init__()
        self._data = data
        self.generator = generator

    def step(self):
        self.generator.shuffle(self._data)

    def data(self):
        return self._data

    def done(self):
        return True
