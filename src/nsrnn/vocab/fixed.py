from .vocabulary import Vocabulary

class FixedVocabulary(Vocabulary):

    def __init__(self, values):
        super().__init__()
        self.values = values

    def contains(self, s):
        return False

    def value(self, i):
        return self.values[i]

    def serialize(self):
        return self.values

    def size(self):
        return len(self.values)

    def is_frozen(self):
        return True
