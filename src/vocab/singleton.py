from .vocabulary import Vocabulary

class SingletonVocabulary(Vocabulary):

    def __init__(self, value):
        super().__init__()
        self._value = value

    def insert(self, s):
        return False

    def contains(self, s):
        return True

    def index(self, s):
        return 0

    def as_index(self, s):
        return 0

    def value(self, i):
        return self._value

    def serialize(self):
        return None

    def size(self):
        return 1

    def is_frozen(self):
        return True
