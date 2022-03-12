class Vocabulary:

    def insert(self, s):
        raise NotImplementedError

    def contains(self, s):
        raise NotImplementedError

    def index(self, s):
        raise NotImplementedError

    def as_index(self, s):
        raise NotImplementedError

    def value(self, i):
        raise NotImplementedError

    def serialize(self):
        raise NotImplementedError

    def size(self):
        raise NotImplementedError

    def frozen(self):
        if self.is_frozen():
            return self
        else:
            return FrozenVocabulary(self)

    def is_frozen(self):
        return False

class FrozenVocabulary(Vocabulary):

    def __init__(self, original):
        self.original = original

    def insert(self, s):
        if not self.original.contains(s):
            raise ValueError('cannot insert new item into frozen vocabulary')

    def contains(self, s):
        return self.original.contains(s)

    def index(self, s):
        return self.original.as_index(s)

    def as_index(self, s):
        return self.original.as_index(s)

    def value(self, i):
        return self.original.value(i)

    def serialize(self):
        return self.original.serialize()

    def size(self):
        return self.original.size()

    def frozen(self):
        return self

    def is_frozen(self):
        return True

    def __getattr__(self, name):
        return getattr(self.original, name)
