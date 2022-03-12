from .vocabulary import Vocabulary
from ._section import SectionMixin, add

class FallbackVocabulary(Vocabulary, SectionMixin):

    def __init__(self, main, fallback):
        if not main.is_frozen():
            raise ValueError('main vocabulary must be frozen')
        super().__init__()
        self.main = main
        self.fallback = fallback
        self.offset = main.size()

    def insert(self, s):
        return not self.main.contains(s) and self.fallback.insert(s)

    def contains(self, s):
        return self.main.contains(s) or self.fallback.contains(s)

    def index(self, s):
        try:
            return self.main.as_index(s)
        except KeyError:
            return self.transform_output_index(self.fallback.index(s))

    def as_index(self, s):
        try:
            return self.main.as_index(s)
        except KeyError:
            return self.transform_output_index(self.fallback.as_index(s))

    def value(self, i):
        if i < self.offset:
            return self.main.value(i)
        else:
            return self.fallback.value(i - self.offset)

    def serialize(self):
        return self.fallback.serialize()

    def size(self):
        return self.offset + self.fallback.size()

    def is_frozen(self):
        return self.fallback.is_frozen()

    def transform_output_index(self, i):
        return add(i, self.offset)

    def section_vocabulary(self, section):
        if section == 0:
            return self.main
        else:
            return self.fallback
