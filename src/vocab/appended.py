from .vocabulary import Vocabulary
from ._section import SectionMixin, add

class AppendedVocabulary(Vocabulary, SectionMixin):

    def __init__(self, first, second, use_first=False):
        if not first.is_frozen():
            raise ValueError('first vocabulary must be frozen')
        super().__init__()
        self.first = first
        self.second = second
        self.offset = first.size()
        self.active = first if use_first else second

    def insert(self, s):
        return self.active.insert(s)

    def contains(self, s):
        return self.active.contains(s)

    def index(self, s):
        return self.transform_output_index(self.active.index(s))

    def as_index(self, s):
        return self.transform_output_index(self.active.as_index(s))

    def value(self, i):
        if i < self.offset:
            return self.first.value(i)
        else:
            return self.second.value(i - self.offset)

    def serialize(self):
        return self.active.serialize()

    def size(self):
        return self.offset + self.second.size()

    def is_frozen(self):
        return self.second.is_frozen()

    def transform_output_index(self, i):
        if self.active is self.second:
            return add(i, self.offset)
        else:
            return i

    def section_vocabulary(self, section):
        if section == 0:
            return self.first
        else:
            return self.second
