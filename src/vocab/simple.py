from .vocabulary import Vocabulary

class SimpleVocabulary(Vocabulary):

    def __init__(self, values=None):
        super().__init__()
        if values is None:
            values = []
        value_to_index = { v : i for i, v in enumerate(values) }
        self.index_to_value = values
        self.value_to_index = value_to_index

    def insert(self, s):
        inserted = s not in self.value_to_index
        if inserted:
            self.value_to_index[s] = len(self.index_to_value)
            self.index_to_value.append(s)
        return inserted

    def contains(self, s):
        return s in self.value_to_index

    def index(self, s):
        result = self.value_to_index.get(s)
        if result is None:
            result = self.value_to_index[s] = len(self.index_to_value)
            self.index_to_value.append(s)
        return result

    def as_index(self, s):
        return self.value_to_index[s]

    def value(self, i):
        return self.index_to_value[i]

    def serialize(self):
        return self.index_to_value

    def size(self):
        return len(self.index_to_value)
