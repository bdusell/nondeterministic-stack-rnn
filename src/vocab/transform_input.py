from .vocabulary import Vocabulary

class TransformInputVocabulary(Vocabulary):

    def insert(self, s):
        return super().insert(self.transform_input(s))

    def contains(self, s):
        return super().contains(self.transform_input(s))

    def index(self, s):
        return super().index(self.transform_input(s))

    def as_index(self, s):
        return super().as_index(self.transform_input(s))

    def transform_input(self):
        raise NotImplementedError
