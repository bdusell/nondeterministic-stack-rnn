class SectionMixin:

    def index_section(self, index):
        return int(index >= self.offset)

    def index_is_in_section_single(self, section, index):
        return self.index_section(index) == section

    def remap_index_single(self, section, index):
        vocab = self.section_vocabulary(section)
        index = self.adjust_section_index(section, index)
        return vocab, index

    def adjust_section_index(self, section, index):
        if section == 1:
            index -= self.offset
        return index

    def index_is_in_section(self, sections, index):
        vocab, index = self.remap_index(sections[:-1], index)
        return (
            vocab is not None and
            vocab.index_is_in_section_single(sections[-1], index)
        )

    def remap_index(self, sections, index):
        vocab = self
        for section in sections:
            if vocab.index_is_in_section_single(section, index):
                vocab, index = vocab.remap_index_single(section, index)
            else:
                return None, None
        return vocab, index

def add(result, offset):
    if isinstance(result, tuple):
        return tuple(x + offset for x in result)
    else:
        return result + offset
