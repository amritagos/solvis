from ase import Atoms

class AtomTagManager:
    def __init__(self, atoms):
        self.tag_index_mapping = self._create_tag_index_mapping(atoms)

    def _create_tag_index_mapping(self, atoms):
        """
        Create a dictionary mapping tags to indices and vice versa in the Atoms object.
        """
        return {atom.tag: (index, atom.tag) for index, atom in enumerate(atoms)}

    def update_tag_index_mapping(self, atoms):
        """
        Update the dictionary mapping tags to indices and vice versa based on the order of Atom objects in the Atoms object.
        """
        self.tag_index_mapping = {atom.tag: (index, atom.tag) for index, atom in enumerate(atoms)}

    def lookup_index_by_tag(self, tag):
        """
        Look up the index of the Atom object with the specified tag.
        """
        return self.tag_index_mapping.get(tag, (None, None))[0]

    def lookup_tag_by_index(self, index):
        """
        Look up the tag of the Atom object at the specified index.
        """
        return self.tag_index_mapping.get(index, (None, None))[1]