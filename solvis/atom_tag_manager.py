from ase import Atoms


class AtomTagManager:
    def __init__(self, atoms: Atoms):

        self.tag_index_mapping = self._create_tag_index_mapping(atoms)

    def _create_tag_index_mapping(self, atoms: Atoms) -> dict[int, int]:
        """Create a dictionary mapping tags to indices and vice versa in the Atoms object.

        Args:
            atoms (Atoms): The ASE atoms object

        Returns:
            dict[int, int]: Key: tag, value: index
        """
        return {atom.tag: (index, atom.tag) for index, atom in enumerate(atoms)}

    def update_tag_index_mapping(self, atoms: Atoms) -> None:
        """Update the dictionary mapping tags to indices and vice versa based on the order of Atom objects in the Atoms object.

        Args:
            atoms (Atoms): The ASE Atoms object
        """
        self.tag_index_mapping = {
            atom.tag: (index, atom.tag) for index, atom in enumerate(atoms)
        }

    def lookup_index_by_tag(self, tag: int) -> int:
        """Look up the index of the Atom object with the specified tag.

        Args:
            tag (int): Desired tag

        Returns:
            int: Index corresponding to the input tag
        """

        return self.tag_index_mapping.get(tag, (None, None))[0]

    def lookup_tag_by_index(self, index: int) -> int:
        """Look up the tag of the Atom object at the specified index.

        Args:
            index (int): index for which you want the tag

        Returns:
            int: tag corresponding to the input index
        """
        return self.tag_index_mapping.get(index, (None, None))[1]
