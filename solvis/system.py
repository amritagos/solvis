from ase import Atoms
from .atom_tag_manager import AtomTagManager
from scipy.spatial import KDTree

class System:
    def __init__(self, atoms: Atoms):
        self.atoms = atoms
        self.bonds = [] # These should be with respect to tags 
        # Box lengths in all three dimensions 
        self.box = atoms.cell.cellpar()[:3]
        # Create unique tags in the Atoms object
        self._generate_unique_tags()
        # Associate tags with indices in the Atoms object
        # The dictionary must be updated every time Atoms is rearranged
        self.tag_manager = AtomTagManager(self.atoms)

    def _generate_unique_tags(self):
        """
        Generate unique tags for each Atom in self.atoms if not already provided.
        Tags are numbered from 1 (not 0) in increasing order, similar in spirit to,
        for example, LAMMPS IDs
        """
        # Check for uniqueness 
        if len(set(self.atoms.get_tags())) != len(self.atoms):
            [atom.set_tags([index + 1]) for index, atom in enumerate(self.atoms)]

    def create_solvation_shell(self, central_pnt_pos, solvent_atom_types, num_neighbours=6):
        from .solvation_shell import SolvationShell
        subset_atoms = self.atoms[atom_indices]
        return SolvationShell(subset_atoms, self.bonds)