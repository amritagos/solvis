from ase import Atoms
from .solvation_shell import SolvationShell

class System:
    def __init__(self, atoms: Atoms):
        self.atoms = atoms
        self.bonds = []
        # Box lengths in all three dimensions 
        self.box = currentframe.get_cell_lengths_and_angles()[:3]

    def create_shell(self, atom_indices):
        subset_atoms = self.atoms[atom_indices]
        return SolvationShell(subset_atoms, self.bonds)