from ase import Atoms
# from .solvation_shell import SolvationShell
from scipy.spatial import KDTree

class System:
    def __init__(self, atoms: Atoms):
        self.atoms = atoms
        self.bonds = []
        # Box lengths in all three dimensions 
        self.box = currentframe.get_cell_lengths_and_angles()[:3]

    def create_solvation_shell(self, central_pnt_pos, solvent_atom_types, num_neighbours=6):
        from .solvation_shell import SolvationShell
        subset_atoms = self.atoms[atom_indices]
        return SolvationShell(subset_atoms, self.bonds)