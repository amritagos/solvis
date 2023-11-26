from ase import Atoms
from .system import System 

class SolvationShell(System):
    """
    Takes the type of the central atom (in LAMMPS there are integral numbers) and the solvent atom type
    along with ASE atoms and bonds (bonds may be empty). This should be initialized such that there are not multiple ions
    If there are multiple atoms with the central atom type, print a warning   
    """
    def __init__(self, atoms: Atoms, bonds, central_atom_type, solvent_atom_type):
        super().__init__(atoms)
        self.bonds = bonds
        self.central_atom_type = central_atom_type
        self.solvent_atom_type = solvent_atom_type

    def build_convex_hull(self):
    """
    Takes the type of the central atom (in LAMMPS there are integral numbers) and the solvent atom type  
    """ 

    def calculate_properties(self):
        # Implement methods to calculate iod, cn, r7dist based on the current atoms and bonds
        # Assign the calculated values to self.iod, self.cn, self.r7dist
        pass