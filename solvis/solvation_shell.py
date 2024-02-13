from ase import Atoms
from .system import System 
import numpy as np 
import math

from .util import minimum_image_shift
from .atom_tag_manager import AtomTagManager

class SolvationShell(System):
    """
    In a solvation shell, the atoms will now refer to solvent atoms. 
    center: Should be coordinates of the central point. If None, then a fake center 
    can be assigned. The solvation shell should have unwrapped coordinates.  
    """
    def __init__(self, solvent_atoms: Atoms, center=None):
        super().__init__(solvent_atoms, expand_box=False)
        # We created the following: self.atoms, self.bonds=[], 
        # self.box_lengths from solvent_atoms
        # self.tag_manager and self.is_expanded_box

        # If the center is not set, then assign a fake center 
        if center is None:
            # Calculate a fake center, assuming unwrapped coordinates 
            self.center =  self._create_fake_center()
        else:
            self.center = center 

    def _create_fake_center(self):
        """
        If a center is not provided, then a 'fake' center, using the atoms, is calculated.
        This function assumes that the positions are unwrapped. 
        """
        return np.mean(self.atoms.get_positions(), axis=0)

    def build_convex_hull(self, num_neighbours):
        """
        Takes the type of the central atom (in LAMMPS there are integral numbers) and the solvent atom type  
        """
        # TODO: rearrange in order of lowest distance to highest distance to center
        # rearrange the tags 
        pass 

    def calculate_properties(self):
        # Implement methods to calculate iod, cn, r7dist based on the current atoms and bonds
        # Assign the calculated values to self.iod, self.cn, self.r7dist
        pass

def create_solvation_shell_from_solvent(solvent_atoms: Atoms, box_lengths, center=None):
    """
    Create a SolvationShell with unwrapped coordinates, 
    which is not formed from a System, but directly from an ASE Atoms
    object with the desired solvent atoms 

    solvent_atoms: ASE object with just the solvent atoms
    box_lengths: Box lengths, needed to get unwrapped coordinates 
    center: Coordinates of the center; if not provided then the first coordinate
    in solvent_atoms is chosen as the anchor point in creating unwrapped coordinates 
    """

    # Shift all the positions by the minimum coordinate in x, y, z
    # but only if the coordinates lie outside the box (0,0,0)(xboxlength, yboxlength,zboxlength).
    # You need to do this or else SciPy will error out when 
    # applying periodic boundary conditions if coordinates are 
    # outside the box (0,0,0)(xboxlength, yboxlength,zboxlength).
    # We use SciPy to find the nearest neighbours 
    x_min, y_min, z_min = np.min(solvent_atoms.get_positions(), axis=0)
    x_max, y_max, z_max = np.max(solvent_atoms.get_positions(), axis=0)
        
    # Check if the minimum coordinates are less than 0 or 
    # if the maximum coordinates are greater than 0 
    if x_min < 0 or y_min < 0 or z_min < 0 or x_max > box_lengths[0] or y_max > box_lengths[1] or z_max > box_lengths[2]:
        solvent_atoms.translate([-x_min,-y_min,-z_min])

    # Clip the coordinates into the periodic box
    # Sometimes atoms seem to migrate slightly out of boxes in LAMMPS
    # This ensures that SciPy's k-nearest neighbours won't fail.
    pos = np.array(solvent_atoms.get_positions())
    for j in range(3):
        pos[:,j] = np.clip(pos[:,j], 0.0, box_lengths[j]*(1.0-1E-16))
    solvent_atoms.set_positions(pos)

    # Unwrap the clusters about a point
    if center is None:
        # If no center is provided,
        anchor_point = solvent_atoms.get_positions()[0] 
    else:
        anchor_point = center

    # Make sure the positions are "unwrapped"
    # with respect to the anchor point
    for index,atom in enumerate(solvent_atoms):
        unwrapped_position = minimum_image_shift(atom.position, anchor_point, box_lengths)
        # Adjust this threshold if needed; check if you need to update the position to 
        # the unwrapped position
        if np.linalg.norm(unwrapped_position - atom.position) > 1e-5:
            atom.position = unwrapped_position

    # Create the SolvationShell
    return SolvationShell(solvent_atoms, center=center)