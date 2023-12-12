from ase import Atoms
from .atom_tag_manager import AtomTagManager
from scipy.spatial import KDTree
import numpy as np 

from .util import minimum_image_shift

class System:
    def __init__(self, atoms: Atoms, expand_box=True):
        self.atoms = atoms
        self.bonds = [] # These should be with respect to tags 
        # Box lengths in all three dimensions 
        self.box_lengths = atoms.cell.cellpar()[:3]
        # Shift all the atoms so that they are inside a box starting from 0 to box length
        self._shift_all_positions_into_box()
        # Create unique tags in the Atoms object
        self._generate_unique_tags()
        # Associate tags with indices in the Atoms object
        # The dictionary must be updated every time Atoms is rearranged
        self.tag_manager = AtomTagManager(self.atoms)
        # For 'expanded' simulation boxes
        if expand_box:
            self.is_expanded_box = True
            self.expanded_atoms = self.atoms.copy()
            # Keep track of the next available tag for expanded_atoms
            self.next_expanded_tag = len(self.atoms) + 1
            self.expanded_tag_manager = AtomTagManager(self.expanded_atoms)
        else:
            self.is_expanded_box = False

    def _generate_unique_tags(self):
        """
        Generate unique tags for each Atom in self.atoms if not already provided.
        Tags are numbered from 1 (not 0) in increasing order, similar in spirit to,
        for example, LAMMPS IDs
        """
        # Check for uniqueness 
        if len(set(self.atoms.get_tags())) != len(self.atoms):
            [atom.set_tags([index + 1]) for index, atom in enumerate(self.atoms)]

    def _shift_all_positions_into_box(self):
        """
        Shift all the positions by the minimum coordinate in x, y, z
        You need to do this or else SciPy will error out when 
        applying periodic boundary conditions if coordinates are 
        outside the box (0,0,0)(xboxlength, yboxlength,zboxlength).
        We use SciPy to find the nearest neighbours 
        """
        xlo = np.min(self.atoms.get_positions()[:,0])
        ylo = np.min(self.atoms.get_positions()[:,1])
        zlo = np.min(self.atoms.get_positions()[:,2])
        self.atoms.translate([-xlo,-ylo,-zlo])

    def add_expanded_box_atoms(self, query_pnt, neigh_atoms:Atoms):
        """
        Given a query point and neighbouring atoms (in an Atoms object), 
        add any Atom object and add to the expanded box if the unwrapped distance
        is greater than half the box length.
        This is needed more for visualization 
        """

        if not self.is_expanded_box:
            print("You cannot expand the box unless you allow it\n")
            return

        added_new_atoms = False
        unwrapped_atoms = Atoms()

        for index,atom in enumerate(neigh_atoms):
            # Get the position, wrapped into the simulation box
            wrapped_position = minimum_image_shift(atom.position, query_pnt, self.box_lengths)
            if np.linalg.norm(wrapped_position - position) > 1e-5:  # Adjust this threshold if needed
                # Add the wrapped atom to self.expanded_atoms with an updated tag
                new_atom = self.atoms[index].copy()
                new_atom.set_tags([self.next_expanded_tag])
                self.expanded_atoms.append(wrapped_atom)

                # Update the next available tag for expanded_atoms
                self.next_expanded_tag += 1

                added_new_atoms = True # Set this flag
                unwrapped_atoms.append(wrapped_atom)
            else:
                unwrapped_atoms.append(atom)

            if added_new_atoms:
                # Rearrage the expanded atom tags 
                self.expanded_tag_manager.rearrange_atoms(self.expanded_atoms)
                return unwrapped_atoms
            else:
                return neigh_atoms

    def create_solvation_shell(self, central_pnt_pos, solvent_atom_types, num_neighbours=6):
        from .solvation_shell import SolvationShell
        subset_atoms = self.atoms[atom_indices]
        return SolvationShell(subset_atoms, self.bonds)