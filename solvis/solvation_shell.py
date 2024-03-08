from ase import Atoms
from .system import System
import numpy as np
import math
import warnings

from .util import (
    minimum_image_shift,
    k_nearest_neighbours,
    nearest_neighbours_within_cutoff,
)
from .atom_tag_manager import AtomTagManager
from .geometric_utils import ConvexHull


class SolvationShell(System):
    """
    In a solvation shell, the atoms will now refer to solvent atoms.
    center: Should be coordinates of the central point. If None, then a fake center
    can be assigned. The solvation shell should have unwrapped coordinates.
    Solvent atoms are arranged in ascending order of increasing distance from the center
    """

    def __init__(self, solvent_atoms: Atoms, center=None):
        super().__init__(solvent_atoms, expand_box=False)
        # We created the following: self.atoms, self.bonds=[],
        # self.box_lengths from solvent_atoms
        # self.tag_manager and self.is_expanded_box

        # If the center is not set, then assign a fake center
        if center is None:
            # Calculate a fake center, assuming unwrapped coordinates
            self.center = self._create_fake_center()
        else:
            self.center = center

    def _create_fake_center(self):
        """
        If a center is not provided, then a 'fake' center, using the atoms, is calculated.
        This function assumes that the positions are unwrapped.
        """
        return np.mean(self.atoms.get_positions(), axis=0)

    def build_convex_hull_k_neighbours(self, num_neighbours):
        """
        Returns a convex hull, created from k nearest neighbours.
        This function does not check the atom type
        """
        # k should not be greater than the number of solvent atoms, and should be greater than 0
        natoms = len(self.atoms)
        try:
            k = num_neighbours
            if num_neighbours > natoms:
                raise ValueError(
                    "k cannot be greater than the number of solvent atoms.\n Setting to maximum value."
                )
        except ValueError as error:
            print(error)
            k = natoms

        # Find k-nearest neighbours and build the convex hull
        pos = self.atoms.get_positions()
        k_nearest_pos = pos[:k]
        convex_hull = ConvexHull(k_nearest_pos)

        return convex_hull

    def calculate_coordination_number_from_center(
        self, cutoff, coordinating_type="all"
    ):
        """
        Calculates the coordination number from the center. The coordinating_type should either
        be the default, 'all', or a list of numbers corresponding to the type or atomic number.
        """
        # Get the Atoms object with all the solvent positions
        if coordinating_type == "all":
            surrounding_atoms = self.atoms
        else:
            # Only when types (numbers) are given in a list, will fail otherwise
            surrounding_atoms = self.atoms[
                [atom.index for atom in self.atoms if atom.number in coordinating_type]
            ]

        # If the cutoff is larger than the distance of the furthest atom in the solvation shell
        # you have a problem, print a warning and change the cutoff
        furthest_pos = surrounding_atoms.get_positions()[-1]
        # The distance is already unwrapped with respect to the center
        r_max = np.linalg.norm(furthest_pos - self.center)
        try:
            if r_max < cutoff:
                raise ValueError(
                    "The cutoff is bigger than the distance of the furthest atom from the center in the solvation center.\n Setting cutoff to maximum distance {}".format(
                        r_max
                    )
                )
        except ValueError as error:
            print(error)
            cutoff = r_max * (1 + 1e-12)
            print("The cutoff has been set to ", cutoff)

        neigh_ind = nearest_neighbours_within_cutoff(
            surrounding_atoms.get_positions(), self.center, cutoff, self.box_lengths
        )
        # Return the coordination number
        return len(neigh_ind)

    def ravg_of_k_neighbours_from_center(self, num_neighbours, coordinating_type="all"):
        """
        Gets the average distance of k neighbours from the center.
        """
        # k should not be greater than the number of solvent atoms, and should be greater than 0
        # Get the Atoms object with all the solvent positions
        if coordinating_type == "all":
            surrounding_atoms = self.atoms
        else:
            # Only when types (numbers) are given in a list, will fail otherwise
            surrounding_atoms = self.atoms[
                [atom.index for atom in self.atoms if atom.number in coordinating_type]
            ]
        # k should not be greater than the number of solvent atoms, and should be greater than 0
        natoms = len(surrounding_atoms)
        try:
            k = num_neighbours
            if num_neighbours > natoms:
                raise ValueError(
                    "k cannot be greater than the number of solvent atoms.\n Setting to maximum value."
                )
        except ValueError as error:
            print(error)
            k = natoms

        pos = surrounding_atoms.get_positions()[:k]
        dist = np.linalg.norm(pos - self.center, axis=1)
        # Return the average of these distances
        return np.mean(dist)

    def r_of_k_th_neighbour_from_center(self, k, coordinating_type="all"):
        """
        Gets radial distance of k-th neighbour from the center.
        """
        # Get the Atoms object with all the solvent positions
        if coordinating_type == "all":
            surrounding_atoms = self.atoms
        else:
            # Only when types (numbers) are given in a list, will fail otherwise
            surrounding_atoms = self.atoms[
                [atom.index for atom in self.atoms if atom.number in coordinating_type]
            ]
        # k should not be greater than the number of solvent atoms, and should be greater than 0
        natoms = len(surrounding_atoms)
        try:
            if k > natoms or k < 1:
                raise ValueError(
                    "k cannot be less than 1 or greater than the number of solvent atoms.\n Setting to maximum value."
                )
        except ValueError as error:
            print(error)
            k = natoms

        k_pos = surrounding_atoms.get_positions()[k - 1]
        # The distance is already unwrapped with respect to the center
        r = np.linalg.norm(k_pos - self.center)
        return r

    def distances_of_k_neighbours_from_center(
        self, num_neighbours, coordinating_type="all"
    ):
        """
        Gets a numPy array of distances of k neighbours from the center.
        """
        # k should not be greater than the number of solvent atoms, and should be greater than 0
        # Get the Atoms object with all the solvent positions
        if coordinating_type == "all":
            surrounding_atoms = self.atoms
        else:
            # Only when types (numbers) are given in a list, will fail otherwise
            surrounding_atoms = self.atoms[
                [atom.index for atom in self.atoms if atom.number in coordinating_type]
            ]
        # k should not be greater than the number of solvent atoms, and should be greater than 0
        natoms = len(surrounding_atoms)
        try:
            k = num_neighbours
            if num_neighbours > natoms:
                raise ValueError(
                    "k cannot be greater than the number of solvent atoms.\n Setting to maximum value."
                )
        except ValueError as error:
            print(error)
            k = natoms

        pos = surrounding_atoms.get_positions()[:k]
        dist = np.linalg.norm(pos - self.center, axis=1)
        return dist


def create_solvation_shell_from_solvent(solvent_atoms: Atoms, box_lengths, center=None):
    """
    Create a SolvationShell with unwrapped coordinates,
    which is not formed from a System, but directly from an ASE Atoms
    object with the desired solvent atoms. These will be arranged in order of distance
    from the center

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

    # Unwrap the clusters about a point
    if center is None:
        # If no center is provided,
        anchor_point = solvent_atoms.get_positions()[0]
    else:
        anchor_point = center

    # Make sure the positions are "unwrapped"
    # with respect to the anchor point
    for index, atom in enumerate(solvent_atoms):
        unwrapped_position = minimum_image_shift(
            atom.position, anchor_point, box_lengths
        )
        # Adjust this threshold if needed; check if you need to update the position to
        # the unwrapped position
        if np.linalg.norm(unwrapped_position - atom.position) > 1e-5:
            atom.position = unwrapped_position

    x_min, y_min, z_min = np.min(solvent_atoms.get_positions(), axis=0)
    x_max, y_max, z_max = np.max(solvent_atoms.get_positions(), axis=0)

    # Check if the minimum coordinates are less than 0 or
    # if the maximum coordinates are greater than 0
    if (
        x_min < 0
        or y_min < 0
        or z_min < 0
        or x_max > box_lengths[0]
        or y_max > box_lengths[1]
        or z_max > box_lengths[2]
    ):
        solvent_atoms.translate([-x_min, -y_min, -z_min])

    # Clip the coordinates into the periodic box
    # Sometimes atoms seem to migrate slightly out of boxes in LAMMPS
    # This ensures that SciPy's k-nearest neighbours won't fail.
    pos = np.array(solvent_atoms.get_positions())
    for j in range(3):
        pos[:, j] = np.clip(pos[:, j], 0.0, box_lengths[j] * (1.0 - 1e-16))
    solvent_atoms.set_positions(pos)

    # Arrange in order of distance from a fake center
    pos = solvent_atoms.get_positions()
    natoms = len(solvent_atoms)
    fake_center = np.mean(pos, axis=0)
    dist, neigh_ind = k_nearest_neighbours(pos, fake_center, natoms, box_lengths)
    solvent_atoms = solvent_atoms[neigh_ind]

    # Create the SolvationShell
    return SolvationShell(solvent_atoms, center=center)
