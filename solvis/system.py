from typing import Union
from ase import Atoms
from .atom_tag_manager import AtomTagManager
import numpy as np
import math
import numpy.typing as npt
import warnings

from .util import minimum_image_shift, k_nearest_neighbours


class System:
    def __init__(self, atoms: Atoms, expand_box: bool = True):
        self.atoms = atoms
        self.bonds = []  # These should be with respect to tags
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

    def _generate_unique_tags(self) -> None:
        """
        Generate unique tags for each Atom in self.atoms if not already provided.
        Tags are numbered from 1 (not 0) in increasing order, similar in spirit to,
        for example, LAMMPS IDs
        """
        # Check for uniqueness
        if len(set(self.atoms.get_tags())) != len(self.atoms):
            self.atoms.set_tags([index + 1 for index, _ in enumerate(self.atoms)])

    def _shift_all_positions_into_box(self) -> None:
        """
        Shift all the positions by the minimum coordinate in x, y, z
        but only if the coordinates lie outside the box (0,0,0)(xboxlength, yboxlength,zboxlength).
        You need to do this or else SciPy will error out when
        applying periodic boundary conditions if coordinates are
        outside the box (0,0,0)(xboxlength, yboxlength,zboxlength).
        We use SciPy to find the nearest neighbours
        """
        # Do nothing if pbcs have not been set
        if False in self.atoms.pbc:
            if self.__class__.__name__ == System:
                warnings.warn(f"PBCs are set to false, for {self.__class__.__name__}")
            return

        x_min, y_min, z_min = np.min(self.atoms.get_positions(), axis=0)
        x_max, y_max, z_max = np.max(self.atoms.get_positions(), axis=0)

        # Check if the minimum coordinates are less than 0 or
        # if the maximum coordinates are greater than 0
        if (
            x_min < 0
            or y_min < 0
            or z_min < 0
            or x_max > self.box_lengths[0]
            or y_max > self.box_lengths[1]
            or z_max > self.box_lengths[2]
        ):
            self.atoms.translate([-x_min, -y_min, -z_min])

        # Clip the coordinates into the periodic box
        pos = np.array(self.atoms.get_positions())
        for j in range(3):
            pos[:, j] = np.clip(pos[:, j], 0.0, self.box_lengths[j] * (1.0 - 1e-16))

        self.atoms.set_positions(pos)

    def add_expanded_box_atoms(
        self, query_pnt: npt.ArrayLike, neigh_atoms: Atoms
    ) -> Atoms:
        """Given a query point and neighbouring atoms (in an Atoms object),
        add any Atom object and add to the expanded box if the unwrapped distance
        is greater than half the box length.
        This is needed more for visualization

        Args:
            query_pnt (npt.ArrayLike): Query point coordinates
            neigh_atoms (Atoms): ASE Atoms object for the neighbouring atoms

        Returns:
            Atoms: Expanded ASE atoms object
        """

        if not self.is_expanded_box:
            print("You cannot expand the box unless you allow it\n")
            return neigh_atoms

        added_new_atoms = False
        unwrapped_atoms = Atoms()

        for index, atom in enumerate(neigh_atoms):
            # Get the position, unwrapped and with respect to the query point
            unwrapped_position = minimum_image_shift(
                atom.position, query_pnt, self.box_lengths
            )
            if (
                np.linalg.norm(unwrapped_position - atom.position) > 1e-5
            ):  # Adjust this threshold if needed
                # Add the new atom to self.expanded_atoms with an updated tag
                new_atom = atom
                new_atom.position = unwrapped_position
                new_atom.tag = self.next_expanded_tag
                self.expanded_atoms.append(new_atom)

                # Update the next available tag for expanded_atoms
                self.next_expanded_tag += 1

                added_new_atoms = True  # Set this flag
                unwrapped_atoms.append(new_atom)
            else:
                unwrapped_atoms.append(atom)

        if added_new_atoms:
            # Rearrage the expanded atom tags
            self.expanded_tag_manager.update_tag_index_mapping(self.expanded_atoms)
            return unwrapped_atoms
        else:
            return neigh_atoms

    def create_solvation_shell_from_center(
        self,
        central_pnt: Union[int, npt.ArrayLike],
        num_neighbours: int,
        contains_solvation_center: bool = False,
        solvent_atom_types: Union[str, list[int]] = "all",
    ):
        """Creates a SolvationShell as a subsystem from System, given a central point.

        Args:
            central_pnt (Union[int, npt.ArrayLike]): If int, this is the index of the Atom in atoms. If
        np.ndarray, it is the coordinates of the central point
            num_neighbours (int): Number of neighbours to return in the solvation shell
            contains_solvation_center (bool, optional): If this is set to True, then it is assumed that
        the center is part of solvent_atoms. An additional check is performed to remove the first neighbour
        if the distance is within 1e-4 of 0.0. Defaults to False.
            solvent_atom_types (Union[str, list[int]], optional): Neighbours will be searched in this subset of atoms. Defaults to "all".

        Returns:
            SolvationShell
        """
        from .solvation_shell import SolvationShell

        if type(central_pnt) is int:
            central_pnt_pos = self.atoms.get_positions()[central_pnt]
        elif type(central_pnt) is np.ndarray:
            central_pnt_pos = central_pnt
        else:
            print("You've used an invalid value for the central_pnt argument\n")
            return
        # Get the Atoms object with all the solvent positions
        if solvent_atom_types == "all":
            solvent_atoms = self.atoms
        else:
            # Only when types (numbers) are given in a list, will fail otherwise
            solvent_atoms = self.atoms[
                [atom.index for atom in self.atoms if atom.number in solvent_atom_types]
            ]

        # Find the k-nearest neighbours from the central point
        if contains_solvation_center:
            dist, neigh_ind = k_nearest_neighbours(
                solvent_atoms.get_positions(),
                central_pnt_pos,
                num_neighbours + 1,
                self.box_lengths,
            )
            if math.isclose(dist[0], 0.0, rel_tol=1e-4):
                dist = dist[1:]
                neigh_ind = neigh_ind[1:]
            else:
                dist = dist[:num_neighbours]
                neigh_ind = neigh_ind[:num_neighbours]
        else:
            dist, neigh_ind = k_nearest_neighbours(
                solvent_atoms.get_positions(),
                central_pnt_pos,
                num_neighbours,
                self.box_lengths,
            )
        # Get an Atoms object with the neighbours such that the positions are unwrapped
        neigh_atoms = self.add_expanded_box_atoms(
            central_pnt_pos, solvent_atoms[neigh_ind]
        )

        return SolvationShell(neigh_atoms, center=central_pnt_pos)
