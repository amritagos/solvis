from ase import Atom, Atoms
from ase.data import chemical_symbols
from typing import Any
import math

from solvis.solvation_shell import SolvationShell
from .util import (
    k_nearest_neighbours,
    minimum_image_distance,
    angle_between_points,
    nearest_neighbours_within_cutoff,
)


def find_intra_water_bonds(
    atoms: Atoms, o_type: int, h_type: int, box_len: Any
) -> list[list[int, int]]:
    """Gets the intramolecular OH bonds in water molecules

    Args:
        atoms (Atoms): ASE Atoms object
        o_type (int): Integer type (LAMMPS type for example) of the O atom (atomic number in ASE)
        h_type (int): Type of the H atom
        box_len (Any): Box size dimensions

    Returns:
        list[list[int, int]]: Intramolecular OH bonds, wherein each bond is represented by tags
    """
    bonds = []
    num_h_per_o = 2

    all_tags = atoms.get_tags()
    all_pos = atoms.get_positions()

    # Find all O atom indices in ASE atoms object
    o_ind = atoms.symbols.search(chemical_symbols[o_type])
    o_tags = all_tags[o_ind]
    o_pos = all_pos[o_ind]

    # Find all H atom indices in ASE atoms object
    h_ind = atoms.symbols.search(chemical_symbols[h_type])
    h_tags = all_tags[h_ind]
    h_pos = all_pos[h_ind]

    for index, (current_o_pos, o_tag) in enumerate(zip(o_pos, o_tags)):
        dist, neigh_ind = k_nearest_neighbours(
            h_pos, current_o_pos, num_h_per_o, box_len
        )
        bonded_h_atom_tags = h_tags[neigh_ind]
        for h_tag in bonded_h_atom_tags:
            bonds.append([o_tag, h_tag])

    return bonds


def find_all_hydrogen_bonds(
    solvation_shell: SolvationShell,
    donor_types: list[int],
    acceptor_types: list[int],
    h_types: list[int],
    donor_H_distance: float = 1.0,
    donor_acceptor_cutoff: float = 3.1,
    ignore_hydrogens: bool = False,
) -> list[list[int, int]]:
    """
    Can find hydrogen bonds between donor and acceptor atom types. The H atom type must be provided as well.
    Hydrogen bonds are found according to a geometric criterion. A donor (D) is an
    electronegative atom, which is bonded to at least one hydrogen. An acceptor (A)
    is an electronegative atom which accepts the hydrogen (H). The criterion is as
    follows: 1) the distance DA between the donor and acceptor should be less than
    or equal to 3.1 Angstroms, which corresponds to the first minimum in the O-O RDF
    of water, and 2) the angle HDA should be less than or equal to 30 degrees.

    Args:
        solvation_shell (SolvationShell): SolvationShell object
        donor_types (list[int]): List of all the integral types of the donor atom(s)
        acceptor_types (list[int]): List of all the integral types of the acceptor atom(s)
        h_types (list[int]): List of all the integral types of the H atoms
        donor_H_distance (float, optional): Intramolecular distance between the donor and H atom. Defaults to 1.0, which gets all the OH bonds in water.
        donor_acceptor_cutoff (float, optional): Distance between the donor and acceptor, taking into account the minimum image convention. Defaults to 3.1. I have found that 3.5 is usually too big a cutoff for water, and 3.0 is too small a cutoff when I have chloride ions in the system.
        ignore_hydrogens (bool, optional): If you set this to false, then the hydrogen bonds returned are between the H and the acceptor atom. Otherwise the bond is created between the acceptor and donor. Defaults to False.

    Returns:
        list[list[int, int]]: List of hydrogen bonds, wherein each bond is represented by tags
    """
    bonds = []  # will hold all found hydrogen bonds

    # Get the chemical symbols corresponding the acceptor types and H types
    donor_symbols = [chemical_symbols[x] for x in donor_types]
    acceptor_symbols = [chemical_symbols[x] for x in acceptor_types]
    h_symbols = [chemical_symbols[x] for x in h_types]

    all_tags = solvation_shell.atoms.get_tags()
    all_pos = solvation_shell.atoms.get_positions()

    # Tags, indices and positions
    donor_all_ind = solvation_shell.atoms.symbols.search(donor_symbols)
    donor_tags = solvation_shell.atoms.get_tags()[donor_all_ind]
    donor_pos = all_pos[donor_all_ind]

    acceptor_all_ind = solvation_shell.atoms.symbols.search(acceptor_symbols)
    acceptor_tags = all_tags[acceptor_all_ind]
    acceptor_pos = all_pos[acceptor_all_ind]

    h_all_ind = solvation_shell.atoms.symbols.search(h_symbols)
    h_tags = all_tags[h_all_ind]
    h_pos = all_pos[h_all_ind]

    # Loop through all donor positions
    for d_query_pos, donor_tag in zip(donor_pos, donor_tags):
        # Find neighbouring acceptor atoms within donor_acceptor_cutoff
        acceptor_neigh = nearest_neighbours_within_cutoff(
            acceptor_pos, d_query_pos, donor_acceptor_cutoff
        )

        for a_ind in acceptor_neigh:
            if donor_tag == acceptor_tags[a_ind]:
                continue

        # Find the closest H atoms to the donor atom within the D-H distance cutoff
        h_ind = nearest_neighbours_within_cutoff(h_pos, d_query_pos, donor_H_distance)

        for current_h in h_ind:
            angle = angle_between_points(
                h_pos[current_h], d_query_pos, acceptor_pos[a_ind]
            )

            if angle > 90:
                angle = 180 - angle

            if angle <= 30:
                bond = (
                    [acceptor_tags[a_ind], h_tags[current_h]]
                    if not ignore_hydrogens
                    else [acceptor_tags[a_ind], donor_tag]
                )
                bonds.append(bond)

    return bonds


def find_hydrogen_bonds_to_donor_or_acceptor(
    atom_tag: int,
    solvation_shell: SolvationShell,
    donor_types: list[int],
    acceptor_types: list[int],
    h_types: list[int],
    donor_H_distance: float = 1.0,
    donor_acceptor_cutoff: float = 3.1,
    ignore_hydrogens: bool = False,
) -> list[list[int, int]]:
    """
    Find hydrogen bonds emanating from or to a particular donor/acceptor in a solvation shell. The donor type must have H atoms bonded to it, within a distance of donor_H_distance. The atom_tag is
    also present in the solvation shell.

    Args:
        atom_tag (int): tag of the atom from which or to which hydrogen bonds will be found. This can be either a donor or an acceptor
        solvation_shell (SolvationShell): SolvationShell object
        donor_types (list[int]): List of all the integral types of the donor atom(s)
        acceptor_types (list[int]): List of all the integral types of the acceptor atom(s)
        h_types (list[int]): List of all the integral types of the H atoms
        donor_H_distance (float, optional): Intramolecular distance between the donor and H atom. Defaults to 1.0, which gets all the OH bonds in water.
        donor_acceptor_cutoff (float, optional): Distance between the donor and acceptor, taking into account the minimum image convention. Defaults to 3.1. I have found that 3.5 is usually too big a cutoff for water, and 3.0 is too small a cutoff when I have chloride ions in the system.
        ignore_hydrogens (bool, optional): If you set this to false, then the hydrogen bonds returned are between the H and the acceptor atom. Otherwise the bond is created between the acceptor and donor. Defaults to False.

    Returns:
        list[list[int, int]]: List of hydrogen bonds, wherein each bond is represented by tags
    """
    # Get the chemical symbols corresponding the acceptor types, donor types and H types
    donor_symbols = [chemical_symbols[x] for x in donor_types]
    acceptor_symbols = [chemical_symbols[x] for x in acceptor_types]
    h_symbols = [chemical_symbols[x] for x in h_types]

    all_tags = solvation_shell.atoms.get_tags()
    all_pos = solvation_shell.atoms.get_positions()

    # Get the position of the atom in question
    atom_query_index = solvation_shell.tag_manager.lookup_index_by_tag(atom_tag)
    query_pos = all_pos[atom_query_index]  # donor/acceptor position

    # Find all acceptor, donor and H atom indices in ASE atoms object
    acceptor_all_ind = solvation_shell.atoms.symbols.search(acceptor_symbols)
    acceptor_tags = all_tags[acceptor_all_ind]
    acceptor_pos = all_pos[acceptor_all_ind]

    donor_all_ind = solvation_shell.atoms.symbols.search(donor_symbols)
    donor_tags = all_tags[donor_all_ind]
    donor_pos = all_pos[donor_all_ind]

    h_all_ind = solvation_shell.atoms.symbols.search(h_symbols)
    h_tags = all_tags[h_all_ind]
    h_pos = all_pos[h_all_ind]

    # Find neighbouring acceptor atoms within donor_acceptor_cutoff
    acceptor_neigh = nearest_neighbours_within_cutoff(
        acceptor_pos, query_pos, donor_acceptor_cutoff
    )
    donor_neigh = nearest_neighbours_within_cutoff(
        donor_pos, query_pos, donor_acceptor_cutoff
    )

    bonds = []

    # Assume that the atom is a donor. Search for acceptors
    for a_ind in acceptor_neigh:
        if atom_tag == acceptor_tags[a_ind]:
            continue  # atom is the same as acceptor here so skip

        # Find the closest H atoms to the donor atom within the D-H distance cutoff
        h_ind = nearest_neighbours_within_cutoff(h_pos, query_pos, donor_H_distance)

        for current_h in h_ind:
            angle = angle_between_points(
                h_pos[current_h], query_pos, acceptor_pos[a_ind]
            )

            if angle > 90:
                angle = 180 - angle

            if angle <= 30:
                bond = (
                    [acceptor_tags[a_ind], h_tags[current_h]]
                    if not ignore_hydrogens
                    else [acceptor_tags[a_ind], atom_tag]
                )
                bonds.append(bond)

    # Assume that the atom is an acceptor. Search for donors
    for d_ind in donor_neigh:
        if atom_tag == donor_tags[d_ind]:
            continue

        # Find the closest H atoms to the donor atom within the D-H distance cutoff
        h_ind = nearest_neighbours_within_cutoff(
            h_pos, donor_pos[d_ind], donor_H_distance
        )

        for current_h in h_ind:
            # Angle HDA
            angle = angle_between_points(h_pos[current_h], donor_pos[d_ind], query_pos)

            if angle > 90:
                angle = 180 - angle

            if angle <= 30:
                bond = (
                    [atom_tag, h_tags[current_h]]
                    if not ignore_hydrogens
                    else [atom_tag, donor_tags[d_ind]]
                )
                bonds.append(bond)

    return bonds
