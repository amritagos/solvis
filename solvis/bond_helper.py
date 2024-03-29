from ase import Atom, Atoms
from ase.data import chemical_symbols
import math
from .util import (
    k_nearest_neighbours,
    minimum_image_distance,
    angle_between_points,
    nearest_neighbours_within_cutoff,
)


def find_intra_water_bonds(atoms, o_type, h_type, box_len):
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


def find_water_hydrogen_bonds(o_atom_tag, solvation_shell, o_type, h_type):
    """
    Find hydrogen bonds, in a solvation_shell with O and H types. The o_atom_tag is
    also present in the solvation shell
    """
    bonds = []  # List of atom tags

    all_tags = solvation_shell.atoms.get_tags()
    all_pos = solvation_shell.atoms.get_positions()

    o_query_index = solvation_shell.tag_manager.lookup_index_by_tag(o_atom_tag)
    o_query_pos = all_pos[o_query_index]

    # Find all O atom indices in ASE atoms object
    o_all_ind = solvation_shell.atoms.symbols.search(chemical_symbols[o_type])
    o_tags = all_tags[o_all_ind]
    o_pos = all_pos[o_all_ind]

    # Find all H atom indices in ASE atoms object
    h_all_ind = solvation_shell.atoms.symbols.search(chemical_symbols[h_type])
    h_tags = all_tags[h_all_ind]
    h_pos = all_pos[h_all_ind]

    # For a given O atom, find the neighbouring O atoms within 3.5 Angstrom
    o_neigh = nearest_neighbours_within_cutoff(o_pos, o_query_pos, 3.0)

    for o_ind in o_neigh:
        # Find the distance
        o_o_dist = minimum_image_distance(o_query_pos, o_pos[o_ind])

        if math.isclose(o_o_dist, 0.0, rel_tol=1e-4):
            continue

        # Find the two closest H atom neighbours of this O atom
        h_dist, h_ind = k_nearest_neighbours(h_pos, o_pos[o_ind], 2)

        for current_h in h_ind:

            # Neglect if the OA distance is greater than 2.42
            o_a_dist = minimum_image_distance(h_pos[current_h], o_query_pos)
            if o_a_dist > 2.42:
                continue

            # Get the angle
            # HDA
            angle = angle_between_points(h_pos[current_h], o_pos[o_ind], o_query_pos)

            if angle > 90:
                angle = 180 - angle

            if angle < 30:
                bonds.append([o_atom_tag, h_tags[current_h]])

    return bonds
