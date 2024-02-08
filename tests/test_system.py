import pytest
import numpy as np
from ase.io import read 
from ase.atoms import Atom, Atoms  

import solvis

@pytest.fixture
def atomic_system():
    ''' 
    Small system consists of 4 H atoms, with two on one of the box and two on the other side.
    Returns a System object
    '''
    box_length = 10.0 # Define the box length

    # Create an Atoms object with 4 atoms in a box
    atoms = Atoms('H4', positions=[
        (9.8, 0.0, 0.0),
        (9.6, 0.0, 0.0),
        (0.1, 0.0, 0.0),
        (0.5, 0.0, 0.0),
    ], pbc=True, cell=[box_length, box_length, box_length])
    # Create a System using the ASE Atoms object we created
    return solvis.system.System(atoms, expand_box=True)

def test_wrapped_in_box(atomic_system):
    ''' Make sure that the coordinates are shifted into a periodic box defined by box lengths,
    and starting from 0.0. This is required for SciPy's nearest neighbours with pbcs to work. 
    '''

    # The positions of atoms in atomic_system do not start from 0.0. Let's shift them by 0.1
    atomic_system.atoms.translate([-0.1,0.0,0.0])
    old_pos = atomic_system.atoms.get_positions()
    
    # Now shift these positions so that they are outside the periodic box starting from 0.0,0.0,0.0
    shifted_pos = 10.0 + old_pos 
    # Testing the function _shift_all_positions_into_box inside the System class
    atomic_system.atoms.set_positions(shifted_pos)
    atomic_system._shift_all_positions_into_box()
    np.testing.assert_allclose(atomic_system.atoms.get_positions(), old_pos)

def test_expanded_box(atomic_system):
    ''' Test that when a query point is chosen, atoms which are 'neighbours'
    are put into an expanded_atoms Atoms object in System, with new tags
    The system consists of 4 H atoms, with two on one of the box and two on the other side.
    After expansion, the new system should contain 6 atoms 
    '''

    # The Atom with coordinates we want to exclude is the first one, with index 0
    index_query_pnt = 0
    query_pnt = atomic_system.atoms.get_positions()[index_query_pnt]
    # This is our query point, which we will exclude from the Atoms object with the neighbours 
    neigh_atoms = atomic_system.atoms[np.arange(len(atomic_system.atoms)) != index_query_pnt]  

    # Expand the simulation box
    atomic_system.add_expanded_box_atoms(query_pnt, neigh_atoms)

    # The original atoms object should have 4 H atoms, and the expanded system should have 6
    atoms_ref_tags = np.array([1, 2, 3, 4]) # These should be the tags
    expanded_atoms_ref_tags = np.array([1, 2, 3, 4, 5, 6]) # Two new atoms should have tags added
    np.testing.assert_array_equal(atomic_system.atoms.get_tags(), atoms_ref_tags)
    np.testing.assert_array_equal(atomic_system.expanded_atoms.get_tags(), expanded_atoms_ref_tags)

def test_create_solvation_shell_from_system_4_atoms_center(atomic_system):
    ''' Test that you can create a solvation cell from a System of 4 atoms, 
    where the first atom is the 'center', and coordinates are unwrapped according to 
    the first atom 
    '''

    # The Atom with coordinates we want to exclude is the first one, with index 0
    index_center = 0

    # Create the solvation shell from the System, such that the first atom is the center
    # and is excluded. There will be three neighbours 
    solvation_shell = atomic_system.create_solvation_shell_from_center(index_center, num_neighbours=3, contains_solvation_center=True,
        solvent_atom_types='all') 

    # Check the tags of the solvation shell
    shell_tags = np.array([2, 5, 6])
    np.testing.assert_array_equal(solvation_shell.atoms.get_tags(), shell_tags)
    # Since we created this solvation shell from a System with the first atom in System,
    # the center should be the first atom's positions
    np.testing.assert_array_equal(solvation_shell.center, atomic_system.atoms.get_positions()[index_center])

    # Check that the tag manager works: each tag should correspond to the index in the atoms object
    # of the SolvationShell
    for index,tag in enumerate(solvation_shell.atoms.get_tags()):
        assert solvation_shell.tag_manager.lookup_index_by_tag(tag) == index 