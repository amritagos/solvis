import pytest
import numpy as np
from ase.io import read 
from ase.atoms import Atom, Atoms
from ase.data import chemical_symbols  
from pathlib import Path 

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
        (9.8, 0, 0),
        (9.6, 0, 0),
        (0.1, 0, 0),
        (0.5, 0, 0),
    ], pbc=True, cell=[box_length, box_length, box_length])
    # Create a System using the ASE Atoms object we created
    return solvis.system.System(atoms, expand_box=True)

@pytest.fixture
def capped_trigonal_prism_system(): 
    ''' Reads in a LAMMPS trajectory of a single step, and gets the solvation shell
    corresponding to O (type 1) solvent atoms surrounding a Fe atom (type 3)
    '''
 
    test_dir = Path(__file__).resolve().parent
    infilename = test_dir / '../resources/single_capped_trigonal_prism.lammpstrj'
    # Read in the current frame 
    atoms = read(infilename, format="lammps-dump-text") # Read in the last frame of the trajectory
    
    ctp_system = solvis.system.System(atoms, expand_box=True)

    return ctp_system

def test_create_solvation_shell_from_scratch_4_atoms_no_center(atomic_system):
    ''' Test that you can create a solvation cell from an ASE atoms object of 4 atoms, 
    The coordinates should be unwrapped using the coordinates of the first atom as a reference 
    '''

    # Create a solvation shell from the ASE atoms object 
    # Don't specify a center 
    # All 4 atoms are now solvent atoms 
    solvation_shell = solvis.solvation_shell.create_solvation_shell_from_solvent(atomic_system.atoms, atomic_system.box_lengths, center=None)

    # There should be 4 'solvent atoms' in the solvation shell
    assert len(solvation_shell.atoms) == 4

    # A "fake" center is created 
    center_ref = np.array([0.4, 0. , 0. ])
    np.testing.assert_allclose(solvation_shell.center, center_ref)

    # Check that the tag manager works: each tag should correspond to the index in the atoms object
    # of the SolvationShell
    for index,tag in enumerate(solvation_shell.atoms.get_tags()):
        assert solvation_shell.tag_manager.lookup_index_by_tag(tag) == index 

def test_solvation_shell_from_system(capped_trigonal_prism_system):
    ''' Test that you can create a solvation cell from a System 
    '''
    # In the LAMMPS trajectory file, the types of atoms are 1, 2 and 3 for O, H and Fe respectively.
    fe_type = 3
    h_type = 2
    o_type = 1

    # Find the Fe ion (there is only one in this system)
    # This will be our central atom 
    fe_ind = capped_trigonal_prism_system.atoms.symbols.search(chemical_symbols[fe_type]) 
    fe_pos_query_pnt = capped_trigonal_prism_system.atoms.get_positions()[fe_ind[0]]

    solvation_shell = capped_trigonal_prism_system.create_solvation_shell_from_center(central_pnt=fe_pos_query_pnt, num_neighbours=7, contains_solvation_center=True,
        solvent_atom_types=[1])

    # Check that the tags are the same as those assigned for the 
    # parent System that the SolvationShell
    # was created from
    for tag in solvation_shell.atoms.get_tags():
        index_solvation_shell  = solvation_shell.tag_manager.lookup_index_by_tag(tag)
        pos_solvation_shell = solvation_shell.atoms.get_positions()[index_solvation_shell]
        # Find the corresponding atom (which should have the same tag)
        # in the System object
        index_system = capped_trigonal_prism_system.expanded_tag_manager.lookup_index_by_tag(tag)
        pos_system = capped_trigonal_prism_system.expanded_atoms.get_positions()[index_system]

        # The positions should basically be the same
        np.testing.assert_array_equal(pos_system, pos_solvation_shell)
