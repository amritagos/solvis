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
        (9.8, 0, 0),
        (9.6, 0, 0),
        (0.1, 0, 0),
        (0.5, 0, 0),
    ], pbc=True, cell=[box_length, box_length, box_length])
    # Create a System using the ASE Atoms object we created
    return solvis.system.System(atoms, expand_box=True)

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