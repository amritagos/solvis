import pytest
import numpy as np
from ase.io import read 
from ase.atoms import Atom, Atoms  

import solvis

def test_expanded_box_system():
    ''' Test that when a query point is chosen, atoms which are 'neighbours'
    are put into an expanded_atoms Atoms object in System, with new tags
    The system consists of 4 H atoms, with two on one of the box and two on the other side.
    After expansion, the new system should contain 6 atoms 
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
    atomic_system = solvis.system.System(atoms, expand_box=True)

    # The Atom with coordinates we want to exclude is the first one, with index 0
    index_query_pnt = 0
    query_pnt = atoms.get_positions()[index_query_pnt]
    # This is our query point, which we will exclude from the Atoms object with the neighbours 
    neigh_atoms = atoms[np.arange(len(atoms)) != index_query_pnt]  

    # Expand the simulation box
    atomic_system.add_expanded_box_atoms(query_pnt, neigh_atoms)

    # The original atoms object should have 4 H atoms, and the expanded system should have 6
    assert len(atomic_system.atoms) == 4
    assert len(atomic_system.expanded_atoms) == 6 