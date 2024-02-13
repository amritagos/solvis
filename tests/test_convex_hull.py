import pytest
import numpy as np
from ase.io import read 
from ase.atoms import Atom, Atoms  
from ase.data import chemical_symbols
from pathlib import Path 

import solvis

@pytest.fixture
def capped_trigonal_prism_solv_system(): 
    ''' 
    Reads in a LAMMPS trajectory of a single step, and gets the solvation shell
    corresponding to O (type 1) solvent atoms surrounding a Fe atom (type 3)
    Returns the convex hull of the closest 6 (O) neighbours.
    '''
    # In the LAMMPS trajectory file, the types of atoms are 1, 2 and 3 for O, H and Fe respectively.
    fe_type = 3
    h_type = 2
    o_type = 1
 
    test_dir = Path(__file__).resolve().parent
    infilename = test_dir / '../resources/single_capped_trigonal_prism.lammpstrj'
    # Read in the current frame 
    atoms = read(infilename, format="lammps-dump-text") # Read in the last frame of the trajectory
    
    ctp_system = solvis.system.System(atoms, expand_box=True)
    # Find the Fe ion (there is only one in this system)
    # This will be our central atom 
    fe_ind = ctp_system.atoms.symbols.search(chemical_symbols[fe_type]) 
    fe_pos_query_pnt = ctp_system.atoms.get_positions()[fe_ind[0]]

    solvation_shell = ctp_system.create_solvation_shell_from_center(central_pnt=fe_pos_query_pnt, num_neighbours=7, contains_solvation_center=True,
        solvent_atom_types=[1])

    solvent_pos = solvation_shell.atoms.get_positions()
    k_near_pos = solvent_pos[:6] # closest six solvent molecules only!
    # Get the convex hull using the scipy wrapper for QHull
    convex_hull = solvis.geometric_utils.ConvexHull(k_near_pos)

    return solvation_shell

def test_convex_hull_creation(capped_trigonal_prism_solv_system):
    ''' 
    Test that a convex hull is created, and that points, faces and edges
    are created.  
    '''
    solvent_pos = capped_trigonal_prism_solv_system.atoms.get_positions()
    k_near_pos = solvent_pos[:6] # closest six solvent molecules only!
    # Get the convex hull using the scipy wrapper for QHull
    convex_hull = solvis.geometric_utils.ConvexHull(k_near_pos)

    # Points should be the same as k_near_pos
    np.testing.assert_array_equal(k_near_pos, convex_hull.points)

    # There should be 8 faces (since the first 6 neighbours form a distorted octahedron)
    assert convex_hull.faces.shape[0] == 8

    # The distorted octahedron has 12 edges 
    edges = convex_hull.get_edges()
    assert edges.shape[0] == 12


def test_sphericity_from_hull(capped_trigonal_prism_solv_system):
    ''' 
    Test the calculation of the sphericity of the 6 nearest neighbours, 
    for a capped trigonal prismatic configuration. 
    '''
    solvent_pos = capped_trigonal_prism_solv_system.atoms.get_positions()
    k_near_pos = solvent_pos[:6] # closest six solvent molecules only!
    # Get the convex hull using the scipy wrapper for QHull
    convex_hull = solvis.geometric_utils.ConvexHull(k_near_pos)
 
    area = convex_hull.area
    vol = convex_hull.volume 

    # Calculate the sphericity from the volume and area 
    sph_value = solvis.util.sphericity(vol,area)
    # Check: 
    # The sphericity should be 0.805005
    sph_ref_value = 0.805005
    assert sph_value == pytest.approx(sph_ref_value, 1e-6)