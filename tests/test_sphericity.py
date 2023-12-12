import pytest
import numpy as np
from ase.io import read 
from ase.atoms import Atom, Atoms  
from ase.data import chemical_symbols
from scipy.spatial import ConvexHull
from pathlib import Path 
from pyvista import PolyData

import solvis

@pytest.fixture
def solvation_shell_from_frame(): 
    ''' Reads in a LAMMPS trajectory of a single step, and gets the solvation shell
    corresponding to O (type 1) solvent atoms surrounding a Fe atom (type 3)
    We will return the 6 nearest neighbours
    This returns the coordinates of just the surrounding O atoms
    '''
    # In the LAMMPS trajectory file, the types of atoms are 1, 2 and 3 for O, H and Fe respectively.
    fe_type = 3
    h_type = 2
    o_type = 1
 
    script_dir = Path(__file__).resolve().parent
    infilename = script_dir / '../resources/single_capped_trigonal_prism.lammpstrj'
    # Read in the current frame 
    currentframe = read(infilename, format="lammps-dump-text") # Read in the last frame of the trajectory
    
    # Get just the O atoms 
    o_atoms = currentframe[[atom.index for atom in currentframe if atom.number==o_type]]
    o_atoms_pos = o_atoms.get_positions()

    # Indices of the Fe atoms
    fe_ind = currentframe.symbols.search(chemical_symbols[fe_type])
    # Here we only have one Fe atom 

    fe_pos_query_pnt = currentframe.get_positions()[fe_ind[0]]

    # Box size lengths 
    box_len = currentframe.cell.cellpar()[:3]

    # Nearest neighbours
    k = 7
    dist, neigh_ind = solvis.util.k_nearest_neighbours(o_atoms_pos, fe_pos_query_pnt, k, box_len)
    solvent_pos = o_atoms_pos[neigh_ind]
    k_near_pos = solvent_pos[:6] # closest six solvent molecules only!

    # Shift the points with respect to the central Fe atom
    # SciPy doesn't have support for periodic boundary conditions
    for i in range(len(k_near_pos)):
        k_near_pos[i] = solvis.util.minimum_image_shift(k_near_pos[i], fe_pos_query_pnt, box_len)

    return k_near_pos

def test_sphericity_calculation(solvation_shell_from_frame):
    ''' Test the calculation of the sphericity of the 6 nearest neighbours, 
    for a capped trigonal prismatic configuration. 
    '''
    # Get the convex hull using the scipy wrapper for QHull
    hull = ConvexHull(solvation_shell_from_frame)
 
    area = hull.area
    vol = hull.volume 

    # Calculate the sphericity from the volume and area 
    sph_value = solvis.util.sphericity(vol,area)
    # Check: 
    # The sphericity should be 0.805005
    sph_ref_value = 0.805005
    assert sph_value == pytest.approx(sph_ref_value, 1e-6)