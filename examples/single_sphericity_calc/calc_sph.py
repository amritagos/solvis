from ase import Atom, Atoms
from ase.data import chemical_symbols
from ase.io import lammpsdata, read 
import numpy as np
from scipy.spatial import ConvexHull
from pathlib import Path 
from pyvista import PolyData

import solvis

# Input filename 
script_dir = Path(__file__).resolve().parent
infilename = script_dir / '../../resources/single_capped_trigonal_prism.lammpstrj'
# In the LAMMPS trajectory file, the types of atoms are 1, 2 and 3 for O, H and Fe respectively.
fe_type = 3
h_type = 2
o_type = 1

# ------------------------------------------------------
# Glean the coordinates from the LAMMPS trajectory file

traj = read(infilename, format="lammps-dump-text", index=':') # Read in the trajectory
num_frames = len(traj) # There is only one frame in the trajectory 

# Read in the current frame 
currentframe = traj[-1] # The last frame (there's only one frame in the trajectory)
# The sphericity should be 0.805005
ref_value = 0.805005

# Delete the H atoms (In this example there are no H atoms, but still) 
del currentframe[[atom.index for atom in currentframe if atom.number==h_type]]

# Get just the O atoms 
o_atoms = currentframe[[atom.index for atom in currentframe if atom.number==o_type]]
o_atoms_pos = o_atoms.get_positions()

# Indices of the Fe atoms
fe_ind = currentframe.symbols.search(chemical_symbols[fe_type])
# Here we only have one Fe atom 

fe_pos_query_pnt = currentframe.get_positions()[fe_ind[0]]

# Box size lengths 
box_len = currentframe.get_cell_lengths_and_angles()[:3]

# Nearest neighbours
k = 7
dist, solvent_pos = solvis.util.k_nearest_neighbours(o_atoms_pos, fe_pos_query_pnt, k, box_len)
k_near_pos = solvent_pos[:6] # closest six solvent molecules only!

# Shift the points with respect to the central Fe atom
# SciPy doesn't have support for periodic boundary conditions
for i in range(len(k_near_pos)):
    k_near_pos[i] = solvis.util.minimum_image_shift(k_near_pos[i], fe_pos_query_pnt, box_len)

# Get the convex hull using the scipy wrapper for QHull
hull = ConvexHull(k_near_pos)
# Number of faces, which is also the number of cells in the subsequent polyhull
num_faces = len(hull.simplices)
 
area = hull.area
vol = hull.volume 

# Calculate the sphericity from the volume and area 
sph_value = solvis.util.sphericity(vol,area)
print('The calculated sphericity is', sph_value, 'which matches the reference value of', ref_value,'\n', sep=' ')