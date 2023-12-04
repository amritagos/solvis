from ase import Atom, Atoms
from ase.data import chemical_symbols
from ase.io import lammpsdata, read 
import numpy as np
from scipy.spatial import ConvexHull
import math 
import pandas as pd
from pathlib import Path 
from os import listdir
from os.path import isfile, join
from pyvista import PolyData
import pyvista as pv 
import solvis
from solvis.visualization import AtomicPlotter

# Input filename 
infilename = '../../resources/octahedral_solvation_shell.lammpstrj'
# In the LAMMPS trajectory file, the types of atoms are 1, 2 and 3 for O, H and Fe respectively.
fe_type = 3
h_type = 2
o_type = 1
imagename = "octahedral_shell.png"
trimmed_img_name = "trimmed.png"

# Bond and atom appearance 
# Decide what kind of gradient shading you want for bonds 
bond_gradient_start = 0.0
bond_radius = 0.15
atom_radius = 0.2
fe_center_radius = 0.3

nearest_neigh = 7

# ------------------------------------------------------
# Glean the coordinates from the LAMMPS trajectory file 

# Read in the current frame 
currentframe = read(infilename, format="lammps-dump-text") # Read in the last frame of the trajectory

pos = currentframe.get_positions()

# Get just the O atoms 
o_atoms = currentframe[[atom.index for atom in currentframe if atom.number==o_type]]
o_atoms_pos = o_atoms.get_positions()

# Indices of the Fe atoms
fe_ind = currentframe.symbols.search(chemical_symbols[fe_type])
# Here we only have one Fe atom 

fe_pos_query_pnt = currentframe.get_positions()[fe_ind[0]]

# Box size lengths 
box_len = currentframe.get_cell_lengths_and_angles()[:3]
# 7 Nearest neighbours
k=nearest_neigh+1 # including the central atom 
dist, all_pos = solvis.util.k_nearest_neighbours(pos, fe_pos_query_pnt, k, box_len)

# Nearest positions excluding the central atom 
solvent_pos = all_pos[1:]
k_near_pos = solvent_pos[:6] # closest six solvent molecules only!

# Get the convex hull using the scipy wrapper for QHull
hull = ConvexHull(k_near_pos)
# Number of faces, which is also the number of cells in the subsequent polyhull
num_faces = len(hull.simplices)
 
area = hull.area
vol = hull.volume 

# Calculate the sphericity from the volume and area 
sph_value = solvis.util.sphericity(vol,area)
print('The calculated sphericity with 6 neighbours is', sph_value, '\n')

# Convert the convex hull into a pyvista PolyData object
faces_pyvistaformat = np.column_stack((3*np.ones((len(hull.simplices), 1), dtype=int), hull.simplices)).flatten()
polyhull = PolyData(k_near_pos, faces_pyvistaformat)

edges = []

# Get the bonds between the central atom (index 0) and the remaining solvent atoms
for i in range(len(solvent_pos)):
    edges.append([0,i+1])

edges = np.array(edges)

# ------------------------------------------------------------

hull_color = "#c7c7c7"
point_colours = ["black","midnightblue", "midnightblue", "midnightblue", "midnightblue", "midnightblue", "midnightblue", "red"]
bond_colours = point_colours[1:]

# For interactive plotting 
pl_inter = AtomicPlotter(interactive_mode=True, depth_peeling=True, shadows=True)
# Add the hull as a mesh 
pl_inter.add_hull(polyhull,color=hull_color, show_edges=True, line_width=4, lighting=True, opacity=0.9)
# Create the bonds corresponding to the edges and add them to the plotter
pl_inter.create_bonds_from_edges(all_pos, edges, single_bond_colors=bond_colours,
    radius=bond_radius, resolution=1,bond_gradient_start=bond_gradient_start)
pl_inter.add_atoms_as_spheres(all_pos[1:], point_colours[1:], radius=atom_radius)
# Add the Fe solvation center as a sphere with a different size 
pl_inter.add_single_atom_as_sphere(all_pos[0], point_colours[0], radius=fe_center_radius)
# Open the interactive window 
pl_inter.interactive_window(delete_actor=True) 

# ---------------------------
# Now render the image (offscreen=True)

pl_render = AtomicPlotter(interactive_mode=False, window_size=[4000,4000], depth_peeling=True, shadows=True)
# Add the hull as a mesh 
pl_render.add_hull(polyhull, color=hull_color, show_edges=True, line_width=4, lighting=True, opacity=0.9)
# Create the bonds corresponding to the edges and add them to the plotter
pl_render.create_bonds_from_edges(all_pos, edges, single_bond_colors=bond_colours,
    radius=bond_radius, resolution=1,bond_gradient_start=bond_gradient_start)
# Add the solvent atoms as spheres
pl_render.add_atoms_as_spheres(all_pos[1:], point_colours[1:], radius=atom_radius)
# Add the Fe solvation center as a sphere with a different size 
pl_render.add_single_atom_as_sphere(all_pos[0], point_colours[0], radius=fe_center_radius)
# Get and set the camera position found from the last frame in the interactive mode
inter_cpos = pl_inter.interactive_camera_position[-1]
print("Camera position will be set to ", inter_cpos)
# Save image 
pl_render.render_image(imagename, inter_cpos)

# Trim the image
trimmed_im = solvis.util.trim(imagename,10)
trimmed_im.save(trimmed_img_name)