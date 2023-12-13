from ase import Atom, Atoms
from ase.data import chemical_symbols
from ase.io import lammpsdata, read 
import numpy as np
from scipy.spatial import ConvexHull
from pathlib import Path 
from pyvista import PolyData

import solvis
from solvis.visualization import AtomicPlotter

# Input filename 
script_dir = Path(__file__).resolve().parent
infilename = script_dir / '../../resources/single_capped_trigonal_prism_unwrapped.lammpstrj'
# In the LAMMPS trajectory file, the types of atoms are 1, 2 and 3 for O, H and Fe respectively.
fe_type = 3
h_type = 2
o_type = 1
imagename = script_dir / "nonoctahedral.png"
trimmed_img_name = script_dir / "trimmed.png"

# Bond and atom appearance 
# Decide what kind of gradient shading you want for bonds 
bond_gradient_start = 0.3
bond_radius = 0.1
atom_radius = 0.1 

# ------------------------------------------------------
# Glean the coordinates from the LAMMPS trajectory file 

# Read in the current frame 
currentframe = read(infilename, format="lammps-dump-text") # Read in the last frame of the trajectory

pos = currentframe.get_positions()

# Make a "fake center"
fake_center = np.mean(pos, axis=0)
# Box size lengths 
box_len = currentframe.get_cell_lengths_and_angles()[:3]
# 7 Nearest neighbours
k=7
dist, neigh_ind = solvis.util.k_nearest_neighbours(pos, fake_center, k, box_len)
k_near_pos = pos[neigh_ind]

# Get the convex hull using the scipy wrapper for QHull
hull = ConvexHull(k_near_pos)
# Number of faces, which is also the number of cells in the subsequent polyhull
num_faces = len(hull.simplices)
 
area = hull.area
vol = hull.volume 

# Calculate the sphericity from the volume and area 
sph_value = solvis.util.sphericity(vol,area)
print('The calculated sphericity with 7 neighbours is', sph_value, '\n')

# Convert the convex hull into a pyvista PolyData object
faces_pyvistaformat = np.column_stack((3*np.ones((len(hull.simplices), 1), dtype=int), hull.simplices)).flatten()
polyhull = PolyData(k_near_pos, faces_pyvistaformat)

faces = hull.simplices

# Get edges as pairs of indices based on the faces
edges = set()

for simplex in faces:
    edges.add(tuple(sorted([simplex[0], simplex[1]])))
    edges.add(tuple(sorted([simplex[1], simplex[2]])))
    edges.add(tuple(sorted([simplex[2], simplex[0]])))

edges = np.array(list(edges))

# ------------------------------------------------------------

mesh_cmap = solvis.util.create_two_color_gradient("#3737d2", "red", gradient_start=0.0) # blue to red gradient
point_colours = ["midnightblue", "midnightblue", "midnightblue", "midnightblue", "midnightblue", "midnightblue", "red"]

# For interactive plotting 
# NOTE: Shadows and opacity won't work together without pbr 
pl_inter = AtomicPlotter(interactive_mode=True, depth_peeling=True, shadows=True)
# Add the hull as a mesh 
pl_inter.add_hull(polyhull,cmap=mesh_cmap,clim=[dist[4],dist[-1]],scalars=dist)
# Create the bonds corresponding to the edges and add them to the plotter
pl_inter.create_bonds_from_edges(k_near_pos, edges, point_colors=point_colours,
    radius=bond_radius, resolution=1,bond_gradient_start=bond_gradient_start)
# Add the atoms as spheres
pl_inter.add_atoms_as_spheres(k_near_pos, point_colours, radius=atom_radius)
# Remove bonds or atoms interactively
pl_inter.interactive_window(delete_actor=True) 

# ---------------------------
# Now render the image (offscreen=True)

pl_render = AtomicPlotter(interactive_mode=False, window_size=[4000,4000], depth_peeling=True, shadows=True)
# Add the hull as a mesh 
pl_render.add_hull(polyhull, cmap=mesh_cmap,clim=[dist[4],dist[-1]],scalars=dist)
# Create the bonds corresponding to the edges and add them to the plotter
pl_render.create_bonds_from_edges(k_near_pos, edges, point_colors=point_colours,
    radius=bond_radius, resolution=1,bond_gradient_start=bond_gradient_start)
# Delete actors selected interactively 
pl_render.plotter.remove_actor(pl_inter.selected_actors)
# Add the atoms as spheres
pl_render.add_atoms_as_spheres(k_near_pos, point_colours, radius=atom_radius)
# Get and set the camera position found from the last frame in the interactive mode
inter_cpos = pl_inter.interactive_camera_position[-1]
print("Camera position will be set to ", inter_cpos)
# Save image 
pl_render.render_image(imagename, inter_cpos)

# Trim the image
trimmed_im = solvis.util.trim(imagename,10)
trimmed_im.save(trimmed_img_name)