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

# ------------------------------------------------------
# Glean the coordinates from the LAMMPS trajectory file 

# Read in the current frame 
currentframe = read(infilename, format="lammps-dump-text") # Read in the last frame of the trajectory
box_len = currentframe.get_cell_lengths_and_angles()[:3]

# Create a SolvationShell object from the solvent 
solvation_shell = solvis.solvation_shell.create_solvation_shell_from_solvent(currentframe, box_len, center=None)
pos = solvation_shell.atoms.get_positions()
# Get the distance of the seven neighbours, from the center
dist = solvation_shell.distances_of_k_neighbours_from_center(num_neighbours=7,coordinating_type='all')

convex_hull = solvation_shell.build_convex_hull_k_neighbours(num_neighbours=7) 
# Calculate the sphericity from the volume and area 
sph_value = solvis.util.sphericity(convex_hull.volume,convex_hull.area)
print('The calculated sphericity with 7 neighbours is', sph_value, '\n')

# Convert the convex hull into a pyvista PolyData object
polyhull = convex_hull.pyvista_hull_from_convex_hull()

# Get a numPy array of edges (n_edges,2)
edges = convex_hull.get_edges()

# ------------------------------------------------------------

# Bond and atom appearance 
o_color = "midnightblue"
seventh_neigh_color = "red"
# Decide what kind of gradient shading you want for bonds 
bond_gradient_start = 0.3
bond_radius = 0.1
atom_radius = 0.1 

# Build the RendererRepresentation object 
render_rep = solvis.render_helper.RendererRepresentation()
# Add the atom type and atom type specific rendering options that could go into the 
# add_single_atom_as_sphere function of the plotter 
render_rep.add_atom_type_rendering(atom_type=o_type,color=o_color, radius=atom_radius)

# Loop through the solvation atoms and add them
atom_string = "atom" 
for solv_atom in solvation_shell.atoms:
    iatom_type = solv_atom.number
    iatom_tag = solv_atom.tag
    iatom_name = atom_string + str(render_rep.num_atoms+1)
    render_rep.add_atom(iatom_name, iatom_tag, iatom_type)

# Change the color of the seventh atom 
seventh_mol_name = list(render_rep.atoms.keys())[-1]
render_rep.update_atom_render_opt(seventh_mol_name, color=seventh_neigh_color) 
# ------------------------------------------------------------

mesh_cmap = solvis.util.create_two_color_gradient("#3737d2", "red", gradient_start=0.0) # blue to red gradient
point_colours = ["midnightblue", "midnightblue", "midnightblue", "midnightblue", "midnightblue", "midnightblue", "red"]

# For interactive plotting 
# NOTE: Shadows and opacity won't work together without pbr 
pl_inter = AtomicPlotter(interactive_mode=True, depth_peeling=True, shadows=True)
# Add the hull as a mesh 
pl_inter.add_hull(polyhull,cmap=mesh_cmap,clim=[dist[4],dist[-1]],scalars=dist)
# Create the bonds corresponding to the edges and add them to the plotter
pl_inter.create_bonds_from_edges(pos, edges, point_colors=point_colours,
    radius=bond_radius, resolution=1,bond_gradient_start=bond_gradient_start)
# Add the atoms as spheres
pl_inter.add_atoms_as_spheres(pos, point_colours, radius=atom_radius)
# Remove bonds or atoms interactively
pl_inter.interactive_window(delete_actor=True) 

# # ---------------------------
# # Now render the image (offscreen=True)

# pl_render = AtomicPlotter(interactive_mode=False, window_size=[4000,4000], depth_peeling=True, shadows=True)
# # Add the hull as a mesh 
# pl_render.add_hull(polyhull, cmap=mesh_cmap,clim=[dist[4],dist[-1]],scalars=dist)
# # Create the bonds corresponding to the edges and add them to the plotter
# pl_render.create_bonds_from_edges(pos, edges, point_colors=point_colours,
#     radius=bond_radius, resolution=1,bond_gradient_start=bond_gradient_start)
# # Delete actors selected interactively 
# pl_render.plotter.remove_actor(pl_inter.selected_actors)
# # Add the atoms as spheres
# pl_render.add_atoms_as_spheres(pos, point_colours, radius=atom_radius)
# # Get and set the camera position found from the last frame in the interactive mode
# inter_cpos = pl_inter.interactive_camera_position[-1]
# print("Camera position will be set to ", inter_cpos)
# # Save image 
# pl_render.render_image(imagename, inter_cpos)

# # Trim the image
# trimmed_im = solvis.util.trim(imagename,10)
# trimmed_im.save(trimmed_img_name)