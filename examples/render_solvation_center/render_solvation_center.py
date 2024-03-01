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
infilename = script_dir / "../../resources/octahedral_solvation_shell.lammpstrj"
# In the LAMMPS trajectory file, the types of atoms are 1, 2 and 3 for O, H and Fe respectively.
fe_type = 3
h_type = 2
o_type = 1
imagename = script_dir / "octahedral_shell.png"
trimmed_img_name = script_dir / "trimmed.png"

# Bond and atom appearance
# Decide what kind of gradient shading you want for bonds
set_gradient = 0.5  # Asymmetric bond gradient , not used here
bond_radius = 0.15
atom_radius = 0.2
fe_center_radius = 0.3

nearest_neigh = 7

# ------------------------------------------------------
# Glean the coordinates from the LAMMPS trajectory file

# Read in the current frame
currentframe = read(
    infilename, format="lammps-dump-text"
)  # Read in the last frame of the trajectory
# Box size lengths
box_len = currentframe.get_cell_lengths_and_angles()[:3]

# Make a System object
ctp_system = solvis.system.System(currentframe, expand_box=True)

# Find the Fe ion (there is only one in this system)
# This will be our central atom
fe_ind = ctp_system.atoms.symbols.search(chemical_symbols[fe_type])
fe_pos_query_pnt = ctp_system.atoms.get_positions()[fe_ind[0]]

# contains_solvation_center should be set to False if the solvent atoms do not contain the center
solvation_shell = ctp_system.create_solvation_shell_from_center(
    central_pnt=fe_pos_query_pnt,
    num_neighbours=7,
    contains_solvation_center=False,
    solvent_atom_types=[o_type],
)

convex_hull = solvation_shell.build_convex_hull_k_neighbours(num_neighbours=6)
# Calculate the sphericity from the volume and area
sph_value = solvis.util.sphericity(convex_hull.volume, convex_hull.area)
print("The calculated sphericity with 6 neighbours is", sph_value, "\n")
# ------------------------------------------------------------

# Bond and atom appearance
o_color = "midnightblue"
center_type = 0
central_point_colour = "black"
seventh_neigh_color = "red"
o_radius = 0.2
fe_center_radius = 0.3
# Decide what kind of gradient shading you want for bonds
bond_opt = dict(radius=0.15, resolution=1)
bondtype = 1

# Build the RendererRepresentation object
render_rep = solvis.render_helper.RendererRepresentation()
# Add the atom type and atom type specific rendering options
render_rep.add_atom_type_rendering(atom_type=o_type, color=o_color, radius=o_radius)
render_rep.add_atom_type_rendering(
    atom_type=center_type, color=central_point_colour, radius=fe_center_radius
)
render_rep.add_bond_type_rendering(bond_type=bondtype, color=o_color, **bond_opt)

# Add the atoms from the solvation shell
solvis.vis_initializers.fill_render_rep_atoms_from_solv_shell(
    render_rep, solvation_shell, include_center=True
)

# Change the color of the seventh atom
seventh_mol_name = list(render_rep.atoms.keys())[-1]
render_rep.update_atom_color(seventh_mol_name, color=seventh_neigh_color)

# Add bonds from the center to the solvent atoms
solvis.vis_initializers.fill_render_rep_bonds_from_solv_shell_center(
    render_rep, solvation_shell, bondtype, colorby="bondtype"
)

# Change the color of the bond from the seventh atom
# should be the last bond
bond_name = list(render_rep.bonds.keys())[-1]
render_rep.update_bond_colors(
    bond_name, a_color=seventh_neigh_color, b_color=seventh_neigh_color
)

# Add the hull
hull_color = "#c7c7c7"
hull_options = dict(
    color=hull_color, show_edges=True, line_width=5, lighting=True, opacity=0.5
)
render_rep.add_hull(**hull_options)
# ------------------------------------------------------------

# For interactive plotting
pl_inter = AtomicPlotter(interactive_mode=True, depth_peeling=True, shadows=False)

# Populate plotter using the RendererRepresentation object
solvis.vis_initializers.populate_plotter_from_solv_shell(
    pl_inter, render_rep, solvation_shell, convex_hull_list=[convex_hull]
)
# Open the interactive window
pl_inter.interactive_window(delete_actor=True)

# Delete the selected actors from the RendererRepresentation object
for actor_name in pl_inter.selected_actors:
    render_rep.delete_actor(actor_name)
# ---------------------------
# Now render the image (offscreen=True)

pl_render = AtomicPlotter(
    interactive_mode=False, window_size=[4000, 4000], depth_peeling=True, shadows=False
)
# Populate plotter using the RendererRepresentation object
solvis.vis_initializers.populate_plotter_from_solv_shell(
    pl_render, render_rep, solvation_shell, convex_hull_list=[convex_hull]
)
# Get and set the camera position found from the last frame in the interactive mode
inter_cpos = pl_inter.interactive_camera_position[-1]
print("Camera position will be set to ", inter_cpos)
# Save image
pl_render.render_image(imagename, inter_cpos)

# Trim the image
trimmed_im = solvis.util.trim(imagename, 10)
trimmed_im.save(trimmed_img_name)
