from ase.io import read
from pathlib import Path

import solvis
from solvis.visualization import AtomicPlotter

# Input filename
script_dir = Path(__file__).resolve().parent
infilename = (
    script_dir / "../../resources/single_capped_trigonal_prism_unwrapped.lammpstrj"
)
# In the LAMMPS trajectory file, the types of atoms are 1, 2 and 3 for O, H and Fe respectively.
fe_type = 3
h_type = 2
o_type = 1
imagename = script_dir / "nonoctahedral.png"
trimmed_img_name = script_dir / "trimmed.png"

# ------------------------------------------------------
# Glean the coordinates from the LAMMPS trajectory file

# Read in the current frame
currentframe = read(
    infilename, format="lammps-dump-text"
)  # Read in the last frame of the trajectory
box_len = currentframe.get_cell_lengths_and_angles()[:3]

# Create a SolvationShell object from the solvent
solvation_shell = solvis.solvation_shell.create_solvation_shell_from_solvent(
    currentframe, box_len, center=None
)
pos = solvation_shell.atoms.get_positions()
# Get the distance of the seven neighbours, from the center
dist = solvation_shell.distances_of_k_neighbours_from_center(
    num_neighbours=7, coordinating_type="all"
)

convex_hull = solvation_shell.build_convex_hull_k_neighbours(num_neighbours=7)
# Calculate the sphericity from the volume and area
sph_value = solvis.util.sphericity(convex_hull.volume, convex_hull.area)
print("The calculated sphericity with 7 neighbours is", sph_value, "\n")

# Get a numPy array of edges (n_edges,2)
edges = convex_hull.get_edges()

# ------------------------------------------------------------

# Bond and atom appearance
o_color = "midnightblue"
seventh_neigh_color = "red"
atom_radius = 0.1
# Decide what kind of gradient shading you want for bonds
bond_opt = dict(radius=0.1, resolution=1, bond_gradient_start=0.3)
bondtype = 1

# Build the RendererRepresentation object
render_rep = solvis.render_helper.RendererRepresentation()
# Add the atom type and atom type specific rendering options
render_rep.add_atom_type_rendering(atom_type=o_type, color=o_color, radius=atom_radius)
render_rep.add_bond_type_rendering(bond_type=bondtype, color=None, **bond_opt)

# Add the atoms from the solvation shell
solvis.vis_initializers.fill_render_rep_atoms_from_solv_shell(
    render_rep, solvation_shell, include_center=False
)

# Change the color of the seventh atom
seventh_mol_name = render_rep.atoms.atoms[-1].name
render_rep.update_atom_color(seventh_mol_name, color=seventh_neigh_color)

# Add the edges as bonds (edges are wrt convex_hull here, with the same order as atoms in solvation_shell)
# Edges should be with tags here, not indices
edges_tags = []
for edge in edges:
    tag1 = solvation_shell.atoms.get_tags()[edge[0]]
    tag2 = solvation_shell.atoms.get_tags()[edge[1]]
    edges_tags.append([tag1, tag2])

# Add the edges as bonds (edges are wrt convex_hull here)
solvis.vis_initializers.fill_render_rep_bonds_from_edges(
    render_rep, edges_tags, bond_type=1, colorby="atomcolor"
)

# Add the hull
mesh_cmap = solvis.util.create_two_color_gradient(
    "#3737d2", "red", gradient_start=0.0
)  # blue to red gradient
hull_options = dict(cmap=mesh_cmap, clim=[dist[4], dist[-1]], scalars=dist)
render_rep.add_hull(**hull_options)
# ------------------------------------------------------------

# For interactive plotting
# NOTE: Shadows and opacity won't work together without pbr
pl_inter = AtomicPlotter(interactive_mode=True, depth_peeling=True, shadows=True)
# Populate plotter using the RendererRepresentation object
solvis.vis_initializers.populate_plotter_from_solv_shell(
    pl_inter, render_rep, solvation_shell, convex_hull_list=[convex_hull]
)
# Remove bonds or atoms interactively
pl_inter.interactive_window(delete_actor=True)
print(pl_inter.selected_actors)

# Delete the selected actors from the RendererRepresentation object
for actor_name in pl_inter.selected_actors:
    render_rep.delete_actor(actor_name)
# ---------------------------
# Now render the image (offscreen=True)

pl_render = AtomicPlotter(
    interactive_mode=False, window_size=[4000, 4000], depth_peeling=True, shadows=True
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
