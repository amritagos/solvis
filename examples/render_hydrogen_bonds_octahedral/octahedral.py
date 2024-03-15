from ase import Atom, Atoms
from ase.data import chemical_symbols
from ase.io import lammpsdata, read
import numpy as np
from scipy.spatial import ConvexHull
from pathlib import Path
from pyvista import PolyData
import pyvista as pv
import math

import solvis
from solvis.visualization import AtomicPlotter

# Input filename
script_dir = Path(__file__).resolve().parent
infilename = script_dir / "input/system-1.data"
# In the LAMMPS trajectory file, the types of atoms are 1, 2 and 3 for O, H and Fe respectively.
fe_type = 3
h_type = 2
o_type = 1
imagename = script_dir / "temp.png"
trimmed_img_name = script_dir / "oct.png"

interactive_session = False  # True
# final_camera_position=None
final_camera_position = [
    (39.903722799943836, 2.7383406565607014, 23.05713579611572),
    (37.99034118652344, 16.987112998962402, 10.5028977394104),
    (-0.9900947779477238, -0.1401620478065384, -0.008181138999240611),
]


# ------------------------------------------------------
# Glean the coordinates from the LAMMPS trajectory file

# Read in the current frame
currentframe = read(
    infilename, format="lammps-data"
)  # Read in the last frame of the trajectory
# Box size lengths
box_len = currentframe.get_cell_lengths_and_angles()[:3]

# Make a System object
full_system = solvis.system.System(currentframe, expand_box=True)

# Find the Fe ion (there is only one in this system)
# This will be our central atom
fe_ind = full_system.atoms.symbols.search(chemical_symbols[fe_type])
fe_pos_query_pnt = full_system.atoms.get_positions()[fe_ind[0]]

# Find solvent atoms only (exclude the Fe atom, which is the center)
solvent_atoms = full_system.atoms[
    [atom.index for atom in full_system.atoms if atom.number in [o_type, h_type]]
]

# Create a solvation shell, with the Fe as the center
solvation_shell = solvis.solvation_shell.create_solvation_shell_from_solvent(
    solvent_atoms, box_len, center=fe_pos_query_pnt
)

# Create a ConvexHull object from the nearest O atom positions
convex_hull = solvation_shell.build_convex_hull_k_neighbours(
    num_neighbours=6, coordinating_type=[o_type]
)

# Calculate the sphericity from the volume and area
sph_value = solvis.util.sphericity(convex_hull.volume, convex_hull.area)
print("The calculated sphericity with 6 neighbours is", sph_value, "\n")
# ------------------------------------------------------------
# Bond and atom appearance
o_color = "midnightblue"
center_type = 0
central_point_colour = "black"
seventh_neigh_color = "red"
o_radius = 0.3
fe_center_radius = 0.4
h_atom_radius = 0.2
h_atom_color = "#d3d3d3"
water_bond_thickness = 0.1
# H bond parameters
h_segment = 0.17545664400649102
hbond_color = "orchid"
hbond_width = 40
# Decide render options for bonds
bond_opt_1 = dict(
    radius=0.15, resolution=1
)  # Fe-O bonds, single color according to bond typpe
bond_opt_2 = dict(
    bond_gradient_start=0.4, radius=water_bond_thickness, resolution=1
)  # O-H bonds, gradient, according to atomcolor type
# Render options for hydrogen bonds
hbond_opt = dict(segment_spacing=h_segment, color=hbond_color, width=hbond_width)

# Build the RendererRepresentation object
render_rep = solvis.render_helper.RendererRepresentation()
# Add the atom type and atom type specific rendering options
render_rep.add_atom_type_rendering(atom_type=o_type, color=o_color, radius=o_radius)
render_rep.add_atom_type_rendering(
    atom_type=h_type, color=h_atom_color, radius=h_atom_radius
)
render_rep.add_atom_type_rendering(
    atom_type=center_type, color=central_point_colour, radius=fe_center_radius
)
# Add Fe-O bonds
render_rep.add_bond_type_rendering(bond_type=1, color=o_color, **bond_opt_1)
# Add O-H intra-water bonds
render_rep.add_bond_type_rendering(bond_type=2, color=h_atom_color, **bond_opt_2)
# Add the hydrogen bond rendering
render_rep.set_hbond_rendering(**hbond_opt)

# Add the atoms from the solvation shell
solvis.vis_initializers.fill_render_rep_atoms_from_solv_shell(
    render_rep, solvation_shell, include_center=True
)

# Change the color of the seventh atom
seventh_mol_tag = solvation_shell.find_k_th_neighbour_from_center(
    k=7, coordinating_type=[o_type]
)
seventh_mol_index = solvation_shell.tag_manager.lookup_index_by_tag(seventh_mol_tag)
seventh_mol_name = render_rep.get_atom_name_from_tag(target_atom_tag=seventh_mol_tag)
render_rep.update_atom_color(seventh_mol_name, color=seventh_neigh_color)

# Add bonds from the center to the solvent atoms
bond1_render_opt = render_rep.bond_type_rendering[1].get("render_options")
for solv_atom in solvation_shell.atoms:
    if solv_atom.number == o_type:
        iatom_tag = solv_atom.tag
        render_rep.add_bond(
            ["center", iatom_tag], bond_type=1, colorby="bondtype", **bond1_render_opt
        )

# Change the color of the bond from the seventh atom
# should be the last bond
bond_name = list(render_rep.bonds.keys())[-1]
render_rep.update_bond_colors(
    bond_name, a_color=seventh_neigh_color, b_color=seventh_neigh_color
)

# Add the intra-water bonds
oh_bonds = solvis.bond_helper.find_intra_water_bonds(
    solvation_shell.atoms, o_type, h_type, box_len
)
solvis.vis_initializers.fill_render_rep_bonds_from_edges(
    render_rep, edges=oh_bonds, bond_type=2, colorby="atomcolor"
)

# Add the hydrogen bonds
hbond_list = solvis.bond_helper.find_water_hydrogen_bonds(seventh_mol_index, solvation_shell.atoms, o_type, h_type, box_len)
# hbond_list2 = [[solvation_shell.atoms[0].tag, seventh_mol_tag]]
# hbond_list = [[seventh_mol_tag,20], [seventh_mol_tag,22]] # 17,15
solvis.vis_initializers.fill_render_rep_hbonds_from_edges(render_rep, edges=hbond_list)
# Add the hull
hull_color = "#c7c7c7"
hull_options = dict(
    color=hull_color, show_edges=True, line_width=5, lighting=True, opacity=0.5
)
render_rep.add_hull(**hull_options)
# ------------------------------------------------------------

pl_render = AtomicPlotter(
    interactive_mode=False, window_size=[4000, 4000], depth_peeling=True, shadows=False
)
# Populate plotter using the RendererRepresentation object
solvis.vis_initializers.populate_plotter_from_solv_shell(
    pl_render, render_rep, solvation_shell, convex_hull_list=[convex_hull]
)
# Set the camera position to desired position
print("Camera position will be set to ", final_camera_position)
# Save image
pl_render.render_image(imagename, final_camera_position)

# Trim the image
trimmed_im = solvis.util.trim(imagename, 10)
trimmed_im.save(trimmed_img_name)
