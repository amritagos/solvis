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

def angle_between_points(H, D, A):
    vector1 = D - H
    vector2 = A - D

    dot_product = np.dot(vector1, vector2)
    magnitude_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)

    # Avoid division by zero
    if magnitude_product == 0:
        return np.nan

    cosine_angle = dot_product / magnitude_product
    angle_radians = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    # Convert angle from radians to degrees
    angle_degrees = np.degrees(angle_radians)
    
    return angle_degrees

def create_line(plotter, start_point, end_point, color, width):

    line_mesh = pv.Line(start_point, end_point)

    # Add the tube created as a bond with a unique name given by actor_name 
    plotter.plotter.add_mesh(
        line_mesh,
        color=color, 
        line_width=width
        )

    return plotter

def create_dashed_line_custom_spacing(plotter, point1, point2, segment_spacing, color, width=None):

    if width is None:
        width = 10
    # Calculate the vector between the two points
    vector = np.array(point2) - np.array(point1)

    # Calculate the total length of the line
    total_length = np.linalg.norm(vector)

    # Calculate the normalized direction vector
    direction = vector / total_length

    # Calculate the number of line segments
    num_segments = int(total_length / segment_spacing)

    # Calculate the actual segment spacing
    actual_spacing = total_length / num_segments

    # Calculate the points for the dashed line with user-defined spacing
    dashed_line_points = [np.array(point1) + i * direction * actual_spacing for i in range(num_segments)]

    # Create PyVista line segments with alternating empty spaces
    lines = []
    for i in range(0, num_segments - 1, 2):
        start_point = dashed_line_points[i]
        end_point = dashed_line_points[min(i + 1, num_segments - 1)]
        plotter = create_line(plotter, start_point, end_point, color, width)

    return plotter 

def render_hydrogen_bonds(plotter, o_atom_index, o_atoms_pos, h_atoms_pos, box_len, dash_length=None, color=None, width=None):

    if color is None:
        color = 'black'

    if width is None:
        width = 10

    # For a given O atom, find the neighbouring O atoms within 3.5 Angstrom
    o_neigh = solvis.util.nearest_neighbours_within_cutoff(o_atoms_pos, o_atoms_pos[o_atom_index], 3.0, box_dimensions=box_len)
    
    for o_ind in o_neigh:
        # Find the distance 
        o_o_dist = solvis.util.minimum_image_distance(o_atoms_pos[o_atom_index], o_atoms_pos[o_ind], box_len)

        if math.isclose(o_o_dist, 0.0, rel_tol=1e-4):
            continue

        # Find the two closest H atom neighbours of this O atom 
        h_dist, h_ind = solvis.util.k_nearest_neighbours(h_atoms_pos, o_atoms_pos[o_ind], 2, box_len)

        for current_h in h_ind:

            # Neglect if the OA distance is greater than 2.42
            o_a_dist = solvis.util.minimum_image_distance(h_atoms_pos[current_h], o_atoms_pos[o_atom_index], box_len)
            if o_a_dist > 2.42:
                continue

            # Get the angle 
            # HDA
            angle = angle_between_points(h_atoms_pos[current_h], o_atoms_pos[o_ind], o_atoms_pos[o_atom_index])
            
            if angle>90:
                angle = 180-angle

            if angle < 30:
                print(o_a_dist, angle)
                print("\n")
                # Plot the hydrogen bond between H and A 
                if dash_length is None:
                    plotter = create_line(plotter, o_atoms_pos[o_atom_index], h_atoms_pos[current_h], color, width) 
                else:
                    plotter = create_dashed_line_custom_spacing(plotter, o_atoms_pos[o_atom_index], h_atoms_pos[current_h], dash_length, color, width)

    return plotter

def render_water_bonds(plotter, o_atoms_pos, o_atom_colors, h_atom_color, h_atoms_pos, box_len, num_h_per_o=2, bond_thickness=0.1):

    for index, pos in enumerate(o_atoms_pos):
        dist, neigh_ind = solvis.util.k_nearest_neighbours(h_atoms_pos, pos, num_h_per_o, box_len)
        bonded_h_atoms = h_atoms_pos[neigh_ind]
        bond_name = "h_bond_" + str(index)
        # plotter.create_bonds_to_point(bonded_h_atoms, pos, single_bond_colors=solvent_point_colours,
        #     radius=bond_thickness, resolution=1)
        plotter.create_bonds_to_point(bonded_h_atoms, pos, point_colors=[h_atom_color, h_atom_color], central_point_color=o_atom_colors[index],
            name=bond_name,bond_gradient_start=0.4, radius=bond_thickness, resolution=1)

    return plotter

# Input filename 
script_dir = Path(__file__).resolve().parent
infilename = script_dir / 'input/system-1.data'
# In the LAMMPS trajectory file, the types of atoms are 1, 2 and 3 for O, H and Fe respectively.
fe_type = 3
h_type = 2
o_type = 1
imagename = script_dir / "temp.png"
trimmed_img_name = script_dir / "oct.png"

# Bond and atom appearance 
# Decide what kind of gradient shading you want for bonds 
set_gradient = 0.5 # Asymmetric bond gradient , not used here 
bond_radius = 0.15 # 0.15
atom_radius = 0.3 # 0.2
fe_center_radius = 0.4 # 0.3
h_atom_radius = 0.2 # 0.1

h_atom_color = "#d3d3d3"

segment_spacing = 0.17545664400649102
hbond_color = "orchid"
hbond_width = 40

nearest_neigh = 6

interactive_session= False # True
# final_camera_position=None
final_camera_position = [(39.903722799943836, 2.7383406565607014, 23.05713579611572),
 (37.99034118652344, 16.987112998962402, 10.5028977394104),
 (-0.9900947779477238, -0.1401620478065384, -0.008181138999240611)]


# ------------------------------------------------------
# Glean the coordinates from the LAMMPS trajectory file 

# Read in the current frame 
currentframe = read(infilename, format="lammps-data") # Read in the last frame of the trajectory

pos = currentframe.get_positions()

# Get just the O atoms 
o_atoms = currentframe[[atom.index for atom in currentframe if atom.number==o_type]]
o_atoms_pos = o_atoms.get_positions()

# Get the H atoms
h_atoms = currentframe[[atom.index for atom in currentframe if atom.number==h_type]]
h_atoms_pos = h_atoms.get_positions()

# Indices of the Fe atoms
fe_ind = currentframe.symbols.search(chemical_symbols[fe_type])
# Here we only have one Fe atom 

fe_pos_query_pnt = currentframe.get_positions()[fe_ind[0]]

# Box size lengths 
box_len = currentframe.get_cell_lengths_and_angles()[:3]
# 6 Nearest neighbours
k=nearest_neigh+1 # including the central atom 
dist, neigh_ind = solvis.util.k_nearest_neighbours(o_atoms_pos, fe_pos_query_pnt, k, box_len)
solvent_pos = o_atoms_pos[neigh_ind]

# Nearest positions excluding the central atom 
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

# ------------------------------------------------------------

hull_color = "#c7c7c7"
solvent_point_colours = ["midnightblue", "midnightblue", "midnightblue", "midnightblue", "midnightblue", "midnightblue", "red"]
central_point_colour = "black"
# H atom colors 
h_colors = []
for index, pos in enumerate(h_atoms_pos):
    h_colors.append(h_atom_color)

if interactive_session:
    # For interactive plotting 
    pl_inter = AtomicPlotter(interactive_mode=True, depth_peeling=True, shadows=False)
    # Add the hull as a mesh 
    pl_inter.add_hull(polyhull,color=hull_color, show_edges=True, line_width=5, lighting=True, opacity=0.5)
    # Create the bonds corresponding to the edges and add them to the plotter
    # pl_inter.create_bonds_to_point(solvent_pos, fe_pos_query_pnt, point_colors=solvent_point_colours, central_point_color=central_point_colour,
    #     radius=bond_radius, resolution=1,asymmetric_gradient_start=set_gradient)
    # If you want single color bonds 
    pl_inter.create_bonds_to_point(solvent_pos, fe_pos_query_pnt, single_bond_colors=solvent_point_colours,
        radius=bond_radius, resolution=1)
    pl_inter.add_atoms_as_spheres(solvent_pos, solvent_point_colours, radius=atom_radius)
    # Add the hydrogens 
    pl_inter.add_atoms_as_spheres(h_atoms_pos, h_colors, name="h",radius=h_atom_radius)
    # bonds inside each water molecule
    pl_inter = render_water_bonds(pl_inter, solvent_pos, solvent_point_colours, h_atom_color, h_atoms_pos, box_len, num_h_per_o=2, bond_thickness=0.1)
    # Add the Fe solvation center as a sphere with a different size 
    pl_inter.add_single_atom_as_sphere(fe_pos_query_pnt, central_point_colour, radius=fe_center_radius, actor_name="center")
    # Hydrogen bonds with the 7th molecule
    pl_inter = render_hydrogen_bonds(pl_inter, 6, solvent_pos, h_atoms_pos, box_len, dash_length=segment_spacing, color=hbond_color)
    pl_inter.interactive_window(delete_actor=True) 

# ---------------------------
# Now render the image (offscreen=True)

pl_render = AtomicPlotter(interactive_mode=False, window_size=[4000,4000], depth_peeling=True, shadows=False)
# Add the hull as a mesh 
pl_render.add_hull(polyhull,color=hull_color, show_edges=True, line_width=5, lighting=True, opacity=0.5)
# Create the bonds corresponding to the edges and add them to the plotter
pl_render.create_bonds_to_point(solvent_pos, fe_pos_query_pnt, single_bond_colors=solvent_point_colours,
    radius=bond_radius, resolution=1)
# Add the solvent atoms as spheres
pl_render.add_atoms_as_spheres(solvent_pos, solvent_point_colours, radius=atom_radius)
# Add the Fe solvation center as a sphere with a different size 
pl_render.add_single_atom_as_sphere(fe_pos_query_pnt, central_point_colour, radius=fe_center_radius, actor_name="center")
# Add the hydrogens 
pl_render.add_atoms_as_spheres(h_atoms_pos, h_colors, name="h",radius=h_atom_radius)
# bonds inside each water molecule
pl_render = render_water_bonds(pl_render, solvent_pos, solvent_point_colours, h_atom_color, h_atoms_pos, box_len, num_h_per_o=2, bond_thickness=0.1)
# Hydrogen bonds with the 7th molecule
pl_render = render_hydrogen_bonds(pl_render,6, solvent_pos, h_atoms_pos, box_len, dash_length=segment_spacing, color=hbond_color, width=hbond_width)
# Get and set the camera position found from the last frame in the interactive mode
if interactive_session:
    cpos = pl_inter.interactive_camera_position[-1]
else:
    if final_camera_position is None:
        cpos = pl_render.plotter.camera_position
    else:
        cpos = final_camera_position
print("Camera position will be set to ", cpos)
pl_render.plotter.camera.zoom(1.33)
# Save image 
pl_render.render_image(imagename, cpos)
# Trim the image
trimmed_im = solvis.util.trim(imagename,10)
trimmed_im.save(trimmed_img_name)