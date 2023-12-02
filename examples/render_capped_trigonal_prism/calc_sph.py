from ase import Atom, Atoms
from ase.data import chemical_symbols
from ase.io import lammpsdata, read 
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial import KDTree
import math 
import pandas as pd
from pathlib import Path 
from os import listdir
from os.path import isfile, join
from pyvista import PolyData
import pyvista as pv
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.pyplot as plt
from pyvista.plotting.opts import ElementType 
import solvis

# Input filename 
infilename = '../../resources/single_capped_trigonal_prism_unwrapped.lammpstrj'
# In the LAMMPS trajectory file, the types of atoms are 1, 2 and 3 for O, H and Fe respectively.
fe_type = 3
h_type = 2
o_type = 1

# Decide what kind of gradient shading you want for bonds 
bond_gradient_start = 0.3

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
dist, neigh_ind = solvis.k_nearest_neighbours(pos, fake_center, k, box_len)
k_near_pos = pos[neigh_ind]

# Get the convex hull using the scipy wrapper for QHull
hull = ConvexHull(k_near_pos)
# Number of faces, which is also the number of cells in the subsequent polyhull
num_faces = len(hull.simplices)
 
area = hull.area
vol = hull.volume 

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

# TODO TODO TODO


# For interactive plotting 
pl = pv.Plotter(off_screen=False) #  window_size=[4000,4000]
pl.image_scale = 4

pl.set_background("white") # Background 
# Colour the mesh faces according to the distance from the center 
# matplotlib_cmap = plt.cm.get_cmap("coolwarm") # Get the colormap from matplotlib
# colors = ["blue", "red"]
colors = ["#3737d2", "red"]
matplotlib_cmap = LinearSegmentedColormap.from_list("mycmap", colors)
# Add the convex hull 
pl.add_mesh(polyhull, 
    cmap=matplotlib_cmap, 
    clim=[dist[4],dist[-1]], 
    scalars=dist,
    show_scalar_bar=False,
    pickable=False,
    **shell_mesh_options)

# Create the bonds corresponding to the edges and add them to the plotter
# Use matplotlib colors for points and bonds!
point_colours = ["midnightblue", "midnightblue", "midnightblue", "midnightblue", "midnightblue", "midnightblue", "red"]# Don't hard code darkblue midnightblue
pl = create_bonds_from_edges(pl, polyhull, edges, point_colours=point_colours, bond_gradient_start=bond_gradient_start,**bond_mesh_options)

# ------------------------------------
# Remove bonds (actors), which are individual meshes 

remove_bonds = []

def callback(actor):
    remove_bonds.append(actor.name)
    print("Removed actor, ", actor.name, "\n")

pl.enable_mesh_picking(callback, use_actor=True, show=True)

# ------------------------------------

cam_positions = [pl.camera_position] # Camera positions we want to render later?

# In order to take the last screenshot and save it to an image
def last_frame_info(plotter, cam_pos, outfilename=None):
    if outfilename is not None:
        plotter.screenshot(outfilename)
    cam_positions.append(plotter.camera_position)
 
pl.show(before_close_callback=lambda pl: last_frame_info(pl,cam_positions), auto_close=False)
# -------------------------------------------------
# Create a separate render image 
pl_render = pv.Plotter(off_screen=True, window_size=[4000,4000])
pl_render.set_background("white") # Background 
pl_render.enable_depth_peeling(10)
pl_render.enable_shadows()

# Colour the mesh faces according to the distance from the center 
pl_render.add_mesh(polyhull, 
    cmap=matplotlib_cmap, 
    clim=[dist[4],dist[-1]], 
    scalars=dist,
    show_scalar_bar=False,
    **shell_mesh_options) 

# Create the bonds from the edges
pl_render = create_bonds_from_edges(pl_render, polyhull, edges, point_colours=point_colours, bond_gradient_start=bond_gradient_start,**bond_mesh_options)
# Remove the bonds 
pl_render.remove_actor(remove_bonds)

# Add the atoms as spheres 
pl_render = add_atoms(pl_render, k_near_pos, point_colours, radius=0.1, **atom_mesh_options)

pl_render.camera_position = cam_positions[-1]
pl_render.camera_set = True
print("Camera position set to ", pl_render.camera_position)
pl_render.screenshot("nonoctahedral.png")

from PIL import Image, ImageChops

def trim(im, border_pixels=0):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, 0)
    #Bounding box given as a 4-tuple defining the left, upper, right, and lower pixel coordinates.
    #If the image is completely empty, this method returns None.
    bbox = diff.getbbox()
    bbox_borders = (bbox[0]-border_pixels, bbox[1]+border_pixels, bbox[2]+border_pixels, bbox[3]-border_pixels) # crop rectangle, as a (left, upper, right, lower)-tuple.
    if bbox:
        return im.crop(bbox)

bg = Image.open("nonoctahedral.png")
trimmed_im = trim(bg,10)
trimmed_im.save("trimmed.png")