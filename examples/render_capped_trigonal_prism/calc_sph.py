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
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from pyvista.plotting.opts import ElementType

def minimum_image_shift(point, reference, box_dimensions):
    """
    Shift the coordinates of a point to have the minimum image distance
    with respect to a reference point, based on periodic boundary conditions.

    Parameters:
    - point: Coordinates of the point to be shifted.
    - reference: Coordinates of the reference point.
    - box_dimensions: List or array of box dimensions [lx, ly, lz].

    Returns:
    - shifted_point: Shifted coordinates of the point.
    """
    delta = point - reference
    for i in range(len(delta)):
        delta[i] -= box_dimensions[i] * round(delta[i] / box_dimensions[i])

    shifted_point = reference + delta
    return shifted_point

def sphericity(vol,area):
    """
    Calculate the sphericity, from the volume and area of the hull  
    """
    sph = (math.pi**(1.0/3.0) * (6* vol)**(2.0/3.0)) / area
    return sph

# def create_tube(point1, point2):
    
#     return tube

# Input filename 
infilename = '../../resources/single_capped_trigonal_prism_unwrapped.lammpstrj'
# In the LAMMPS trajectory file, the types of atoms are 1, 2 and 3 for O, H and Fe respectively.
fe_type = 3
h_type = 2
o_type = 1

# Read in the current frame 
currentframe = read(infilename, format="lammps-dump-text") # Read in the last frame of the trajectory

pos = currentframe.get_positions()

# Make a "fake center"
fake_center = np.mean(pos, axis=0)
# Box size lengths 
box_len = currentframe.get_cell_lengths_and_angles()[:3]
# KDTree of O atom positions
kdtree = KDTree(pos, boxsize=box_len)
# Indices of the k nearest neighbours 
k = 7 
dist, neigh_ind = kdtree.query(fake_center,k)
k_near_pos = pos[neigh_ind]

# Get the convex hull using the scipy wrapper for QHull
hull = ConvexHull(k_near_pos)
# Number of faces, which is also the number of cells in the subsequent polyhull
num_faces = len(hull.simplices)
 
area = hull.area
vol = hull.volume 

# Calculate the sphericity from the volume and area 
sph_value = sphericity(vol,area)
print('The calculated sphericity with 7 neighbours is', sph_value, '\n')

# Visualize the convex hull with pyvista
faces = np.column_stack((3*np.ones((len(hull.simplices), 1), dtype=int), hull.simplices)).flatten()
polyhull = PolyData(k_near_pos, faces)

# This returns a PolyData object!!
edges = polyhull.extract_feature_edges(non_manifold_edges=False)

# Create a tube for each line inside edges
bond_tubes = []
# breakpoint()

# For interactive plotting 
pl = pv.Plotter(off_screen=False) #  window_size=[4000,4000]
pl.image_scale = 4

remove_edges = []

def callback(edge):
    print("selected, ", edge, " and type", type(edge)) 
    remove_edges.append(edge)

# Pick edge
# pl.enable_cell_picking(callback=callback)
pl.enable_element_picking(callback=callback, mode=ElementType.EDGE)

pl.set_background("white") # Background 
# Colour the mesh faces according to the distance from the center 
matplotlib_cmap = plt.cm.get_cmap("bwr") # Get the colormap from matplotlib
pl.add_mesh(polyhull, 
    line_width=3, 
    show_edges=False,
    opacity=0.85,
    metallic=True,
    show_scalar_bar=False,
    specular=0.7,
    ambient=0.3,cmap=matplotlib_cmap, 
    clim=[dist[4],dist[-1]], 
    scalars=dist,)

pl.add_mesh(edges, color='red', line_width=10, pickable=True)

cam_positions = [pl.camera_position] # Camera positions we want to render later?

# In order to take the last screenshot and save it to an image
def last_frame_info(plotter, cam_pos, outfilename=None):
    if outfilename is not None:
        plotter.screenshot(outfilename)
    cam_positions.append(plotter.camera_position)
 
pl.show(before_close_callback=lambda pl: last_frame_info(pl,cam_positions), auto_close=False)

breakpoint()

# Create a separate render image 
pl_render = pv.Plotter(off_screen=True, window_size=[4000,4000])
pl_render.set_background("white") # Background 

# Colour the mesh faces according to the distance from the center 
pl_render.add_mesh(polyhull, 
    line_width=3, 
    show_edges=False,
    opacity=0.85,
    metallic=True,
    show_scalar_bar=False,
    specular=0.7,
    ambient=0.3,cmap=matplotlib_cmap, 
    clim=[dist[4],dist[-1]], 
    scalars=dist) 

pl_render.add_mesh(edges, color='red', line_width=10) 

pl_render.camera_position = cam_positions[-1]
pl_render.camera_set = True
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