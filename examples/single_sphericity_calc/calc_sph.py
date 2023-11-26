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

# Input filename 
infilename = '../../resources/single_capped_trigonal_prism.lammpstrj'
# In the LAMMPS trajectory file, the types of atoms are 1, 2 and 3 for O, H and Fe respectively.
fe_type = 3
h_type = 2
o_type = 1

# Read in the current frame 
currentframe = read(infilename, format="lammps-dump-text") # Read in the last frame of the trajectory
# The sphericity should be 0.805005, as calculated by OVITO when taking a probe radius of 7. 
ref_value = 0.805005

# Delete the H atoms (In this example there are no H atoms, but still) 
del currentframe[[atom.index for atom in currentframe if atom.number==h_type]]

# Shift all the positions by the minimum coordinate in x, y, z
xlo = np.min(currentframe.get_positions()[:,0])
ylo = np.min(currentframe.get_positions()[:,1])
zlo = np.min(currentframe.get_positions()[:,2])
currentframe.translate([-xlo,-ylo,-zlo])

# Get just the O atoms 
o_atoms = currentframe[[atom.index for atom in currentframe if atom.number==o_type]]
o_atoms_pos = o_atoms.get_positions()

# Indices of the Fe atoms
fe_ind = currentframe.symbols.search(chemical_symbols[fe_type])
# Here we only have one Fe atom 

fe_pos_query_pnt = currentframe.get_positions()[fe_ind[0]]

# Box size lengths 
box_len = currentframe.get_cell_lengths_and_angles()[:3]

# KDTree of O atom positions
kdtree = KDTree(o_atoms_pos, boxsize=box_len)
# Indices of the k nearest neighbours 
k = 7 
dist, neigh_ind = kdtree.query(fe_pos_query_pnt,k)

# Extract the 6 nearest neighbour positons 
k_near_pos = o_atoms_pos[neigh_ind[:6]]

# Shift the points with respect to the central Fe atom
# SciPy doesn't have support for periodic boundary conditions
for i in range(len(k_near_pos)):
    k_near_pos[i] = minimum_image_shift(k_near_pos[i], fe_pos_query_pnt, box_len)

# Get the convex hull using the scipy wrapper for QHull
hull = ConvexHull(k_near_pos)
# Number of faces, which is also the number of cells in the subsequent polyhull
num_faces = len(hull.simplices)
 
area = hull.area
vol = hull.volume 

# Calculate the sphericity from the volume and area 
sph_value = sphericity(vol,area)
print('The calculated sphericity is', sph_value, 'which matches the reference value of', ref_value,'\n', sep=' ')

# Visualize the convex hull with pyvista
faces = np.column_stack((3*np.ones((len(hull.simplices), 1), dtype=int), hull.simplices)).flatten()
polyhull = PolyData(k_near_pos, faces)
pl = pv.Plotter(off_screen=False)
pl.image_scale = 4
pl.set_background("white") # Background 
# Colour the mesh faces according to the distance from the center 
matplotlib_cmap = plt.cm.get_cmap("bwr") # Get the colormap from matplotlib
pl.add_mesh(polyhull, 
    line_width=3, 
    show_edges=True,
    scalars=dist[:6],
    cmap=matplotlib_cmap,
    opacity=0.85,
    clim=[2.04,2.12],
    metallic=True,
    show_scalar_bar=False,) 
# Add the six nearest neighbours as points on the hull
# pl.add_points( k_near_pos,
#     color='black',
#     render_points_as_spheres=True,
#     point_size=15,
#     show_vertices=True,
#     lighting=True)
# Add the seventh neighbour, as a single particle in red 
seventh_mol_pos = o_atoms_pos[neigh_ind[-1]]
seventh_mol_pos = minimum_image_shift(seventh_mol_pos, fe_pos_query_pnt, box_len)
pl.add_points( seventh_mol_pos,
    color='red',
    render_points_as_spheres=True,
    point_size=15,
    show_vertices=True,
    lighting=True)
cam_positions = [pl.camera_position] # Camera positions we want to render later?
def last_frame_info(plotter, cam_pos, outfilename=None):
    if outfilename is not None:
        plotter.screenshot(outfilename)
    cam_positions.append(plotter.camera_position)
# In order to take the last screenshot and save it to an image 
# pl.show(before_close_callback=lambda pl: last_frame_info(pl,cam_positions,"nonoctahedral.png"))
pl.show(before_close_callback=lambda pl: last_frame_info(pl,cam_positions))
print(cam_positions[-1])
pl.set_camera = cam_positions[-1]
pl.screenshot("nonoctahedral.png")

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