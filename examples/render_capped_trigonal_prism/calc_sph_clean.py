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

def make_interpolated_points(pointa, pointb, num_points):
    """Interpolate points between pointa and pointb"""
    t = np.linspace(0, 1, num_points + 2)[1:-1]  # Exclude endpoints to avoid duplicating pointa and pointb
    x_interp = (1 - t) * pointa[0] + t * pointb[0]
    y_interp = (1 - t) * pointa[1] + t * pointb[1]
    z_interp = (1 - t) * pointa[2] + t * pointb[2]
    points = np.column_stack((x_interp, y_interp, z_interp))
    # Insert the end points 
    points = np.insert(points, 0, pointa, axis=0)
    points = np.insert(points, len(points), pointb, axis=0)
    return points

# Create a list of padded edges, with each point connected to the next 
def edges_with_padding_adjacent(points):
    n_segments = len(points)

    edge_list = []

    for i in range(n_segments-1):
        edge_list.append( [i, i+1] )

    edge_list = np.array(edge_list)
    # We must "pad" the edges to indicate to vtk how many points per edge 
    padding = np.empty(edges.shape[0], int) * 2
    padding[:] = 2
    edges_w_padding = np.vstack((padding, edges.T)).T
    return edges_w_padding 

def distance_projected(pointa, pointb, pointc, pointd):
    # Calculate the direction vector of the line defined by C and D
    direction_vector = pointd - pointc
    
    # Calculate the projection of A and B onto the line
    projection_A = pointc + np.dot(pointa - pointc, direction_vector) / np.dot(direction_vector, direction_vector) * direction_vector
    projection_B = pointc + np.dot(pointb - pointc, direction_vector) / np.dot(direction_vector, direction_vector) * direction_vector
    
    # Calculate the distance between the projections
    projected_distance = np.linalg.norm(projection_B - projection_A)
    
    return projected_distance

def create_bonds_from_edges(plotter, hull:PolyData, edges, point_colours=None, single_bond_color=None):
    """
    Create bonds from faces in a PolyData object 
    """ 

    num_points = 5
    points = hull.points 

    for idx, bond in enumerate(edges):
        a_index = bond[0]
        b_index = bond[1]
        pointa = points[a_index]
        pointb = points[b_index]
        # Create matplotlib colormap
        colors = [point_colours[a_index], point_colours[b_index]]
        m_cmap = LinearSegmentedColormap.from_list("mycmap", colors)
        #
        nodes = make_interpolated_points(pointa, pointb, num_points)
        edges = edges_with_padding_adjacent(nodes)
        #
        cylinder = pv.PolyData(nodes, edges)
        tube = pv.Tube(pointa=pointa, pointb=pointb, resolution=1, radius=0.1, n_sides=15)
        # Create color ids
        color_ids = [] 
        for point in tube.points:
            color_ids.append(distance_projected(tube.points[0], point, tube.points[0], tube.points[-1]))
        color_ids = np.array(color_ids)
        plotter.add_mesh(tube, 
        show_edges=False,
        metallic=True,
        smooth_shading=True,
        specular=0.7,
        ambient=0.3,
        scalars=color_ids,
        cmap=m_cmap,
        show_scalar_bar=False,
        pickable=True,
        name="tube"+str(idx))

    return plotter 

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

# For interactive plotting 
pl = pv.Plotter(off_screen=False) #  window_size=[4000,4000]
pl.image_scale = 4

pl.set_background("white") # Background 
# Colour the mesh faces according to the distance from the center 
matplotlib_cmap = plt.cm.get_cmap("coolwarm") # Get the colormap from matplotlib
# Add the convex hull 
pl.add_mesh(polyhull, 
    line_width=3, 
    show_edges=False,
    opacity=0.85,
    metallic=True,
    show_scalar_bar=False,
    specular=0.7,
    ambient=0.3,cmap=matplotlib_cmap, 
    clim=[dist[4],dist[-1]], 
    scalars=dist,
    pickable=False)

# Create the bonds corresponding to the edges and add them to the plotter
point_colours = ["midnightblue", "midnightblue", "midnightblue", "midnightblue", "midnightblue", "midnightblue", "red"]# Don't hard code darkblue
pl = create_bonds_from_edges(pl, polyhull, edges, point_colours=point_colours)

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

# Create the bonds from the edges
pl_render = create_bonds_from_edges(pl_render, polyhull, edges, point_colours=point_colours)
# Remove the bonds 
pl_render.remove_actor(remove_bonds)

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