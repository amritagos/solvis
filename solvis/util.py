import numpy as np
import math 
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageChops
from scipy.spatial import KDTree

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

def k_nearest_neighbours(data, query_pnt, k, box_dimensions=None):
    """
    Return the distances and a numpy ndarray corresponding to the k-nearest neighbours, 
    given a query point. 
    
    Parameters:
    - data: Coordinates of points to be searched for neighbours.
    - query_pnt: Coordinates of central 'query' point from nearest neighbours will be calculated.
    - k: Number of nearest neighbours 
    - box_dimensions: List or array of box dimensions [lx, ly, lz].

    Returns:
    - (dist, k_nearest_pos): Distances and positions of the k-nearest neighbours.

    """
    # Create the KDTree using data
    if box_dimensions is not None:
        kdtree = KDTree(data, boxsize=box_dimensions)
    else:
        kdtree = KDTree(data) # no periodic boundary conditions 

    # Find the nearest neighbours from the query point
    # neigh_ind refers to the indices in data corresponding to the neighbours 
    dist, neigh_ind = kdtree.query(query_pnt,k)

    return (dist,data[neigh_ind])

def distance_projected(pointa, pointb, pointc, pointd):
    # Calculate the direction vector of the line defined by C and D
    direction_vector = pointd - pointc
    
    # Calculate the projection of A and B onto the line
    projection_A = pointc + np.dot(pointa - pointc, direction_vector) / np.dot(direction_vector, direction_vector) * direction_vector
    projection_B = pointc + np.dot(pointb - pointc, direction_vector) / np.dot(direction_vector, direction_vector) * direction_vector
    
    # Calculate the distance between the projections
    projected_distance = np.linalg.norm(projection_B - projection_A)
    
    return projected_distance

def create_two_color_gradient(color1:str, color2:str, gradient_start=0.0):
    """
    Create and return a matplotlib color gradient, given two colours (the same colour)
    is fine too. 

    Parameters:
    - color1: Any valid matplotlib color
    - color2: The second matplotlib color the gradient goes up to 
    - gradient_start: Where the mixing of the two colors will start. (implemented in the
    form of node positions equidistant from the end points). By default, this is 0.0
    Valid input are numbers from 0 to 0.5 (no mixing and sharp edge between both colors) 

    Returns:
    - colormap: Matplotlib colormap.

    """
    if gradient_start==0.0:
        colors = [color1, color2]
        m_cmap = LinearSegmentedColormap.from_list("mycmap", colors)
    elif gradient_start>0.0 and gradient_start<=0.5:
        nodes = [0.0, gradient_start, 1.0-gradient_start, 1.0]
        colors = [color1, color1, color2, color2]
        m_cmap = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes,colors)))
    else:
        # Print warning
        print("You entered an invalid value of gradient_start= ", gradient_start,"for creating a two-color gradient. Reverting to default.\n")
        colors = [color1, color2]
        m_cmap = LinearSegmentedColormap.from_list("mycmap", colors)

    return m_cmap

# Trim an image with some tolerance 
def trim(imagefilename, border_pixels=0):
    im = Image.open(imagefilename)
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, 0)
    #Bounding box given as a 4-tuple defining the left, upper, right, and lower pixel coordinates.
    #If the image is completely empty, this method returns None.
    bbox = diff.getbbox()
    bbox_borders = (bbox[0]-border_pixels, bbox[1]-border_pixels, bbox[2]+border_pixels, bbox[3]+border_pixels) # crop rectangle, as a (left, upper, right, lower)-tuple.
    if bbox:
        return im.crop(bbox_borders)

# Merge options, to be used for kwargs and default dictionary values
# For all options inside default_options, if override has the value, set it
# or else keep the default_options value. Ignore all other keys inside override 
def merge_options(default_options, override):
    result = {}
    for key in default_options:
        if key in override:
            result[key] = override[key]
        else:
            result[key] = default_options[key]
    return result