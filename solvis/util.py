import numpy as np
import math 

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