import numpy as np
import math
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageChops
from scipy.spatial import KDTree


def sanitize_positions_into_periodic_box(atoms, box_lengths):
    """
    Make sure all atom coordinates fit into the periodic boundary box,
    given an ASE atoms object and the box lengths.
    If pbcs are not set then return the atoms object itself.
    If set_tol is set to True, then atoms are shifted with a tolerance of 1E-16
    It is recommended to keep the value of set_tol to False, unless SciPy fails.
    """

    if False in atoms.pbc:
        print("Warning: PBCs are set to false.\n")
        return

    x_min, y_min, z_min = np.min(atoms.get_positions(), axis=0)
    x_max, y_max, z_max = np.max(atoms.get_positions(), axis=0)

    # Check if the minimum coordinates are less than 0 or
    # if the maximum coordinates are greater than 0
    if (
        x_min < 0
        or y_min < 0
        or z_min < 0
        or x_max > box_lengths[0]
        or y_max > box_lengths[1]
        or z_max > box_lengths[2]
    ):
        atoms.translate([-x_min, -y_min, -z_min])

    # Clip the coordinates into the periodic box
    pos = np.array(atoms.get_positions())
    for j in range(3):
        pos[:, j] = np.clip(pos[:, j], 0.0, box_lengths[j] * (1.0 - 1e-16))
    atoms.set_positions(pos)

    return atoms


def minimum_image_distance(pointa, pointb, box_dimensions=None):
    delta = pointa - pointb

    if box_dimensions is None:
        return np.linalg.norm(delta)

    for i in range(len(delta)):
        delta[i] -= box_dimensions[i] * round(delta[i] / box_dimensions[i])

    return np.linalg.norm(delta)


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


def get_first_timesteps(dump_file):
    """
    Get the timesteps from the LAMMPS trajectory file, since ASE
    does not read this in directly

    Parameters:
    - dump_file: Path to the LAMMPS trajectory file, must be in human-readable dump format

    Returns:
    - Values of the first two timesteps
    """
    times = []
    with open(dump_file, "r") as f:
        for line in f:
            if line.strip() == "ITEM: TIMESTEP":
                timestep_val = int(next(f).strip())
                times.append(timestep_val)
                if len(times) == 2:
                    break
    return times


def sphericity(vol, area):
    """
    Calculate the sphericity, from the volume and area of the hull
    """
    sph = (math.pi ** (1.0 / 3.0) * (6 * vol) ** (2.0 / 3.0)) / area
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
    - (dist, neigh_ind): Distances and indices of the k-nearest neighbours.

    """
    # Create the KDTree using data
    if box_dimensions is not None:
        kdtree = KDTree(data, boxsize=box_dimensions)
    else:
        kdtree = KDTree(data)  # no periodic boundary conditions

    # Find the nearest neighbours from the query point
    # neigh_ind refers to the indices in data corresponding to the neighbours
    dist, neigh_ind = kdtree.query(query_pnt, k)

    return (dist, neigh_ind)


def nearest_neighbours_within_cutoff(data, query_pnt, cutoff, box_dimensions=None):
    """
    Return the indices of the nearest neighbours within a cutoff, from a given query point
    given a query point.

    Parameters:
    - data: Coordinates of points to be searched for neighbours.
    - query_pnt: Coordinates of central 'query' point from nearest neighbours will be calculated.
    - cutoff: Distance cutoff
    - box_dimensions: List or array of box dimensions [lx, ly, lz].

    Returns:
    - neigh_ind: list of indices

    """
    # Create the KDTree using data
    if box_dimensions is not None:
        kdtree = KDTree(data, boxsize=box_dimensions)
    else:
        kdtree = KDTree(data)  # no periodic boundary conditions

    # Find the coordination number within a cutoff
    neigh_ind = kdtree.query_ball_point(query_pnt, r=cutoff, return_length=False)

    return neigh_ind


def distance_projected(pointa, pointb, pointc, pointd):
    # Calculate the direction vector of the line defined by C and D
    direction_vector = pointd - pointc

    # Calculate the projection of A and B onto the line
    projection_A = (
        pointc
        + np.dot(pointa - pointc, direction_vector)
        / np.dot(direction_vector, direction_vector)
        * direction_vector
    )
    projection_B = (
        pointc
        + np.dot(pointb - pointc, direction_vector)
        / np.dot(direction_vector, direction_vector)
        * direction_vector
    )

    # Calculate the distance between the projections
    projected_distance = np.linalg.norm(projection_B - projection_A)

    return projected_distance


def create_two_color_gradient(color1: str, color2: str, gradient_start=0.0):
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
    if gradient_start == 0.0:
        colors = [color1, color2]
        m_cmap = LinearSegmentedColormap.from_list("mycmap", colors)
    elif gradient_start > 0.0 and gradient_start <= 0.5:
        nodes = [0.0, gradient_start, 1.0 - gradient_start, 1.0]
        colors = [color1, color1, color2, color2]
        m_cmap = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))
    else:
        # Print warning
        print(
            "You entered an invalid value of gradient_start= ",
            gradient_start,
            "for creating a two-color gradient. Reverting to default.\n",
        )
        colors = [color1, color2]
        m_cmap = LinearSegmentedColormap.from_list("mycmap", colors)

    return m_cmap


def create_asymmetric_two_color_gradient(color1: str, color2: str, gradient_start=0.1):
    """
    Create and return a matplotlib color gradient, given two colours (the same colour)
    is fine too.

    Parameters:
    - color1: Any valid matplotlib color
    - color2: The second matplotlib color the gradient goes up to
    - gradient_start: Where the mixing of the two colors will start. By default, this is 0.1
    Valid input are numbers from 0 to 1.0, not included

    Returns:
    - colormap: Matplotlib colormap.

    """
    if gradient_start == 0.0:
        colors = [color1, color2]
        m_cmap = LinearSegmentedColormap.from_list("mycmap", colors)
    elif gradient_start > 0.0 and gradient_start < 1.0:
        nodes = [0.0, gradient_start, 1.0]
        colors = [color1, color2, color2]
        m_cmap = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))
    else:
        # Print warning
        print(
            "You entered an invalid value of gradient_start= ",
            gradient_start,
            "for creating an asymmetric two-color gradient\n Reverting to smooth gradient.\n",
        )
        colors = [color1, color2]
        m_cmap = LinearSegmentedColormap.from_list("mycmap", colors)

    return m_cmap


# Trim an image with some tolerance
def trim(imagefilename, border_pixels=0):
    im = Image.open(imagefilename)
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, 0)
    # Bounding box given as a 4-tuple defining the left, upper, right, and lower pixel coordinates.
    # If the image is completely empty, this method returns None.
    bbox = diff.getbbox()
    bbox_borders = (
        bbox[0] - border_pixels,
        bbox[1] - border_pixels,
        bbox[2] + border_pixels,
        bbox[3] + border_pixels,
    )  # crop rectangle, as a (left, upper, right, lower)-tuple.
    if bbox:
        return im.crop(bbox_borders)


# Merge options, to be used for kwargs and default dictionary values
# For all options inside default_options, if override has the value, set it
# or else keep the default_options value. Ignore all other keys inside override
def merge_options(default_options, override_dict):
    result = {}
    for key in default_options:
        if key in override_dict:
            result[key] = override_dict[key]
        else:
            result[key] = default_options[key]
    return result


# Merge options, to be used for kwargs and bond/atom render options set already
# For all options inside default_options, if override has the value, set it
# or else keep the default_options value. Also keep all extra keys inside override
def merge_options_keeping_override(old_options, override_dict):
    result = old_options

    for key in override_dict:
        result[key] = override_dict[key]
    return result


def angle_between_points(H, D, A):
    """
    Find the angle, in degrees, between the points H, D and A
    """
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
