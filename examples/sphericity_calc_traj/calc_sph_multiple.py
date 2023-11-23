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
    with open(dump_file, 'r') as f:
        for line in f:
            if line.strip() == "ITEM: TIMESTEP":
                timestep_val = int(next(f).strip())
                times.append(timestep_val)
                if len(times)==2:
                    break
    return times

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

# Some variables that need to be defined for reading in and writing out 
infilename = '../../resources/solution_box.lammpstrj'
out_dir = "output"
outfilename = "sph_data.txt" # save to output directory
cutoff = 2.50
# In the LAMMPS trajectory file, the types of atoms are 1, 2, 3 and 4 for O, H, Fe and Cl respectively.
fe_type = 3
h_type = 2
o_type = 1
cl_type = 4 
# If you want to do the calculation for all central (Fe) ions, set this to true
calc_all_ions = False
num_ions_calc = 1 # Number of ions for which these will be calculated 

# Create the output directory
path = Path(out_dir) 
path.mkdir(parents=True, exist_ok=True)
output_file = join(out_dir, outfilename)

# Read in the trajectory using ASE
traj = read(infilename, format="lammps-dump-text", index=':') # Read in the trajectory
num_frames = len(traj)

# Get the actual timestep value from the file
time_vals = get_first_timesteps(infilename)
first_step = time_vals[0]
time_inc = time_vals[1]-time_vals[0]
last_step_excl = first_step + num_frames*time_inc
time = np.arange(first_step, last_step_excl, time_inc)
num_frames = len(traj) # Number of frames 

# Create pandas dataframe
df = pd.DataFrame(time,columns=['time'])
# Make columns for each ion 
firstframe = traj[0]
if calc_all_ions:
    num_feions = firstframe.symbols.search(chemical_symbols[fe_type]).shape[0]
else:
    num_feions = num_ions_calc

# Loop through the ions, creating a column for the sphericity,
# and CN for each ion 
for i in range(num_feions):
    df[f'sph{i}'] = None 
    df[f'cn{i}'] = None 

# Loop through all the frames 
for iframe,currentframe in enumerate(traj):

    print("Analyzing frame ", time[iframe])

    # Delete the H and Cl atoms
    del currentframe[[atom.index for atom in currentframe if atom.number in {h_type, cl_type}]]

    # Shift all the positions by the minimum coordinate in x, y, z
    xlo = np.min(currentframe.get_positions()[:,0])
    ylo = np.min(currentframe.get_positions()[:,1])
    zlo = np.min(currentframe.get_positions()[:,2])
    currentframe.translate([-xlo,-ylo,-zlo])

    # Get just the O atoms 
    o_atoms = currentframe[[atom.index for atom in currentframe if atom.number==o_type]]
    o_atoms_pos = o_atoms.get_positions()

    # Box size lengths for the current frame 
    box_len = currentframe.get_cell_lengths_and_angles()[:3]

    # Indices of the Fe atoms
    fe_ind = currentframe.symbols.search(chemical_symbols[fe_type])

    # KDTree of O atom positions
    kdtree = KDTree(o_atoms_pos, boxsize=box_len)
    # Indices of the k nearest neighbours 
    k = 6 

    # LOOP through all Fe ions 
    for j_ion in range(num_feions):
        # Current Fe atom 
        fe_pos_query_pnt = currentframe.get_positions()[fe_ind[j_ion]]
    
        dist, neigh_ind = kdtree.query(fe_pos_query_pnt,k)

        # Find the coordination number within a cutoff 
        cn_val = kdtree.query_ball_point(fe_pos_query_pnt,r=cutoff,return_length=True)

        # Extract the k nearest neighbour positons 
        k_near_pos = o_atoms_pos[neigh_ind]

        # Shift the points with respect to the central Fe atom
        # SciPy doesn't have support for periodic boundary conditions
        for i in range(len(k_near_pos)):
            k_near_pos[i] = minimum_image_shift(k_near_pos[i], fe_pos_query_pnt, box_len)

        # Get the convex hull using the scipy wrapper for QHull
        hull = ConvexHull(k_near_pos) # 

        # Get the area and volume 
        area = hull.area
        vol = hull.volume 

        # Calculate the sphericity 
        sph_val = sphericity(vol,area)

        # Update the pandas dataframe
        df.loc[iframe,f'sph{j_ion}'] = sph_val
        df.loc[iframe,f'cn{j_ion}'] = cn_val

df.to_csv(output_file, index=False, header=True, mode='w', line_terminator='\n', sep=' ')