import pandas as pd
from ase import Atom, Atoms
from ase.data import chemical_symbols
from ase.io import lammpsdata, read 
import numpy as np
from scipy.spatial import ConvexHull
from pathlib import Path 
from pyvista import PolyData

import solvis

# Some variables that need to be defined for reading in and writing out 
script_dir = Path(__file__).resolve().parent
infilename = script_dir / '../../resources/solution_box.lammpstrj'
out_dir = script_dir / "output"
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
k = 6 # Number of nearest neighbours to consider for generating the convex hull

# Create the output directory
out_dir.mkdir(parents=True, exist_ok=True)
output_file = out_dir / outfilename

# ------------------------------------------------------
# Process a LAMMPS trajectory file 

# Read in the trajectory using ASE
traj = read(infilename, format="lammps-dump-text", index=':') # Read in the trajectory
num_frames = len(traj)

# Get the actual timestep value from the file
time_vals = solvis.util.get_first_timesteps(infilename)
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

    # Shift all the positions by the minimum coordinate in x, y, z
    # You need to do this or else SciPy will error out when 
    # applying periodic boundary conditions if coordinates are outside the box (0,0,0)(xboxlength, yboxlength,zboxlength)
    xlo = np.min(currentframe.get_positions()[:,0])
    ylo = np.min(currentframe.get_positions()[:,1])
    zlo = np.min(currentframe.get_positions()[:,2])
    currentframe.translate([-xlo,-ylo,-zlo])

    # Delete the H and Cl atoms
    del currentframe[[atom.index for atom in currentframe if atom.number in {h_type, cl_type}]]

    # Get just the O atoms 
    o_atoms = currentframe[[atom.index for atom in currentframe if atom.number==o_type]]
    o_atoms_pos = o_atoms.get_positions()

    # Box size lengths for the current frame 
    box_len = currentframe.get_cell_lengths_and_angles()[:3]

    # Indices of the Fe atoms
    fe_ind = currentframe.symbols.search(chemical_symbols[fe_type])

    # We will find the 6 nearest neighbours of each Fe atom and the coordination number
    # within a cutoff
    # LOOP through all Fe ions 
    for j_ion in range(num_feions):

        # Current Fe atom 
        fe_pos_query_pnt = currentframe.get_positions()[fe_ind[j_ion]]

        # Extract the k (6) nearest neighbour positons 
        dist, neigh_ind = solvis.util.k_nearest_neighbours(o_atoms_pos, fe_pos_query_pnt, k, box_len)
        k_near_pos = o_atoms_pos[neigh_ind]

        # Find the coordination number within a cutoff 
        cn_val = solvis.util.nearest_neighbours_within_cutoff(o_atoms_pos, fe_pos_query_pnt, cutoff, box_len)

        # Shift the points with respect to the central Fe atom
        # SciPy doesn't have support for periodic boundary conditions
        for i in range(len(k_near_pos)):
            k_near_pos[i] = solvis.util.minimum_image_shift(k_near_pos[i], fe_pos_query_pnt, box_len)

        # Get the convex hull using the scipy wrapper for QHull
        hull = ConvexHull(k_near_pos) # 

        # Get the area and volume 
        area = hull.area
        vol = hull.volume 

        # Calculate the sphericity 
        sph_val = solvis.util.sphericity(vol,area)

        # Update the pandas dataframe
        df.loc[iframe,f'sph{j_ion}'] = sph_val
        df.loc[iframe,f'cn{j_ion}'] = cn_val

df.to_csv(output_file, index=False, header=True, mode='w', line_terminator='\n', sep=' ')