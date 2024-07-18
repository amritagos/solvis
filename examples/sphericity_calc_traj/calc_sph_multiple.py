import pandas as pd
from ase.data import chemical_symbols
from ase.io import read
import numpy as np
from scipy.stats import sem
from pathlib import Path

import solvis

# Some variables that need to be defined for reading in and writing out
script_dir = Path(__file__).resolve().parent
infilename = script_dir / "../../resources/solution_box.lammpstrj"
out_dir = script_dir / "output"
outfilename = "sph_data.txt"  # save to output directory
cutoff = 2.60
# In the LAMMPS trajectory file, the types of atoms are 1, 2, 3 and 4 for O, H, Fe and Cl respectively.
fe_type = 3
h_type = 2
o_type = 1
cl_type = 4
# If you want to do the calculation for all central (Fe) ions, set this to true
calc_all_ions = False
num_ions_calc = 1  # Number of ions for which these will be calculated

# Create the output directory
out_dir.mkdir(parents=True, exist_ok=True)
output_file = out_dir / outfilename

# ------------------------------------------------------
# Process a LAMMPS trajectory file

# Read in the trajectory using ASE
traj = read(infilename, format="lammps-dump-text", index=":")  # Read in the trajectory
num_frames = len(traj)

# Get the actual timestep value from the file
time_vals = solvis.util.get_first_timesteps(infilename)
first_step = time_vals[0]
time_inc = time_vals[1] - time_vals[0]
last_step_excl = first_step + num_frames * time_inc
time = np.arange(first_step, last_step_excl, time_inc)
num_frames = len(traj)  # Number of frames

# Create pandas dataframe
df = pd.DataFrame(time, columns=["time"])
# Make columns for each ion
firstframe = traj[0]
if calc_all_ions:
    num_feions = firstframe.symbols.search(chemical_symbols[fe_type]).shape[0]
else:
    num_feions = num_ions_calc

# Loop through the ions, creating a column for the sphericity,
# and CN for each ion
for i in range(num_feions):
    df[f"sph{i}"] = None
    df[f"r6{i}"] = None
    df[f"cn{i}"] = None

# Loop through all the frames
for iframe, currentframe in enumerate(traj):

    print("Analyzing frame ", time[iframe])

    ctp_system = solvis.system.System(currentframe, expand_box=True)
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

        # contains_solvation_center should be set to False if the solvent atoms do not contain the center
        solvation_shell = ctp_system.create_solvation_shell_from_center(
            central_pnt=fe_pos_query_pnt,
            num_neighbours=7,
            contains_solvation_center=False,
            solvent_atom_types=[o_type],
        )

        # Find the coordination number within a cutoff
        cn_val = solvation_shell.calculate_coordination_number_from_center(cutoff)

        # Average distance of the first six neighbours
        r6_val = solvation_shell.ravg_of_k_neighbours_from_center(num_neighbours=6)

        convex_hull = solvation_shell.build_convex_hull_k_neighbours(num_neighbours=6)
        # Calculate the sphericity from the volume and area
        sph_val = solvis.util.sphericity(convex_hull.volume, convex_hull.area)

        # Update the pandas dataframe
        df.loc[iframe, f"sph{j_ion}"] = sph_val
        df.loc[iframe, f"r6{j_ion}"] = r6_val  # distance of 6 closest neighbours
        df.loc[iframe, f"cn{j_ion}"] = cn_val

df.to_csv(output_file, index=False, header=True, mode="w", sep=" ")

# Filter non-octahedral and octahedral data
# Octahedral if sph >= 0.836694
# Non-octahedral if sph < 0.836694
df_oct = df[df["sph0"] >= 0.836694]
df_non_oct = df[df["sph0"] < 0.836694]

# Mean and standard error of the mean for the r6 value
# mean_r6_oct = df_oct['r60'].mean() # r6 average for octahedral
# stdev_r6_oct = sem(df_oct['r60']) # standard error of the mean for octahedral
mean_r6_non_oct = df_non_oct["r60"].mean()  # r6 average for non-octahedral
stdev_r6_non_oct = sem(
    df_non_oct["r60"]
)  # standard error of the mean for non-octahedral

# print(f"Mean of first six neighbour distance for octahedral state: {mean_r6_oct}, Standard Deviation: {stdev_r6_oct}\n")
print(
    f"Mean of first six neighbour distance for non-octahedral state: {mean_r6_non_oct}, Standard Deviation: {stdev_r6_non_oct}\n"
)
