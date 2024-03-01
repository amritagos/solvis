from ase import Atom, Atoms
from ase.data import chemical_symbols
from ase.io import lammpsdata, read
import numpy as np
from scipy.spatial import ConvexHull
from pathlib import Path
from pyvista import PolyData

import solvis

# Input filename
script_dir = Path(__file__).resolve().parent
infilename = script_dir / "../../resources/single_capped_trigonal_prism.lammpstrj"
# In the LAMMPS trajectory file, the types of atoms are 1, 2 and 3 for O, H and Fe respectively.
fe_type = 3
h_type = 2
o_type = 1
# The sphericity should be 0.805005
ref_value = 0.805005

# ------------------------------------------------------
# Glean the coordinates from the LAMMPS trajectory file

traj = read(infilename, format="lammps-dump-text", index=":")  # Read in the trajectory
num_frames = len(traj)  # There is only one frame in the trajectory

# Read in the current frame
currentframe = traj[-1]  # The last frame (there's only one frame in the trajectory)
# Box size lengths
box_len = currentframe.get_cell_lengths_and_angles()[:3]

# Make a System object
ctp_system = solvis.system.System(currentframe, expand_box=True)

# Find the Fe ion (there is only one in this system)
# This will be our central atom
fe_ind = ctp_system.atoms.symbols.search(chemical_symbols[fe_type])
fe_pos_query_pnt = ctp_system.atoms.get_positions()[fe_ind[0]]

# contains_solvation_center should be set to False if the solvent atoms do not contain the center
solvation_shell = ctp_system.create_solvation_shell_from_center(
    central_pnt=fe_pos_query_pnt,
    num_neighbours=7,
    contains_solvation_center=False,
    solvent_atom_types=[o_type],
)
convex_hull = solvation_shell.build_convex_hull_k_neighbours(num_neighbours=6)

# Calculate the sphericity from the volume and area
sph_value = solvis.util.sphericity(convex_hull.volume, convex_hull.area)
print(
    "The calculated sphericity is",
    sph_value,
    "which matches the reference value of",
    ref_value,
    "\n",
    sep=" ",
)
