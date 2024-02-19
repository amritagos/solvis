import pytest
import numpy as np
from ase.io import read 
from ase.atoms import Atom, Atoms
from ase.data import chemical_symbols  
from pathlib import Path 

import solvis

@pytest.fixture
def capped_trigonal_prism_solv_system(): 
    ''' Reads in a LAMMPS trajectory of a single step, and gets the solvation shell
    corresponding to O (type 1) solvent atoms surrounding a Fe atom (type 3)
    '''
 	# In the LAMMPS trajectory file, the types of atoms are 1, 2 and 3 for O, H and Fe respectively.
    fe_type = 3
    h_type = 2
    o_type = 1

    test_dir = Path(__file__).resolve().parent
    infilename = test_dir / '../resources/single_capped_trigonal_prism.lammpstrj'
    # Read in the current frame 
    atoms = read(infilename, format="lammps-dump-text") # Read in the last frame of the trajectory
    
    ctp_system = solvis.system.System(atoms, expand_box=True)

    fe_ind = ctp_system.atoms.symbols.search(chemical_symbols[fe_type]) 
    fe_pos_query_pnt = ctp_system.atoms.get_positions()[fe_ind[0]]

    solvation_shell = ctp_system.create_solvation_shell_from_center(central_pnt=fe_pos_query_pnt, num_neighbours=7, contains_solvation_center=True,
    	solvent_atom_types=[1])

    return solvation_shell

def test_renderer_helper_ctp_system(capped_trigonal_prism_solv_system):
    ''' 
    Test that the RendererRepresentation can accurately describe elements required for plotting. 
    The closest six O atoms should be a dark blue, and the seventh closest should be red.
    We should be able to delete elements from a RendererRepresentation object as well.
    '''

    # In the LAMMPS trajectory file, the types of atoms are 1, 2 and 3 for O, H and Fe respectively.
    # There are no H atoms in this particular trajectory
    fe_type = 3
    o_type = 1
    # Desired colors and radii for atoms
    fe_color = "black"
    fe_radius = 0.3 

    # Find the Fe ion (there is only one in this system)
    fe_ind = capped_trigonal_prism_solv_system.atoms.symbols.search(chemical_symbols[fe_type])
    # O atom indices
    o_ind = capped_trigonal_prism_solv_system.atoms.symbols.search(chemical_symbols[o_type])

    render_rep = solvis.render_helper.RendererRepresentation()

    # Add the atom type and atom type specific rendering options that could go into the 
    # add_single_atom_as_sphere function of the plotter 
    render_rep.add_atom_type_rendering(atom_type=fe_type,color=fe_color, radius=fe_radius)
    fe_atom_options = render_rep.get_rendering_options(atom_type=fe_type)
    assert fe_atom_options == {'color': 'black', 'radius': 0.3}
    
    # render_rep.add_atom_type_rendering(atom_type=fe_type)
    # options = render_rep.get_rendering_options(atom_type=fe_type)
    # pl_inter = solvis.visualization.AtomicPlotter(interactive_mode=True, depth_peeling=True, shadows=False)
    # pl_inter.add_single_atom_as_sphere([0,0,0], color=fe_color, radius=0.1, **options)