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
    o_color = "midnightblue"
    o_radius = 0.2

    # Find the Fe ion (there is only one in this system)
    fe_ind = capped_trigonal_prism_solv_system.atoms.symbols.search(chemical_symbols[fe_type])
    # O atom indices
    o_ind = capped_trigonal_prism_solv_system.atoms.symbols.search(chemical_symbols[o_type])

    render_rep = solvis.render_helper.RendererRepresentation()

    # Add the atom type and atom type specific rendering options that could go into the 
    # add_single_atom_as_sphere function of the plotter 
    render_rep.add_atom_type_rendering(atom_type=fe_type,color=fe_color, radius=fe_radius)
    fe_atom_options = render_rep.get_atom_rendering_options(atom_type=fe_type)
    assert fe_atom_options == {'color': 'black', 'radius': 0.3}
    # Atom type of the O atoms
    render_rep.add_atom_type_rendering(atom_type=o_type,color=o_color, radius=o_radius)
    o_atom_options = render_rep.get_atom_rendering_options(atom_type=o_type)
    assert o_atom_options == {'color': 'midnightblue', 'radius': 0.2}


    # Now add the Fe atom as the center 
    render_rep.add_atom("fe", None, fe_type) # Handle atom_tag=None for solvation center! TODO
    # Loop through the solvation atoms and add them
    solvent_string = "oatom" 
    for i,solv_atom in enumerate(capped_trigonal_prism_solv_system.atoms):
        iatom_type = solv_atom.number
        iatom_tag = solv_atom.tag
        iatom_name = solvent_string + str(i)
        render_rep.add_atom(iatom_name, iatom_tag, iatom_type)

    # There should be 8 keys inside atoms in the renderer represenation
    assert len(render_rep.atoms.keys()) == len(capped_trigonal_prism_solv_system.atoms)+1

    # Retreive the indices and coord using atom tags from the solvation shell object
    for i,name in enumerate(render_rep.atoms.keys()):
        tag = render_rep.get_tag_from_atom_name(name)
        if tag == None:
            coord = capped_trigonal_prism_solv_system.center
        else:
            index = capped_trigonal_prism_solv_system.tag_manager.lookup_index_by_tag(tag)
            # This should correspond to the i-1 th solvation atom in the solvation system object
            assert index == i-1
            coord = capped_trigonal_prism_solv_system.atoms.get_positions()[index]

    # Change the colour of the furthest solvent atom to red
    seventh_mol_name = list(render_rep.atoms.keys())[-1]
    render_rep.update_atom_render_opt(seventh_mol_name, color="red")
    seventh_mol_options = render_rep.get_render_info_from_atom_name(seventh_mol_name)
    assert seventh_mol_options == {'color': 'red', 'radius': 0.2}
    
    # render_rep.add_atom_type_rendering(atom_type=fe_type)
    # options = render_rep.get_atom_rendering_options(atom_type=fe_type)
    # pl_inter = solvis.visualization.AtomicPlotter(interactive_mode=True, depth_peeling=True, shadows=False)
    # pl_inter.add_single_atom_as_sphere([0,0,0], color=fe_color, radius=0.1, **options)