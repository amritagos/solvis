import pytest
import numpy as np
from ase.io import read
from ase.atoms import Atom, Atoms
from ase.data import chemical_symbols
from pathlib import Path

import solvis


@pytest.fixture
def capped_trigonal_prism_solv_system():
    """Reads in a LAMMPS trajectory of a single step, and gets the solvation shell
    corresponding to O (type 1) solvent atoms surrounding a Fe atom (type 3)
    """
    # In the LAMMPS trajectory file, the types of atoms are 1, 2 and 3 for O, H and Fe respectively.
    fe_type = 3
    h_type = 2
    o_type = 1

    test_dir = Path(__file__).resolve().parent
    infilename = test_dir / "../resources/single_capped_trigonal_prism.lammpstrj"
    # Read in the current frame
    atoms = read(
        infilename, format="lammps-dump-text"
    )  # Read in the last frame of the trajectory

    ctp_system = solvis.system.System(atoms, expand_box=True)

    fe_ind = ctp_system.atoms.symbols.search(chemical_symbols[fe_type])
    fe_pos_query_pnt = ctp_system.atoms.get_positions()[fe_ind[0]]

    solvation_shell = ctp_system.create_solvation_shell_from_center(
        central_pnt=fe_pos_query_pnt,
        num_neighbours=7,
        contains_solvation_center=True,
        solvent_atom_types=[1],
    )

    return solvation_shell


def test_renderer_helper_ctp_system(capped_trigonal_prism_solv_system):
    """
    Test that the RendererRepresentation can accurately describe elements required for plotting.
    The closest six O atoms should be a dark blue, and the seventh closest should be red.
    We should be able to delete elements from a RendererRepresentation object as well.
    """

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
    fe_ind = capped_trigonal_prism_solv_system.atoms.symbols.search(
        chemical_symbols[fe_type]
    )
    # O atom indices
    o_ind = capped_trigonal_prism_solv_system.atoms.symbols.search(
        chemical_symbols[o_type]
    )

    render_rep = solvis.render_helper.RendererRepresentation()

    # Add the atom type and atom type specific rendering options that could go into the
    # add_single_atom_as_sphere function of the plotter
    render_rep.add_atom_type_rendering(
        atom_type=fe_type, color=fe_color, radius=fe_radius
    )
    assert render_rep.atom_type_rendering[fe_type].get("color") == "black"
    fe_atom_options = render_rep.get_atom_rendering_options(atom_type=fe_type)
    assert fe_atom_options == {"radius": 0.3}
    # Atom type of the O atoms
    render_rep.add_atom_type_rendering(atom_type=o_type, color=o_color, radius=o_radius)
    assert render_rep.atom_type_rendering[o_type].get("color") == "midnightblue"
    o_atom_options = render_rep.get_atom_rendering_options(atom_type=o_type)
    assert o_atom_options == {"radius": 0.2}

    # Now add the Fe atom as the center
    render_rep.add_atom(
        "center", fe_type, color=None, actor_name="fe"
    )  # Handle atom_tag='center' for solvation center! TODO
    assert render_rep.num_atoms == 1
    # Loop through the solvation atoms and add them
    for i, solv_atom in enumerate(capped_trigonal_prism_solv_system.atoms):
        iatom_type = solv_atom.number
        iatom_tag = solv_atom.tag
        render_rep.add_atom(iatom_tag, iatom_type)

    # There should be 8 atoms in the renderer represenation
    assert render_rep.num_atoms == len(capped_trigonal_prism_solv_system.atoms) + 1

    # Retreive the indices and coord using atom tags from the solvation shell object
    for i, name in enumerate(render_rep.atoms.keys()):
        tag = render_rep.get_tag_from_atom_name(name)
        if tag == "center":
            coord = capped_trigonal_prism_solv_system.center
        else:
            index = capped_trigonal_prism_solv_system.tag_manager.lookup_index_by_tag(
                tag
            )
            # This should correspond to the i-1 th solvation atom in the solvation system object
            assert index == i - 1
            coord = capped_trigonal_prism_solv_system.atoms.get_positions()[index]

    # Change the colour of the furthest solvent atom to red
    seventh_mol_name = list(render_rep.atoms.keys())[-1]
    render_rep.update_atom_color(seventh_mol_name, color="red")
    seventh_mol_options = render_rep.get_render_info_from_atom_name(seventh_mol_name)
    assert seventh_mol_options == {"radius": 0.2}
    # Check that you can access the new color of this atom from the atom tag
    # This can be used for getting bond colors
    seventh_mol_tag = render_rep.atoms.get(seventh_mol_name).get("tag")
    seventh_mol_color = render_rep.find_color_by_atom_tag(seventh_mol_tag)
    assert seventh_mol_color == "red"

    # Create a bond between the first and last atoms
    # Coloured by the atom color, so a_color='midnightblue' and b_color='red'
    tag1 = capped_trigonal_prism_solv_system.atoms[0].tag
    tag2 = capped_trigonal_prism_solv_system.atoms[-1].tag
    assigned_bond_type = 1
    render_rep.add_bond(
        [tag1, tag2], assigned_bond_type, colorby="atomcolor", resolution=1
    )
    bond_info = {
        "tags": [3, 10],
        "type": 1,
        "a_color": "midnightblue",
        "b_color": "red",
        "render_options": {"resolution": 1},
    }
    assert render_rep.bonds["bond_1"] == bond_info

    # Add a bond type color
    render_rep.add_bond_type_rendering(bond_type=assigned_bond_type, color="dimgrey")
    assert render_rep.bond_type_rendering[assigned_bond_type].get("color") == "dimgrey"

    # Add a bond between the first and second atoms, coloured according to the bond_type color
    tag3 = capped_trigonal_prism_solv_system.atoms[1].tag
    render_rep.add_bond(
        [tag1, tag3], assigned_bond_type, colorby="bondtype", resolution=1
    )
    bond2_info = {
        "tags": [3, 4],
        "type": 1,
        "a_color": "dimgrey",
        "b_color": "dimgrey",
        "render_options": {"resolution": 1},
    }
    assert render_rep.bonds["bond_2"] == bond2_info

    # Change the color of bond2 to black and red
    render_rep.update_bond_colors(actor_name="bond_2", a_color="black", b_color="red")
    assert render_rep.bonds["bond_2"].get("a_color") == "black"
    assert render_rep.bonds["bond_2"].get("b_color") == "red"
    # The other bond colours shouldn't have changed
    assert render_rep.bonds["bond_1"].get("a_color") == "midnightblue"
    assert render_rep.bonds["bond_1"].get("b_color") == "red"

    # Add hull rendering options
    hull_options = dict(
        color="grey", show_edges=True, line_width=5, lighting=True, opacity=0.5
    )
    render_rep.add_hull(**hull_options)
    assert render_rep.hulls["hull_1"].get("color") == "grey"
    # Test that you can change the hull color to blue
    render_rep.update_hull_render_opt("hull_1", color="skyblue")
    assert render_rep.hulls["hull_1"].get("color") == "skyblue"

    # Test that you can delete things from the RendererRepresentation object
    assert "fe" in render_rep.atoms.keys()
    render_rep.delete_atom("fe")
    assert "fe" not in render_rep.atoms.keys()
    # Delete the bonds
    render_rep.delete_bond("bond_1")
    render_rep.delete_bond("bond_2")
    assert len(render_rep.bonds) == 0
    # Delete the hull
    render_rep.delete_hull("hull_1")
    assert len(render_rep.hulls) == 0
    # This number refers to the number of hulls added, not the actual number of hulls
    assert render_rep.num_hulls == 1

    # render_rep.add_atom_type_rendering(atom_type=fe_type)
    # options = render_rep.get_atom_rendering_options(atom_type=fe_type)
    # pl_inter = solvis.visualization.AtomicPlotter(interactive_mode=True, depth_peeling=True, shadows=False)
    # pl_inter.add_single_atom_as_sphere([0,0,0], color=fe_color, radius=0.1, **options)


def test_renderer_helper_populate(capped_trigonal_prism_solv_system):
    """
    Test that the RendererRepresentation can be filled using a solvation system.
    """

    # In the LAMMPS trajectory file, the types of atoms are 1, 2 and 3 for O, H and Fe respectively.
    # There are no H atoms in this particular trajectory
    fe_type = 3
    o_type = 1
    # Desired colors and radii for atoms
    fe_color = "black"
    fe_radius = 0.3
    o_color = "midnightblue"
    o_radius = 0.2

    render_rep = solvis.render_helper.RendererRepresentation()

    # Set the atom and bond types
    # The center is of atom type 0
    render_rep.add_atom_type_rendering(atom_type=0, color=fe_color, radius=fe_radius)
    render_rep.add_atom_type_rendering(atom_type=o_type, color=o_color, radius=o_radius)
    # Bond type for a bond type of 1 (start from 1)
    bond_opt = dict(radius=0.1, resolution=1, bond_gradient_start=0.3)
    render_rep.add_bond_type_rendering(
        bond_type=1, color="dimgrey", **bond_opt
    )  # color is optional

    # Fill the render_rep object
    solvis.vis_initializers.fill_render_rep_atoms_from_solv_shell(
        render_rep, capped_trigonal_prism_solv_system, include_center=True
    )
    # There should be 8 atoms in the renderer represenation
    assert render_rep.num_atoms == len(capped_trigonal_prism_solv_system.atoms) + 1

    # Change the colour of the seventh atom
    seventh_mol_name = list(render_rep.atoms.keys())[-1]
    render_rep.update_atom_color(seventh_mol_name, color="red")
    seventh_mol_options = render_rep.get_render_info_from_atom_name(seventh_mol_name)

    # Add bonds from the center
    solvis.vis_initializers.fill_render_rep_bonds_from_solv_shell_center(
        render_rep, capped_trigonal_prism_solv_system, bond_type=1, colorby="atomcolor"
    )
    assert render_rep.bonds["bond_1"] == {
        "tags": ["center", 3],
        "type": 1,
        "a_color": "black",
        "b_color": "midnightblue",
        "render_options": {"radius": 0.1, "resolution": 1, "bond_gradient_start": 0.3},
    }
    assert render_rep.bonds["bond_7"] == {
        "tags": ["center", 10],
        "type": 1,
        "a_color": "black",
        "b_color": "red",
        "render_options": {"radius": 0.1, "resolution": 1, "bond_gradient_start": 0.3},
    }


def test_renderer_hbond(capped_trigonal_prism_solv_system):
    """
    Test hydrogen bonding options for the RendererRepresentation object
    """

    render_rep = solvis.render_helper.RendererRepresentation()

    # The default hbond render options should be:
    assert render_rep.hbond_rendering == {
        "segment_spacing": 0.175,
        "color": "grey",
        "width": 10.0,
    }

    # Change the color in the hbond render options
    render_rep.set_hbond_rendering(color="orchid")
    assert render_rep.hbond_rendering == {
        "segment_spacing": 0.175,
        "color": "orchid",
        "width": 10.0,
    }
