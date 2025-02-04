import pytest
from ase.data import chemical_symbols
from ase.io import read
from pathlib import Path
import solvis
from solvis.visualization import AtomicPlotter
import pyvista as pv


def test_single_hbond(verify_image_cache):
    """
    Test that you can render a hydrogen bond between a Cl ion and water molecule
    """
    test_dir = Path(__file__).resolve().parent
    infilename = test_dir / "../resources/single_hbond.lammpstrj"
    # In the LAMMPS data file, the types of atoms are 1, 2 and 4 for O, H and Cl respectively.
    fe_type = 3
    cl_type = 4
    h_type = 2
    o_type = 1
    cl_o_cutoff = 3.3
    final_camera_position = [
        (30.956671343923677, 14.718458105502862, 22.676697734907997),
        (25.868474006652832, 22.091593742370605, 22.799850463867188),
        (-0.0986735300038882, -0.08465568627086995, 0.9915124554228151),
    ]
    # ------------------------------------------------------
    # Glean the coordinates from the LAMMPS trajectory file

    # Read in the current frame
    currentframe = read(
        infilename, format="lammps-dump-text"
    )  # Read in the last frame of the trajectory
    # Box size lengths
    box_len = currentframe.cell.cellpar()[:3]

    # Create a SolvationShell object from the solvent
    solvation_shell = solvis.solvation_shell.create_solvation_shell_from_solvent(
        currentframe, box_len, center=None
    )
    # ------------------------------------------------------------
    # Bond and atom appearance
    o_color = "midnightblue"
    o_radius = 0.3
    cl_radius = 0.4
    cl_color = "red"
    h_atom_radius = 0.2
    h_atom_color = "#d3d3d3"
    water_bond_thickness = 0.1
    # H bond parameters
    h_segment = 0.17545664400649102
    hbond_color = "orchid"
    hbond_width = 40
    # Decide render options for bonds
    bond_opt_1 = dict(
        bond_gradient_start=0.4, radius=water_bond_thickness, resolution=1
    )  # O-H bonds, gradient, according to atomcolor type
    # Render options for hydrogen bonds
    hbond_opt = dict(segment_spacing=h_segment, color=hbond_color, width=hbond_width)

    # Build the RendererRepresentation object
    render_rep = solvis.render_helper.RendererRepresentation()
    # Add the atom type and atom type specific rendering options
    render_rep.add_atom_type_rendering(atom_type=o_type, color=o_color, radius=o_radius)
    render_rep.add_atom_type_rendering(
        atom_type=h_type, color=h_atom_color, radius=h_atom_radius
    )
    render_rep.add_atom_type_rendering(
        atom_type=cl_type, color=cl_color, radius=cl_radius
    )
    # Add O-H intra-water bonds
    render_rep.add_bond_type_rendering(bond_type=2, color=h_atom_color, **bond_opt_1)
    # Add the hydrogen bond rendering
    render_rep.set_hbond_rendering(**hbond_opt)

    # Add the atoms from the solvation shell
    solvis.vis_initializers.fill_render_rep_atoms_from_solv_shell(
        render_rep, solvation_shell, include_center=False
    )

    # Add the intra-water bonds
    oh_bonds = solvis.bond_helper.find_intra_water_bonds(
        solvation_shell.atoms, o_type, h_type, box_len
    )
    solvis.vis_initializers.fill_render_rep_bonds_from_edges(
        render_rep, edges=oh_bonds, bond_type=2, colorby="atomcolor"
    )

    # Add the hydrogen bonds
    hbond_list = solvis.bond_helper.find_all_hydrogen_bonds(
        solvation_shell,
        [o_type],
        [cl_type, o_type],
        [h_type],
        donor_H_distance=1.0,
        donor_acceptor_cutoff=cl_o_cutoff,
        ignore_hydrogens=False,
    )
    solvis.vis_initializers.fill_render_rep_hbonds_from_edges(
        render_rep, edges=hbond_list
    )

    # Add the atom labels
    # atom label for O
    # Find the O (there is only one in this system)
    o_ind = solvation_shell.atoms.symbols.search(chemical_symbols[o_type])
    o_tag = solvation_shell.atoms.get_tags()[o_ind[0]]
    o_actor_name = render_rep.get_atom_name_from_tag(target_atom_tag=o_tag)
    render_rep.set_atom_label(o_actor_name, "O")
    # Add the label for Cl (there is only one)
    cl_ind = solvation_shell.atoms.symbols.search(chemical_symbols[cl_type])
    cl_tag = solvation_shell.atoms.get_tags()[cl_ind[0]]
    cl_actor_name = render_rep.get_atom_name_from_tag(target_atom_tag=cl_tag)
    render_rep.set_atom_label(cl_actor_name, "Cl")
    # ------------------------------------------------------------

    pl_render = AtomicPlotter(
        interactive_mode=False,
        window_size=[4000, 4000],
        depth_peeling=False,
        shadows=False,
    )
    # Populate plotter using the RendererRepresentation object
    solvis.vis_initializers.populate_plotter_from_solv_shell(
        pl_render,
        render_rep,
        solvation_shell,
        convex_hull_list=None,
        always_visible=True,
        show_points=False,
        font_size=186,
    )
    # Add a description of the image
    pl_render.plotter.add_text(
        "D-A <= 3.3 Angstrom; angle HDA <=30",
        position="upper_left",
        font_size=186,
        color="black",
        font="arial",
        shadow=False,
        name=None,
        viewport=False,
        orientation=0.0,
        font_file=None,
        render=True,
    )
    # Set the camera position
    pl_render.plotter.camera_position = final_camera_position
    pl_render.plotter.camera_set = True

    # Verify image
    pl_render.plotter.show(auto_close=True, interactive=False)
    pl_render.close()
