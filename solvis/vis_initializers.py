from .solvation_shell import SolvationShell
from .render_helper import RendererRepresentation
from .visualization import AtomicPlotter
from .geometric_utils import ConvexHull


def fill_render_rep_atoms_from_solv_shell(render_rep, solv_shell, include_center=False):
    """
    Fill up a given RendererRepresentation object with atoms from a SolvationShell.
    The center can optionally be included. The atom types must have ALREADY been set
    if you want atom type colors etc included.
    """
    if include_center:
        render_rep.add_atom(atom_tag="center", atom_type=0, actor_name="center")

    for solv_atom in solv_shell.atoms:
        iatom_type = solv_atom.number
        iatom_tag = solv_atom.tag
        render_rep.add_atom(atom_tag=iatom_tag, atom_type=iatom_type)


def fill_render_rep_bonds_from_solv_shell_center(
    render_rep, solv_shell, bond_type, colorby
):
    """
    Fill up a given RendererRepresentation object with bonds, emanating from the center of a SolvationShell.
    The bond type must have ALREADY been set.
    colorby: "atomcolor" or "bondtype"
    """
    bond_render_opt = render_rep.bond_type_rendering[bond_type].get("render_options")

    for solv_atom in solv_shell.atoms:
        iatom_tag = solv_atom.tag
        render_rep.add_bond(
            ["center", iatom_tag], bond_type, colorby=colorby, **bond_render_opt
        )


def coord_from_tag(atom_tag, solv_shell):
    """
    Get the coordinate, given an atom tag (if center, then use SolvationShell center)
    """
    if atom_type == "center":
        return solv_shell.center
    else:  # TODO some error handling?
        index = solv_shell.tag_manager.lookup_tag_by_index(atom_tag)
        return solv_shell.atoms.get_positions()[index]


def populate_plotter_from_solv_shell(
    plotter, render_rep, solv_shell, convex_hull_list=None
):
    """
    AFTER initializing the AtomicPlotter object, fill it up with atoms, bonds and hulls from
    the render_rep, referencing elements inside the SolvationShell object
    Optionally, add a convex hull. The convex hulls should be a ConvexHull object
    """
    # Add the atoms
    # Loop through all atoms
    for actor_name in render_rep.atoms.keys():
        atom_tag = render_rep.atoms[actor_name]["tag"]
        atom_coord = coord_from_tag(atom_tag, solv_shell)
        render_opt = render_rep.atoms[actor_name]["render_options"]
        atom_color = render_rep.atoms[actor_name]["color"]
        # Add the atom to the plotter
        plotter.add_single_atom_as_sphere(
            point=atom_coord, color=atom_color, actor_name=actor_name, **render_opt
        )

    # Add the bonds
    # Loop through all bonds
    bond_info = {
        "tags": bond_tags,
        "type": bond_type,
        "a_color": a_color,
        "b_color": b_color,
        "render_options": options,
    }
    for actor_name in render_rep.bonds.keys():
        bond_tags = render_rep.bonds[actor_name]["tags"]
        coord_a = coord_from_tag(bond_tags[0], solv_shell)
        coord_b = coord_from_tag(bond_tags[1], solv_shell)
        a_color = render_rep.bonds[actor_name]["a_color"]
        b_color = render_rep.bonds[actor_name]["b_color"]
        render_opt = render_rep.bonds[actor_name]["render_options"]
        # Add the bond to the plotter
        plotter.add_bond(
            pointa=coord_a,
            pointb=coord_b,
            a_color=a_color,
            b_color=b_color,
            actor_name=actor_name,
            **render_opt
        )

    if convex_hull_list is None and len(render_rep.hulls.keys()) > 0:
        print("Convex hull list not provided")
        return
        # Convert the convex hull into a pyvista PolyData object
        polyhull = convex_hull.pyvista_hull_from_convex_hull()
    # Add the hulls
    for i, actor_name in enumerate(render_rep.hulls.keys()):
        # Convert the convex hull into a pyvista PolyData object
        polyhull = convex_hull_list[i].pyvista_hull_from_convex_hull()
        render_opt = render_rep.hulls[actor_name]["render_options"]
        # Add the hull to the plotter
        plotter.add_hull(polyhull, **render_opt)
