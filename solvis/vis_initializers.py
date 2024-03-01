from .solvation_shell import SolvationShell
from .render_helper import RendererRepresentation
from .visualization import AtomicPlotter

def fill_render_rep_atoms_from_solv_shell(render_rep, solv_shell, include_center=False):
    """
    Fill up a given RendererRepresentation object with atoms from a SolvationShell.
    The center can optionally be included. The atom types must have ALREADY been set 
    if you want atom type colors etc included. 
    """
    if include_center:
        render_rep.add_atom(atom_tag='center', atom_type=0, actor_name='center')

    for solv_atom in solv_shell.atoms:
        iatom_type = solv_atom.number
        iatom_tag = solv_atom.tag
        render_rep.add_atom(atom_tag=iatom_tag, atom_type=iatom_type)