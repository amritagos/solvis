class RendererRepresentation:
    def __init__(self):
        self.atoms = {}
        self.bonds = {}
        self.hydrogen_bonds = {}
        self.atom_type_rendering = {}

    def add_atom_type_rendering(self, atom_type, **render_options):
        self.atom_type_rendering[atom_type] = render_options

    def add_atom(self, actor_name, atom_tag, atom_type, **render_options):
        """
        render_options overrides the render options provided by atom type 
        """
        atom_info = {'tag': atom_tag, 'type': atom_type}
        self.atoms[actor_name] = atom_info

    def add_bond(self, actor_name, bond_tags, bond_type):
        bond_info = {'tags': bond_tags, 'type': bond_type}
        self.bonds[actor_name] = bond_info

    def add_hydrogen_bond(self, actor_name, atom_tags, bond_type):
        hydrogen_bond_info = {'tags': atom_tags, 'type': bond_type}
        self.hydrogen_bonds[actor_name] = hydrogen_bond_info

    def get_atom_info(self, actor_name):
        return self.atoms.get(actor_name)

    def get_bond_info(self, actor_name):
        return self.bonds.get(actor_name)

    def get_hydrogen_bond_info(self, actor_name):
        return self.hydrogen_bonds.get(actor_name)

    def get_rendering_options(self, atom_type):
        return self.atom_type_rendering.get(atom_type, {})

