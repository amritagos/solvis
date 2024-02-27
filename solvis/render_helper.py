from .util import merge_options

class RendererRepresentation:
    def __init__(self):
        self.atoms = {}
        self.bonds = {}
        self.hydrogen_bonds = {}
        self.atom_type_rendering = {}
        self.num_atoms = 0
        self.num_bonds = 0
        self.num_hbonds = 0

    def add_atom_type_rendering(self, atom_type, **render_options):
        """
        Render options that you can put into add_single_atom_as_sphere in the plotter object 
        """
        self.atom_type_rendering[atom_type] = render_options

    def add_atom(self, actor_name, atom_tag, atom_type, **render_options):
        """
        render_options overrides the render options provided by atom type. TODO: change render_options too 
        """
        # Get the render options for the atom type, if they exist
        if atom_type in self.atom_type_rendering:
            options = merge_options(self.atom_type_rendering[atom_type], render_options)
        else:
            print("Warning: No atom type render options set for atom type", atom_type)
            options = render_options
        atom_info = {'tag': atom_tag, 'type': atom_type, 'render_options': options}
        self.atoms[actor_name] = atom_info
        # Add to the number of atoms 
        self.num_atoms += 1

    def add_bond(self, actor_name, bond_tags, bond_type):
        bond_info = {'tags': bond_tags, 'type': bond_type}
        self.bonds[actor_name] = bond_info
        self.num_bonds += 1

    def add_hydrogen_bond(self, actor_name, atom_tags, bond_type):
        hydrogen_bond_info = {'tags': atom_tags, 'type': bond_type}
        self.hydrogen_bonds[actor_name] = hydrogen_bond_info
        self.num_hbonds += 1

    def get_render_info_from_atom_name(self, actor_name):
        return self.atoms.get(actor_name).get('render_options')

    def get_tag_from_atom_name(self, actor_name):
        return self.atoms.get(actor_name).get('tag')

    def get_atom_rendering_options(self, atom_type):
        return self.atom_type_rendering.get(atom_type, {})

    def get_bond_info(self, actor_name):
        return self.bonds.get(actor_name)

    def get_hydrogen_bond_info(self, actor_name):
        return self.hydrogen_bonds.get(actor_name)

    def update_atom_render_opt(self, actor_name, **render_options):
        """
        Function to replace the value of a key in a flat dictionary.

        Parameters:
        - actor_name: The key in the atoms dict whose value needs to be replaced.
        - render_options: The new render options.

        Returns:
        - True if the replacement was successful, False if actor_name was not found.
        """
        if actor_name in self.atoms:
            # Merge with previous options, overwriting
            options = merge_options(self.atoms.get(actor_name).get('render_options'), render_options)
            self.atoms[actor_name]['render_options'] = options
            return True
        else:
            return False

    def find_color_option_by_atom_tag(self, target_atom_tag):
        """
        Function to find the 'color' option in 'render_options' based on the value of 'tag' in a nested dictionary.

        Parameters:
        - target_atom_tag: The value of 'tag' to search for.

        Returns:
        - The value associated with the 'color' option if found, otherwise None.
        """
        for key, value in self.atoms.items():
            if isinstance(value, dict):
                if 'tag' in value and value['tag'] == target_atom_tag:
                    if 'render_options' in value and 'color' in value['render_options']:
                        return value['render_options']['color']
        return None