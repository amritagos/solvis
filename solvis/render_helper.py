from .util import merge_options_keeping_override


class RendererRepresentation:
    def __init__(self):
        self.atoms = {}
        self.bonds = {}
        self.hydrogen_bonds = {}
        self.hulls = {}
        self.atom_type_rendering = {}
        self.bond_type_rendering = {}
        # These numbers do not reflect the actual numbers of such items
        # since they can be deleted. They are just there to bookkeep and make sure
        # you always have a unique actor_name
        self.num_atoms = 0
        self.num_bonds = 0
        self.num_hbonds = 0
        self.num_hulls = 0
        self.default_bond_color = "black"
        self.default_atom_color = "black"

    def add_atom_type_rendering(self, atom_type, color=None, **render_options):
        """
        Render options that you can put into add_single_atom_as_sphere in the plotter object
        """
        if color is None:
            color = self.default_atom_color
        # color should not be in render_options
        if color in render_options:
            del render_options["color"]
        atom_info = dict(color=color, render_options=render_options)
        self.atom_type_rendering[atom_type] = atom_info

    def add_bond_type_rendering(self, bond_type, color=None, **render_options):
        """
        Assign a particular color to all bonds of a particular bond type
        """
        if color is None:
            color = self.default_bond_color
        # color should not be in render_options
        if color in render_options:
            del render_options["color"]
        bond_info = dict(color=color, render_options=render_options)
        self.bond_type_rendering[bond_type] = bond_info

    def add_atom(
        self, atom_tag, atom_type, color=None, actor_name=None, **render_options
    ):
        """
        render_options overrides the render options provided by atom type.
        If you don't provide actor_name, a name is assigned using the number of atoms
        """
        # Get the render options for the atom type, if they exist
        if atom_type in self.atom_type_rendering:
            options = merge_options_keeping_override(
                self.atom_type_rendering[atom_type]["render_options"], render_options
            )
            if color is None:
                color = self.atom_type_rendering[atom_type]["color"]
        else:
            print("Warning: No atom type render options set for atom type", atom_type)
            options = render_options
            if color is None:
                print("Warning: Setting atom color to default.")
                color = self.default_atom_color
        self.num_atoms += 1
        if actor_name is not None:
            name = actor_name
        else:
            name = "atom_" + str(self.num_atoms)
        atom_info = {
            "tag": atom_tag,
            "type": atom_type,
            "color": color,
            "render_options": options,
        }
        self.atoms[name] = atom_info
        # Add to the number of atoms

    def add_hull(self, actor_name=None, **render_options):
        """
        for options to: add_hull(self,hull:PolyData, **color_or_additional_mesh_options)
        """
        self.num_hulls += 1
        if actor_name is not None:
            name = actor_name
        else:
            name = "hull_" + str(self.num_hulls)
        self.hulls[name] = render_options

    def add_hydrogen_bond(self, actor_name, atom_tags, bond_type):
        hydrogen_bond_info = {"tags": atom_tags, "type": bond_type}
        self.hydrogen_bonds[actor_name] = hydrogen_bond_info
        self.num_hbonds += 1

    def get_render_info_from_atom_name(self, actor_name):
        return self.atoms.get(actor_name).get("render_options")

    def get_color_from_atom_name(self, actor_name):
        return self.atoms.get(actor_name).get("color")

    def get_tag_from_atom_name(self, actor_name):
        """
        Get the atom tag, given the atom actor name in the self.atoms dictionary
        """
        return self.atoms.get(actor_name).get("tag")

    def get_atom_rendering_options(self, atom_type):
        return self.atom_type_rendering.get(atom_type).get("render_options")

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
            options = merge_options_keeping_override(
                self.atoms.get(actor_name).get("render_options"), render_options
            )
            self.atoms[actor_name]["render_options"] = options
            return True
        else:
            return False

    def update_atom_color(self, actor_name, color):
        """
        Function to change the color of an atom, given the actor_name

        Parameters:
        - actor_name: The key in the atoms dict whose value needs to be replaced.
        - render_options: The new render options.

        Returns:
        - True if the replacement was successful, False if actor_name was not found.
        """
        if actor_name in self.atoms:
            self.atoms[actor_name]["color"] = color
            return True
        else:
            return False

    def update_bond_render_opt(self, actor_name, **render_options):
        """
        Function to replace the value of a key in a flat dictionary.

        Parameters:
        - actor_name: The key in the atoms dict whose value needs to be replaced.
        - render_options: The new render options.

        Returns:
        - True if the replacement was successful, False if actor_name was not found.
        """
        if actor_name in self.bonds:
            # Merge with previous options, overwriting
            options = merge_options_keeping_override(
                self.bonds.get(actor_name).get("render_options"), render_options
            )
            self.bonds[actor_name]["render_options"] = options
            return True
        else:
            return False

    def update_hull_render_opt(self, actor_name, **render_options):
        """
        Function to replace the value of a key for the hull rendering options.

        Parameters:
        - actor_name: The key in the atoms dict whose value needs to be replaced.
        - render_options: The new render options.

        Returns:
        - True if the replacement was successful, False if actor_name was not found.
        """
        if actor_name in self.hulls:
            # Merge with previous options, overwriting
            options = merge_options_keeping_override(
                self.hulls.get(actor_name), render_options
            )
            self.hulls[actor_name] = options
            return True
        else:
            return False

    def find_color_by_atom_tag(self, target_atom_tag):
        """
        Function to find the 'color' based on the value of 'tag' in a nested dictionary.

        Parameters:
        - target_atom_tag: The value of 'tag' to search for.

        Returns:
        - The value associated with the 'color' option if found, otherwise None.
        """
        for key, value in self.atoms.items():
            if isinstance(value, dict):
                if "tag" in value and value["tag"] == target_atom_tag:
                    if "color" in value:
                        return value["color"]
        return None

    def add_bond(
        self,
        bond_tags,
        bond_type,
        colorby="atomcolor",
        actor_name=None,
        **render_options
    ):
        """
        render_options can override any of the following (default options are:)
        radius=0.1, resolution=1, bond_gradient_start=0.0, asymmetric_gradient_start=None,**mesh_options_override_default
        colorby: "atomcolor" or "bondtype". If not any of these options, set to the default color.
        TODO: test for bondtype also. Also function for updating colors of an existing bond
        """
        # We need a_color and b_color to render the bond
        if colorby == "atomcolor":
            # Get a_color and b_color from the atom colors saved in the atoms dict
            a_color = self.find_color_by_atom_tag(bond_tags[0])
            b_color = self.find_color_by_atom_tag(bond_tags[-1])
        elif colorby == "bondtype":
            if bond_type in self.bond_type_rendering.keys():
                a_color = self.bond_type_rendering[bond_type].get("color")
                b_color = a_color
            else:
                print(
                    "Warning: Setting bonds to default since colorby was bondtype, but bond_type color was not set\n"
                )
                a_color = self.default_bond_color
                b_color = a_color
        else:
            a_color = self.default_bond_color
            b_color = a_color
        # TODO: error handling here
        # Get the render options for the atom type, if they exist
        if bond_type in self.bond_type_rendering:
            options = merge_options_keeping_override(
                self.bond_type_rendering[bond_type].get("render_options"),
                render_options,
            )
        else:
            print("Warning: No bond type render options set for bond type", bond_type)
            options = render_options
        self.num_bonds += 1
        if actor_name is not None:
            name = actor_name
        else:
            name = "bond_" + str(self.num_bonds)
        bond_info = {
            "tags": bond_tags,
            "type": bond_type,
            "a_color": a_color,
            "b_color": b_color,
            "render_options": options,
        }
        self.bonds[name] = bond_info

    def update_bond_colors(self, actor_name, a_color, b_color):
        """
        Function to change the bond colors of a given bond actor

        Parameters:
        - actor_name: The key in the atoms dict whose value needs to be replaced.
        - render_options: The new render options.

        Returns:
        - True if the replacement was successful, False if actor_name was not found.
        """
        if actor_name in self.bonds:
            self.bonds[actor_name]["a_color"] = a_color
            self.bonds[actor_name]["b_color"] = b_color
            return True
        else:
            return False

    def delete_atom(self, actor_name):
        """
        Delete an atom from the self.atoms dict, given the actor name (key).
        If not found, returns False
        """
        if actor_name in self.atoms:
            del self.atoms[actor_name]
            return True
        else:
            return False

    def delete_bond(self, actor_name):
        """
        Delete a bond from the self.bonds dict, given the actor name (key).
        If not found, returns False
        """
        if actor_name in self.bonds:
            del self.bonds[actor_name]
            return True
        else:
            return False

    def delete_hull(self, actor_name):
        """
        Delete a hull from the self.bonds dict, given the actor name (key).
        If not found, returns False
        """
        if actor_name in self.hulls:
            del self.hulls[actor_name]
            return True
        else:
            return False
