from typing import Any, Optional
from .util import merge_options_keeping_override, merge_options
import warnings


class RenderingUnitTypeInfo:
    def __init__(self, color: Optional[str] = None, **render_options):
        """Contains rendering information about an atom type or bond type

        Args:
            color (str, optional): Color corresponding to an integral atom type or bond type, which will be applied to all atoms of this atom type. Defaults to None.
            render_options: Additional PyVista rendering options for the atom
        """
        if color is None:
            color = "black"  # Default atom or bond color
        self.color = color
        if self.color in render_options:
            del render_options["color"]
        self.render_options = render_options


class SingleAtomInfo:
    def __init__(
        self,
        name: str,
        tag: int,
        atom_type: int,
        color: str,
        label: Optional[str] = None,
        **render_options,
    ):
        """
        Rendering information for a single atom.

        Args:
            name (str): PyVista actor name for the atom
            tag (int): Atom tag
            type (int): Atom type
            color (str): Color of the atom. Must correspond to a color recognized by PyVista
            label (Optional[str], optional): Text label for the atom. Defaults to None, meaning the atom will have no label.
        """
        self.name = name
        self.tag = tag
        self.type = atom_type
        self.color = color
        self.label = label
        self.render_options = render_options


class AllAtomInfo:
    def __init__(self):
        """
        Rendering information for all atoms. Contains a list of SingleAtomInfo objects (and some other things)
        """
        self.atoms = []
        # These numbers do not reflect the actual numbers of such items
        # since they can be deleted. They are just there to bookkeep and make sure
        # you always have a unique actor_name
        self.num_atoms = 0
        self.default_atom_color = "black"

    def _first(self, iterable) -> Optional[SingleAtomInfo]:
        """Returns the first match, if any

        Returns:
            Optional[SingleAtomInfo]: Returns None if not found
        """
        for item in iterable:
            return item
        return None

    def find_atom_with_name(self, actor_name: str) -> Optional[SingleAtomInfo]:
        """Get the first SingleAtomInfo object such that the actor name matches the target actor_name

        Args:
            actor_name (str): Target actor name to match against

        Returns:
            Optional[SingleAtomInfo]: Either the SingleAtomInfo object or None
        """
        return self._first(x for x in self.atoms if x.name == actor_name)

    def find_atom_with_tag(self, tag: int) -> Optional[SingleAtomInfo]:
        """Get the first SingleAtomInfo object such that the atom tag matches the target tag

        Args:
            tag (str): Target atom tag to match against

        Returns:
            Optional[SingleAtomInfo]: Either the SingleAtomInfo object or None
        """
        return self._first(x for x in self.atoms if x.tag == tag)

    def find_first_atom_index_with_name(self, actor_name: str) -> Optional[str]:
        """Find the index of the atom info corresponding to a given actor name

        Args:
            actor_name (str): The PyVista actor name

        Returns:
            Optional[str]: Returns None if not found, or else returns the index in atoms
        """
        return self._first(i for i, x in enumerate(self.atoms) if x.name == actor_name)

    def add_atom(
        self,
        tag: int,
        atom_type: int,
        color: str,
        actor_name: Optional[str] = None,
        label: Optional[str] = None,
        **render_options,
    ) -> None:
        """Add a new atom

        Args:
            name (str): If you don't provide actor_name, a name is assigned using the number of atoms. Defaults to None; i.e. a name is automatically assigned.
            tag (int): Atom tag
            type (int): Atom type
            color (str): Color of the atom. Must be accepted by PyVista
            label (Optional[str], optional): Text label for the atom. Defaults to None; i.e. no label.
            render_options: PyVista rendering options
        """
        self.num_atoms += 1
        # Get the name of the actor
        if actor_name is not None:
            name = actor_name
        else:
            name = "atom_" + str(self.num_atoms)
        # If the actor name already exists, ovewrite it, while printing a warning
        existing_index = self.find_first_atom_index_with_name(name)
        new_atom_info = SingleAtomInfo(
            name=name,
            tag=tag,
            atom_type=atom_type,
            color=color,
            label=label,
            **render_options,
        )
        if existing_index is not None:
            # overwrite it
            warnings.warn(
                f"Overwriting atom information values for an actor with name {name}"
            )
            self.atoms[existing_index] = new_atom_info
        else:
            # Add a new atom
            self.atoms.append(new_atom_info)


class RendererRepresentation:
    """Class for handling all the rendering information for each atom (rendered as a sphere),
    bond (rendered as a cylinder), hull (convex shape) and hydrogen bond (dotted line).
    """

    def __init__(self):
        self.atoms = AllAtomInfo()
        self.bonds = {}
        self.hydrogen_bonds = {}
        self.hulls = {}
        self.atom_type_rendering = {}
        self.bond_type_rendering = {}

        self.num_bonds = 0
        self.num_hbonds = 0
        self.num_hulls = 0
        self.default_bond_color = "black"

        # Hydrogen bond rendering
        self.hbond_rendering = dict(segment_spacing=0.175, color="grey", width=10.0)

    def add_atom_type_rendering(
        self, atom_type: int, color: Optional[str] = None, **render_options
    ):
        """Render options that you can put into add_single_atom_as_sphere in the plotter object

        Args:
            atom_type (int): Integral atom type
            color (str, optional): Color corresponding to the atom type. Must be a color recognized by PyVista. Defaults to a color black if None.
            render_options: Additional PyVista rendering options for the atom, rendered as a sphere
        """
        self.atom_type_rendering[atom_type] = RenderingUnitTypeInfo(
            color=color, **render_options
        )

    def add_bond_type_rendering(
        self, bond_type: int, color: Optional[str] = None, **render_options
    ):
        """Assign a particular color to all bonds of a particular bond type

        Args:
            bond_type (int): Integral bond type
            color (Optional[str], optional): Color corresponding to the bond type. Must be a color recognized by PyVista. Defaults to a color black if None.
            render_options: Additional PyVista rendering options for the bond, rendered as a cylinder
        """
        self.bond_type_rendering[bond_type] = RenderingUnitTypeInfo(
            color=color, **render_options
        )

    def set_hbond_rendering(self, **render_options):
        """
        Set the segment_spacing, color or width of hydrogen bonds
        Rendered as a dashed line

        Args:
            render_options: Rendering options for the hydrogen bond, rendered as a dashed line
        """
        options = merge_options(self.hbond_rendering, render_options)
        self.hbond_rendering = options

    """
        render_options overrides the render options provided by atom type.
        If you don't provide actor_name, a name is assigned using the number of atoms
        """

    def add_atom(
        self,
        atom_tag: int,
        atom_type: int,
        color: Optional[str] = None,
        actor_name: Optional[str] = None,
        label: Optional[str] = None,
        **render_options,
    ):
        """
        Adds an atom to the RendererRepresentation object. All options you enter will override properties set by the atom type.

        Args:
            atom_tag (int): The tag of the atom
            atom_type (int): The atom type
            color (Optional[str], optional): Color of the atom. This overrides the color set by the atom type. Defaults to None, which means the color set by the atom type is used, if the atom type information has previously been set.
            actor_name (Optional[str], optional): The name of the PyVista actor. If you don't provide actor_name, a name is assigned using the number of atoms. Defaults to None; i.e. a name is automatically assigned.
            label (Optional[str], optional): A text label of the atom. Defaults to None.
            render_options: Additional PyVista options that overrides the render options provided by the atom type
        """
        # Get the render options for the atom type, if they exist
        if atom_type in self.atom_type_rendering:
            options = merge_options_keeping_override(
                self.atom_type_rendering[atom_type].render_options, render_options
            )
            if color is None:
                color = self.atom_type_rendering[atom_type].color
        else:
            warnings.warn(f"No atom type render options set for atom type {atom_type}")
            options = render_options
            if color is None:
                warnings.warn("Setting atom color to default.")
                color = self.default_atom_color
        self.atoms.add_atom(
            tag=atom_tag,
            atom_type=atom_type,
            color=color,
            actor_name=actor_name,
            label=label,
            **options,
        )

    def add_hull(self, actor_name: Optional[str] = None, **render_options) -> None:
        """for options to: add_hull(self,hull:PolyData, **color_or_additional_mesh_options)

        Args:
            actor_name (Optional[str], optional): The name of the PyVista actor. If you don't provide one, a name is
            generated using the number of hulls added. Defaults to None.
        """
        self.num_hulls += 1
        if actor_name is not None:
            name = actor_name
        else:
            name = "hull_" + str(self.num_hulls)
        self.hulls[name] = render_options

    """
        
        """

    def add_hydrogen_bond(
        self, bond_tags: list[int], actor_name: Optional[str] = None, **render_options
    ) -> None:
        """Adds a hydrogen bond. The kwargs render_options can override any of the following (default options are:)
        segment_spacing=0.175, color="grey", width=10.0
        Hydrogen bonds will be rendered as dashed lines

        Args:
            bond_tags (list[int]): Tags corresponding to the two atoms in the bond.
            actor_name (Optional[str], optional): The name of the PyVista actor. If you don't provide one, a name is
            generated using the number of hydrogen bonds added. Defaults to None.
        """
        options = merge_options(self.hbond_rendering, render_options)

        self.num_hbonds += 1
        if actor_name is not None:
            name = actor_name
        else:
            name = "hbond_" + str(self.num_hbonds)

        hydrogen_bond_info = {"tags": bond_tags, "render_options": options}
        self.hydrogen_bonds[name] = hydrogen_bond_info

    def get_render_info_from_atom_name(self, actor_name: str) -> Optional[dict]:
        """Obtain the rendering information, given the actor name of an atom

        Args:
            actor_name (str): Actor name

        Returns:
            Optional[dict[str]]: Dictionary containing rendering options, or None if the actor_name cannot be found.
        """
        atom_info = self.atoms.find_atom_with_name(actor_name=actor_name)
        if atom_info is None:
            return None
        else:
            return atom_info.render_options

    def get_color_from_atom_name(self, actor_name: str) -> Optional[str]:
        """Obtain the color of a particular atom, given its actor name

        Args:
            actor_name (str): Actor name of the atom

        Returns:
            Optional[str]: Color of the atom, or None if actor_name cannot be found.
        """
        atom_info = self.atoms.find_atom_with_name(actor_name=actor_name)
        if atom_info is None:
            return None
        else:
            return atom_info.color

    def get_tag_from_atom_name(self, actor_name: str) -> Optional[int]:
        """Get the atom tag of an atom, given its actor name

        Args:
            actor_name (str): Actor name of the atom

        Returns:
            Optional[int]: Atom tag or None of actor_name cannot be found
        """
        atom_info = self.atoms.find_atom_with_name(actor_name=actor_name)
        if atom_info is None:
            return None
        else:
            return atom_info.tag

    def get_atom_name_from_tag(self, target_atom_tag: int) -> Optional[str]:
        """Get the actor name, given the tag of the atom

        Args:
            target_atom_tag (int): Atom tag

        Returns:
            Optional[str]: The actor name, or None if it does not exist
        """
        atom_info = self.atoms.find_atom_with_tag(target_atom_tag)
        if atom_info is not None:
            return atom_info.name
        else:
            return None

    def get_atom_rendering_options(self, atom_type: int) -> Optional[dict]:
        """Get the rendering options given the atom type

        Args:
            atom_type (int): Atom type of a particular atom

        Returns:
            Optional[dict]:  Dictionary containing rendering options, or None if the actor_name cannot be found.
        """
        return self.atom_type_rendering.get(atom_type).render_options

    def update_atom_render_opt(self, actor_name: str, **render_options) -> bool:
        """Update the rendering options of an atom, given the actor name

        Args:
            actor_name (str): PyVista actor name of the atom
            render_options: The new render options.

        Returns:
            bool: True if the replacement was successful, False if actor_name was not found.
        """
        atom_index = self.atoms.find_first_atom_index_with_name(actor_name=actor_name)
        if atom_index is None:
            return False  # the atom corresponding to the name was not found
        else:
            # Merge with previous options, overwriting
            options = merge_options_keeping_override(
                self.atoms.atoms[atom_index].render_options, render_options
            )
            self.atoms.atoms[atom_index].render_options = options
            return True

    """
        Function to change the color of an atom, given the actor_name

        Parameters:
        actor_name: The key in the atoms dict whose value needs to be replaced.
        render_options: The new render options.

        Returns:
        - True if the replacement was successful, False if actor_name was not found.
        """

    def update_atom_color(self, actor_name: str, color: str) -> bool:
        """Function to change the color of an atom, given the actor_name

        Args:
            actor_name (str): The PyVista actor name
            color (str): The new color

        Returns:
            bool: True if the replacement was successful, False if the actor_name was not found
        """
        atom_index = self.atoms.find_first_atom_index_with_name(actor_name=actor_name)
        if atom_index is None:
            return False  # the atom corresponding to the name was not found
        else:
            self.atoms.atoms[atom_index].color = color
            return True

    def set_atom_label(self, actor_name: str, label: str) -> bool:
        """Function to change or set the label of an atom, given the actor_name

        Args:
            actor_name (str): The (unique) PyVista actor name
            label (str): The text of the label.

        Returns:
            bool: True if the replacement was successful, False if actor_name was not found.
        """
        atom_index = self.atoms.find_first_atom_index_with_name(actor_name=actor_name)
        if atom_index is None:
            return False  # the atom corresponding to the name was not found
        else:
            self.atoms.atoms[atom_index].label = label
            return True

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

    def find_color_by_atom_tag(self, target_atom_tag: int) -> Optional[str]:
        """Function to find the 'color' of an atom, based on the value of the 'tag' of the atom

        Args:
            target_atom_tag (int): The atom tag of the atom

        Returns:
            Optional[str]: The value associated with color of the atom, otherwise None.
        """
        atom_info = self.atoms.find_atom_with_tag(target_atom_tag)
        if atom_info is not None:
            return atom_info.color
        else:
            return None

    def add_bond(
        self,
        bond_tags: list[int],
        bond_type: int,
        colorby: str = "atomcolor",
        actor_name: Optional[str] = None,
        **render_options,
    ):
        """Add a bond to the RendererRepresentation object. The kwargs render_options can override any of the following (default options are:)
        radius=0.1, resolution=1, bond_gradient_start=0.0, asymmetric_gradient_start=None,**mesh_options_override_default
        colorby: "atomcolor" or "bondtype". If not any of these options, set to the default color.

        Args:
            bond_tags (list[int]): Tags of the two atoms in the bond
            bond_type (int): Integral bond type of the bond
            colorby (str, optional): Decides how the bond will be coloured. Defaults to "atomcolor".
            actor_name (Optional[str], optional): Name of the bond actor. If not provided, then a unique actor name
            is created based on the number of bonds already added. Defaults to None.
        """
        # We need a_color and b_color to render the bond
        if colorby == "atomcolor":
            # Get a_color and b_color from the atom colors saved in the atoms dict
            a_color = self.find_color_by_atom_tag(bond_tags[0])
            b_color = self.find_color_by_atom_tag(bond_tags[-1])
        elif colorby == "bondtype":
            if bond_type in self.bond_type_rendering.keys():
                a_color = self.bond_type_rendering[bond_type].color
                b_color = a_color
            else:
                warnings.warn(
                    "Setting bonds to default since colorby was bondtype, but bond_type color was not set"
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
                self.bond_type_rendering[bond_type].render_options,
                render_options,
            )
        else:
            warnings.warn(f"No bond type render options set for bond type {bond_type}")
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

    def update_bond_colors(self, actor_name: str, a_color: str, b_color: str) -> bool:
        """Function to change the bond colors of a given bond actor

        Args:
            actor_name (str): The actor name of the bond, whose colours need to be changed.
            a_color (str): End point ("A") color of the bond.
            b_color (str): End point ("B) color of the bond. a_color and b_color can be the same.
            render_options: The new render options

        Returns:
            bool: True if successfully changed.
        """
        if actor_name in self.bonds:
            self.bonds[actor_name]["a_color"] = a_color
            self.bonds[actor_name]["b_color"] = b_color
            return True
        else:
            return False

    def delete_actor(self, actor_name: str) -> bool:
        """Delete an actor from atoms, bonds or hulls

        Args:
            actor_name (str): Name of the actor

        Returns:
            bool: True if successfully deleted, otherwise False
        """
        # Search for the atom in self.atoms
        atom_index = self.atoms.find_first_atom_index_with_name(actor_name=actor_name)
        # the name was found in self.atoms
        if atom_index is not None:
            self.atoms.atoms.pop(atom_index)
            return True

        # Check for bonds, hulls and hydrogen bonds
        if actor_name in self.bonds:
            del self.bonds[actor_name]
            return True
        elif actor_name in self.hulls:
            del self.hulls[actor_name]
            return True
        elif actor_name in self.hydrogen_bonds:
            del self.hydrogen_bonds[actor_name]
            return True
        else:
            return False

    def delete_atom(self, actor_name: str) -> bool:
        """Delete an atom, given the actor name

        Args:
            actor_name (str): PyVista actor name of the atom

        Returns:
            bool: True if atom was found and deleted, otherwise False if not found
        """
        # Search for the atom in self.atoms
        atom_index = self.atoms.find_first_atom_index_with_name(actor_name=actor_name)
        # the name was found in self.atoms
        if atom_index is not None:
            self.atoms.atoms.pop(atom_index)
            return True
        else:
            return False

    def delete_bond(self, actor_name: str) -> bool:
        """Delete a bond from the self.bonds dict, given the actor name (key).
        If not found, returns False

        Args:
            actor_name (str): PyVista actor name

        Returns:
            bool: True if successfully deleted, otherwise Falses
        """
        if actor_name in self.bonds:
            del self.bonds[actor_name]
            return True
        else:
            return False

    def delete_hull(self, actor_name: str) -> bool:
        """Delete a hull from the self.hulls dict, given the actor name (key).
        If not found, returns False

        Args:
            actor_name (str): PyVista actor name

        Returns:
            bool: True if successfully deleted, otherwise Falses
        """
        if actor_name in self.hulls:
            del self.hulls[actor_name]
            return True
        else:
            return False
