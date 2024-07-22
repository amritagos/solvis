from ase import Atom, Atoms
from ase.data import chemical_symbols
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial import KDTree
import math
from pathlib import Path
from pyvista import PolyData
import pyvista as pv
import matplotlib.pyplot as plt
from pyvista.plotting.opts import ElementType

from .util import (
    create_two_color_gradient,
    distance_projected,
    merge_options,
    create_asymmetric_two_color_gradient,
)


class AtomicPlotter:
    """Plots atoms, bonds and convex hulls using PyVista
    if interactive mode is on, the window_size will be ignored
    Interactive mode means the atoms and bonds will be pickable
    depth_peeling: If set to True, then depth peeling will be enabled for nicer translucent
    images
    shadows: If set to True, shadows will be enabled"""

    def __init__(
        self,
        interactive_mode=False,
        window_size=[4000, 4000],
        depth_peeling=True,
        shadows=True,
    ):

        self.interactive_mode = interactive_mode

        if self.interactive_mode:
            self.plotter = pv.Plotter(off_screen=False)
            self.plotter.image_scale = (
                4  # For nicer resolution if you take a screenshot
            )
        else:
            # The plotter will render images offscreen
            self.plotter = pv.Plotter(off_screen=True, window_size=window_size)

        # More options for the plotter
        self.plotter.set_background("white")  # Make transparent later

        if depth_peeling:
            self.plotter.enable_depth_peeling(10)
        if shadows:
            self.plotter.enable_shadows()

        # Default options for atoms, bonds and hulls
        # Dictionary for solvation shell mesh (add_mesh) options
        self.hull_mesh_options = dict(
            line_width=3,
            show_edges=False,
            opacity=0.5,
            metallic=True,
            specular=0.7,
            ambient=0.3,
            show_scalar_bar=False,
            pickable=False,
        )
        # Dictionary for atoms, bonds (add_mesh) options
        self.atom_bond_mesh_options = dict(
            show_edges=False,
            metallic=True,
            smooth_shading=True,
            specular=0.7,
            ambient=0.3,
            show_scalar_bar=False,
        )

    def add_hull(self, hull: PolyData, **color_or_additional_mesh_options):
        """
        TODO: pass it a hull object created in solvis and not the PyVista object

        color_or_additional_mesh_options: Arguments you can pass onto the PyVista add_mesh function
        Put color and colormap options here, or override default mesh options"""

        mesh_options = self.hull_mesh_options  # Default options
        mesh_options.update(
            color_or_additional_mesh_options
        )  # Update with user-defined options

        self.plotter.add_mesh(hull, **mesh_options)

    def add_bond(
        self,
        pointa,
        pointb,
        a_color: str,
        b_color: str,
        actor_name: str,
        radius=0.1,
        resolution=1,
        bond_gradient_start=0.0,
        asymmetric_gradient_start=None,
        **mesh_options_override_default
    ):
        """
        Plot a bond between two points (say A and B), each of which is a numPy array with the x,y,z coordinates

        pointa: Coordinates of end point (atom) A
        pointb: Coordinates of end point (atom) B
        a_color: Color of end point (atom) A
        b_color: Color of end point (atom) B
        actor_name: Give a unique name to the actor corresponding to the bond
        radius: Radius of the tube created for the bond (Default 0.1)
        resolution: Resolution of the tube created for the bond (Default 1), see PolyVista tube for more details
        bond_gradient_start: (Default 0.0) Where the mixing of the two colors will start.
        asymmetric_gradient_start: If this is set, then the colormap is not split equally between two colors. Should be in between 0 and 1
        mesh_options_override_default: Arguments in the PyVista add_mesh function,
        that will override the default atom_bond_mesh_options."""

        # Only override the default options with user-defined options
        mesh_options = merge_options(
            self.atom_bond_mesh_options, mesh_options_override_default
        )

        # If interactive mode is on, make the bond pickable
        if self.interactive_mode:
            mesh_options.update({"pickable": True})

        # Create the colormap to be used for bond colouring
        if asymmetric_gradient_start is None:
            # Symmetric gradient
            m_cmap = create_two_color_gradient(a_color, b_color, bond_gradient_start)
        else:
            m_cmap = create_asymmetric_two_color_gradient(
                a_color, b_color, asymmetric_gradient_start
            )

        # Each tube is between the two input points
        tube = pv.Tube(
            pointa=pointa,
            pointb=pointb,
            resolution=resolution,
            radius=radius,
            n_sides=15,
        )
        # Create color ids
        color_ids = []
        for point in tube.points:
            color_ids.append(
                distance_projected(
                    tube.points[0], point, tube.points[0], tube.points[-1]
                )
            )
        color_ids = np.array(color_ids)

        # Add the tube created as a bond with a unique name given by actor_name
        self.plotter.add_mesh(
            tube, scalars=color_ids, cmap=m_cmap, name=actor_name, **mesh_options
        )

    def add_atoms_as_spheres(
        self, pointset, colors, radius=0.1, name=None, **mesh_options_override_default
    ):
        """
        Plot points (i.e. atoms) as spheres

        pointset: ndarray of coordinates every point inside pointset
        colors: Color of each point. Must be the same length as the number of points
        radius: Radius of the Sphere created for the atom (Default 0.1)
        mesh_options_override_default: Arguments in the PyVista add_mesh function,
        that will override the default atom_bond_mesh_options."""

        # Only override the default options with user-defined options
        mesh_options = merge_options(
            self.atom_bond_mesh_options, mesh_options_override_default
        )

        # If interactive mode is on, make the bond pickable
        if self.interactive_mode:
            mesh_options.update({"pickable": True})

        # Go over all the points inside pointset
        for i, (point, color) in enumerate(zip(pointset, colors)):
            sphere = pv.Sphere(radius=radius, center=point)
            if name is None:
                actor_name = "p_" + str(i)
            else:
                actor_name = "p_" + name + str(i)
            # Add the actor
            self.plotter.add_mesh(
                sphere,
                color=color,
                name=actor_name,
                **mesh_options,
            )

    def add_single_atom_as_sphere(
        self,
        point,
        color: str,
        actor_name: str = None,
        radius=0.1,
        **mesh_options_override_default
    ):
        """
        Plot a single atom as a PyVista Sphere

        point: ndarray of coordinates every point inside pointset
        color: Color of the atom
        actor_name: Unique name of the actor. Must be set in interactive mode if you want to correctly select and do things with the atom
        radius: Radius of the Sphere created for the atom (Default 0.1)
        mesh_options_override_default: Arguments in the PyVista add_mesh function,
        that will override the default atom_bond_mesh_options."""

        # Only override the default options with user-defined options
        mesh_options = merge_options(
            self.atom_bond_mesh_options, mesh_options_override_default
        )

        # If interactive mode is on, make the atom pickable
        if self.interactive_mode:
            mesh_options.update({"pickable": True})

        # Set a unique actor name if you want the point to be selected
        # and if you want to do things with it
        if actor_name is None:
            print("Warning: Setting name for point.\n")
            actor_name = "point"

        # Create the sphere
        sphere = pv.Sphere(radius=radius, center=point)

        self.plotter.add_mesh(
            sphere,
            color=color,
            name=actor_name,
            **mesh_options,
        )

    def add_single_line(self, start_point, end_point, color, width, actor_name=None):
        """
        Render a single straight line, given two end point (coordinate arrays)
        I have not used this for hydrogen bonds, and this is not included in the
        RendererRepresentation object
        """
        if actor_name is None:
            name_opt = dict()
        else:
            name_opt = dict(name=actor_name)

        line_mesh = pv.Line(start_point, end_point)

        # Add the tube created as a bond with a unique name given by actor_name
        self.plotter.add_mesh(line_mesh, color=color, line_width=width, **name_opt)

    def create_dashed_line_custom_spacing(
        self, point1, point2, segment_spacing, color, width=None, actor_name=None
    ):
        """
        Create a dashed line (usually for hydrogen bonds) between two points
        """
        if width is None:
            width = 10
        if actor_name is None:
            name_opt = dict()
        else:
            name_opt = dict(name=actor_name)

        # Calculate the vector between the two points
        vector = np.array(point2) - np.array(point1)

        # Calculate the total length of the line
        total_length = np.linalg.norm(vector)

        # Calculate the normalized direction vector
        direction = vector / total_length

        # Calculate the number of line segments
        num_segments = int(total_length / segment_spacing)

        # Calculate the actual segment spacing
        actual_spacing = total_length / num_segments

        # Calculate the points for the dashed line with user-defined spacing
        dashed_line_points = [
            np.array(point1) + i * direction * actual_spacing
            for i in range(num_segments)
        ]

        # Create PyVista line segments with alternating empty spaces
        edges = []
        for i in range(0, num_segments - 1, 2):
            edges.append([2, i, min(i + 1, num_segments - 1)])
        dashed_line_points = np.array(dashed_line_points)
        lines = np.hstack(edges)
        line_mesh = pv.PolyData(dashed_line_points, lines=lines)
        # Add the lines
        self.plotter.add_mesh(line_mesh, color=color, line_width=width, **name_opt)

    def create_bonds_from_edges(
        self,
        pointset,
        edges,
        point_colors=None,
        single_bond_colors=None,
        radius=0.1,
        resolution=1,
        bond_gradient_start=0.0,
        asymmetric_gradient_start=None,
        name=None,
        **mesh_options_override_default
    ):
        """
        Create bonds from a list of bonds, using

        pointset: ndarray of coordinates every point inside pointset
        edges: Bonds such that each index corresponds to a point inside pointset
        point_colors: list of colors corresponding to the color of each point
        single_bond_colors: Colors corresponding to all bonds. If you set this, point_colors will be ignored
        radius: Radius of the tube created for the bond (Default 0.1)
        resolution: Resolution of the tube created for the bond (Default 1), see PolyVista tube for more details
        bond_gradient_start: (Default 0.0) Where the mixing of the two colors will start.
        asymmetric_gradient_start: If you want an asymmetric bond gradient
        mesh_options_override_default: Arguments in the PyVista add_mesh function,
        that will override the default atom_bond_mesh_options."""

        if single_bond_colors is not None:
            set_single_color = True
        else:
            set_single_color = False

        for idx, bond in enumerate(edges):
            a_index = bond[0]
            b_index = bond[1]
            pointa = pointset[a_index]
            pointb = pointset[b_index]
            if name is None:
                actor_name = "tube" + str(idx)
            else:
                actor_name = "tube_" + name + str(idx)
            if set_single_color:
                colora = single_bond_colors[idx]
                colorb = colora
            else:
                colora = point_colors[a_index]
                colorb = point_colors[b_index]
            # Create the bond
            self.add_bond(
                pointa,
                pointb,
                colora,
                colorb,
                actor_name,
                radius,
                resolution,
                bond_gradient_start,
                asymmetric_gradient_start,
                **mesh_options_override_default,
            )

    def change_atom_color(self, atom_actor_name, new_color):
        """
        Change the atom color directly. Not recommended; you should also change the color
        in the RendererRepresentation object when using this command
        """

        self.plotter.renderer.actors[atom_actor_name].prop.color = new_color

    def create_bonds_to_point(
        self,
        pointset,
        central_point,
        point_colors=None,
        central_point_color=None,
        single_bond_colors=None,
        radius=0.1,
        resolution=1,
        bond_gradient_start=0.0,
        asymmetric_gradient_start=None,
        name=None,
        **mesh_options_override_default
    ):
        """
        Create bonds from of each point in a list of points, to a particular point

        pointset: ndarray of coordinates every point inside pointset,to be bonded to a given point (called central_point here)
        central_point: Point to be bonded to
        point_colors: list of colors corresponding to the color of each point inside pointset
        central_point_color : You must set this to some color if using point_colors
        single_bond_colors: Colors corresponding to all bonds. If you set this, point_colors will be ignored
        radius: Radius of the tube created for the bond (Default 0.1)
        resolution: Resolution of the tube created for the bond (Default 1), see PolyVista tube for more details
        bond_gradient_start: (Default 0.0) Where the mixing of the two colors will start.
        asymmetric_gradient_start: If you want an asymmetric bond gradient
        mesh_options_override_default: Arguments in the PyVista add_mesh function,
        that will override the default atom_bond_mesh_options."""

        if single_bond_colors is not None:
            set_single_color = True
        else:
            set_single_color = False
            if central_point_color is None:
                central_point_color = "black"

        for idx, point in enumerate(pointset):
            if name is None:
                actor_name = "tube" + str(idx)
            else:
                actor_name = "tube_" + name + str(idx)
            if set_single_color:
                colora = single_bond_colors[idx]
                colorb = colora
            else:
                colora = central_point_color
                colorb = point_colors[idx]
            # Create the bond
            self.add_bond(
                central_point,
                point,
                colora,
                colorb,
                actor_name,
                radius,
                resolution,
                bond_gradient_start,
                asymmetric_gradient_start,
                **mesh_options_override_default,
            )

    def interactive_window(self, delete_actor=True):
        if not self.interactive_mode:
            print("You are not in interactive mode.\n")
            return
        else:
            self.selected_actors = []

            if delete_actor:
                show_actor = False
            else:
                show_actor = True

            # Callback for selecting actors
            def select_actors(actor):
                self.selected_actors.append(actor.name)
                print("Selected actor, ", actor.name, "\n")
                if delete_actor:
                    self.plotter.remove_actor(actor)

            self.plotter.enable_mesh_picking(
                select_actors, use_actor=True, show=show_actor
            )

            # Also get the camera position as the last position before closing
            self.interactive_camera_position = []

            # Key event for camera positions
            # Key s
            def save_cam_pos():
                self.interactive_camera_position.append(self.plotter.camera.position)

            # Add key event to the plotter
            self.plotter.add_key_event("s", save_cam_pos)

            # Callback for getting the last camera position
            def last_frame_info(plotter):
                self.interactive_camera_position.append(plotter.camera_position)

            # Interactive window
            self.plotter.show(before_close_callback=last_frame_info, auto_close=False)

    def render_image(self, imagename, camera_position=None):
        """Renders an image, optionally with the user-defined camera position"""
        if camera_position is not None:
            self.plotter.camera_position = camera_position
            self.plotter.camera_set = True
        # Take the screenshot
        self.plotter.screenshot(imagename, return_img=False)

    def close(self):
        """Close the plotter window"""
        self.plotter.close()
