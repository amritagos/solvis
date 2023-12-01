from ase import Atom, Atoms
from ase.data import chemical_symbols
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial import KDTree
import math 
import pandas as pd
from pathlib import Path 
from pyvista import PolyData
import pyvista as pv
import matplotlib.pyplot as plt
from pyvista.plotting.opts import ElementType

from .util import create_two_color_gradient, distance_projected

class AtomicPlotter:
     """Plots atoms, bonds and convex hulls using PyVista
     if interactive mode is on, the window_size will be ignored
     Interactive mode means the atoms and bonds will be pickable
     depth_peeling: If set to True, then depth peeling will be enabled for nicer translucent
     images
     shadows: If set to True, shadows will be enabled """
     def __init__(self, interactive_mode=False, window_size=[4000,4000], depth_peeling=True, shadows=True):
         self.interactive_mode = interactive_mode 

         if self.interactive_mode:
            self.plotter = pv.Plotter(off_screen=False)
            self.plotter.image_scale = 4 # For nicer resolution if you take a screenshot
        else:
            # The plotter will render images offscreen
            self.plotter = pv.Plotter(off_screen=True, window_size=window_size)

        # More options for the plotter
        self.plotter.set_background("white") # Make transparent later

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
            show_scalar_bar=False
            )

    def add_hull(hull:PolyData, **color_or_additional_mesh_options):
        """ 
        TODO: pass it a hull object created in solvis and not the PyVista object
        
        color_or_additional_mesh_options: Arguments you can pass onto the PyVista add_mesh function
        Put color and colormap options here, or override default mesh options"""
        
        mesh_options = self.hull_mesh_options # Default options
        mesh_options.update(color_or_additional_mesh_options) # Update with user-defined options

        self.plotter.add_mesh(hull, **mesh_options)

    def add_bond(pointa, pointb, a_color:str, b_color:str, actor_name:str, radius=0.1, resolution=1,bond_gradient_start=0.0, **mesh_options_override_default):
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
        mesh_options_override_default: Arguments in the PyVista add_mesh function,
        that will override the default atom_bond_mesh_options. """

        # Only override the default options with user-defined options
        mesh_options = {k: mesh_options_override_default[k] for k in mesh_options_override_default if k in self.atom_bond_mesh_options}

        # If interactive mode is on, make the bond pickable
        if self.interactive_mode:
            mesh_options.update({"pickable":True})

        # Create the colormap to be used for bond colouring
        m_cmap = create_two_color_gradient(a_color, b_color, bond_gradient_start)

        # Each tube is between the two input points 
        tube = pv.Tube(pointa=pointa, pointb=pointb, resolution=resolution, radius=radius, n_sides=15)
        # Create color ids
        color_ids = [] 
        for point in tube.points:
            color_ids.append(distance_projected(tube.points[0], point, tube.points[0], tube.points[-1]))
        color_ids = np.array(color_ids)

        # Add the tube created as a bond with a unique name given by actor_name 
        self.plotter.add_mesh(
        tube, 
        scalars=color_ids,
        cmap=m_cmap,
        name=actor_name,
        **mesh_options)

    def add_atoms_as_spheres(pointset, colors, radius=0.1, **mesh_options_override_default):
        """ 
        Plot points (i.e. atoms) as spheres 

        pointset: ndarray of coordinates every point inside pointset
        colors: Color of each point. Must be the same length as the number of points
        radius: Radius of the Sphere created for the atom (Default 0.1)
        mesh_options_override_default: Arguments in the PyVista add_mesh function,
        that will override the default atom_bond_mesh_options. """

        # Only override the default options with user-defined options
        mesh_options = {k: mesh_options_override_default[k] for k in mesh_options_override_default if k in self.atom_bond_mesh_options}

        # If interactive mode is on, make the bond pickable
        if self.interactive_mode:
            mesh_options.update({"pickable":True})

        # Go over all the points inside pointset
        for i, (point, color) in enumerate(zip(pointset, colors)):
            sphere = pv.Sphere(radius=radius, center=point)
            actor_name = "p_"+str(i)
            # Add the actor
            self.plotter.add_mesh(sphere,
                color=color, 
                name=actor_name,
                **mesh_options,
                )

    def add_single_atom_as_sphere(point, color:str, actor_name:str=None, radius=0.1, **mesh_options_override_default):
        """ 
        Plot a single atom as a PyVista Sphere

        point: ndarray of coordinates every point inside pointset
        color: Color of the atom
        actor_name: Unique name of the actor. Must be set in interactive mode if you want to correctly select and do things with the atom 
        radius: Radius of the Sphere created for the atom (Default 0.1)
        mesh_options_override_default: Arguments in the PyVista add_mesh function,
        that will override the default atom_bond_mesh_options. """

        # Only override the default options with user-defined options
        mesh_options = {k: mesh_options_override_default[k] for k in mesh_options_override_default if k in self.atom_bond_mesh_options}

        # If interactive mode is on, make the bond pickable
        if self.interactive_mode:
            mesh_options.update({"pickable":True})

        # Set a unique actor name if you want the point to be selected 
        # and if you want to do things with it
        if actor_name is None:
            print("Warning: Setting name for point.\n")
            actor_name = "point" 

        # Create the sphere
        sphere = pv.Sphere(radius=radius, center=point)

        self.plotter.add_mesh(sphere,
            color=color, 
            name=actor_name,
            **mesh_options,
            )

    # TODO
    # def create_bonds_from_edges(pointset, colors, radius=0.1, **mesh_options_override_default):
    #     """ 
    #     Plot points (i.e. atoms) as spheres 

    #     pointset: ndarray of coordinates every point inside pointset
    #     colors: Color of each point. Must be the same length as the number of points
    #     radius: Radius of the Sphere created for the atom (Default 0.1)
    #     mesh_options_override_default: Arguments in the PyVista add_mesh function,
    #     that will override the default atom_bond_mesh_options. """

    #     # Only override the default options with user-defined options
    #     mesh_options = {k: mesh_options_override_default[k] for k in mesh_options_override_default if k in self.atom_bond_mesh_options}

    #     # If interactive mode is on, make the bond pickable
    #     if self.interactive_mode:
    #         mesh_options.update({"pickable":True})

    #     # Go over all the points inside pointset
    #     for i, (point, color) in enumerate(zip(pointset, colors)):
    #         sphere = pv.Sphere(radius=radius, center=point)
    #         actor_name = "p_"+str(i)
    #         # Add the actor
    #         self.plotter.add_mesh(sphere,
    #             color=color, 
    #             name=actor_name,
    #             **mesh_options,
    #             )