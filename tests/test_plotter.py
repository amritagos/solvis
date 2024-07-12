import pytest
import numpy as np
from ase.io import read
from ase.atoms import Atom, Atoms
from ase.data import chemical_symbols
from pathlib import Path

import solvis


def test_plotter_simple():
    """
    Test that the plotter can change colors of atoms.
    """
    pl_inter = solvis.visualization.AtomicPlotter(
        interactive_mode=True, depth_peeling=True, shadows=False
    )
    # Single atom (as a mesh)
    pl_inter.add_single_atom_as_sphere(
        [0, 0, 0], color="black", radius=0.1, actor_name="point0"
    )
    # Test the position
    np.testing.assert_array_equal(
        [0, 0, 0], pl_inter.plotter.renderer.actors["point0"].position
    )
    # Test that the colour is black
    assert pl_inter.plotter.renderer.actors["point0"].prop.color == "black"
    # Change the color to blue and check it
    pl_inter.change_atom_color(atom_actor_name="point0", new_color="blue")
    assert pl_inter.plotter.renderer.actors["point0"].prop.color == "blue"
