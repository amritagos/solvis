import numpy as np
import numpy.typing as npt
import scipy
from pyvista import PolyData


class ConvexHull:
    def __init__(self, points: npt.ArrayLike):
        """
        Create a convex hull. Note that the convex hull in SciPy (which is a wrapper around Qhull)
        does not have support for periodic boundary conditions. Make sure the coordinates are unwrapped.
        If you are creating a ConvexHull from SolvationShell, this should be taken care of.
        """
        # The input coordinates
        self.points = points
        # The actual convex hull (wrapper arouund SciPy)
        hull = scipy.spatial.ConvexHull(points)

        # The faces of the convex hull. Indices correspond to
        # indices of the input points
        self.faces = hull.simplices

        # Get the surface area and the enclosed volume from the convex hull
        self.area = hull.area
        self.volume = hull.volume

    def pyvista_hull_from_convex_hull(self) -> PolyData:
        """Get a PyVista PolyData object, converting from SciPy's convex hull

        Returns:
            PolyData: PyVista PolyData object
        """
        faces_pyvistaformat = np.column_stack(
            (3 * np.ones((len(self.faces), 1), dtype=int), self.faces)
        ).flatten()
        polyhull = PolyData(self.points, faces_pyvistaformat)
        return polyhull

    def get_edges(self) -> npt.ArrayLike:
        """Convenience function for getting a numPy array of edges from the faces, of shape (n_edges,2).

        Returns:
            npt.ArrayLike: NumPy array of edges
        """
        edges = set()

        for simplex in self.faces:
            edges.add(tuple(sorted([simplex[0], simplex[1]])))
            edges.add(tuple(sorted([simplex[1], simplex[2]])))
            edges.add(tuple(sorted([simplex[2], simplex[0]])))

        edges = np.array(list(edges))
        return edges
