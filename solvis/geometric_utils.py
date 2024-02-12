import numpy as np
from scipy.spatial import ConvexHull
from pyvista import PolyData

class ConvexHull:
	def __init__(self, points):
		"""
		Create a convex hull. Note that the convex hull in SciPy (which is a wrapper around Qhull)
		does not have support for periodic boundary conditions. Make sure the coordinates are unwrapped.
		If you are creating a ConvexHull from SolvationShell, this should be taken care of. 
		"""
		# The input coordinates
		self.points = points 
		# The actual convex hull (wrapper arouund SciPy)
		self.hull = ConvexHull(self.points)

		# The faces of the convex hull. Indices correspond to 
		# indices of the input points
		self.faces = self.hull.simplices

		# Get the surface area and the enclosed volume from the convex hull
		self.area = self.hull.area
		self.volume = self.hull.volume

		def pyvista_hull_from_convex_hull(self):
        """
        Return a PyVista PolyData object, converting from SciPy's convex hull 
        """
        faces_pyvistaformat = np.column_stack((3*np.ones((len(self.faces), 1), dtype=int), self.faces)).flatten()
		polyhull = PolyData(self.points, faces_pyvistaformat)
        return polyhull

		def get_edges(self):
        """
        Get a numPy array of edges from the faces, of shape (n_edges,2). 
        """

        edges = set()

		for simplex in self.faces:
    		edges.add(tuple(sorted([simplex[0], simplex[1]])))
    		edges.add(tuple(sorted([simplex[1], simplex[2]])))
    		edges.add(tuple(sorted([simplex[2], simplex[0]])))

		edges = np.array(list(edges))
        return edges
