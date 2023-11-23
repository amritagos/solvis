# solvis

## About

Various ways to analyze and visualize solvation shell structures, using free codes. Meant primarily for analyzing the outputs produced by LAMMPS. 

## Installation

We use [`micromamba`](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) as the package manager, but feel free to use your own poison. Create and activate the environment. 

```bash
micromamba create -f environment.yml
micromamba activate solvisenv
```

## Examples 

- single_sphericity_calc: This calculates the sphericity for a convex hull constructed from the six nearest neighbours of an Fe atom. The input trajectory file has a single frame, containing only 7 oxygen atoms and 1 Fe atom in the LAMMPS dump text format, with atom types of 1 and 3, respectively. The sphericity is calculated by first finding the six nearest neighbours using k-nearest neighbours, constructing the convex hull using the `ConvexHull` function from `SciPy` (that wraps Qhull), and getting the enclosed volume and surface area. Visualization is done in `PyVista`, such that the surface is coloured according to the distance from the central Fe atom. The point corresponding to the seventh furthest molecule is also plotted.

- sphericity_calc_traj: Calculation of sphericity values for a convex hull surrounding a single Fe ion, starting from a LAMMPS trajectory file with multiple frames. A text output file is written out containing sphericity values. 