# solvis

<img src="https://github.com/amritagos/solvis/blob/main/branding/logo/logo.png?raw=true" width="300" />

## About

Various ways to analyze and visualize solvation shell structures, which wraps [`PyVista`](https://docs.pyvista.org/version/stable/). Meant primarily for analyzing the outputs produced by LAMMPS here. 

## Installation 

### From PyPI

You can install `solvis` from PyPI like so: 

```bash
pip install solvis-tools
```

### From source

We use [`micromamba`](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) as the package manager, but feel free to use your own poison. Create and activate the environment. 

```bash
micromamba create -f environment.yml
micromamba activate solvisenv
```

In order to install the code, run the following in the top-level directory:

```bash
pip install -e .
```

## Examples 

Each example is self-contained. Mostly, they show how to handle interactive plotting.
To close the interactive window, press q. 

## Tests

To run tests (which are inside the `tests` directory), written with `pytest`, run the following command from the top-level directory: 

```bash
pytest -v
```

In order to debug tests using `pdb`, you can write the command `breakpoint()` inside the `Python` files (in `tests`) wherever you want to set a breakpoint. Then, run `pytest --pdb`. This will stop the code at the line where you put the `breakpoint()` command. 

To see more verbose output from `pytest`, including tests that pass, you can run `pytest -rA`.

Note that `test_hydrogen_bond.py` and `test_plot_octahedral_shell.py` actually compare the results of images created for a single hydrogen bond and for an octahedral shell showing hydrogen bonds formed by the acceptor seventh molecule, respectively. The images compared against are present in the top-level `image_cache_dir` directory.

To view a coverage report, run the following from the top-level directory: 

```bash
pytest --cov=solvis tests/
```

## Image Gallery
<p float="left">
    <img src="https://github.com/amritagos/solvis/blob/main/resources/non_octahedral_shape.png?raw=true" width="200" />
    <img src="https://github.com/amritagos/solvis/blob/main/resources/octahedral_shell.png?raw=true" width="200" />
</p>
<p float="left">
    <img src="https://github.com/amritagos/solvis/blob/main/resources/shell_with_hbonds.png?raw=true" width="200" />
    <img src="https://github.com/amritagos/solvis/blob/main/resources/hbond_non_oct.png?raw=true" width="200" />
</p>