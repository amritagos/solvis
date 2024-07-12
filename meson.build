project('solvis')

py_mod = import('python')
py = py_mod.find_installation(pure: false)

# solvis, main package
py.install_sources([
    'solvis/__init__.py',
    'solvis/atom_tag_manager.py',
    'solvis/bond_helper.py',
    'solvis/geometric_utils.py',
    'solvis/render_helper.py',
    'solvis/solvation_shell.py',
    'solvis/system.py',
    'solvis/util.py',
    'solvis/vis_initializers.py',
    'solvis/visualization.py'
  ],
  pure: false, # install next to compiled extension
  subdir: 'solvis'
)