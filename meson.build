project('solvis')

py_mod = import('python')
py = py_mod.find_installation(pure: true) # This is pure Python, so maybe this should be true

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
  pure: true, # no compiled extension
  subdir: 'solvis'
)