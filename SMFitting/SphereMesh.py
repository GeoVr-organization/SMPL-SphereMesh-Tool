"""Stand-in for src/sphere_mesh.py, the one SphereMesh implementation, which
the GUI tool shares with this folder.

The notebooks and scripts here import SphereMesh as a top-level module (they
put this folder on sys.path), so this file keeps that import working:
`import SphereMesh` hands back the src.sphere_mesh module object itself, not a
copy. A flag set on it, such as debugClosestSphereVerbose, therefore reaches
the code that reads it. Edit src/sphere_mesh.py, not this file.
"""

import importlib
import sys
from pathlib import Path

_repositoryRoot = Path(__file__).resolve().parent.parent
if str(_repositoryRoot) not in sys.path:
    sys.path.insert(0, str(_repositoryRoot))

sys.modules[__name__] = importlib.import_module("src.sphere_mesh")
