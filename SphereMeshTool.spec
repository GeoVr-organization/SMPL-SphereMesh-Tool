# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the SphereMesh tool. Run it through ``build_exe.py``, not directly.

``build_exe.py`` passes two switches through the environment:
    SPHEREMESH_BUNDLE_MODELS  "1" bundles SMPL_models/ (default), "0" leaves it out
    SPHEREMESH_CONSOLE        "1" keeps a console window attached, "0" is windowed (default)

Why one-dir and not one-file: torch alone is ~490 MB. A one-file build would unpack all of
it into a temporary directory on every launch, costing tens of seconds each time. A one-dir
build starts at once, at the price of shipping a folder instead of a single exe.

Where the contents come from: this file was reconstructed from the Analysis/EXE tables a
previous build left in build/SphereMeshTool/ (the original was never committed, because
.gitignore used to ignore every *.spec). Entries here are what that build bundled.
"""

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

project_root = Path(SPECPATH)

bundle_models = os.environ.get("SPHEREMESH_BUNDLE_MODELS", "1") == "1"
show_console = os.environ.get("SPHEREMESH_CONSOLE", "0") == "1"

# Packages whose Python sources are shipped as data files as well as compiled into the
# archive, and whose submodules are all pulled in, because they are imported lazily.
whole_packages = ["pyvista", "pyvistaqt", "vtkmodules", "smpl_pytorch"]

# Distributions whose metadata must be present for importlib.metadata lookups at runtime.
# Note "vtk", not "vtkmodules": the wheel's distribution name differs from its package.
metadata_distributions = ["pyvista", "pyvistaqt", "vtk", "smpl_pytorch"]

datas = [
    # Sphere templates, random_pose.csv, beta offsets. Seeded next to the exe on first run
    # by src/app_paths.py, so edits there never touch the bundle.
    (str(project_root / "data"), "data"),
    # The JSON <-> engine-text converters MainWindow imports by file path (resource_path).
    (str(project_root / "scripts" / "json2txt.py"), "scripts"),
    (str(project_root / "scripts" / "txt2json.py"), "scripts"),
]
if bundle_models:
    datas.append((str(project_root / "SMPL_models"), "SMPL_models"))

hiddenimports = [
    "PyQt6.QtSvg",
    "PyQt6.QtPrintSupport",
    "matplotlib.backends.backend_qtagg",
]

for package_name in whole_packages:
    datas += collect_data_files(package_name, include_py_files=True)
    hiddenimports += collect_submodules(package_name)
for distribution_name in metadata_distributions:
    datas += copy_metadata(distribution_name)

# Not used at runtime. polyscope is imported only by the standalone src/sphere_beta_tool.py,
# and bpy only inside a script string that an external Blender process runs. The Qt and
# notebook entries stop PyInstaller dragging in a second Qt binding or the Jupyter stack.
excluded_modules = [
    "tkinter",
    "polyscope",
    "bpy",
    "PyQt5",
    "PySide2",
    "PySide6",
    "pytest",
    "IPython",
    "notebook",
    "jupyter",
]

analysis = Analysis(
    [str(project_root / "main.py")],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excluded_modules,
    noarchive=False,
    optimize=0,
)
python_archive = PYZ(analysis.pure)

executable = EXE(
    python_archive,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="SphereMeshTool",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=show_console,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

collection = COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    name="SphereMeshTool",
)
