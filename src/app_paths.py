"""Filesystem locations that differ between running from source and running frozen.

Why this exists
---------------
Every path in this project used to be derived from ``Path(__file__).parent.parent``, which
is correct from a source checkout and wrong inside a PyInstaller bundle. Two separate
problems hide behind that one expression:

*Read-only resources* (sphere templates, ``random_pose.csv``, the converter scripts) are
unpacked by PyInstaller into a private directory it advertises as ``sys._MEIPASS``. In a
one-file build that directory is a temporary extraction that is deleted on exit, so it can
be read but never written.

*Writable state* (saved sphere configurations and their timestamped archives) must therefore
go somewhere else entirely. Writing a save into ``sys._MEIPASS`` appears to succeed and then
vanishes when the application closes -- the worst kind of failure, because the user is told
the save worked. Writable state goes next to the executable instead.

From a source checkout both roots collapse back to the repository root, so behaviour is
unchanged and nothing needs a frozen build to be tested.

Lookup order for data files
---------------------------
``data_file`` prefers the writable directory over the bundled one, so a user can drop an
edited template next to the executable and have it win over the shipped copy without
rebuilding. ``smpl_models_dir`` works the same way, which additionally lets the SMPL ``.npz``
files be updated, or supplied separately, without a rebuild.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

__all__ = [
    "is_frozen",
    "resource_root",
    "resource_path",
    "app_root",
    "user_data_dir",
    "data_file",
    "smpl_models_dir",
]

_SOURCE_ROOT = Path(__file__).resolve().parent.parent


def is_frozen() -> bool:
    """True when running from a PyInstaller bundle rather than a source checkout."""
    return getattr(sys, "frozen", False)


def resource_root() -> Path:
    """Root of the read-only bundled files.

    PyInstaller sets ``sys._MEIPASS`` to the directory it unpacked the bundle into. In a
    one-file build that is a temporary directory removed on exit; treat everything under it
    as read-only.
    """
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    return _SOURCE_ROOT


def app_root() -> Path:
    """Directory the application should treat as its own, for writable state.

    Next to the executable when frozen, the repository root otherwise.
    """
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return _SOURCE_ROOT


def resource_path(*parts: str) -> Path:
    """Path to a bundled read-only file, e.g. ``resource_path("data", "template.json")``."""
    return resource_root().joinpath(*parts)


def _seed_user_data(directory: Path) -> None:
    """Copy the bundled data files next to the executable the first time it runs.

    Without this the writable directory starts empty while the sphere templates sit inside
    the bundle, so every file dialog opens on an empty folder and the shipped templates look
    missing. Seeding also gives the user somewhere to edit a template, since files here take
    precedence over the bundled copies.

    Only files that are not already present are written, so a user's edits are never
    overwritten on a later run. Failures are ignored: if the install directory is read-only
    the application still works, reading everything from the bundle.
    """
    if not is_frozen():
        return
    bundled = resource_path("data")
    if not bundled.is_dir():
        return
    try:
        for source in bundled.rglob("*"):
            if not source.is_file():
                continue
            destination = directory / source.relative_to(bundled)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copy2(source, destination)
    except OSError:
        pass


def user_data_dir() -> Path:
    """Writable directory for saves and archives, created and seeded on first use."""
    directory = app_root() / "data"
    if directory.is_dir():
        return directory
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        return directory
    _seed_user_data(directory)
    return directory


def data_file(*parts: str) -> Path:
    """Locate a file under ``data/``, preferring a user copy over the bundled one.

    Returns the bundled path when neither exists, so callers report a missing-file error
    naming the location they would have shipped it in.
    """
    candidate = app_root().joinpath("data", *parts)
    if candidate.exists():
        return candidate
    bundled = resource_path("data", *parts)
    if bundled.exists():
        return bundled
    return candidate if is_frozen() else bundled


def smpl_models_dir() -> Path:
    """Directory holding the SMPL ``.npz`` body models.

    A copy next to the executable wins over the bundled one so the models can be replaced,
    or supplied separately, without rebuilding. Builds made with ``--no-models`` ship no
    bundled copy at all and rely on this.
    """
    candidate = app_root() / "SMPL_models"
    if candidate.is_dir():
        return candidate
    return resource_path("SMPL_models")
