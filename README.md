# SMPL-SphereMesh-Tool

## Installation

The project uses [uv](https://docs.astral.sh/uv/) to manage the virtual environment.

### 1. Create the virtual environment and install dependencies

```
uv venv
uv pip install -r requirements.txt
```

### 2. Provide the SMPL model files

You need the SMPL model files in `SMPL_models/` (e.g.
`basicmodel_f_lbs_10_207_0_v1.0.0.npz`), which must be obtained from the official
[SMPL website](https://smpl.is.tue.mpg.de/) under their license.

## Running

```powershell
uv run python main.py
```

## Building a standalone executable

PyInstaller is used to produce a Windows build that runs without Python installed.

```powershell
uv pip install pyinstaller
uv run python build_exe.py
```

The result is `dist/SphereMeshTool/`, containing `SphereMeshTool.exe` alongside the
libraries it needs. **Ship the whole folder** — the `.exe` on its own will not start.

```powershell
Compress-Archive -Path dist\SphereMeshTool -DestinationPath SphereMeshTool.zip
```

### Build options

| Command | Effect |
| --- | --- |
| `python build_exe.py` | Bundle everything, no console window |
| `python build_exe.py --console` | Keep a console attached so startup errors are readable |
| `python build_exe.py --no-models` | Leave `SMPL_models/` out; copy it next to the `.exe` instead |
| `python build_exe.py --clean` | Discard PyInstaller's caches first |

### Checking a build

A packaged GUI application that cannot find its data files fails with nothing on screen,
which looks the same as not launching at all. To get the answer as text:

```powershell
dist\SphereMeshTool\SphereMeshTool.exe --selftest
```

It prints the resolved paths, confirms the SMPL model and sphere template are present and
loadable, and exits non-zero if anything is missing.

### Notes

- It is a **one-dir** build, not one-file. torch alone is ~490 MB; a one-file build would
  unpack that to a temporary directory on every launch, costing tens of seconds each time.
- The SMPL `.npz` files are licensed data obtained separately (see above). They are bundled
  by default so the build is self-contained, which is fine for internal use — but check the
  SMPL license before distributing a build outside your organisation, or use `--no-models`
  and have recipients supply their own.
- Saved sphere configurations are written to a `data/` folder **next to the executable**,
  not inside the bundle. Templates are read from the bundle, but a `data/` or `SMPL_models/`
  folder placed next to the `.exe` takes precedence, so either can be updated without
  rebuilding.
