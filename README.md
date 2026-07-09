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
