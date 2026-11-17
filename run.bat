@echo off
rem Launch the SMPL SphereMesh Tool from source (no build needed).
rem Pass extra arguments through, e.g. run.bat --selftest
cd /d "%~dp0"
uv run python main.py %*
if errorlevel 1 pause
