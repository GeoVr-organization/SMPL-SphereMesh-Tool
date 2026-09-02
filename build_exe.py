"""Build the SphereMesh tool into a standalone Windows folder with an .exe inside.

    python build_exe.py                 # bundle everything, windowed
    python build_exe.py --no-models     # leave SMPL_models out, load it from beside the exe
    python build_exe.py --console       # keep a console window so tracebacks are visible
    python build_exe.py --clean         # discard PyInstaller's caches first

The result is dist/SphereMeshTool/, containing SphereMeshTool.exe next to the libraries it
needs. Ship the whole folder; the exe alone will not run. See SphereMeshTool.spec for why
this is a one-dir rather than a one-file build.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SPEC = PROJECT_ROOT / "SphereMeshTool.spec"


def directory_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 * 1024)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--no-models",
        action="store_true",
        help="Do not bundle SMPL_models/; the app then expects it next to the executable",
    )
    parser.add_argument(
        "--console",
        action="store_true",
        help="Keep a console window attached, so startup errors are readable",
    )
    parser.add_argument("--clean", action="store_true", help="Clear PyInstaller caches first")
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip running the built executable's --selftest afterwards",
    )
    args = parser.parse_args()

    if not SPEC.is_file():
        print(f"Missing {SPEC}", file=sys.stderr)
        return 1

    environment = dict(os.environ)
    environment["SPHEREMESH_BUNDLE_MODELS"] = "0" if args.no_models else "1"
    environment["SPHEREMESH_CONSOLE"] = "1" if args.console else "0"

    # Clear the previous output ourselves rather than letting PyInstaller's --noconfirm do
    # it, so a locked directory produces an explanation instead of a bare WinError 32 out
    # of shutil. Windows refuses to remove a directory that any process has open, and the
    # usual culprit is a terminal sitting inside it or the previous build still running.
    previous_output = PROJECT_ROOT / "dist" / "SphereMeshTool"
    if previous_output.exists():
        try:
            shutil.rmtree(previous_output)
        except PermissionError as error:
            print(
                f"Cannot replace {previous_output}: {error}\n\n"
                "Something is holding that folder open. Close any terminal whose working "
                "directory is inside it, close Explorer windows showing it, and make sure "
                "SphereMeshTool.exe is not still running, then try again.",
                file=sys.stderr,
            )
            return 1

    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        str(SPEC),
        "--noconfirm",
        "--distpath",
        str(PROJECT_ROOT / "dist"),
        "--workpath",
        str(PROJECT_ROOT / "build"),
    ]
    if args.clean:
        command.append("--clean")

    print("Bundling SMPL models:", "no" if args.no_models else "yes")
    print("Console window      :", "yes" if args.console else "no")
    print("Running:", " ".join(command))
    started = time.perf_counter()
    result = subprocess.run(command, cwd=PROJECT_ROOT, env=environment)
    elapsed = time.perf_counter() - started

    if result.returncode != 0:
        print(f"\nBuild failed after {elapsed:.0f}s.", file=sys.stderr)
        return result.returncode

    output_dir = PROJECT_ROOT / "dist" / "SphereMeshTool"
    executable = output_dir / "SphereMeshTool.exe"
    print(f"\nBuilt in {elapsed:.0f}s.")
    print(f"  folder : {output_dir}  ({directory_size_mb(output_dir):.0f} MB)")
    print(f"  exe    : {executable}  ({'present' if executable.is_file() else 'MISSING'})")
    if args.no_models:
        print(
            "\nBuilt without the SMPL models. Copy SMPL_models/ next to SphereMeshTool.exe "
            "before running, or the body model will fail to load."
        )

    if not args.no_verify and executable.is_file():
        # A PyInstaller build can succeed and still be unrunnable: an over-eager exclude, a
        # module resolved at runtime that static analysis never saw, a data file left out.
        # Windowed builds show none of that -- they just fail to appear. Run the self-test
        # here so a broken bundle is caught now rather than on someone else's machine.
        print("\nVerifying the build (--selftest)...")
        verification = subprocess.run(
            [str(executable), "--selftest"],
            cwd=output_dir,
            capture_output=True,
            text=True,
            timeout=600,
        )
        output = (verification.stdout or "") + (verification.stderr or "")
        print("\n".join("  " + line for line in output.strip().splitlines()[-25:]))
        if verification.returncode != 0:
            print(f"\nSelf-test FAILED (exit {verification.returncode}). The bundle is not usable.")
            if not args.console:
                print("Rebuild with --console to see startup errors in a terminal.")
            return verification.returncode

    print("\nDistribute the whole folder. Zip it with:")
    print(f'  powershell Compress-Archive -Path "{output_dir}" -DestinationPath SphereMeshTool.zip')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
