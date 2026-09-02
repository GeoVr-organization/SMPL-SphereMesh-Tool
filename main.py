import sys

from PyQt6.QtWidgets import QApplication, QMessageBox

from src.app_paths import app_root, is_frozen, resource_root, smpl_models_dir, user_data_dir
from src.smpl_models import MALE, load_smpl_model, smpl_model_path, spheremesh_template_path
from src.ui_widgets import MainWindow

_MISSING_MODELS_HINT = (
    "The SMPL body model could not be loaded.\n\n"
    "Expected: {path}\n\n"
    "If this build was made without the models bundled, copy the SMPL_models folder next "
    "to the executable and start it again."
)


def selftest() -> int:
    """Check that a build can find everything it needs, without opening a window.

    Exists because a packaged GUI application that cannot locate its data files fails at
    startup with nothing on screen, which is indistinguishable from not launching at all.
    Run ``SphereMeshTool.exe --selftest`` after a build to get the answer as text and an
    exit code.
    """
    print(f"frozen           : {is_frozen()}")
    print(f"resource root    : {resource_root()}")
    print(f"app root         : {app_root()}")
    print(f"writable data dir: {user_data_dir()}")
    print(f"SMPL models dir  : {smpl_models_dir()}")

    failures = []
    for label, path in (
        ("male body model", smpl_model_path(MALE)),
        ("male sphere template", spheremesh_template_path(MALE)),
    ):
        ok = path.is_file()
        print(f"  [{'OK ' if ok else 'MISSING'}] {label}: {path}")
        if not ok:
            failures.append(label)

    try:
        load_smpl_model(MALE)
        print("  [OK ] SMPL model loads")
    except Exception as error:  # noqa: BLE001 - report whatever went wrong, do not crash
        print(f"  [FAIL] SMPL model failed to load: {error}")
        failures.append("SMPL model load")

    try:
        import pyvista  # noqa: F401
        import vtkmodules.all  # noqa: F401

        print("  [OK ] pyvista and VTK import")
    except Exception as error:  # noqa: BLE001
        print(f"  [FAIL] pyvista/VTK import: {error}")
        failures.append("pyvista/VTK")

    # A bundle can import PyQt6 fine and still be unable to open a window, because the Qt
    # platform plugin (qwindows.dll) travels separately from the Python modules and is a
    # common casualty of packaging. Constructing a QApplication is what actually loads it.
    try:
        application = QApplication.instance() or QApplication([sys.argv[0], "-platform", "windows"])
        print(f"  [OK ] Qt initialises, platform plugin: {application.platformName()!r}")
    except Exception as error:  # noqa: BLE001
        print(f"  [FAIL] Qt could not initialise: {error}")
        failures.append("Qt platform plugin")

    if failures:
        print(f"\nSelf-test FAILED: {', '.join(failures)}")
        return 1
    print("\nSelf-test passed.")
    return 0


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()

    app = QApplication(sys.argv)

    # Start on the male body model; the Body Model switch in the UI loads the other one.
    startup_gender = MALE
    try:
        smpl_model = load_smpl_model(startup_gender)
    except (FileNotFoundError, OSError) as error:
        QMessageBox.critical(
            None,
            "SphereMesh Tool",
            _MISSING_MODELS_HINT.format(path=smpl_model_path(startup_gender)) + f"\n\n{error}",
        )
        return 1

    window = MainWindow(smpl_model, startup_gender)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
