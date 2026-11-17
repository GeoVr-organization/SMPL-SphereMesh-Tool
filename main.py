import sys
import time

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPixmap
from PyQt6.QtWidgets import QApplication, QMessageBox, QSplashScreen

from src.app_paths import app_root, is_frozen, resource_root, smpl_models_dir, user_data_dir

# The torch / SMPL / VTK imports take several seconds, so they are done inside main() after the
# splash screen is up; importing them at module level would leave nothing on screen meanwhile.

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
    from src.smpl_models import MALE, load_smpl_model, smpl_model_path, startup_spheremesh_template_path

    print(f"frozen           : {is_frozen()}")
    print(f"resource root    : {resource_root()}")
    print(f"app root         : {app_root()}")
    print(f"writable data dir: {user_data_dir()}")
    print(f"SMPL models dir  : {smpl_models_dir()}")

    failures = []
    for label, path in (
        ("male body model", smpl_model_path(MALE)),
        ("male sphere template", startup_spheremesh_template_path(MALE)),
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


def _make_splash() -> QSplashScreen:
    """Build a plain dark splash screen that shows startup progress messages."""
    splash_pixmap = QPixmap(420, 110)
    splash_pixmap.fill(QColor("#202124"))
    splash = QSplashScreen(splash_pixmap)
    splash.show()
    return splash


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()

    startup_clock = time.perf_counter()
    app = QApplication(sys.argv)
    splash = _make_splash()

    def report_progress(message: str) -> None:
        """Print a timestamped startup step to the console and show it on the splash."""
        print(f"[startup {time.perf_counter() - startup_clock:5.1f}s] {message}", flush=True)
        splash.showMessage(
            f"SphereMesh Tool\n{message}",
            Qt.AlignmentFlag.AlignCenter,
            QColor("#e8eaed"),
        )
        app.processEvents()

    report_progress("Loading torch and the SMPL library...")
    from src.smpl_models import MALE, load_smpl_model, smpl_model_path

    report_progress("Loading the 3D viewer (PyVista / VTK)...")
    from src.ui_widgets import MainWindow

    # Start on the male body model; the Body Model switch in the UI loads the other one.
    startup_gender = MALE
    try:
        report_progress("Loading the SMPL body model...")
        smpl_model = load_smpl_model(startup_gender)
    except (FileNotFoundError, OSError) as error:
        splash.close()
        QMessageBox.critical(
            None,
            "SphereMesh Tool",
            _MISSING_MODELS_HINT.format(path=smpl_model_path(startup_gender)) + f"\n\n{error}",
        )
        return 1

    report_progress("Building the window and the sphere template...")
    window = MainWindow(smpl_model, startup_gender)
    window.show()
    splash.finish(window)
    report_progress("Ready.")
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
