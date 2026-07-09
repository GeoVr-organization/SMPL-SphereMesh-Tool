import sys
from smpl_pytorch.smpl import load_numpy_params, SMPL
from pathlib import Path
import numpy as np
import pyvista as pv
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSlider, QLabel, QPushButton, QCheckBox, QComboBox, QMenuBar, QMenu
from PyQt6.QtCore import Qt
from pyvistaqt import QtInteractor

from src.ui_widgets import MainWindow


if __name__ == "__main__":

     # Load the SMPL model using the load_pickle_smpl function (path relative to this script)
    script_dir = Path(__file__).resolve().parent
    model_path = script_dir / 'SMPL_models' / 'basicmodel_f_lbs_10_207_0_v1.0.0.npz'
    smpl_model = SMPL(load_numpy_params(model_path))


    app = QApplication(sys.argv)
    window = MainWindow(smpl_model)
    window.show()
    sys.exit(app.exec())
