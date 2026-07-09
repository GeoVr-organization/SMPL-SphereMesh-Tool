import csv
import importlib.util
import json
import random
import tempfile
import os
from pathlib import Path
import numpy as np
import pyvista as pv
import vtk
from matplotlib.colors import LinearSegmentedColormap
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QStackedLayout, QHBoxLayout, QSlider, QLabel, QPushButton, QCheckBox, QComboBox, QMenuBar, QMenu, QGroupBox, QDoubleSpinBox, QFileDialog, QMessageBox, QDialog, QDialogButtonBox, QFormLayout, QScrollArea, QListWidget, QLineEdit, QPlainTextEdit
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QKeyEvent, QShortcut, QKeySequence
from pyvistaqt import QtInteractor
import torch as tc

from src.sphere_manager import SphereManager
from smpl_pytorch.smpl import JOINT_INDEX_TO_LABEL
from src.export_smpl_fbx import export_smpl_to_fbx
from src.smpl_forward_compat import my_function_forward

from src.utils import measure_time

# Initial SMPL shape coefficients (10). Used as-is (no clipping). Edit to change startup shape.
DEFAULT_BETA: tuple[float, ...] = (
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0
    )
# Slider internal value I represents beta = I/10. Wide range avoids clipping loaded defaults/text betas.
BETA_SLIDER_INT_MIN = -20
BETA_SLIDER_INT_MAX = 20

_RANDOM_POSE_CSV = Path(__file__).resolve().parent.parent / "data" / "random_pose.csv"


def _random_pose_csv_column_order():
    """Header column order for ``data/random_pose.csv`` (72 SMPL pose axes)."""
    cols = []
    for i in range(24):
        jn = JOINT_INDEX_TO_LABEL[i]
        for d in range(3):
            cols.append(f"theta_{jn}_{d}")
    return cols


def _slider_int_for_beta(v: float) -> int:
    """Map beta to slider integer, clamped only to Qt-safe slider range (not shape clipping)."""
    x = int(round(float(v) * 10))
    return max(BETA_SLIDER_INT_MIN, min(BETA_SLIDER_INT_MAX, x))


def _format_beta_label(v: float) -> str:
    if v == 0.0:
        return "0"
    av = abs(v)
    if av >= 1000 or av < 1e-3:
        return f"{v:.6g}"
    return f"{v:.6f}".rstrip("0").rstrip(".")


def default_beta_tensor() -> tc.Tensor:
    """Return ``DEFAULT_BETA`` as a float64 tensor (exact values, no clipping)."""
    if len(DEFAULT_BETA) != 10:
        raise ValueError("DEFAULT_BETA must have length 10")
    return tc.tensor([float(x) for x in DEFAULT_BETA], dtype=tc.float64)


class Skeleton:
    """Simple skeleton class to store joint and kinematic tree information"""

    def __init__(self, joints, kintree, Rs, *, rs_are_global: bool = True):
        """
        Args:
            joints: Joint positions in world space (J, 3).
            kintree: Parent index per joint.
            Rs: Global (J, 3, 3) world rotations per joint, or per-bone locals if
                ``rs_are_global=False`` (then composed along ``kintree`` as in classic SMPL).
        """
        self.joints = joints  # Joint positions (already in world space)
        self.kintree = np.asarray(kintree, dtype=np.int64).reshape(-1)
        self.Rs = np.asarray(Rs, dtype=np.float64)
        if rs_are_global:
            self.global_Rs = self.Rs
        else:
            self.global_Rs = self._compute_global_rotations(self.Rs, self.kintree)

    def _compute_global_rotations(self, Rs, kintree):
        """Compute global rotation matrices by accumulating along the kinematic chain."""
        global_Rs = np.zeros_like(Rs)
        global_Rs[0] = Rs[0]

        for i in range(1, len(kintree)):
            parent_idx = int(kintree[i])
            if parent_idx < 0:
                global_Rs[i] = Rs[i]
                continue
            global_Rs[i] = np.dot(global_Rs[parent_idx], Rs[i])

        return global_Rs
    
    def get_joint_rotation(self, joint_idx):
        """Get global rotation matrix for a specific joint"""
        return self.global_Rs[joint_idx]
    
    def transform_point(self, local_point, joint_idx):
        """Transform a point from joint local space to world space"""
        R = self.global_Rs[joint_idx]
        joint_pos = self.joints[joint_idx]
        return np.dot(R, local_point) + joint_pos


class AddConnectionDialog(QDialog):
    """Pick 2 or 3 sphere indices (distinct) to form a connection."""

    def __init__(self, parent, num_spheres: int):
        super().__init__(parent)
        self.setWindowTitle("Add connection")
        self._num_spheres = num_spheres
        layout = QFormLayout(self)
        self._count_combo = QComboBox()
        self._count_combo.addItem("2 spheres", 2)
        if num_spheres >= 3:
            self._count_combo.addItem("3 spheres", 3)
        layout.addRow("Connection:", self._count_combo)
        self._index_combos = []
        for k in range(3):
            cb = QComboBox()
            for j in range(num_spheres):
                cb.addItem(str(j), j)
            self._index_combos.append(cb)
            layout.addRow(f"Sphere index {k + 1}:", cb)
            cb.setVisible(k < 2)
        self._count_combo.currentIndexChanged.connect(self._sync_visibility)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
        self._sync_visibility()

    def _sync_visibility(self):
        n = self._count_combo.currentData()
        if n is None:
            n = 2
        for k in range(3):
            self._index_combos[k].setVisible(k < n)

    def get_indices(self):
        n = self._count_combo.currentData()
        if n is None:
            n = 2
        return [self._index_combos[k].currentData() for k in range(n)]


class RemoveConnectionDialog(QDialog):
    """Pick an existing connection to remove, with optional sphere-index filter."""

    def __init__(self, parent, connections):
        super().__init__(parent)
        self.setWindowTitle("Remove connection")
        self._connections = [tuple(int(i) for i in conn) for conn in connections]
        self._filtered_map = []

        layout = QVBoxLayout(self)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Filter by sphere index:"))
        self._filter_edit = QLineEdit()
        self._filter_edit.setPlaceholderText("e.g. 0")
        filter_row.addWidget(self._filter_edit)
        self._filter_btn = QPushButton("Apply")
        self._filter_btn.clicked.connect(self._refresh_list)
        filter_row.addWidget(self._filter_btn)
        layout.addLayout(filter_row)

        self._list = QListWidget()
        layout.addWidget(self._list)

        self._filter_edit.returnPressed.connect(self._refresh_list)
        self._list.itemDoubleClicked.connect(lambda *_: self.accept())

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._refresh_list()

    def _refresh_list(self):
        text = self._filter_edit.text().strip()
        filter_index = None
        if text:
            try:
                filter_index = int(text)
            except ValueError:
                filter_index = None

        self._list.clear()
        self._filtered_map = []
        for conn in self._connections:
            if filter_index is not None and filter_index not in conn:
                continue
            self._filtered_map.append(conn)
            self._list.addItem(str(list(conn)))

        if self._list.count() > 0:
            self._list.setCurrentRow(0)

    def get_selected_connection(self):
        row = self._list.currentRow()
        if row < 0 or row >= len(self._filtered_map):
            return None
        return self._filtered_map[row]


class BetaSliderGroup(QGroupBox):
    """Group box containing 10 beta sliders with randomize and reset buttons"""
    
    valuesChanged = pyqtSignal(tc.Tensor)  # Signal emitted when any slider changes
    
    def __init__(self, parent=None):
        super().__init__("▶ Beta Controls", parent)

        self.beta = default_beta_tensor().clone()
        self.sliders_list = []
        self.slider_labels = []
        
        self.setCheckable(True)
        self.setChecked(False)
        self.toggled.connect(self._toggle_and_update_arrow)
        
        self._setup_ui()
        
        # Hide all widgets initially since we start collapsed
        self._toggle_visibility(False)
    
    def _setup_ui(self):
        """Setup the UI components"""
        layout = QVBoxLayout()
        
        # Create 10 horizontal sliders
        for i in range(10):
            dec = float(self.beta[i].item())
            iv = _slider_int_for_beta(dec)
            # Create label for each slider
            label = QLabel(f"Beta {i+1}: {_format_beta_label(dec)}")
            self.slider_labels.append(label)

            # Create slider (scaled by 10 to allow 0.1 steps)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(BETA_SLIDER_INT_MIN)
            slider.setMaximum(BETA_SLIDER_INT_MAX)
            slider.setSingleStep(1)
            slider.setPageStep(10)
            slider.valueChanged.connect(lambda value, idx=i: self._on_slider_changed(idx, value))
            self.sliders_list.append(slider)
            slider.blockSignals(True)
            slider.setValue(iv)
            slider.blockSignals(False)

            # Add label and slider to layout
            layout.addWidget(label)
            layout.addWidget(slider)

        load_row = QHBoxLayout()
        self.beta_line_edit = QLineEdit()
        self.beta_line_edit.setPlaceholderText(
            "10 betas, comma-separated (e.g. 0, 0.5, -1.2, …)"
        )
        load_row.addWidget(self.beta_line_edit, stretch=1)
        load_betas_button = QPushButton("Load betas")
        load_betas_button.clicked.connect(self.load_betas_from_text)
        load_row.addWidget(load_betas_button)
        layout.addLayout(load_row)

        # Create horizontal layout for buttons
        button_layout = QHBoxLayout()
        
        random_button = QPushButton("Randomize")
        random_button.clicked.connect(self.randomize)
        button_layout.addWidget(random_button)
        
        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(self.reset)
        button_layout.addWidget(reset_button)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def _on_slider_changed(self, idx, value):
        """Handle slider value changes"""
        decimal_value = value / 10.0
        self.slider_labels[idx].setText(
            f"Beta {idx+1}: {_format_beta_label(decimal_value)}"
        )
        self.beta[idx] = decimal_value
        self.valuesChanged.emit(self.beta)
    
    def randomize(self):
        """Randomize all slider values"""
        for idx, slider in enumerate(self.sliders_list):
            random_value = np.round(np.random.uniform(-2, 2), 1)
            slider.blockSignals(True)
            slider.setValue(int(random_value * 10))
            slider.blockSignals(False)
            self.beta[idx] = random_value
            self.slider_labels[idx].setText(
                f"Beta {idx+1}: {_format_beta_label(random_value)}"
            )
        self.valuesChanged.emit(self.beta)

    def load_betas_from_text(self):
        """Parse 10 comma-separated floats from the line edit and apply as betas."""
        text = self.beta_line_edit.text().strip()
        if not text:
            QMessageBox.warning(self, "Beta", "Enter 10 comma-separated numbers.")
            return
        parts = [p.strip() for p in text.split(",") if p.strip() != ""]
        if len(parts) != 10:
            QMessageBox.warning(
                self,
                "Beta",
                f"Expected exactly 10 values, got {len(parts)}.",
            )
            return
        try:
            raw = [float(p) for p in parts]
        except ValueError:
            QMessageBox.warning(
                self, "Beta", "Could not parse one or more values as numbers."
            )
            return
        for idx, v in enumerate(raw):
            v = float(v)
            self.beta[idx] = v
            slider_value = _slider_int_for_beta(v)
            self.sliders_list[idx].blockSignals(True)
            self.sliders_list[idx].setValue(slider_value)
            self.sliders_list[idx].blockSignals(False)
            self.slider_labels[idx].setText(f"Beta {idx+1}: {_format_beta_label(v)}")
        self.valuesChanged.emit(self.beta)

    def reset(self):
        """Reset all slider values to 0"""
        for idx, slider in enumerate(self.sliders_list):
            slider.blockSignals(True)
            slider.setValue(0)
            slider.blockSignals(False)
            self.beta[idx] = 0.0
            self.slider_labels[idx].setText(f"Beta {idx+1}: 0.0")
        self.valuesChanged.emit(self.beta)
    
    def _toggle_and_update_arrow(self, checked):
        """Toggle visibility and update arrow direction"""
        if checked:
            self.setTitle("▼ Beta Controls")
        else:
            self.setTitle("▶ Beta Controls")
        self._toggle_visibility(checked)
    
    def _toggle_visibility(self, checked):
        """Toggle visibility of group contents"""
        for i in range(self.layout().count()):
            item = self.layout().itemAt(i)
            if item.widget():
                item.widget().setVisible(checked)
            elif item.layout():
                for j in range(item.layout().count()):
                    widget = item.layout().itemAt(j).widget()
                    if widget:
                        widget.setVisible(checked)
    
    def get_beta(self):
        """Get current beta values"""
        return self.beta

class ThetaSliderGroup(QGroupBox):
    """Group box containing a joint selector dropdown and slider with reset button"""
    
    valuesChanged = pyqtSignal(tc.Tensor)  # Signal emitted when slider changes
    
    def __init__(self, parent=None):
        super().__init__("▶ Theta Controls", parent)
        
        self.theta = tc.zeros(72, dtype=tc.float64)
        self.current_joint_idx = 0
        self.current_dimension = 0  # 0=X, 1=Y, 2=Z
        
        self.setCheckable(True)
        self.setChecked(False)
        self.toggled.connect(self._toggle_and_update_arrow)
        
        self._setup_ui()
        
        # Hide all widgets initially since we start collapsed
        self._toggle_visibility(False)
    
    def _setup_ui(self):
        """Setup the UI components"""
        layout = QVBoxLayout()
        
        # Create horizontal layout for joint selector
        joint_layout = QHBoxLayout()
        
        # Create joint selector dropdown
        joint_label = QLabel("Joint:")
        self.joint_selector = QComboBox()
        # Add 24 joints with their names
        for i in range(24):
            joint_name = JOINT_INDEX_TO_LABEL.get(i, f"Joint {i}")
            self.joint_selector.addItem(f"{i}: {joint_name}")
        self.joint_selector.currentIndexChanged.connect(self._on_selection_changed)
        
        joint_layout.addWidget(joint_label)
        joint_layout.addWidget(self.joint_selector)
        layout.addLayout(joint_layout)
        
        # Create horizontal layout for dimension selector
        dim_layout = QHBoxLayout()
        
        # Create dimension selector dropdown
        dim_label = QLabel("Axis:")
        self.dimension_selector = QComboBox()
        self.dimension_selector.addItem("X")
        self.dimension_selector.addItem("Y")
        self.dimension_selector.addItem("Z")
        self.dimension_selector.currentIndexChanged.connect(self._on_selection_changed)
        
        dim_layout.addWidget(dim_label)
        dim_layout.addWidget(self.dimension_selector)
        layout.addLayout(dim_layout)
        
        # Create slider with label
        self.slider_label = QLabel("Theta 0: 0.0")
        layout.addWidget(self.slider_label)
        
        # Create slider (scaled by 10 to allow 0.1 steps)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(-31)  # -π ≈ -3.14
        self.slider.setMaximum(31)   # π ≈ 3.14
        self.slider.setValue(0)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(10)
        self.slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self.slider)
        
        # Create reset button
        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(self.reset)
        layout.addWidget(reset_button)

        random_pose_button = QPushButton("Random pose (CSV)")
        random_pose_button.clicked.connect(self.apply_random_pose_from_csv)
        layout.addWidget(random_pose_button)
        
        self.setLayout(layout)
    
    def _get_current_theta_idx(self):
        """Calculate the current theta index based on joint and dimension"""
        return self.current_joint_idx * 3 + self.current_dimension
    
    def _on_selection_changed(self):
        """Handle joint or dimension selector changes"""
        self.current_joint_idx = self.joint_selector.currentIndex()
        self.current_dimension = self.dimension_selector.currentIndex()
        
        # Update slider to show current theta value for this joint and dimension
        theta_idx = self._get_current_theta_idx()
        current_value = int(self.theta[theta_idx].item() * 10)
        self.slider.blockSignals(True)
        self.slider.setValue(current_value)
        self.slider.blockSignals(False)
        decimal_value = current_value / 10.0
        
        joint_name = JOINT_INDEX_TO_LABEL.get(self.current_joint_idx, f"Joint {self.current_joint_idx}")
        axis_name = ['X', 'Y', 'Z'][self.current_dimension]
        self.slider_label.setText(f"{joint_name} {axis_name} (θ{theta_idx}): {decimal_value:.1f}")
    
    def _on_slider_changed(self, value):
        """Handle slider value changes"""
        decimal_value = value / 10.0
        theta_idx = self._get_current_theta_idx()
        
        joint_name = JOINT_INDEX_TO_LABEL.get(self.current_joint_idx, f"Joint {self.current_joint_idx}")
        axis_name = ['X', 'Y', 'Z'][self.current_dimension]
        self.slider_label.setText(f"{joint_name} {axis_name} (θ{theta_idx}): {decimal_value:.1f}")
        
        self.theta[theta_idx] = decimal_value
        self.valuesChanged.emit(self.theta)
    
    def reset(self):
        """Reset all theta values to 0"""
        self.theta = tc.zeros(72, dtype=tc.float64)
        self.slider.blockSignals(True)
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        
        theta_idx = self._get_current_theta_idx()
        joint_name = JOINT_INDEX_TO_LABEL.get(self.current_joint_idx, f"Joint {self.current_joint_idx}")
        axis_name = ['X', 'Y', 'Z'][self.current_dimension]
        self.slider_label.setText(f"{joint_name} {axis_name} (θ{theta_idx}): 0.0")
        
        self.valuesChanged.emit(self.theta)
    
    def apply_random_pose_from_csv(self):
        """Pick a random pose row from ``data/random_pose.csv`` and apply thetas."""
        path = _RANDOM_POSE_CSV
        if not path.is_file():
            QMessageBox.warning(self, "Random pose", f"Pose file not found:\n{path}")
            return
        cols = _random_pose_csv_column_order()
        tensors = []
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    vals = [float(row[c]) for c in cols]
                except (KeyError, TypeError, ValueError):
                    continue
                if len(vals) != 72:
                    continue
                tensors.append(tc.tensor(vals, dtype=tc.float64))
        if not tensors:
            QMessageBox.warning(self, "Random pose", "No valid pose rows in CSV.")
            return
        self.theta = random.choice(tensors)
        self._on_selection_changed()
        self.valuesChanged.emit(self.theta)
    
    def _toggle_and_update_arrow(self, checked):
        """Toggle visibility and update arrow direction"""
        if checked:
            self.setTitle("▼ Theta Controls")
        else:
            self.setTitle("▶ Theta Controls")
        self._toggle_visibility(checked)
    
    def _toggle_visibility(self, checked):
        """Toggle visibility of group contents"""
        for i in range(self.layout().count()):
            item = self.layout().itemAt(i)
            if item.widget():
                item.widget().setVisible(checked)
            elif item.layout():
                for j in range(item.layout().count()):
                    widget = item.layout().itemAt(j).widget()
                    if widget:
                        widget.setVisible(checked)
    
    def get_theta(self):
        """Get current theta values"""
        return self.theta

class InfoPanel(QGroupBox):
    """Group box containing information about the current state"""
    
    def __init__(self, parent):
        super().__init__("Info Panel", parent)
        self.grow_click_callback = None  # set by MainWindow to pass mesh when Grow is clicked
        self.toggle_clip_planes_panel_callback = None  # set by MainWindow to show/hide clip planes panel
        self._setup_ui()
    
    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Top section: Selected Spheremesh ID and buttons
        top_section = QHBoxLayout()
        top_section.setSpacing(8)
        
        # Left side: Selected info
        info_layout = QVBoxLayout()
        info_layout.setSpacing(2)
        self.spheremesh_id_label = QLabel("Spheremesh: None")
        self.spheremesh_id_label.setStyleSheet("font-weight: bold; font-size: 11pt;")
        self.selected_sphere_label = QLabel("Selected: None")
        self.selected_sphere_label.setStyleSheet("font-size: 10pt;")
        info_layout.addWidget(self.spheremesh_id_label)
        info_layout.addWidget(self.selected_sphere_label)
        conn_hdr = QLabel("Connections:")
        conn_hdr.setStyleSheet("font-size: 9pt; color: #555555;")
        info_layout.addWidget(conn_hdr)
        self.connections_text = QPlainTextEdit()
        self.connections_text.setReadOnly(True)
        self.connections_text.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.connections_text.setFixedHeight(88)
        self.connections_text.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.connections_text.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.connections_text.setPlaceholderText("(none)")
        self.connections_text.setStyleSheet(
            "font-size: 10pt; color: #333333; background-color: rgba(255,255,255,160);"
        )
        info_layout.addWidget(self.connections_text)
        
        top_section.addLayout(info_layout, stretch=1)
        
        # Right side: Buttons
        button_layout = QVBoxLayout()
        button_layout.setSpacing(4)
        self.delete_button = QPushButton("Delete")
        self.delete_button.setFixedSize(70, 28)
        self.delete_button.setStyleSheet("""
            QPushButton {
                background-color: #ff6b6b;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #ff5252;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #888888;
            }
        """)
        self.delete_button.clicked.connect(self._on_delete_clicked)
        
        button_layout.addWidget(self.delete_button)

        
        # Initially disable buttons
        self.delete_button.setEnabled(False)
        
        top_section.addLayout(button_layout)
        main_layout.addLayout(top_section)
        
        # Separator line
        separator = QLabel()
        separator.setStyleSheet("background-color: rgba(100, 100, 100, 180); min-height: 1px; max-height: 1px;")
        main_layout.addWidget(separator)
        
        # Sphere 1 Info Box
        sphere1_box = QGroupBox("Selected sphere")
        sphere1_box.setStyleSheet("""
            QGroupBox {
            background-color: rgba(230, 245, 255, 200);
            border: 1px solid rgba(100, 150, 200, 150);
            border-radius: 6px;
            margin-top: 8px;
            padding-top: 8px;
            font-weight: bold;
            color: #000000;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 8px;
            padding: 0 4px;
            color: #000000;
        }
        QLabel {
            color: #333333;
        }
        """)
        sphere1_layout = QVBoxLayout()
        sphere1_layout.setSpacing(4)
        sphere1_layout.setContentsMargins(8, 8, 8, 8)
        
        center1_row = QHBoxLayout()
        center1_row.addWidget(QLabel("Center:"))
        self.sphere1_center_x = QDoubleSpinBox()
        self.sphere1_center_y = QDoubleSpinBox()
        self.sphere1_center_z = QDoubleSpinBox()

        self.sphere1_center_x.setRange(-100.0, 100.0)
        self.sphere1_center_x.setDecimals(3)
        self.sphere1_center_x.setSingleStep(0.001)
        self.sphere1_center_x.setStyleSheet("font-size: 9pt; min-width: 78px; border: 2px solid #c00; border-radius: 4px;")

        self.sphere1_center_y.setRange(-100.0, 100.0)
        self.sphere1_center_y.setDecimals(3)
        self.sphere1_center_y.setSingleStep(0.001)
        self.sphere1_center_y.setStyleSheet("font-size: 9pt; min-width: 78px; border: 2px solid #0a0; border-radius: 4px;")
        
        self.sphere1_center_z.setRange(-100.0, 100.0)
        self.sphere1_center_z.setDecimals(3)
        self.sphere1_center_z.setSingleStep(0.001)
        self.sphere1_center_z.setStyleSheet("font-size: 9pt; min-width: 78px; border: 2px solid #00a; border-radius: 4px;")

        center1_row.addWidget(self.sphere1_center_x)
        center1_row.addWidget(self.sphere1_center_y)
        center1_row.addWidget(self.sphere1_center_z)

        sphere1_layout.addLayout(center1_row)

        center2_row = QHBoxLayout()
        center2_row.addWidget(QLabel("Radius:"))
        self.sphere1_radius = QDoubleSpinBox()
        self.sphere1_radius.setRange(0.001, 100.0)
        self.sphere1_radius.setDecimals(4)
        self.sphere1_radius.setSingleStep(0.001)
        self.sphere1_radius.setStyleSheet("font-size: 9pt; min-width: 78px; border: 2px solid #666; border-radius: 4px;")
        center2_row.addWidget(self.sphere1_radius)

        self.sphere1_grow_button = QPushButton("Grow")
        self.sphere1_grow_button.setFixedSize(70, 28)
        self.sphere1_grow_button.setStyleSheet("""
            QPushButton { background-color: #7bc96f; color: white; border: none; border-radius: 4px; font-weight: bold; }
            QPushButton:hover { background-color: #6ab85f; }
            QPushButton:disabled { background-color: #cccccc; color: #888888; }
        """)
        self.sphere1_grow_button.clicked.connect(lambda: self._on_grow_clicked(0))

        center2_row.addWidget(self.sphere1_grow_button)

        self.sphere1_to_joint_btn = QPushButton("To joint")
        self.sphere1_to_joint_btn.setFixedSize(70, 28)
        self.sphere1_to_joint_btn.setStyleSheet("""
            QPushButton { background-color: #9b59b6; color: white; border: none; border-radius: 4px; font-weight: bold; }
            QPushButton:hover { background-color: #8e44ad; }
            QPushButton:disabled { background-color: #cccccc; color: #888888; }
        """)
        self.sphere1_to_joint_btn.setToolTip("Move this sphere to the closest SMPL joint")
        self.sphere1_to_joint_btn.clicked.connect(lambda: self._on_move_to_joint_clicked(0))
        center2_row.addWidget(self.sphere1_to_joint_btn)

        sphere1_layout.addLayout(center2_row)

        self.sphere1_lock_btn = QPushButton("Lock")
        self.sphere1_lock_btn.setFixedSize(70, 28)
        self.sphere1_lock_btn.setStyleSheet("""
            QPushButton {
                background-color: #4ecdc4;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45b7b0;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #888888;
            }
        """)
        self.sphere1_lock_btn.clicked.connect(lambda: self._on_sphere_lock_clicked(0))
        sphere1_layout.addWidget(self.sphere1_lock_btn)
        self.sphere1_lock_btn.setEnabled(False)

        


        
        sphere1_box.setLayout(sphere1_layout)
        self.sphere1_box = sphere1_box
        main_layout.addWidget(sphere1_box)
        
        # Sphere 2 Info Box
        sphere2_box = QGroupBox("Sphere 2")
        sphere2_box.setStyleSheet("""
            QGroupBox {
                background-color: rgba(230, 245, 255, 200);
                border: 1px solid rgba(100, 150, 200, 150);
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 8px;
                font-weight: bold;
                color: #000000;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
                color: #000000;
            }
            QLabel {
                color: #333333;
            }
        """)
        sphere2_layout = QVBoxLayout()
        sphere2_layout.setSpacing(4)
        sphere2_layout.setContentsMargins(8, 8, 8, 8)
        
        center2_row = QHBoxLayout()
        center2_row.addWidget(QLabel("Center:"))
        self.sphere2_center_x = QDoubleSpinBox()
        self.sphere2_center_y = QDoubleSpinBox()
        self.sphere2_center_z = QDoubleSpinBox()
        self.sphere2_center_x.setRange(-100.0, 100.0)
        self.sphere2_center_x.setDecimals(3)
        self.sphere2_center_x.setSingleStep(0.001)
        self.sphere2_center_x.setStyleSheet("font-size: 9pt; min-width: 60px; border: 2px solid #c00; border-radius: 4px;")
        self.sphere2_center_y.setRange(-100.0, 100.0)
        self.sphere2_center_y.setDecimals(3)
        self.sphere2_center_y.setSingleStep(0.001)
        self.sphere2_center_y.setStyleSheet("font-size: 9pt; min-width: 60px; border: 2px solid #0a0; border-radius: 4px;")
        self.sphere2_center_z.setRange(-100.0, 100.0)
        self.sphere2_center_z.setDecimals(3)
        self.sphere2_center_z.setSingleStep(0.001)
        self.sphere2_center_z.setStyleSheet("font-size: 9pt; min-width: 60px; border: 2px solid #00a; border-radius: 4px;")
        center2_row.addWidget(self.sphere2_center_x)
        center2_row.addWidget(self.sphere2_center_y)
        center2_row.addWidget(self.sphere2_center_z)
        sphere2_layout.addLayout(center2_row)


        

        center2_row = QHBoxLayout()
        center2_row.addWidget(QLabel("Radius:"))
        self.sphere2_radius = QDoubleSpinBox()
        self.sphere2_radius.setRange(0.001, 100.0)
        self.sphere2_radius.setDecimals(4)
        self.sphere2_radius.setSingleStep(0.01)
        self.sphere2_radius.setStyleSheet("font-size: 9pt; min-width: 60px; border: 2px solid #666; border-radius: 4px;")
        center2_row.addWidget(self.sphere2_radius)
        self.sphere2_grow_button = QPushButton("Grow")
        self.sphere2_grow_button.setFixedSize(70, 28)
        self.sphere2_grow_button.setStyleSheet("""
            QPushButton { background-color: #7bc96f; color: white; border: none; border-radius: 4px; font-weight: bold; }
            QPushButton:hover { background-color: #6ab85f; }
            QPushButton:disabled { background-color: #cccccc; color: #888888; }
        """)
        self.sphere2_grow_button.clicked.connect(lambda: self._on_grow_clicked(1))
        
        center2_row.addWidget(self.sphere2_grow_button)

        self.sphere2_to_joint_btn = QPushButton("To joint")
        self.sphere2_to_joint_btn.setFixedSize(70, 28)
        self.sphere2_to_joint_btn.setStyleSheet("""
            QPushButton { background-color: #9b59b6; color: white; border: none; border-radius: 4px; font-weight: bold; }
            QPushButton:hover { background-color: #8e44ad; }
            QPushButton:disabled { background-color: #cccccc; color: #888888; }
        """)
        self.sphere2_to_joint_btn.setToolTip("Move this sphere to the closest SMPL joint")
        self.sphere2_to_joint_btn.clicked.connect(lambda: self._on_move_to_joint_clicked(1))
        center2_row.addWidget(self.sphere2_to_joint_btn)

        sphere2_layout.addLayout(center2_row)

        self.sphere2_lock_btn = QPushButton("Lock")
        self.sphere2_lock_btn.setFixedSize(70, 28)
        self.sphere2_lock_btn.setStyleSheet("""
            QPushButton {
                background-color: #4ecdc4;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45b7b0;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #888888;
            }
        """)
        self.sphere2_lock_btn.clicked.connect(lambda: self._on_sphere_lock_clicked(1))
        sphere2_layout.addWidget(self.sphere2_lock_btn)
        self.sphere2_lock_btn.setEnabled(False)




        sphere2_box.setLayout(sphere2_layout)
        self.sphere2_box = sphere2_box
        main_layout.addWidget(sphere2_box)
        
        # Sphere 3 Info Box
        self.sphere3_box = QGroupBox("Sphere 3")
        self.sphere3_box.setStyleSheet("""
            QGroupBox {
                background-color: rgba(230, 245, 255, 200);
                border: 1px solid rgba(100, 150, 200, 150);
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 8px;
                font-weight: bold;
                color: #000000;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
                color: #000000;
            }
            QLabel {
                color: #333333;
            }
        """)
        sphere3_layout = QVBoxLayout()
        sphere3_layout.setSpacing(4)
        sphere3_layout.setContentsMargins(8, 8, 8, 8)
        
        center3_row = QHBoxLayout()
        center3_row.addWidget(QLabel("Center:"))
        self.sphere3_center_x = QDoubleSpinBox()
        self.sphere3_center_y = QDoubleSpinBox()
        self.sphere3_center_z = QDoubleSpinBox()
        self.sphere3_center_x.setRange(-100.0, 100.0)
        self.sphere3_center_x.setDecimals(3)
        self.sphere3_center_x.setSingleStep(0.001)
        self.sphere3_center_x.setStyleSheet("font-size: 9pt; min-width: 60px; border: 2px solid #c00; border-radius: 4px;")
        self.sphere3_center_y.setRange(-100.0, 100.0)
        self.sphere3_center_y.setDecimals(3)
        self.sphere3_center_y.setSingleStep(0.001)
        self.sphere3_center_y.setStyleSheet("font-size: 9pt; min-width: 60px; border: 2px solid #0a0; border-radius: 4px;")
        self.sphere3_center_z.setRange(-100.0, 100.0)
        self.sphere3_center_z.setDecimals(3)
        self.sphere3_center_z.setSingleStep(0.001)
        self.sphere3_center_z.setStyleSheet("font-size: 9pt; min-width: 60px; border: 2px solid #00a; border-radius: 4px;")
        center3_row.addWidget(self.sphere3_center_x)
        center3_row.addWidget(self.sphere3_center_y)
        center3_row.addWidget(self.sphere3_center_z)
        sphere3_layout.addLayout(center3_row)


        center3_row = QHBoxLayout()
        center3_row.addWidget(QLabel("Radius:"))
        self.sphere3_radius = QDoubleSpinBox()
        self.sphere3_radius.setRange(0.001, 100.0)
        self.sphere3_radius.setDecimals(4)
        self.sphere3_radius.setSingleStep(0.01)
        self.sphere3_radius.setStyleSheet("font-size: 9pt; min-width: 60px; border: 2px solid #666; border-radius: 4px;")
        center3_row.addWidget(self.sphere3_radius)
        self.sphere3_grow_button = QPushButton("Grow")
        self.sphere3_grow_button.setFixedSize(70, 28)
        self.sphere3_grow_button.setStyleSheet("""
            QPushButton { background-color: #7bc96f; color: white; border: none; border-radius: 4px; font-weight: bold; }
            QPushButton:hover { background-color: #6ab85f; }
            QPushButton:disabled { background-color: #cccccc; color: #888888; }
        """)
        self.sphere3_grow_button.clicked.connect(lambda: self._on_grow_clicked(2))
        center3_row.addWidget(self.sphere3_grow_button)

        self.sphere3_to_joint_btn = QPushButton("To joint")
        self.sphere3_to_joint_btn.setFixedSize(70, 28)
        self.sphere3_to_joint_btn.setStyleSheet("""
            QPushButton { background-color: #9b59b6; color: white; border: none; border-radius: 4px; font-weight: bold; }
            QPushButton:hover { background-color: #8e44ad; }
            QPushButton:disabled { background-color: #cccccc; color: #888888; }
        """)
        self.sphere3_to_joint_btn.setToolTip("Move this sphere to the closest SMPL joint")
        self.sphere3_to_joint_btn.clicked.connect(lambda: self._on_move_to_joint_clicked(2))
        center3_row.addWidget(self.sphere3_to_joint_btn)

        sphere3_layout.addLayout(center3_row)

        self.sphere3_lock_btn = QPushButton("Lock")
        self.sphere3_lock_btn.setFixedSize(70, 28)
        self.sphere3_lock_btn.setStyleSheet("""
            QPushButton {
                background-color: #4ecdc4;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45b7b0;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #888888;
            }
        """)
        self.sphere3_lock_btn.clicked.connect(lambda: self._on_sphere_lock_clicked(2))
        sphere3_layout.addWidget(self.sphere3_lock_btn)
        self.sphere3_lock_btn.setEnabled(False)


        self.sphere3_box.setLayout(sphere3_layout)
        main_layout.addWidget(self.sphere3_box)
        
        self.sphere1_grow_button.setEnabled(False)
        self.sphere2_grow_button.setEnabled(False)
        self.sphere3_grow_button.setEnabled(False)
        self.sphere1_to_joint_btn.setEnabled(False)
        self.sphere2_to_joint_btn.setEnabled(False)
        self.sphere3_to_joint_btn.setEnabled(False)
        # Hide sphere 3 box by default (shown only for triples)
        self.sphere3_box.setVisible(False)
        
        self._block_center_signals = False
        self._block_radius_signals = False
        self.sphere1_center_x.valueChanged.connect(lambda: self._on_center_spinbox_changed(0))
        self.sphere1_center_y.valueChanged.connect(lambda: self._on_center_spinbox_changed(0))
        self.sphere1_center_z.valueChanged.connect(lambda: self._on_center_spinbox_changed(0))
        self.sphere2_center_x.valueChanged.connect(lambda: self._on_center_spinbox_changed(1))
        self.sphere2_center_y.valueChanged.connect(lambda: self._on_center_spinbox_changed(1))
        self.sphere2_center_z.valueChanged.connect(lambda: self._on_center_spinbox_changed(1))
        self.sphere3_center_x.valueChanged.connect(lambda: self._on_center_spinbox_changed(2))
        self.sphere3_center_y.valueChanged.connect(lambda: self._on_center_spinbox_changed(2))
        self.sphere3_center_z.valueChanged.connect(lambda: self._on_center_spinbox_changed(2))
        self.sphere1_radius.valueChanged.connect(lambda v: self._on_radius_spinbox_changed(0, v))
        self.sphere2_radius.valueChanged.connect(lambda v: self._on_radius_spinbox_changed(1, v))
        self.sphere3_radius.valueChanged.connect(lambda v: self._on_radius_spinbox_changed(2, v))
        
        main_layout.addStretch()
        self.setLayout(main_layout)
        self.adjustSize()
        
        self.spheremesh_id_label.setStyleSheet("font-weight: bold; font-size: 11pt; color: #000000;")
        self.selected_sphere_label.setStyleSheet("font-size: 10pt; color: #333333;")

    
    def update_selected_info(self):
        """Update the selected sphere information"""
        if not hasattr(self, 'sphere_manager') or self.sphere_manager is None:
            return

        data = self.sphere_manager.get_selected_sphere_display_data()
        if data is not None:
            gidx = data["sphere_index"]
            self.spheremesh_id_label.setText(f"Sphere #{gidx}")
            twin = data.get("twin_index")
            inc = data.get("incident_connections") or []
            twin_text = f"Twin: {twin}" if twin is not None else "Twin: None"
            self.selected_sphere_label.setText(twin_text)
            if inc:
                self.connections_text.setPlainText("\n".join(str(p) for p in inc))
            else:
                self.connections_text.setPlainText("No connection (orphan)")

            centers = data["centers"]
            radii = data["radii"]
            self._block_center_signals = True
            self.sphere1_center_x.setValue(centers[0][0])
            self.sphere1_center_y.setValue(centers[0][1])
            self.sphere1_center_z.setValue(centers[0][2])
            self.sphere1_box.setVisible(True)
            self.sphere2_box.setVisible(False)
            self.sphere3_box.setVisible(False)
            self._block_center_signals = False
            self._block_radius_signals = True
            self.sphere1_radius.setValue(radii[0])
            self._block_radius_signals = False

            self.delete_button.setEnabled(True)
            self.sphere1_grow_button.setEnabled(True)
            self.sphere2_grow_button.setEnabled(False)
            self.sphere3_grow_button.setEnabled(False)
            self.sphere1_to_joint_btn.setEnabled(True)
            self.sphere2_to_joint_btn.setEnabled(False)
            self.sphere3_to_joint_btn.setEnabled(False)
            self.sphere1_radius.setEnabled(True)
            self.sphere2_radius.setEnabled(False)
            self.sphere3_radius.setEnabled(False)
            locked_per_sphere = data.get("locked_per_sphere", [False, False, False])
            self.update_lock_buttons(locked_per_sphere, 1)
        else:
            self.spheremesh_id_label.setText("Spheremesh: None")
            self.selected_sphere_label.setText("Selected: None")
            self.connections_text.clear()
            self._block_center_signals = True
            for sb in (self.sphere1_center_x, self.sphere1_center_y, self.sphere1_center_z,
                      self.sphere2_center_x, self.sphere2_center_y, self.sphere2_center_z,
                      self.sphere3_center_x, self.sphere3_center_y, self.sphere3_center_z):
                sb.setValue(0.0)
            self._block_center_signals = False
            self._block_radius_signals = True
            self.sphere1_radius.setValue(0.0)
            self.sphere2_radius.setValue(0.0)
            self.sphere3_radius.setValue(0.0)
            self._block_radius_signals = False
            self.sphere1_radius.setEnabled(False)
            self.sphere2_radius.setEnabled(False)
            self.sphere3_radius.setEnabled(False)
            self.sphere1_box.setVisible(True)
            self.sphere2_box.setVisible(False)
            self.sphere3_box.setVisible(False)
            self.delete_button.setEnabled(False)
            self.update_lock_buttons([False, False, False], 0)
            self.sphere1_grow_button.setEnabled(False)
            self.sphere2_grow_button.setEnabled(False)
            self.sphere3_grow_button.setEnabled(False)
            self.sphere1_to_joint_btn.setEnabled(False)
            self.sphere2_to_joint_btn.setEnabled(False)
            self.sphere3_to_joint_btn.setEnabled(False)

        self.adjustSize()

    def _on_radius_spinbox_changed(self, sphere_index, value):
        """Apply numeric radius change to the selected group's sphere."""
        if getattr(self, '_block_radius_signals', True):
            return
        if not hasattr(self, 'sphere_manager') or self.sphere_manager is None:
            return
        data = self.sphere_manager.get_selected_sphere_display_data()
        if data is None or sphere_index >= data["num_slots"]:
            return
        g = data["endpoint_indices"][sphere_index]
        self.sphere_manager.set_sphere_radius_global(g, value)

    def _on_center_spinbox_changed(self, sphere_index):
        """Apply numeric center change to the selected group's sphere."""
        if getattr(self, '_block_center_signals', True):
            return
        if not hasattr(self, 'sphere_manager') or self.sphere_manager is None:
            return
        data = self.sphere_manager.get_selected_sphere_display_data()
        if data is None or sphere_index >= data["num_slots"]:
            return
        if sphere_index == 0:
            x, y, z = self.sphere1_center_x.value(), self.sphere1_center_y.value(), self.sphere1_center_z.value()
        elif sphere_index == 1:
            x, y, z = self.sphere2_center_x.value(), self.sphere2_center_y.value(), self.sphere2_center_z.value()
        else:
            x, y, z = self.sphere3_center_x.value(), self.sphere3_center_y.value(), self.sphere3_center_z.value()
        g = data["endpoint_indices"][sphere_index]
        self.sphere_manager.set_sphere_center_global(g, [x, y, z])

    def _on_delete_clicked(self):
        """Handle delete button click."""
        if hasattr(self, 'sphere_manager') and self.sphere_manager is not None:
            if self.sphere_manager.delete_selected_sphere():
                self.update_selected_info()

    def _on_sphere_lock_clicked(self, sphere_index):
        """Handle per-sphere lock button click (locks/unlocks that sphere in the selected group)."""
        if hasattr(self, 'sphere_manager') and self.sphere_manager is not None:
            self.sphere_manager.toggle_lock_sphere_at_display_slot(sphere_index)
            self.update_selected_info()

    def _on_grow_clicked(self, sphere_index):
        """Handle grow button click for the given sphere; delegates to grow_click_callback (set by MainWindow) so mesh can be passed."""
        if self.grow_click_callback is not None:
            self.grow_click_callback(sphere_index)
        elif hasattr(self, 'sphere_manager') and self.sphere_manager is not None:
            self.sphere_manager.grow_sphere(mesh=None, sphere_index=sphere_index)

    def _on_move_to_joint_clicked(self, sphere_index):
        """Handle 'To joint' button: move this sphere to the closest SMPL joint (callback set by MainWindow)."""
        if getattr(self, 'move_to_joint_callback', None) is not None:
            self.move_to_joint_callback(sphere_index)

    def update_lock_buttons(self, locked_per_sphere, num_spheres):
        """Update per-sphere lock button text and style. locked_per_sphere: list of 3 bools; num_spheres: 0, 1, 2, or 3."""
        locked_style = """
            QPushButton {
                background-color: #ffa07a;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #ff8c61; }
            QPushButton:disabled { background-color: #cccccc; color: #888888; }
        """
        unlocked_style = """
            QPushButton {
                background-color: #4ecdc4;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #45b7b0; }
            QPushButton:disabled { background-color: #cccccc; color: #888888; }
        """
        for i, btn in enumerate([self.sphere1_lock_btn, self.sphere2_lock_btn, self.sphere3_lock_btn]):
            enabled = (i + 1) <= num_spheres
            btn.setEnabled(enabled)
            is_locked = bool(locked_per_sphere[i]) if i < len(locked_per_sphere) else False
            btn.setText("Unlock" if is_locked else "Lock")
            btn.setStyleSheet(locked_style if is_locked else unlocked_style)


class ClipPlanesPanel(QGroupBox):
    """Panel for clip planes: add X/Y/Z plane pairs, reset, and set depth (position) per plane."""

    def __init__(self, parent):
        super().__init__("Clip Planes", parent)
        self._main_window = None  # set by MainWindow after creation
        self._axis_widgets = {}  # axis -> (row_widget, spin_lo, spin_hi)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        btn_layout = QHBoxLayout()
        self.btn_x = QPushButton("Add X Planes")
        self.btn_y = QPushButton("Add Y Planes")
        self.btn_z = QPushButton("Add Z Planes")
        self.btn_x.clicked.connect(self._on_add_x)
        self.btn_y.clicked.connect(self._on_add_y)
        self.btn_z.clicked.connect(self._on_add_z)
        btn_layout.addWidget(self.btn_x)
        btn_layout.addWidget(self.btn_y)
        btn_layout.addWidget(self.btn_z)
        layout.addLayout(btn_layout)

        self.reset_btn = QPushButton("Reset (remove all clip planes)")
        self.reset_btn.clicked.connect(self._on_reset)
        layout.addWidget(self.reset_btn)

        for axis in ('x', 'y', 'z'):
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 4, 0, 4)
            row_layout.addWidget(QLabel(f"{axis.upper()} low:"))
            spin_lo = QDoubleSpinBox()
            spin_lo.setDecimals(4)
            spin_lo.setRange(-1e6, 1e6)
            spin_lo.setSingleStep(0.01)
            spin_lo.valueChanged.connect(lambda v, a=axis, i=0: self._on_depth_changed(a, i, v))
            row_layout.addWidget(spin_lo)
            row_layout.addWidget(QLabel(f"high:"))
            spin_hi = QDoubleSpinBox()
            spin_hi.setDecimals(4)
            spin_hi.setRange(-1e6, 1e6)
            spin_hi.setSingleStep(0.01)
            spin_hi.valueChanged.connect(lambda v, a=axis, i=1: self._on_depth_changed(a, i, v))
            row_layout.addWidget(spin_hi)
            row.setVisible(False)
            self._axis_widgets[axis] = (row, spin_lo, spin_hi)
            layout.addWidget(row)

        layout.addStretch()

    def set_main_window(self, main_window):
        self._main_window = main_window

    def _on_add_x(self):
        if self._main_window:
            self._main_window._add_clip_planes_axis('x')

    def _on_add_y(self):
        if self._main_window:
            self._main_window._add_clip_planes_axis('y')

    def _on_add_z(self):
        if self._main_window:
            self._main_window._add_clip_planes_axis('z')

    def _on_reset(self):
        if self._main_window:
            self._main_window._reset_clip_planes()

    def _on_depth_changed(self, axis, plane_index, value):
        if self._main_window:
            self._main_window._set_clip_plane_depth(axis, plane_index, value)

    def set_axis_active(self, axis, pos_lo, pos_hi, bounds_lo, bounds_hi):
        """Show depth inputs for this axis and set their range and values."""
        if axis not in self._axis_widgets:
            return
        row, spin_lo, spin_hi = self._axis_widgets[axis]
        spin_lo.blockSignals(True)
        spin_hi.blockSignals(True)
        spin_lo.setRange(bounds_lo, bounds_hi)
        spin_hi.setRange(bounds_lo, bounds_hi)
        spin_lo.setValue(pos_lo)
        spin_hi.setValue(pos_hi)
        spin_lo.blockSignals(False)
        spin_hi.blockSignals(False)
        row.setVisible(True)

    def set_axis_inactive(self, axis):
        if axis in self._axis_widgets:
            self._axis_widgets[axis][0].setVisible(False)

    def set_all_axes_inactive(self):
        for axis in ('x', 'y', 'z'):
            self.set_axis_inactive(axis)

    def update_depth_value(self, axis, plane_index, value):
        """Update spinbox value from widget drag (no signal)."""
        if axis not in self._axis_widgets:
            return
        spin = self._axis_widgets[axis][1 + plane_index]  # spin_lo or spin_hi
        spin.blockSignals(True)
        spin.setValue(value)
        spin.blockSignals(False)


class MainWindow(QMainWindow):

    smpl_model = None
    theta = tc.zeros(72, dtype=tc.float64)
    sphere_manager = None
    mesh_actor = None

    def __init__(self, smpl_model):
        super().__init__()

        self.smpl_model = smpl_model
        self.beta = default_beta_tensor()

        # Ortho axis views: repeat same letter (X/Y/Z) flips viewing direction
        self._camera_align_axis = None
        self._camera_align_flip = False

        self.setWindowTitle("PyQt6 + PyVista (Interactive)")
        self.resize(1000, 600)

        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")
        load_topology_action = file_menu.addAction("Load Topology…")
        load_topology_action.setShortcut("Ctrl+O")
        load_topology_action.triggered.connect(self.load_topology_from_file)
        import_spheremesh_action = file_menu.addAction("Import SphereMesh…")
        import_spheremesh_action.triggered.connect(self.import_spheremesh_from_file)
        import_spheres_only_action = file_menu.addAction("Import SphereMesh (no connections)…")
        import_spheres_only_action.triggered.connect(self.import_spheres_only_from_file)

        edit_menu = menubar.addMenu("Edit")
        edit_menu.addAction("Undo")
        edit_menu.addAction("Redo")

        # Create the main frame
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout(self.central_widget)

        # --- LEFT PANEL: UI Controls ---
        self.ui_scroll = QScrollArea()
        self.ui_scroll.setWidgetResizable(True)
        self.ui_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.ui_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.ui_scroll.setStyleSheet("QScrollArea { border: none; }")
        self.ui_scroll.setFixedWidth(220)

        self.ui_panel = QWidget()
        self.ui_layout = QVBoxLayout(self.ui_panel)
        self.ui_panel.setMinimumWidth(200)
        self.ui_scroll.setWidget(self.ui_panel)
        
        # Checkbox to show/hide SMPL model
        self.checkbox_show_model = QCheckBox("Show SMPL Model")
        self.checkbox_show_model.setChecked(True)
        self.checkbox_show_model.stateChanged.connect(self.toggle_smpl_model)
        
        # Checkbox to render/destroy spheremeshes
        self.checkbox_show_spheremeshes = QCheckBox("Show Spheremeshes")
        self.checkbox_show_spheremeshes.setChecked(False)
        self.checkbox_show_spheremeshes.stateChanged.connect(self.toggle_spheremeshes)

        # Checkbox to show per-vertex heatmap on SMPL mesh (with legend)
        self.checkbox_show_heatmap = QCheckBox("Show heatmap")
        self.checkbox_show_heatmap.setChecked(False)
        self.checkbox_show_heatmap.stateChanged.connect(self.toggle_heatmap)

        # Checkbox to show/hide sphere widgets (draggable spheres)
        self.checkbox_show_sphere_widgets = QCheckBox("Show sphere widgets")
        self.checkbox_show_sphere_widgets.setChecked(True)
        self.checkbox_show_sphere_widgets.stateChanged.connect(self.toggle_sphere_widgets)

        # Checkbox to mirror manual sphere edits to X-axis twin.
        self.checkbox_mirror_edits = QCheckBox("Mirror edits")
        self.checkbox_mirror_edits.setChecked(True)
        self.checkbox_mirror_edits.setToolTip(
            "When enabled, manual edits on a sphere are mirrored to its X-axis twin."
        )

        # Checkbox to show/hide SMPL skeleton (bones + joints)
        self.checkbox_show_skeleton = QCheckBox("Show skeleton")
        self.checkbox_show_skeleton.setChecked(False)
        self.checkbox_show_skeleton.stateChanged.connect(self.toggle_skeleton)

        # Inflate preview: scale every sphere radius by a factor.
        self.checkbox_inflate = QCheckBox("Inflate")
        self.checkbox_inflate.setChecked(False)
        self.checkbox_inflate.setToolTip(
            "Temporarily scales all sphere radii by the inflate factor. Uncheck to restore each sphere's radius."
        )
        self.checkbox_uniform_radius_preview = QCheckBox("Uniform radius 0.2 (preview)")
        self.checkbox_uniform_radius_preview.setChecked(False)
        self.checkbox_uniform_radius_preview.setToolTip(
            "Temporarily sets all sphere radii to 0.2 for layout/preview. Uncheck to restore each sphere's radius."
        )
        self.inflate_label = QLabel("Inflate scale: 1.10x")
        self.inflate_label.setStyleSheet("color: black;")
        self.inflate_slider = QSlider(Qt.Orientation.Horizontal)
        self.inflate_slider.setMinimum(10)   # 0.10x
        self.inflate_slider.setMaximum(300)  # 3.00x
        self.inflate_slider.setValue(110)    # 1.10x
        self.inflate_slider.valueChanged.connect(self._on_inflate_value_changed)
        
        # Opacity slider for spheremeshes
        self.opacity_label = QLabel("Spheremesh Opacity: 0.50")
        self.opacity_label.setStyleSheet("color: black;")
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setMinimum(1)
        self.opacity_slider.setMaximum(100)
        self.opacity_slider.setValue(50)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        
        # Opacity slider for model
        self.model_opacity_label = QLabel("Model Opacity: 0.90")
        self.model_opacity_label.setStyleSheet("color: black;")
        self.model_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.model_opacity_slider.setMinimum(1)
        self.model_opacity_slider.setMaximum(100)
        self.model_opacity_slider.setValue(90)
        self.model_opacity_slider.valueChanged.connect(self._on_model_opacity_changed)
        
        self.ui_layout.addWidget(self.checkbox_show_model)
        self.ui_layout.addWidget(self.checkbox_show_spheremeshes)
        self.ui_layout.addWidget(self.checkbox_show_heatmap)
        self.ui_layout.addWidget(self.checkbox_show_sphere_widgets)
        self.ui_layout.addWidget(self.checkbox_mirror_edits)
        self.ui_layout.addWidget(self.checkbox_show_skeleton)
        self.ui_layout.addWidget(self.checkbox_inflate)
        self.ui_layout.addWidget(self.checkbox_uniform_radius_preview)
        self.ui_layout.addWidget(self.inflate_label)
        self.ui_layout.addWidget(self.inflate_slider)
        self.ui_layout.addWidget(self.opacity_label)
        self.ui_layout.addWidget(self.opacity_slider)
        self.ui_layout.addWidget(self.model_opacity_label)
        self.ui_layout.addWidget(self.model_opacity_slider)
        
        self.add_sphere_button = QPushButton("Add Sphere")
        self.add_sphere_button.clicked.connect(self.add_sphere_clicked)
        self.ui_layout.addWidget(self.add_sphere_button)

        self.add_connection_button = QPushButton("Add Connection…")
        self.add_connection_button.clicked.connect(self.add_connection_clicked)
        self.ui_layout.addWidget(self.add_connection_button)
        self.remove_connection_button = QPushButton("Remove Connection…")
        self.remove_connection_button.clicked.connect(self.remove_connection_clicked)
        self.ui_layout.addWidget(self.remove_connection_button)

        self.selected_to_joint_button = QPushButton("Selected → closest joint")
        self.selected_to_joint_button.setToolTip(
            "Move the currently selected sphere to the nearest SMPL joint (same pose as the body)"
        )
        self.selected_to_joint_button.setStyleSheet("""
            QPushButton { background-color: #9b59b6; color: white; border: none; border-radius: 4px;
                font-weight: bold; padding: 6px; }
            QPushButton:hover { background-color: #8e44ad; }
            QPushButton:disabled { background-color: #cccccc; color: #888888; }
        """)
        self.selected_to_joint_button.clicked.connect(self.move_selected_to_closest_joint_clicked)
        self.selected_to_joint_button.setEnabled(False)
        self.ui_layout.addWidget(self.selected_to_joint_button)

        self.reassign_parents_button = QPushButton("Reassign all parents")
        self.reassign_parents_button.setToolTip(
            "Recalculate primary and secondary parent joints, rest offsets, and LBS weights for every sphere."
        )
        self.reassign_parents_button.clicked.connect(self.reassign_all_parent_joints_clicked)
        self.ui_layout.addWidget(self.reassign_parents_button)

        self.check_symmetry_button = QPushButton("Check X Symmetry")
        self.check_symmetry_button.setToolTip(
            "Validate mirrored sphere pairs across X axis (center, radius, and mirrored connections)."
        )
        self.check_symmetry_button.clicked.connect(self.check_x_symmetry_clicked)
        self.ui_layout.addWidget(self.check_symmetry_button)
        
        # Add save button
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_file)
        self.ui_layout.addWidget(self.save_button)

        # Add Export FBX button
        self.export_fbx_button = QPushButton("Export FBX")
        self.export_fbx_button.clicked.connect(self.export_smpl_fbx)
        self.ui_layout.addWidget(self.export_fbx_button)

        # Add Export OBJ button
        self.export_obj_button = QPushButton("Export OBJ")
        self.export_obj_button.clicked.connect(self.export_smpl_obj)
        self.ui_layout.addWidget(self.export_obj_button)

        # Load beta offset CSVs (beta*.csv) from fixed offsets directory.
        self.load_beta_offsets_button = QPushButton("Load Beta Offsets")
        self.load_beta_offsets_button.setToolTip(
            "Load beta*.csv files from python/SphereMeshTool/data/offsets "
            "(sphere_id, dx, dy, dz, dr) to drive sphere positions/radii."
        )
        self.load_beta_offsets_button.clicked.connect(self.load_beta_offsets_clicked)
        self.ui_layout.addWidget(self.load_beta_offsets_button)

        # Add beta slider group
        self.beta_slider_group = BetaSliderGroup()
        self.beta_slider_group.valuesChanged.connect(self._on_beta_changed)
        self.ui_layout.addWidget(self.beta_slider_group)
        
        # Add theta slider group
        self.theta_slider_group = ThetaSliderGroup()
        self.theta_slider_group.valuesChanged.connect(self._on_theta_changed)
        self.ui_layout.addWidget(self.theta_slider_group)
        
        self.ui_layout.addStretch()  # Push items to top

        # --- CENTER + RIGHT PANEL ---
        center_right_container = QWidget()
        center_right_layout = QVBoxLayout(center_right_container)
        center_right_layout.setContentsMargins(0, 0, 0, 0)

        # --- PyVista Plotter ---
        self.plotter = QtInteractor(center_right_container)
        self.plotter.add_axes()
        self.plotter.camera_position = 'xy'

        center_right_layout.addWidget(self.plotter.interactor)

        # --- Info Panel (OVERLAY) ---
        self.info_panel = InfoPanel(center_right_container)
        self.info_panel.setMinimumSize(280, 390)
        self.info_panel.raise_()

        self.info_panel.setStyleSheet("""
            InfoPanel {
                background-color: rgba(255, 255, 255, 245);
                border: 2px solid rgba(80, 80, 80, 200);
                border-radius: 10px;
                padding: 8px;
            }
            InfoPanel > QLabel {
                background-color: transparent;
            }
        """)

        # Clip plane state (must exist before ClipPlanesPanel so reset works)
        self._clip_planes_vtk = {}
        self._clip_plane_actors = {}  # axis -> [actor_lo, actor_hi] (visible plane meshes, no widgets)
        self._saved_interactor_style = None  # restore when clip planes are reset

        # --- Clip Planes Panel (OVERLAY, bottom-left of 3D view) ---
        self.clip_planes_panel = ClipPlanesPanel(center_right_container)
        self.clip_planes_panel.setMinimumSize(320, 100)
        self.clip_planes_panel.setStyleSheet("""
            ClipPlanesPanel, QGroupBox {
                background-color: rgba(255, 255, 255, 245);
                border: 2px solid rgba(80, 80, 80, 200);
                border-radius: 10px;
                padding: 8px;
            }
        """)
        self.clip_planes_panel.set_main_window(self)

        # --- Add to main layout ---
        self.layout.addWidget(self.ui_scroll)
        self.layout.addWidget(center_right_container)

        def _on_sphere_manager_ui_refresh():
            self.info_panel.update_selected_info()
            self._sync_selected_to_joint_toolbar()

        # Initialize sphere manager with callback
        self.sphere_manager = SphereManager(self.plotter, _on_sphere_manager_ui_refresh)

        # Pass SMPL LBS weights once; they are pose/shape independent.
        if hasattr(self.smpl_model, "weights") and self.smpl_model.weights is not None:
            self.sphere_manager.set_skinning_weights(
                self.smpl_model.weights.detach().cpu().numpy()
            )

        self.checkbox_inflate.stateChanged.connect(self.toggle_inflate)
        self.checkbox_uniform_radius_preview.stateChanged.connect(self.toggle_uniform_radius_preview)
        self.checkbox_mirror_edits.stateChanged.connect(self.toggle_mirror_edits)
        self.sphere_manager.set_mirror_edits_enabled(self.checkbox_mirror_edits.isChecked())
        
        # Link sphere manager to info panel
        self.info_panel.sphere_manager = self.sphere_manager
        
        # Set lock button callback
        self.sphere_manager.set_lock_button_callback(self.info_panel.update_lock_buttons)
        
        # Set refocus callback to restore keyboard focus
        self.sphere_manager.set_refocus_callback(self.setFocus)

        verts, joints, Rs, T = my_function_forward(self.smpl_model, self.beta, self.theta)
        _smpl_rs_are_global = T is not None
        if Rs.shape[-1] == 4:
            Rs = Rs[..., :3, :3]

        # Convert vertices to numpy and create PyVista mesh
        verts_np = verts.detach().cpu().numpy()
        faces_np = self.smpl_model.faces
        
        # Create PyVista mesh from vertices and faces
        faces_pv = np.hstack([np.full((faces_np.shape[0], 1), 3), faces_np]).flatten()
        mesh = pv.PolyData(verts_np, faces_pv)
        
        self.mesh_actor = self.plotter.add_mesh(mesh, show_edges=False, color='lightgray')
        self.mesh_actor.GetProperty().SetOpacity(0.90)
        self.heatmap_scalar_bar_actor = None  # legend in top left when show heatmap is on
        self._skeleton_bone_actors = []
        self._skeleton_joint_actors = []
        self._skeleton_visible = False

        # InfoPanel Grow button: use MainWindow handler so we can pass the SMPL mesh
        self.info_panel.grow_click_callback = self._handle_grow_clicked
        self.info_panel.move_to_joint_callback = self._move_sphere_to_closest_joint
        self.info_panel.toggle_clip_planes_panel_callback = self._toggle_clip_planes_panel_visibility

        # Register heatmap callback so sphere_manager can trigger mesh/legend update
      
        
        # Initialize skeleton information in sphere manager
        self._update_skeleton(verts, joints, Rs, rs_are_global=_smpl_rs_are_global)
        
        # Position info panel (top-right) and clip planes panel (bottom-left)
        def on_resize(event):
            self._position_info_panel(event, center_right_container)
            self._position_clip_planes_panel(center_right_container)
        center_right_container.resizeEvent = on_resize

        load_path = Path(__file__).resolve().parent.parent / "data" / "template.json"
        try:
            self.sphere_manager.load_from_json(str(load_path))
            self.sphere_manager.initialize_radius_scaling_from_mesh(mesh)
        except (ValueError, OSError, json.JSONDecodeError) as e:
            print(f"Could not load {load_path}: {e}")

        for key_str, axis_name in (("X", "x"), ("Y", "y"), ("Z", "z")):
            sc = QShortcut(QKeySequence(key_str), self)
            sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
            sc.setAutoRepeat(False)
            sc.activated.connect(lambda a=axis_name: self._on_camera_axis_shortcut(a))

    def _scene_focal_center_and_span(self):
        """Center and max extent of scene for camera positioning."""
        b = self._get_scene_bounds()
        cx = 0.5 * (b[0] + b[1])
        cy = 0.5 * (b[2] + b[3])
        cz = 0.5 * (b[4] + b[5])
        span = max(b[1] - b[0], b[3] - b[2], b[5] - b[4], 0.5)
        return (cx, cy, cz), float(span)

    def _on_camera_axis_shortcut(self, axis: str):
        if self._camera_align_axis == axis:
            self._camera_align_flip = not self._camera_align_flip
        else:
            self._camera_align_axis = axis
            self._camera_align_flip = False
        self._apply_axis_aligned_camera()

    def _apply_axis_aligned_camera(self):
        if self._camera_align_axis is None or not hasattr(self, "plotter"):
            return
        c, span = self._scene_focal_center_and_span()
        dist = max(span * 1.5, 1.0)
        flip = self._camera_align_flip
        cx, cy, cz = c
        focal = (cx, cy, cz)
        if self._camera_align_axis == "x":
            if not flip:
                pos = (cx - dist, cy, cz)
            else:
                pos = (cx + dist, cy, cz)
            up = (0.0, 0.0, 1.0)
        elif self._camera_align_axis == "y":
            if not flip:
                pos = (cx, cy - dist, cz)
            else:
                pos = (cx, cy + dist, cz)
            up = (0.0, 0.0, 1.0)
        else:
            if not flip:
                pos = (cx, cy, cz - dist)
            else:
                pos = (cx, cy, cz + dist)
            up = (0.0, 1.0, 0.0)
        self.plotter.camera_position = (pos, focal, up)
        self.plotter.reset_camera_clipping_range()
        self.plotter.render()

    def _get_scene_bounds(self):
        """Return (xmin, xmax, ymin, ymax, zmin, zmax) from plotter or main mesh."""
        try:
            b = self.plotter.bounds
            if b is not None and len(b) == 6:
                return b
        except Exception:
            pass
        if hasattr(self, 'mesh_actor') and self.mesh_actor is not None:
            mesh = self.mesh_actor.GetMapper().GetInput()
            if mesh is not None:
                return mesh.GetBounds()
        return (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)

    def _apply_clip_planes_to_all_actors(self):
        """Build a vtkPlaneCollection from all active clip planes and set it on every mesh mapper."""
        collection = vtk.vtkPlaneCollection()
        for axis in ('x', 'y', 'z'):
            if axis in self._clip_planes_vtk:
                plow, phigh = self._clip_planes_vtk[axis]
                collection.AddItem(plow)
                collection.AddItem(phigh)
        for name, actor in self.plotter.actors.items():
            mapper = actor.GetMapper() if hasattr(actor, 'GetMapper') else None
            if mapper is not None and hasattr(mapper, 'SetClippingPlanes'):
                if collection.GetNumberOfItems() > 0:
                    mapper.SetClippingPlanes(collection)
                else:
                    mapper.RemoveAllClippingPlanes()
        self.plotter.render()

    def _add_clip_planes_axis(self, axis):
        """Add two visible (non-interactive) clip planes for the given axis. Position only via panel."""
        if axis in self._clip_plane_actors:
            return  # already added
        bounds = self._get_scene_bounds()
        axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
        lo, hi = bounds[axis_idx * 2], bounds[axis_idx * 2 + 1]
        span = max(hi - lo, 0.1)
        pos_lo = lo + span * 0.25
        pos_hi = hi - span * 0.25
        normals = {'x': ((1, 0, 0), (-1, 0, 0)), 'y': ((0, 1, 0), (0, -1, 0)), 'z': ((0, 0, 1), (0, 0, -1))}[axis]
        origin_lo = [bounds[0], bounds[2], bounds[4]]
        origin_lo[axis_idx] = pos_lo
        origin_hi = [bounds[0], bounds[2], bounds[4]]
        origin_hi[axis_idx] = pos_hi
        plane_lo = vtk.vtkPlane()
        plane_lo.SetNormal(normals[0])
        plane_lo.SetOrigin(origin_lo)
        plane_hi = vtk.vtkPlane()
        plane_hi.SetNormal(normals[1])
        plane_hi.SetOrigin(origin_hi)
        self._clip_planes_vtk[axis] = (plane_lo, plane_hi)

        # Visible plane meshes (no widgets — position only via panel)
        def plane_size(b, ax):
            if ax == 0:
                return max(b[3] - b[2], 0.1), max(b[5] - b[4], 0.1)
            if ax == 1:
                return max(b[1] - b[0], 0.1), max(b[5] - b[4], 0.1)
            return max(b[1] - b[0], 0.1), max(b[3] - b[2], 0.1)
        size_lo = plane_size(bounds, axis_idx)
        plo_mesh = pv.Plane(center=origin_lo, direction=normals[0], i_size=size_lo[0], j_size=size_lo[1])
        phi_mesh = pv.Plane(center=origin_hi, direction=normals[1], i_size=size_lo[0], j_size=size_lo[1])
        actor_lo = self.plotter.add_mesh(plo_mesh,opacity=0.0, edge_color='cyan', show_edges=True)
        actor_hi = self.plotter.add_mesh(phi_mesh,opacity=0.0, edge_color='cyan', show_edges=True)
        self._clip_plane_actors[axis] = [actor_lo, actor_hi]

        self._apply_clip_planes_to_all_actors()
        if hasattr(self, 'clip_planes_panel'):
            self.clip_planes_panel.set_axis_active(axis, pos_lo, pos_hi, lo, hi)
        self.adjustSize()

    def _set_clip_plane_depth(self, axis, plane_index, value):
        """Set the position (depth) of one clip plane and update the visible plane mesh."""
        if axis not in self._clip_planes_vtk or axis not in self._clip_plane_actors:
            return
        axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
        plane = self._clip_planes_vtk[axis][plane_index]
        origin = list(plane.GetOrigin())
        origin[axis_idx] = value
        plane.SetOrigin(origin)
        bounds = self._get_scene_bounds()
        normals = {'x': ((1, 0, 0), (-1, 0, 0)), 'y': ((0, 1, 0), (0, -1, 0)), 'z': ((0, 0, 1), (0, 0, -1))}[axis]
        def plane_size(b, ax):
            if ax == 0:
                return max(b[3] - b[2], 0.1), max(b[5] - b[4], 0.1)
            if ax == 1:
                return max(b[1] - b[0], 0.1), max(b[5] - b[4], 0.1)
            return max(b[1] - b[0], 0.1), max(b[3] - b[2], 0.1)
        size = plane_size(bounds, axis_idx)
        new_plane = pv.Plane(center=origin, direction=normals[plane_index], i_size=size[0], j_size=size[1])
        actor = self._clip_plane_actors[axis][plane_index]
        mapper = actor.GetMapper()
        if mapper and mapper.GetInput():
            vtk_data = getattr(new_plane, '_vtk_obj', getattr(new_plane, 'VTKObject', new_plane))
            mapper.GetInput().ShallowCopy(vtk_data)
        self._apply_clip_planes_to_all_actors()

    def _reset_clip_planes(self):
        """Remove all clip planes and visible plane meshes (reset to starting state)."""
        for axis in list(self._clip_plane_actors.keys()):
            for actor in self._clip_plane_actors[axis]:
                self.plotter.remove_actor(actor)
        self._clip_planes_vtk.clear()
        self._clip_plane_actors.clear()
        if hasattr(self, 'clip_planes_panel'):
            self.clip_planes_panel.set_all_axes_inactive()
        self._apply_clip_planes_to_all_actors()
        self.adjustSize()
    
    def _position_info_panel(self, event, container):
        """Position the info panel in the top-right corner"""
        container_width = container.width()
        panel_width = self.info_panel.width()
        margin = 10
        self.info_panel.move(container_width - panel_width - margin, margin)
        self.info_panel.raise_()

    def _position_clip_planes_panel(self, container):
        """Position the clip planes panel in the bottom-left corner of the 3D view."""
        margin = 10
        self.clip_planes_panel.move(container.width() - self.clip_planes_panel.width() - margin, container.height() - self.clip_planes_panel.height() - margin)
        self.clip_planes_panel.raise_()

    def _toggle_clip_planes_panel_visibility(self):
        """Toggle Clip Planes panel visibility. Returns True if panel is now visible."""
        vis = not self.clip_planes_panel.isVisible()
        self.clip_planes_panel.setVisible(vis)
        return vis

    def _update_skeleton(self, verts, joints, Rs, *, rs_are_global: bool = True):
        """Update skeleton information in sphere manager"""
        joints_np = joints.detach().cpu().numpy()
        Rs_np = Rs.detach().cpu().numpy()
        if Rs_np.shape[-1] == 4:
            Rs_np = Rs_np[..., :3, :3]
        kintree = self.smpl_model.parents

        # Create current skeleton
        self.sphere_manager.skeleton = Skeleton(joints_np, kintree, Rs_np, rs_are_global=rs_are_global)

        # Initialize rest skeleton if not already set (must match self.beta so FK rest_offset matches the mesh).
        if self.sphere_manager.rest_skeleton is None:
            # Reference pose: same shape as the loaded mesh (self.beta), zero pose (theta).
            theta_rest = tc.zeros(72, dtype=tc.float64)
            verts_rest, joints_rest, Rs_rest, T_rest = my_function_forward(
                self.smpl_model, self.beta, theta_rest
            )
            rest_rs_global = T_rest is not None
            if Rs_rest.shape[-1] == 4:
                Rs_rest = Rs_rest[..., :3, :3]

            joints_rest_np = joints_rest.detach().cpu().numpy()
            Rs_rest_np = Rs_rest.detach().cpu().numpy()

            self.sphere_manager.rest_skeleton = Skeleton(
                joints_rest_np, kintree, Rs_rest_np, rs_are_global=rest_rs_global
            )
            # Store rest-pose vertex positions so assign_parent_joint can look up skin weights.
            self.sphere_manager.set_rest_mesh_points(
                verts_rest.detach().cpu().numpy()
            )
            # Preserve existing parent/rest_offset bindings so spheres follow the updated
            # skeleton after beta changes. Only auto-assign bindings when missing.
            for i in range(self.sphere_manager.get_sphere_count()):
                sp = self.sphere_manager.spheres[i]
                if (
                    sp.get_parent_joint() is None
                    or sp.get_rest_offset() is None
                    or sp.get_secondary_joint() is None
                ):
                    self.sphere_manager.assign_parent_joint(i)
        if self._skeleton_visible:
            self._update_skeleton_actors()

    def _clear_skeleton_actors(self):
        for actor in self._skeleton_bone_actors:
            self.plotter.remove_actor(actor)
        for actor in self._skeleton_joint_actors:
            self.plotter.remove_actor(actor)
        self._skeleton_bone_actors.clear()
        self._skeleton_joint_actors.clear()

    def _rebuild_skeleton_actors(self):
        self._clear_skeleton_actors()
        if not self.sphere_manager or self.sphere_manager.skeleton is None:
            return
        joints = np.asarray(self.sphere_manager.skeleton.joints)
        kintree = np.asarray(self.sphere_manager.skeleton.kintree)
        if joints.size == 0 or kintree.size == 0:
            return

        # Render bones as black tubes between parent-child joints.
        for i in range(1, len(kintree)):
            parent_idx = int(kintree[i])
            if parent_idx < 0 or parent_idx >= len(joints):
                continue
            c0 = joints[parent_idx]
            c1 = joints[i]
            line = pv.Line(c0, c1)
            tube = line.tube(radius=0.001, n_sides=4)
            actor = self.plotter.add_mesh(
                tube, color="black", opacity=1.0, smooth_shading=True
            )
            self._skeleton_bone_actors.append(actor)

        # Render each joint as a blue low-resolution sphere.
        for center in joints:
            sphere = pv.Sphere(
                radius=0.005,
                center=center,
                theta_resolution=6,
                phi_resolution=6,
            )
            actor = self.plotter.add_mesh(
                sphere, color="blue", opacity=1.0, smooth_shading=True
            )
            self._skeleton_joint_actors.append(actor)

        self._set_skeleton_visibility(self._skeleton_visible)

    def _update_skeleton_actors(self):
        """Fast path: keep actors, only update mesh data."""
        if not self.sphere_manager or self.sphere_manager.skeleton is None:
            return
        joints = np.asarray(self.sphere_manager.skeleton.joints)
        kintree = np.asarray(self.sphere_manager.skeleton.kintree)
        if joints.size == 0 or kintree.size == 0:
            return

        expected_bones = max(len(kintree) - 1, 0)
        expected_joints = len(joints)
        if (
            len(self._skeleton_bone_actors) != expected_bones
            or len(self._skeleton_joint_actors) != expected_joints
        ):
            self._rebuild_skeleton_actors()
            return

        for i in range(1, len(kintree)):
            parent_idx = int(kintree[i])
            if parent_idx < 0 or parent_idx >= len(joints):
                continue
            c0 = joints[parent_idx]
            c1 = joints[i]
            line = pv.Line(c0, c1)
            tube = line.tube(radius=0.001, n_sides=4)
            actor = self._skeleton_bone_actors[i - 1]
            actor.GetMapper().SetInputData(
                getattr(tube, "_vtk_obj", getattr(tube, "VTKObject", tube))
            )

        for i, center in enumerate(joints):
            sphere = pv.Sphere(
                radius=0.005,
                center=center,
                theta_resolution=6,
                phi_resolution=6,
            )
            actor = self._skeleton_joint_actors[i]
            actor.GetMapper().SetInputData(
                getattr(sphere, "_vtk_obj", getattr(sphere, "VTKObject", sphere))
            )

    def _set_skeleton_visibility(self, visible):
        self._skeleton_visible = bool(visible)
        for actor in self._skeleton_bone_actors:
            actor.SetVisibility(1 if self._skeleton_visible else 0)
        for actor in self._skeleton_joint_actors:
            actor.SetVisibility(1 if self._skeleton_visible else 0)
    
    def _on_beta_changed(self, beta):
        """Handle beta value changes from slider group"""
        self.beta = beta
        if self.sphere_manager:
            self.sphere_manager.rest_skeleton = None
        #measure_time(lambda:self.update_smplmodel(update_radius=True,update_pose=False),label="update_smplmodel")
        self.update_smplmodel(update_radius=True,update_pose=False)

    def _on_theta_changed(self, theta):
        """Handle theta value changes from slider group"""
        self.theta = theta
        self.update_smplmodel(update_radius=False,update_pose=True)
    
    def _on_opacity_changed(self, value):
        """Handle opacity slider changes"""
        opacity = value / 100.0
        self.opacity_label.setText(f"Spheremesh Opacity: {opacity:.2f}")
        if self.sphere_manager:
            self.sphere_manager.set_spheremesh_opacity(opacity)
            self.plotter.render()
    
    def _on_model_opacity_changed(self, value):
        """Handle model opacity slider changes"""
        opacity = value / 100.0
        self.model_opacity_label.setText(f"Model Opacity: {opacity:.2f}")
        if hasattr(self, 'mesh_actor') and self.mesh_actor is not None:
            self.mesh_actor.GetProperty().SetOpacity(opacity)
            self.plotter.render()

    def update_smplmodel(self, update_radius=True,update_pose=True):
        verts, joints, Rs, T = my_function_forward(self.smpl_model, self.beta, self.theta)
        rs_are_global = T is not None
        if Rs.shape[-1] == 4:
            Rs = Rs[..., :3, :3]
        # Update the mesh in the plotter
        verts_np = verts.detach().cpu().numpy()
        
        # Update mesh points in-place instead of recreating
        if hasattr(self, 'mesh_actor') and self.mesh_actor is not None:
            # Get the current mesh from the actor
            mesh = self.mesh_actor.GetMapper().GetInput()
            # Update the points
            mesh.points = verts_np
            # Mark as modified to trigger render update
            mesh.Modified()
            
            # Update skeleton in sphere manager
            self._update_skeleton(verts, joints, Rs, rs_are_global=rs_are_global)

            # CSV blendshapes when loaded; otherwise FK + mesh-based radii.
            if self.sphere_manager.has_beta_offsets_loaded():
                self.sphere_manager.apply_beta_offsets(self.beta)
                if update_pose:
                    self.sphere_manager.update_sphere_positions()
            else:
                #measure_time(lambda:self.sphere_manager.update_sphere_positions(),label="update_sphere_positions")
                self.sphere_manager.update_sphere_positions()
                if update_radius:
                    mesh_wrapped = pv.wrap(mesh)
                    #self.sphere_manager.update_sphere_radius(mesh_wrapped)

            self.plotter.render()

    def _get_json_load_path_from_dialog(self, title: str):
        """Return a JSON file path ready for sphere_manager loading (supports .txt via conversion)."""
        data_dir = str(Path(__file__).resolve().parent.parent / "data")
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            title,
            data_dir,
            "SphereMesh files (*.json *.txt);;JSON files (*.json);;Text files (*.txt);;All files (*)",
        )
        if not filepath:
            return None, None

        path = Path(filepath)
        tmp_path = None
        if path.suffix.lower() == ".txt":
            txt2json_path = Path(__file__).resolve().parent.parent / "scripts" / "txt2json.py"
            spec = importlib.util.spec_from_file_location("txt2json", txt2json_path)
            txt2json_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(txt2json_mod)

            text = path.read_text(encoding="utf-8")
            payload = txt2json_mod.convert_text_to_json_payload(text)

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, encoding="utf-8"
            ) as tmp:
                json.dump(payload, tmp)
                tmp_path = tmp.name

            return tmp_path, tmp_path
        return filepath, None

    def _post_load_refresh(self):
        mesh = None
        if hasattr(self, "mesh_actor") and self.mesh_actor is not None:
            mesh_vtk = self.mesh_actor.GetMapper().GetInput()
            if mesh_vtk is not None:
                mesh = pv.wrap(mesh_vtk)
        if mesh is not None:
            self.sphere_manager.initialize_radius_scaling_from_mesh(mesh)

    def load_topology_from_file(self):
        """Load topology from file and replace existing spheres/connections."""
        tmp_path = None
        try:
            load_path, tmp_path = self._get_json_load_path_from_dialog("Load Topology")
            if not load_path:
                return
            self.sphere_manager.load_from_json(load_path)
            self._post_load_refresh()
        except Exception as e:
            QMessageBox.critical(self, "Load Topology", f"Failed to load file:\n{e}")
        finally:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def import_spheremesh_from_file(self):
        """Import another SphereMesh from file and append it to the current one."""
        tmp_path = None
        try:
            load_path, tmp_path = self._get_json_load_path_from_dialog("Import SphereMesh")
            if not load_path:
                return
            self.sphere_manager.append_from_json(load_path)
            self._post_load_refresh()

        except Exception as e:
            QMessageBox.critical(self, "Import SphereMesh", f"Failed to import file:\n{e}")
        finally:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def import_spheres_only_from_file(self):
        """Import spheres from file without importing any connections."""
        tmp_path = None
        try:
            load_path, tmp_path = self._get_json_load_path_from_dialog(
                "Import SphereMesh (no connections)"
            )
            if not load_path:
                return
            self.sphere_manager.append_spheres_only_from_json(load_path)
            self._post_load_refresh()
        except Exception as e:
            QMessageBox.critical(
                self,
                "Import SphereMesh (no connections)",
                f"Failed to import file:\n{e}",
            )
        finally:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def save_file(self):
        # Save the mesh
        print("Save file method called")
        save_path = Path(__file__).resolve().parent.parent / "data" / "save1.json"
        self.sphere_manager.save_to_json(str(save_path))

    def load_beta_offsets_clicked(self):
        """Load all beta*.csv files from the fixed data/offsets directory."""
        data_dir = Path(__file__).resolve().parent.parent / "data" / "offsets"
        if self.sphere_manager.get_sphere_count() == 0:
            QMessageBox.warning(
                self,
                "Load Beta Offsets",
                "No spheres are loaded yet.\n"
                "Place and save spheres first, then load beta offsets.",
            )
            return
        if not data_dir.is_dir():
            QMessageBox.warning(
                self,
                "Load Beta Offsets",
                f"Offsets directory not found:\n{data_dir}",
            )
            return

        self.sphere_manager.load_beta_offsets_from_dir(str(data_dir))
        loaded = self.sphere_manager.offsets_pos  # Dict[int, ndarray]
        if loaded:
            indices = sorted(loaded.keys())
            QMessageBox.information(
                self,
                "Load Beta Offsets",
                f"Loaded {len(indices)} beta offset mode(s): {indices}\n"
                f"for {self.sphere_manager.get_sphere_count()} spheres.\n\n"
                "Use the Beta Controls sliders to deform the sphere configuration.",
            )
        else:
            QMessageBox.warning(
                self,
                "Load Beta Offsets",
                f"No valid beta*.csv files found in:\n{str(data_dir)}",
            )

    def export_smpl_fbx(self):
        """Export current SMPL model to FBX with skeleton and vertex weights."""
        if not hasattr(self, "smpl_model") or self.smpl_model is None:
            QMessageBox.warning(self, "Export FBX", "No SMPL model loaded.")
            return
        verts, joints, Rs, _ = my_function_forward(self.smpl_model, self.beta, self.theta)
        Rs = Rs[..., :3, :3]
        verts_np = verts.detach().cpu().numpy()
        joints_np = joints.detach().cpu().numpy()
        faces = np.asarray(self.smpl_model.faces, dtype=np.int32)
        parents = np.asarray(self.smpl_model.parents, dtype=np.int32)
        joint_names = [JOINT_INDEX_TO_LABEL[i] for i in range(24)]
        weights_np = self.smpl_model.weights.cpu().numpy()

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export SMPL as FBX", "", "FBX (*.fbx)"
        )
        if not filepath:
            return
        success, message = export_smpl_to_fbx(
            filepath,
            verts_np,
            faces,
            joints_np,
            parents,
            joint_names,
            weights_np,
        )
        if success:
            QMessageBox.information(self, "Export FBX", message)
        else:
            QMessageBox.critical(self, "Export FBX", message)

    def export_smpl_obj(self):
        """Export current SMPL model mesh as OBJ (vertices and faces only)."""
        if not hasattr(self, "smpl_model") or self.smpl_model is None:
            QMessageBox.warning(self, "Export OBJ", "No SMPL model loaded.")
            return
        verts, _, Rs, _ = my_function_forward(self.smpl_model, self.beta, self.theta)
        Rs = Rs[..., :3, :3]
        verts_np = verts.detach().cpu().numpy()
        faces = np.asarray(self.smpl_model.faces, dtype=np.int32)

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export SMPL as OBJ", "", "OBJ (*.obj)"
        )
        if not filepath:
            return
        try:
            with open(filepath, "w") as f:
                for v in verts_np:
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                for face in faces:
                    # OBJ uses 1-based indices
                    f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")
            QMessageBox.information(
                self, "Export OBJ", f"Saved SMPL mesh to:\n{filepath}"
            )
        except OSError as e:
            QMessageBox.critical(
                self, "Export OBJ", f"Failed to write file:\n{e}"
            )

    def add_sphere_clicked(self):
        if self.sphere_manager:
            self.sphere_manager.add_sphere()
            self.info_panel.update_selected_info()
            self.plotter.render()

    def _sync_selected_to_joint_toolbar(self):
        if not hasattr(self, "selected_to_joint_button"):
            return
        sm = self.sphere_manager
        ok = (
            sm is not None
            and sm.get_sphere_count() > 0
            and sm.get_selected_sphere_index() is not None
        )
        self.selected_to_joint_button.setEnabled(ok)

    def move_selected_to_closest_joint_clicked(self):
        """Toolbar: snap selected sphere center to closest SMPL joint (current pose)."""
        if not self.sphere_manager:
            return
        data = self.sphere_manager.get_selected_sphere_display_data()
        if not data:
            return
        self._move_sphere_to_closest_joint(data["selected_local_index"])

    def reassign_all_parent_joints_clicked(self):
        """Toolbar: recompute parent joints/rest offsets for all spheres."""
        if not self.sphere_manager:
            return
        self.sphere_manager.reassign_parent_joints_all()
        self.info_panel.update_selected_info()
        self.plotter.render()

    def check_x_symmetry_clicked(self):
        """Validate sphere/connectivity symmetry across X axis and show a report."""
        if not self.sphere_manager:
            return
        result = self.sphere_manager.check_x_symmetry(pos_tol=0.02, radius_tol=1e-4)
        issues = result["issues"]
        if result["ok"]:
            QMessageBox.information(
                self,
                "X Symmetry Check",
                "Configuration is symmetric across X axis within tolerance.",
            )
            return

        max_lines = 20
        shown = issues[:max_lines]
        extra = len(issues) - len(shown)
        text = "\n".join(f"- {line}" for line in shown)
        if extra > 0:
            text += f"\n- ... and {extra} more issue(s)"
        QMessageBox.warning(
            self,
            "X Symmetry Check",
            f"Found {len(issues)} symmetry issue(s):\n\n{text}",
        )

    def add_connection_clicked(self):
        if not self.sphere_manager:
            return
        n = self.sphere_manager.get_sphere_count()
        if n < 2:
            QMessageBox.information(self, "Connection", "Add at least two spheres first.")
            return
        dlg = AddConnectionDialog(self, n)
        if not dlg.exec():
            return
        indices = dlg.get_indices()
        if len(set(indices)) != len(indices):
            QMessageBox.warning(self, "Connection", "Sphere indices must be distinct.")
            return
        if not self.sphere_manager.try_add_connection(tuple(indices)):
            QMessageBox.warning(
                self,
                "Connection",
                "Could not create connection (invalid geometry or duplicate indices).",
            )
        self.info_panel.update_selected_info()
        self.plotter.render()

    def remove_connection_clicked(self):
        if not self.sphere_manager:
            return
        if not self.sphere_manager.connections:
            QMessageBox.information(self, "Remove Connection", "No connections to remove.")
            return
        conns = [tuple(c.sphere_indices) for c in self.sphere_manager.connections]
        dlg = RemoveConnectionDialog(self, conns)
        if not dlg.exec():
            return
        selected = dlg.get_selected_connection()
        if selected is None:
            QMessageBox.information(self, "Remove Connection", "No connection selected.")
            return
        conn_obj = None
        for c in self.sphere_manager.connections:
            if tuple(c.sphere_indices) == tuple(selected):
                conn_obj = c
                break
        if conn_obj is None or not self.sphere_manager.delete_connection(conn_obj):
            QMessageBox.warning(self, "Remove Connection", "Could not remove selected connection.")
            return
        self.info_panel.update_selected_info()
        self.plotter.render()

    def _handle_grow_clicked(self, sphere_index):
        """Called when InfoPanel Grow button is clicked for a sphere; passes SMPL mesh and sphere index to manager."""
        if not self.sphere_manager:
            return
        mesh = None
        if hasattr(self, 'mesh_actor') and self.mesh_actor is not None:
            mesh_vtk = self.mesh_actor.GetMapper().GetInput()
            if mesh_vtk is not None:
                mesh = pv.wrap(mesh_vtk)
        self.sphere_manager.grow_sphere(mesh, sphere_index)
        self.plotter.render()

    def _move_sphere_to_closest_joint(self, sphere_index):
        """Move the given sphere in the selected group to the closest SMPL joint position."""
        if not self.sphere_manager or not hasattr(self, "smpl_model") or self.smpl_model is None:
            return
        data = self.sphere_manager.get_selected_sphere_display_data()
        if not data or sphere_index >= data["num_slots"]:
            return
        _, joints, Rs, _ = my_function_forward(self.smpl_model, self.beta, self.theta)
        Rs = Rs[..., :3, :3]
        joints_np = joints.detach().cpu().numpy()
        gidx = data["endpoint_indices"][sphere_index]
        center = np.array(data["centers"][sphere_index], dtype=np.float64)
        distances = np.linalg.norm(joints_np - center, axis=1)
        j = int(np.argmin(distances))
        joint_pos = joints_np[j].tolist()
        self.sphere_manager.set_sphere_center_global(gidx, joint_pos)
        self.info_panel.update_selected_info()
        self.plotter.render()

    def toggle_smpl_model(self, state):
        """Toggle SMPL model visibility"""
        if hasattr(self, 'mesh_actor') and self.mesh_actor is not None:
            if state == Qt.CheckState.Checked.value:
                # Show the mesh
                self.mesh_actor.SetVisibility(True)
            else:
                # Hide the mesh
                self.mesh_actor.SetVisibility(False)
            
            self.plotter.render()
    
    def toggle_spheremeshes(self, state):
        """Toggle spheremesh rendering"""
        if not self.sphere_manager:
            return
        visible = state == Qt.CheckState.Checked.value
        self.sphere_manager.set_spheremesh_visibility(visible)
        self.plotter.render()

    def toggle_heatmap(self, state):
        """Toggle per-vertex heatmap on SMPL mesh (triggers sphere_manager)."""
        if not self.sphere_manager:
            return
        visible = state == Qt.CheckState.Checked.value
        self._on_show_heatmap(visible)
        self.plotter.render()

    def toggle_sphere_widgets(self, state):
        """Toggle visibility of sphere widgets (draggable spheres)."""
        if not self.sphere_manager:
            return
        visible = state == Qt.CheckState.Checked.value
        self.sphere_manager.set_sphere_widget_visibility(visible)
        self.plotter.render()

    def toggle_mirror_edits(self, state):
        """Toggle mirroring of manual sphere edits to X-axis twin."""
        if not self.sphere_manager:
            return
        enabled = state == Qt.CheckState.Checked.value
        self.sphere_manager.set_mirror_edits_enabled(enabled)

    def toggle_skeleton(self, state):
        """Toggle SMPL skeleton visibility (bones + joints)."""
        visible = state == Qt.CheckState.Checked.value
        if visible:
            self._skeleton_visible = True
            self._rebuild_skeleton_actors()
        else:
            self._set_skeleton_visibility(False)
        self.plotter.render()

    def _inflate_radius_value(self):
        return self.inflate_slider.value() / 100.0

    def _on_inflate_value_changed(self, value):
        scale = value / 100.0
        self.inflate_label.setText(f"Inflate scale: {scale:.2f}x")
        if not self.sphere_manager:
            return
        if self.checkbox_inflate.isChecked():
            self.sphere_manager.set_uniform_radius_override(True, scale, mode="scale")
            self.plotter.render()

    def toggle_inflate(self, state):
        """Temporarily scale all sphere radii by inflate factor; restore on uncheck."""
        if not self.sphere_manager:
            return
        active = state == Qt.CheckState.Checked.value
        if active and self.checkbox_uniform_radius_preview.isChecked():
            self.checkbox_uniform_radius_preview.blockSignals(True)
            self.checkbox_uniform_radius_preview.setChecked(False)
            self.checkbox_uniform_radius_preview.blockSignals(False)
        self.sphere_manager.set_uniform_radius_override(
            active, self._inflate_radius_value(), mode="scale"
        )
        self.plotter.render()

    def toggle_uniform_radius_preview(self, state):
        """Temporarily force all sphere radii to 0.02; restore on uncheck."""
        if not self.sphere_manager:
            return
        active = state == Qt.CheckState.Checked.value
        if active and self.checkbox_inflate.isChecked():
            self.checkbox_inflate.blockSignals(True)
            self.checkbox_inflate.setChecked(False)
            self.checkbox_inflate.blockSignals(False)
        self.sphere_manager.set_uniform_radius_override(active, 0.02, mode="absolute")
        self.plotter.render()

    def _on_show_heatmap(self, visible):
        """Callback from sphere_manager: color SMPL mesh per-vertex and show/remove legend."""
        if not hasattr(self, 'mesh_actor') or self.mesh_actor is None:
            return
        mesh_vtk = self.mesh_actor.GetMapper().GetInput()
        mesh = pv.wrap(mesh_vtk)
        mapper = self.mesh_actor.GetMapper()
        heatmap_title = "heatmap"
        if visible:
            heat = self.sphere_manager.get_heatmap(mesh)

            mesh["heatmap"] = heat

            # Cut off top percentile: 95 = top 5% use "above range" (gray) color
            percentile_cutoff =55.0
            vmin = float(np.min(heat))
            vmax = float(np.percentile(heat, percentile_cutoff))

       

            # Build lookup table with 0 = green (diverging: red -> green -> blue)
            t_zero = (0.0 - vmin) / (vmax - vmin) if vmax != vmin else 0.5
            print(t_zero)
            t_zero = float(np.clip(t_zero, 0.0, 1.0))
            cmap_custom = LinearSegmentedColormap.from_list(
                'green_at_zero',
                [(0.0, (1, 0, 0)), (t_zero, (0, 1, 0)), (1.0, (0, 0, 1))],  # red, green at 0, blue
                N=256,
            )
            lut = pv.LookupTable(cmap=cmap_custom, scalar_range=(vmin, vmax))
            lut.SetUseAboveRangeColor(True)
            lut.SetAboveRangeColor(0.5, 0.5, 0.5, 1.0)
            mapper.SetScalarRange(vmin, vmax)
            mapper.SetLookupTable(lut)

            
            mesh.Modified()
            mapper.SetInputData(mesh_vtk)
            mapper.SetScalarModeToUsePointData()
            mapper.SelectColorArray("heatmap")
            mapper.ScalarVisibilityOn()
            
            # Use plotter's remove_scalar_bar so internal state is cleared (fixes re-add on 3rd click)
            if heatmap_title in self.plotter.scalar_bars:
                self.plotter.remove_scalar_bar(title=heatmap_title, render=False)
            self.heatmap_scalar_bar_actor = self.plotter.add_scalar_bar(
                title=heatmap_title,
                mapper=mapper,
                position_x=0.02,
                position_y=0.5,
                height=0.5,
                above_label='Above cutoff',
                vertical=True,
                n_labels=5,
            )
            self.plotter.update_scalar_bar_range([vmin, vmax])
        else:
            if heatmap_title in self.plotter.scalar_bars:
                self.plotter.remove_scalar_bar(title=heatmap_title, render=False)
            self.heatmap_scalar_bar_actor = None
            mapper.ScalarVisibilityOff()
        self.plotter.render()

    def toggle_grid(self):
        # Toggle grid visibility
        pass

    def keyPressEvent(self, event: QKeyEvent):
        """Handle keyboard events for sphere navigation"""
        if event.key() == Qt.Key.Key_Q:
            self.select_previous_sphere()
        elif event.key() == Qt.Key.Key_E:
            self.select_next_sphere()
        else:
            super().keyPressEvent(event)
    
    def select_previous_sphere(self):
        if not self.sphere_manager or self.sphere_manager.get_sphere_count() == 0:
            return
        self.sphere_manager.select_previous_sphere()
        self.info_panel.update_selected_info()
        self.plotter.render()

    def select_next_sphere(self):
        if not self.sphere_manager or self.sphere_manager.get_sphere_count() == 0:
            return
        self.sphere_manager.select_next_sphere()
        self.info_panel.update_selected_info()
        self.plotter.render()

