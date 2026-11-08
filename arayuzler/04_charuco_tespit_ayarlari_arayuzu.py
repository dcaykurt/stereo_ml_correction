"""
ChArUco / AprilTag Tespit Ayarlari Arayuzu
==========================================
CLAHE, blur ve ArUco dedektor parametrelerini canli ayarlar.
Stereo (ChArUco / AprilTag) ve 3. kamera (AprilTag) icin ayri profiller;
config.py'ye kaydedilir (veri_toplama_arayuzu ve capture scriptleri okur).

Kullanim:
    python charuco_tespit_ayarlari_arayuzu.py
"""

import importlib
import sys

import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QSlider, QGroupBox, QGridLayout, QDoubleSpinBox, QScrollArea, QComboBox,
    QPushButton, QMessageBox, QCheckBox,
)
from PySide6.QtCore import QThread, Signal, Qt, QTimer
from PySide6.QtGui import QImage, QPixmap

import config
from config import (
    LEFT_CAM_INDEX, RIGHT_CAM_INDEX, THIRD_CAM_INDEX,
    FLIP_LEFT_CAM, FLIP_RIGHT_CAM,
    ARUCO_DICT_TYPE, SQUARES_X, SQUARES_Y, SQUARE_LENGTH, MARKER_LENGTH,
    APRILTAG_DICT, get_detect_tuning, apply_detector_parameters, preprocess_gray,
)
from camera_utils import open_cameras, open_third_camera
from detect_tuning_io import persist_detect_tuning


def tuning_from_worker(w):
    return {
        'clahe_enabled': w.clahe_enabled,
        'clahe_clip': w.clahe_clip,
        'clahe_grid': w.clahe_grid,
        'blur_enabled': w.blur_enabled,
        'blur_k': w.blur_k,
        'thresh_win_min': w.thresh_win_min,
        'thresh_win_max': w.thresh_win_max,
        'thresh_win_step': w.thresh_win_step,
        'thresh_constant': w.thresh_constant,
        'min_perimeter': w.min_perimeter,
        'max_perimeter': w.max_perimeter,
        'poly_accuracy': w.poly_accuracy,
        'min_corner_dist': w.min_corner_dist,
        'min_corners': w.min_corners,
    }


class CharucoTespitAyarlariWorker(QThread):
    """camera_mode: 'stereo' | 'stereo_apriltag' | 'third_cam'"""
    frame_ready = Signal(np.ndarray, object, int, int)

    def __init__(self, camera_mode='stereo'):
        super().__init__()
        self.camera_mode = camera_mode
        self.running = False
        tuning = get_detect_tuning(
            'third_cam' if camera_mode == 'third_cam' else 'stereo'
        )
        self._load_tuning(tuning)

    def _load_tuning(self, tuning):
        self.clahe_enabled = tuning.get('clahe_enabled', True)
        self.clahe_clip = tuning.get('clahe_clip', 0.5)
        self.clahe_grid = tuning.get('clahe_grid', 22)
        self.blur_enabled = tuning.get('blur_enabled', True)
        self.blur_k = tuning.get('blur_k', 1)
        self.thresh_win_min = tuning.get('thresh_win_min', 46)
        self.thresh_win_max = tuning.get('thresh_win_max', 135)
        self.thresh_win_step = tuning.get('thresh_win_step', 10)
        self.thresh_constant = tuning.get('thresh_constant', 2)
        self.min_perimeter = tuning.get('min_perimeter', 0.01)
        self.max_perimeter = tuning.get('max_perimeter', 6.0)
        self.poly_accuracy = tuning.get('poly_accuracy', 0.12)
        self.min_corner_dist = tuning.get('min_corner_dist', 0.08)
        self.min_corners = tuning.get('min_corners', 12)

    def _tuning_dict(self):
        return tuning_from_worker(self)

    def _preprocess(self, gray):
        return preprocess_gray(gray, self._tuning_dict())

    def _build_charuco_detector(self):
        dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
        board = cv2.aruco.CharucoBoard(
            (SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, dictionary
        )
        params = cv2.aruco.DetectorParameters()
        apply_detector_parameters(params, self._tuning_dict())
        charuco_params = cv2.aruco.CharucoParameters()
        return cv2.aruco.CharucoDetector(board, charuco_params, params)

    def _build_apriltag_detector(self):
        dictionary = cv2.aruco.getPredefinedDictionary(APRILTAG_DICT)
        params = cv2.aruco.DetectorParameters()
        apply_detector_parameters(params, self._tuning_dict())
        return cv2.aruco.ArucoDetector(dictionary, params)

    def run(self):
        capl = capr = cap3 = None
        try:
            if self.camera_mode == 'third_cam':
                cap3 = open_third_camera(THIRD_CAM_INDEX)
                if not cap3.isOpened():
                    self.frame_ready.emit(
                        np.zeros((270, 540, 3), dtype=np.uint8), None, 0, 0
                    )
                    return
            else:
                capl, capr = open_cameras(LEFT_CAM_INDEX, RIGHT_CAM_INDEX)
                if not capl.isOpened() or not capr.isOpened():
                    blank = np.zeros((270, 540, 3), dtype=np.uint8)
                    self.frame_ready.emit(blank, blank, 0, 0)
                    return

            self.running = True
            while self.running:
                if self.camera_mode == 'third_cam':
                    ok, f3 = cap3.read()
                    if not ok or f3 is None:
                        continue
                    detector = self._build_apriltag_detector()
                    gray = self._preprocess(cv2.cvtColor(f3, cv2.COLOR_BGR2GRAY))
                    corners, ids, _ = detector.detectMarkers(gray)
                    n = len(ids) if ids is not None else 0
                    show = f3.copy()
                    if corners is not None and n > 0:
                        cv2.aruco.drawDetectedMarkers(show, corners, ids)
                    show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
                    self.frame_ready.emit(show, None, n, 0)

                elif self.camera_mode == 'stereo_apriltag':
                    r1, r2 = capl.grab(), capr.grab()
                    if not (r1 and r2):
                        continue
                    _, fl = capl.retrieve()
                    _, fr = capr.retrieve()
                    if FLIP_LEFT_CAM:
                        fl = cv2.flip(fl, -1)
                    if FLIP_RIGHT_CAM:
                        fr = cv2.flip(fr, -1)
                    detector = self._build_apriltag_detector()
                    gray_l = self._preprocess(cv2.cvtColor(fl, cv2.COLOR_BGR2GRAY))
                    gray_r = self._preprocess(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
                    corners_l, ids_l, _ = detector.detectMarkers(gray_l)
                    corners_r, ids_r, _ = detector.detectMarkers(gray_r)
                    n_l = len(ids_l) if ids_l is not None else 0
                    n_r = len(ids_r) if ids_r is not None else 0
                    show_l = fl.copy()
                    show_r = fr.copy()
                    if corners_l is not None and n_l > 0:
                        cv2.aruco.drawDetectedMarkers(show_l, corners_l, ids_l)
                    if corners_r is not None and n_r > 0:
                        cv2.aruco.drawDetectedMarkers(show_r, corners_r, ids_r)
                    show_l = cv2.cvtColor(show_l, cv2.COLOR_BGR2RGB)
                    show_r = cv2.cvtColor(show_r, cv2.COLOR_BGR2RGB)
                    self.frame_ready.emit(show_l, show_r, n_l, n_r)

                else:
                    r1, r2 = capl.grab(), capr.grab()
                    if not (r1 and r2):
                        continue
                    _, fl = capl.retrieve()
                    _, fr = capr.retrieve()
                    if FLIP_LEFT_CAM:
                        fl = cv2.flip(fl, -1)
                    if FLIP_RIGHT_CAM:
                        fr = cv2.flip(fr, -1)

                    detector = self._build_charuco_detector()
                    gray_l = self._preprocess(cv2.cvtColor(fl, cv2.COLOR_BGR2GRAY))
                    gray_r = self._preprocess(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
                    corners_l, ids_l, _, _ = detector.detectBoard(gray_l)
                    corners_r, ids_r, _, _ = detector.detectBoard(gray_r)
                    n_l = len(corners_l) if corners_l is not None else 0
                    n_r = len(corners_r) if corners_r is not None else 0

                    show_l = fl.copy()
                    show_r = fr.copy()
                    if corners_l is not None and n_l > 0:
                        cv2.aruco.drawDetectedCornersCharuco(show_l, corners_l, ids_l)
                    if corners_r is not None and n_r > 0:
                        cv2.aruco.drawDetectedCornersCharuco(show_r, corners_r, ids_r)
                    show_l = cv2.cvtColor(show_l, cv2.COLOR_BGR2RGB)
                    show_r = cv2.cvtColor(show_r, cv2.COLOR_BGR2RGB)
                    self.frame_ready.emit(show_l, show_r, n_l, n_r)
        finally:
            for cap in (capl, capr, cap3):
                if cap is not None and cap.isOpened():
                    cap.release()

    def stop(self):
        self.running = False
        self.wait()


class CharucoTespitAyarlariArayuzu(QMainWindow):
  PROFILE_STEREO = 'stereo'
  PROFILE_THIRD = 'third_cam'

  def __init__(self):
    super().__init__()
    self.setWindowTitle("ChArUco / AprilTag Tespit Ayarlari")
    self.setMinimumWidth(1100)

    self._profile = self.PROFILE_STEREO
    self._saved_tuning = {
      self.PROFILE_STEREO: get_detect_tuning('stereo'),
      self.PROFILE_THIRD: get_detect_tuning('third_cam'),
    }

    main = QWidget()
    self.setCentralWidget(main)
    top_layout = QHBoxLayout(main)

    cam_panel = QVBoxLayout()
    self.cmb_camera = QComboBox()
    self.cmb_camera.addItem("Stereo (ChArUco)", "stereo")
    self.cmb_camera.addItem("Stereo (AprilTag)", "stereo_apriltag")
    self.cmb_camera.addItem("3. Kamera (AprilTag)", self.PROFILE_THIRD)
    self.cmb_camera.currentIndexChanged.connect(self.on_camera_changed)

    bar = QHBoxLayout()
    bar.addWidget(QLabel("<b>Kamera:</b>"))
    bar.addWidget(self.cmb_camera, 1)
    self.btn_save = QPushButton("Ayarlari config.py'ye Kaydet")
    self.btn_save.setStyleSheet("font-weight:bold; padding:6px;")
    self.btn_save.clicked.connect(self.save_to_config)
    bar.addWidget(self.btn_save)
    cam_panel.addLayout(bar)

    self.lbl_l = QLabel()
    self.lbl_l.setFixedSize(540, 304)
    self.lbl_r = QLabel()
    self.lbl_r.setFixedSize(540, 304)
    self.lbl_cam_title_l = QLabel("<b>Sol Kamera</b>")
    self.lbl_cam_title_r = QLabel("<b>Sag Kamera</b>")
    self.lbl_info = QLabel("Bekleniyor...")
    self.lbl_info.setStyleSheet(
      "font-size:15px; font-weight:bold; padding:6px; background:#222; color:#0f0;"
    )
    cam_panel.addWidget(self.lbl_cam_title_l)
    cam_panel.addWidget(self.lbl_l)
    cam_panel.addWidget(self.lbl_cam_title_r)
    cam_panel.addWidget(self.lbl_r)
    cam_panel.addWidget(self.lbl_info)
    top_layout.addLayout(cam_panel)

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFixedWidth(420)
    scroll_widget = QWidget()
    param_layout = QVBoxLayout(scroll_widget)

    g_pre = QGroupBox("On-Isleme (CLAHE + Blur)")
    gl_pre = QGridLayout(g_pre)
    self.cb_clahe = QCheckBox("CLAHE acik")
    self.cb_clahe.setChecked(True)
    gl_pre.addWidget(self.cb_clahe, 0, 0, 1, 3)
    self.sl_clahe_clip = self._add_dslider(gl_pre, 1, "CLAHE clipLimit", 0.5, 10.0, 0.5, 0.5)
    self.sl_clahe_grid = self._add_slider(gl_pre, 2, "CLAHE gridSize", 2, 32, 22)
    self.cb_blur = QCheckBox("Gaussian blur acik")
    self.cb_blur.setChecked(True)
    gl_pre.addWidget(self.cb_blur, 3, 0, 1, 3)
    self.sl_blur_k = self._add_slider(gl_pre, 4, "Blur kernel", 1, 15, 1)
    param_layout.addWidget(g_pre)

    g_det = QGroupBox("ArUco Dedektor Parametreleri")
    gl_det = QGridLayout(g_det)
    self.sl_tw_min = self._add_slider(gl_det, 0, "threshWinMin", 3, 53, 46)
    self.sl_tw_max = self._add_slider(gl_det, 1, "threshWinMax", 3, 153, 135)
    self.sl_tw_step = self._add_slider(gl_det, 2, "threshWinStep", 2, 30, 10)
    self.sl_th_const = self._add_slider(gl_det, 3, "threshConstant", 1, 30, 2)
    self.sl_min_per = self._add_dslider(gl_det, 4, "minPerimeterRate", 0.01, 0.2, 0.01, 0.01)
    self.sl_max_per = self._add_dslider(gl_det, 5, "maxPerimeterRate", 1.0, 10.0, 6.0, 0.5)
    self.sl_poly = self._add_dslider(gl_det, 6, "polyApproxAccuracy", 0.01, 0.2, 0.12, 0.01)
    self.sl_corner_dist = self._add_dslider(gl_det, 7, "minCornerDistRate", 0.01, 0.3, 0.08, 0.01)
    param_layout.addWidget(g_det)

    g_misc = QGroupBox("Diger")
    gl_misc = QGridLayout(g_misc)
    self.lbl_min_corners = QLabel("MIN_CORNERS (yalnizca stereo ChArUco)")
    gl_misc.addWidget(self.lbl_min_corners, 0, 0, 1, 3)
    self.sl_min_corners = self._add_slider(gl_misc, 1, "MIN_CORNERS", 1, 24, 12)
    param_layout.addWidget(g_misc)

    param_layout.addStretch()
    scroll.setWidget(scroll_widget)
    top_layout.addWidget(scroll)

    self._apply_tuning_to_ui(self._saved_tuning[self._profile])
    self._start_worker()

    self.timer = QTimer()
    self.timer.timeout.connect(self.sync_params)
    self.timer.start(200)

  def _camera_mode(self):
    return self.cmb_camera.currentData()

  def _tuning_profile(self):
    """config.py'deki profil adi (kayit icin)."""
    return self.PROFILE_THIRD if self._camera_mode() == self.PROFILE_THIRD else self.PROFILE_STEREO

  def _start_worker(self):
    if hasattr(self, 'worker') and self.worker is not None:
      self.worker.stop()
    self.worker = CharucoTespitAyarlariWorker(self._camera_mode())
    self.worker.frame_ready.connect(self.update_frames)
    self.worker.start()

  def on_camera_changed(self):
    self._stash_current_profile()
    self._profile = self._tuning_profile()
    self._apply_tuning_to_ui(self._saved_tuning[self._profile])
    is_third = self._camera_mode() == self.PROFILE_THIRD
    is_stereo_apriltag = self._camera_mode() == 'stereo_apriltag'
    self.lbl_cam_title_l.setText(
      "<b>3. Kamera</b>" if is_third else "<b>Sol Kamera</b>"
    )
    self.lbl_cam_title_r.setVisible(not is_third)
    self.lbl_r.setVisible(not is_third)
    self.lbl_min_corners.setEnabled(not is_third and not is_stereo_apriltag)
    self.sl_min_corners.setEnabled(not is_third and not is_stereo_apriltag)
    self._start_worker()

  def _stash_current_profile(self):
    if hasattr(self, 'worker') and self.worker is not None:
      self.sync_params()
      key = self._tuning_profile() if hasattr(self, '_profile') else self.PROFILE_STEREO
      self._saved_tuning[key] = tuning_from_worker(self.worker)

  def _apply_tuning_to_ui(self, t):
    self.cb_clahe.setChecked(t.get('clahe_enabled', True))
    self.sl_clahe_clip.setValue(t.get('clahe_clip', 0.5))
    self.sl_clahe_grid.setValue(t.get('clahe_grid', 22))
    self.cb_blur.setChecked(t.get('blur_enabled', True))
    self.sl_blur_k.setValue(t.get('blur_k', 1))
    self.sl_tw_min.setValue(t.get('thresh_win_min', 46))
    self.sl_tw_max.setValue(t.get('thresh_win_max', 135))
    self.sl_tw_step.setValue(t.get('thresh_win_step', 10))
    self.sl_th_const.setValue(t.get('thresh_constant', 2))
    self.sl_min_per.setValue(t.get('min_perimeter', 0.01))
    self.sl_max_per.setValue(t.get('max_perimeter', 6.0))
    self.sl_poly.setValue(t.get('poly_accuracy', 0.12))
    self.sl_corner_dist.setValue(t.get('min_corner_dist', 0.08))
    self.sl_min_corners.setValue(t.get('min_corners', 12))

  def save_to_config(self):
    self._stash_current_profile()
    profile = self._tuning_profile()
    try:
      persist_detect_tuning(profile, self._saved_tuning[profile])
      importlib.reload(config)
      QMessageBox.information(
        self, "Kaydedildi",
        f"'{profile}' ayarlari config.py dosyasina yazildi.\n"
        "Diger uygulamalar bir sonraki calistirmada bu degerleri kullanir.",
      )
    except Exception as e:
      QMessageBox.critical(self, "Kayit Hatasi", str(e))

  def _add_slider(self, grid, row, label, vmin, vmax, vdefault):
    grid.addWidget(QLabel(label), row, 0)
    sl = QSlider(Qt.Horizontal)
    sl.setRange(vmin, vmax)
    sl.setValue(vdefault)
    lbl = QLabel(str(vdefault))
    sl.valueChanged.connect(lambda v, l=lbl: l.setText(str(v)))
    grid.addWidget(sl, row, 1)
    grid.addWidget(lbl, row, 2)
    return sl

  def _add_dslider(self, grid, row, label, vmin, vmax, vdefault, step):
    grid.addWidget(QLabel(label), row, 0)
    sp = QDoubleSpinBox()
    sp.setRange(vmin, vmax)
    sp.setValue(vdefault)
    sp.setSingleStep(step)
    grid.addWidget(sp, row, 1, 1, 2)
    return sp

  def sync_params(self):
    if not hasattr(self, 'worker') or self.worker is None:
      return
    w = self.worker
    w.clahe_enabled = self.cb_clahe.isChecked()
    w.clahe_clip = self.sl_clahe_clip.value()
    w.clahe_grid = max(2, self.sl_clahe_grid.value())
    w.blur_enabled = self.cb_blur.isChecked()
    w.blur_k = self.sl_blur_k.value()
    w.thresh_win_min = self.sl_tw_min.value()
    w.thresh_win_max = self.sl_tw_max.value()
    w.thresh_win_step = self.sl_tw_step.value()
    w.thresh_constant = self.sl_th_const.value()
    w.min_perimeter = self.sl_min_per.value()
    w.max_perimeter = self.sl_max_per.value()
    w.poly_accuracy = self.sl_poly.value()
    w.min_corner_dist = self.sl_corner_dist.value()
    w.min_corners = self.sl_min_corners.value()

  def cv2qt(self, img, w, h):
    ih, iw, ch = img.shape
    qimg = QImage(img.data, iw, ih, ch * iw, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.scaled(w, h, Qt.KeepAspectRatio))

  def update_frames(self, show_l, show_r, n_l, n_r):
    if self._camera_mode() == self.PROFILE_THIRD:
      self.lbl_l.setPixmap(self.cv2qt(show_l, 540, 304))
      mc = max(1, self.worker.min_corners)
      ok = n_l >= 1
      color = "#0f0" if ok else "#f00"
      self.lbl_info.setText(f"AprilTag: {n_l} adet  |  Min: 1")
      self.lbl_info.setStyleSheet(
        f"font-size:15px; font-weight:bold; padding:6px; background:#222; color:{color};"
      )
      return

    self.lbl_l.setPixmap(self.cv2qt(show_l, 540, 304))
    self.lbl_r.setPixmap(self.cv2qt(show_r, 540, 304))
    if self._camera_mode() == 'stereo_apriltag':
      color = "#0f0" if (n_l > 0 or n_r > 0) else "#f00"
      self.lbl_info.setText(f"Sol: {n_l} tag  |  Sag: {n_r} tag")
      self.lbl_info.setStyleSheet(
        f"font-size:15px; font-weight:bold; padding:6px; background:#222; color:{color};"
      )
      return

    mc = self.worker.min_corners
    ok_l = "OK" if n_l >= mc else "X"
    ok_r = "OK" if n_r >= mc else "X"
    color = "#0f0" if (n_l >= mc and n_r >= mc) else "#f00"
    self.lbl_info.setText(f"Sol: {n_l} kose {ok_l}  |  Sag: {n_r} kose {ok_r}  |  Min: {mc}")
    self.lbl_info.setStyleSheet(
      f"font-size:15px; font-weight:bold; padding:6px; background:#222; color:{color};"
    )

  def closeEvent(self, event):
    if hasattr(self, 'worker') and self.worker is not None:
      self.worker.stop()
    event.accept()


if __name__ == '__main__':
  app = QApplication(sys.argv)
  w = CharucoTespitAyarlariArayuzu()
  w.show()
  sys.exit(app.exec())
